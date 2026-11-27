"""Mirror the evaluator's public state into Weights & Biases (humeval-wandb.timer, every 10 minutes).

Everything logged comes from the evaluator's PUBLIC endpoints (/v1/info, /v1/periods, /v1/queue,
/v1/results, /v1/leaderboard), the same data any miner can read, so nothing private can leak through it: no hidden
input, no secret, no balance or spending. One run per period (``evaluator-<period>``), resumed on every tick.

    python scripts/wandb_report.py --api http://127.0.0.1:8032 [--project subnet32-humanizer] [--entity itsai-dev]

Logged each tick:
* period: id, inputs, the public commitment and Merkle root (fixed before anyone is evaluated);
* intake: open or closed, queue waiting/running, evaluations in the last 24 h, the price card, the quote;
* crown: the current king, the number of paid kings, the signed leaderboard's sequence and age;
* results: every finished submission of the period (S, crowned, quality and detector aggregates) as a table,
  and each new one as a point (result/S, result/crowned and every other public number);
* events, one line each in the run's Logs tab: a submission was queued (UID, hotkey, name), started, left
  the queue without a result (failed or cancelled), finished (its whole public result row); the king changed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

from humanizer_eval.paths import var_dir


def get(http: httpx.Client, path: str) -> dict:
    r = http.get(path)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def snapshot(api: str) -> dict:
    with httpx.Client(base_url=api.rstrip("/"), timeout=30) as http:
        info = get(http, "/v1/info")
        periods = get(http, "/v1/periods")
        results = get(http, "/v1/results")
        queue = get(http, "/v1/queue")
        board = get(http, "/v1/leaderboard")
    payload = json.loads(board["payload"]) if board.get("payload") else {}
    return {"info": info, "periods": periods, "results": results, "board": payload, "queue": queue}


def metrics(s: dict, now: float) -> dict:
    info, board = s["info"], s["board"]
    pid = info.get("period_id")
    per = (s["periods"].get("periods") or {}).get(pid) or {}
    limits, queue = info.get("limits") or {}, info.get("queue") or {}
    king = board.get("king") or {}
    out = {
        "period/n_inputs": info.get("n_inputs"),
        "intake/open": int(bool(limits.get("submissions_open"))),
        "intake/queue_waiting": queue.get("waiting"),
        "intake/queue_running": queue.get("running"),
        "intake/evals_last_24h": limits.get("evals_last_24h"),
        "intake/quote_usdc": float(info.get("quote_total_usdc") or 0),
        "crown/n_kings": len(board.get("kings") or []),
        "crown/king_delta_s": king.get("delta_s"),
        "leaderboard/seq": board.get("seq"),
        "leaderboard/age_s": round(now - float(board["issued_at"]), 1) if board.get("issued_at") else None,
        "results/n_finished": len((s["results"] or {}).get("results") or []),
    }
    text = {
        "period/id": pid,
        "period/commitment": per.get("commitment"),
        "period/merkle_root": per.get("merkle_root"),
        "price_card/version": (info.get("price_card") or {}).get("version"),
        "crown/king_hotkey": king.get("hotkey"),
        "crown/kings": json.dumps(board.get("kings") or []),
    }
    return {k: v for k, v in out.items() if v is not None}, {k: v for k, v in text.items() if v is not None}


def events(s: dict, prev: dict, seen: set) -> tuple[list[dict], dict]:
    """What happened since the last tick, from public data only; and what to remember for the next tick."""
    board = s["board"]
    queue = {q["submission_id"]: q for q in ((s.get("queue") or {}).get("queue") or [])}
    rows = (s["results"] or {}).get("results") or []
    done = {r.get("submission_id") for r in rows}
    before = prev.get("queue") or {}
    out: list[dict] = []
    who = ("uid", "hotkey", "name")
    for sid, q in queue.items():
        if sid not in before:
            out.append({"event": "queued", "submission_id": sid, **{k: q.get(k) for k in (*who, "state")}})
        if q.get("state") == "running" and (before.get(sid) or {}).get("state") != "running":
            out.append({"event": "started", "submission_id": sid, **{k: q.get(k) for k in who}})
    for sid, q in before.items():
        if sid not in queue and sid not in done:
            out.append({"event": "left_queue", "submission_id": sid, **{k: q.get(k) for k in who},
                        "note": "no public result: failed, cancelled or refunded"})  # fmt: skip
    key = lambda r: r.get("finished_ts") or 0  # noqa: E731
    for r in sorted((r for r in rows if r.get("submission_id") not in seen), key=key):
        out.append({"event": "finished", **r})
    king = (board.get("king") or {}).get("hotkey")
    if prev and prev.get("king") != king:
        out.append({"event": "king", "from": prev.get("king"), "to": king, "kings": board.get("kings")})
    keep = {sid: {k: q.get(k) for k in (*who, "state")} for sid, q in queue.items()}
    return out, {"queue": keep, "king": king}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8032")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT") or "subnet32-humanizer")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or "itsai-dev")
    ap.add_argument("--state", type=Path, default=None, help="which results were logged already")
    args = ap.parse_args()
    import wandb  # only this script needs it

    now = time.time()
    s = snapshot(args.api)
    pid = s["info"].get("period_id") or "none"
    state_p = args.state or var_dir() / "ops" / "wandb_report.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    seen = set(state.get(pid, []))
    nums, text = metrics(s, now)
    evs, counters = events(s, (state.get("_counters") or {}).get(pid) or {}, seen)
    run = wandb.init(project=args.project, entity=args.entity, id=f"evaluator-{pid}",
                     name=f"evaluator-{pid}", resume="allow", config={"netuid": 32, "period_id": pid, **text},
                     allow_val_change=True)  # fmt: skip
    rows = (s["results"] or {}).get("results") or []
    log: dict = {**nums}
    if rows:
        keys = sorted(
            {k for r in rows for k in r if isinstance(r[k], (int, float, str, bool)) or r[k] is None}
        )
        log["results/table"] = wandb.Table(columns=keys, data=[[r.get(k) for k in keys] for r in rows])
    new = [r for r in rows if r.get("submission_id") not in seen]
    for r in sorted(new, key=lambda r: r.get("finished_ts") or 0):  # one point per finished submission
        point = {f"result/{k}": float(v) for k, v in r.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "finished_ts"}  # fmt: skip
        if "crowned" in r:
            point["result/crowned"] = int(bool(r.get("crowned")))
        if point:
            run.log(point)
        seen.add(r.get("submission_id"))
    run.summary.update(text)
    run.log(log)
    # the run's Logs tab (W&B captures stdout only while the run is open): the events, then the tick
    for e in evs:
        print(json.dumps({"ts": round(now), "period": pid, **e}, default=str), flush=True)
    print(json.dumps({"ts": round(now), "period": pid, "event": "tick", **nums}), flush=True)
    run.finish()
    state.setdefault("_counters", {})[pid] = counters
    state[pid] = sorted(x for x in seen if x)
    state_p.parent.mkdir(parents=True, exist_ok=True)
    state_p.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
