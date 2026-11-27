"""SN32 humanizer competition (20% of the weights): a separate section of the SN32 leaderboard Space.

It reads only the evaluator's public API (Space variable HUMANIZER_API, e.g. https://eval.example):
/v1/info, /v1/leaderboard, /v1/results and /v1/kings, cached for a minute. It shares nothing with the
detection leaderboard: if the evaluator is down or this module fails, only this section shows a message.
Source of truth: humanizer/space/humanizer_tab.py in It-s-AI/llm-detection (branch v4).
"""

from __future__ import annotations

import datetime
import html
import json
import os
import re
import time
import urllib.request

import gradio as gr
import pandas as pd

API = os.environ.get("HUMANIZER_API", "").rstrip("/")
TTL_S = 60
TIMEOUT_S = 10
SHARES = (0.85, 0.10, 0.05)
_cache: dict = {"t": 0.0, "v": None}

INTRO = """<p>The humanizer competition gets 20% of SN32's miner emission: miners submit a model that
rewrites AI text so the crown detectors take it for human writing while the meaning and quality are kept. Each
submission is scored on the period's hidden texts; a challenger is crowned when its score <b>S</b>
beats the king's on the same texts with confidence. The king gets 85%, the two previous kings 10% and
5% (a missing previous king's share goes to the king). While the owner's baseline is king, the 20%
goes to detection. Bundles become public 48 hours after the coronation.</p>"""


TEST_BANNER = """<div style="border:2px solid #d97706;border-radius:8px;padding:10px 14px;margin:6px 0">
<b>TEST LAUNCH.</b> Testnet, the free It's AI detector instead of the crown detectors. Results and crowns
here do not carry over to the production launch.</div>"""
CLOSED_BANNER = """<div style="border:2px solid #d97706;border-radius:8px;padding:10px 14px;margin:6px 0">
<b>TEST LAUNCH: code and rules only.</b> Submissions open at the production launch on mainnet. Read the
rules, the miner guide and the reference humanizer in
<a href="https://github.com/It-s-AI/llm-detection/tree/v4/humanizer">llm-detection/humanizer</a> and get your
bundle ready.
</div>"""


def _get(path: str):
    req = urllib.request.Request(API + path, headers={"User-Agent": "sn32-leaderboard-space"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode())


def _fetch():
    if not API.startswith("https://"):
        raise RuntimeError("the evaluator's address is not configured yet (Space variable HUMANIZER_API)")
    now = time.time()
    if _cache["v"] is not None and now - _cache["t"] < TTL_S:
        return _cache["v"]
    info, env, res, kings = (_get(p) for p in ("/v1/info", "/v1/leaderboard", "/v1/results", "/v1/kings"))
    v = (info, json.loads(env["payload"]), res, kings)
    _cache.update(t=now, v=v)
    return v


def _hk(h) -> str:
    s = str(h or "")
    return s[:6] + "…" + s[-4:] if len(s) > 14 else s


def _num(x, d: int = 4):
    return round(float(x), d) if isinstance(x, (int, float)) and x == x and abs(x) != float("inf") else None


def _ts(t) -> str:
    if t is None:
        return ""
    try:
        d = (datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc) if isinstance(t, (int, float))
             else datetime.datetime.fromisoformat(str(t)))  # fmt: skip
    except (ValueError, OverflowError):
        return ""
    return d.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


def _link(url, text: str) -> str:
    if isinstance(url, str) and re.fullmatch(r"https://huggingface\.co/[A-Za-z0-9._/@-]+", url):
        return f"[{text}]({url})"
    return text


def _summary(info: dict, lb: dict) -> str:
    q = info.get("queue") or {}
    king = (lb.get("king") or {}).get("hotkey")
    net = info.get("network")
    rows = [
        ("Network", f"{'testnet' if net == 'test' else net} · netuid {info.get('netuid')}" if net else None),
        ("Period", info.get("period_id")),
        ("Texts per evaluation", info.get("n_inputs")),
        ("Crown detectors", ", ".join(info.get("crown_detectors") or [])),
        ("Queue (waiting / running)", f"{q.get('waiting', 0)} / {q.get('running', 0)}"),
        ("Current king", _hk(king) if king else "owner baseline"),
    ]
    cells = "".join(
        f"<tr><td><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
        if v is not None
    )
    return f"<table>{cells}</table>"


def _payouts(lb: dict) -> pd.DataFrame:
    king = (lb.get("king") or {}).get("hotkey")
    rows = []
    kings = [h for h in (lb.get("kings") or []) if isinstance(h, str)][: len(SHARES)]
    for i, h in enumerate(kings):
        w = SHARES[i] + (sum(SHARES[len(kings) :]) if i == 0 else 0.0)
        role = "king" if h == king or i == 0 else "previous king"
        rows.append([_hk(h), role, f"{100 * w:.1f}%"])
    return pd.DataFrame(rows, columns=["Hotkey", "Role", "Share of the humanizer 20%"])


def _results(res: dict, dets: list[str]) -> pd.DataFrame:
    items = sorted(
        res.get("results") or [], key=lambda r: -(r.get("S") if isinstance(r.get("S"), (int, float)) else -1)
    )
    cols = ["#", "Name", "Hotkey", "S", "ΔS vs king", "Lower bound", "Crowned", "Quality 1/0.75/0.5/0",
            *[f"E {d}" for d in dets], "Finished (UTC)"]  # fmt: skip
    rows = []
    for i, r in enumerate(items, 1):
        lv = r.get("levels") or {}
        q = f"{lv.get('1', 0)}/{lv.get('0.75', 0)}/{lv.get('0.5', 0)}/{lv.get('0', 0)}" if lv else ""
        e = [_num(((r.get("detectors") or {}).get(d) or {}).get("E"), 3) for d in dets]
        rows.append([i, r.get("name") or "", _hk(r.get("hotkey")), _num(r.get("S")), _num(r.get("delta")),
                     _num(r.get("lower_bound")), "👑" if r.get("crowned") else "", q, *e,
                     _ts(r.get("finished_ts"))])  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def _bundle(b: dict | None) -> str:
    b = b or {}
    src = b.get("source")
    if src == "baseline":
        return "owner baseline"
    if src == "pending":
        due = b.get("public_at")
        return "being published" if _ts(due) and _ts(due) <= _ts(time.time()) else f"public at {_ts(due)}"
    if src == "miner":
        return _link(b.get("url"), "miner's repo")
    if src == "mirror":
        return _link(b.get("url"), "our copy")
    return "unavailable"


def _kings(k: dict) -> pd.DataFrame:
    rows = []
    for e in k.get("kings") or []:
        base = (e.get("bundle") or {}).get("source") == "baseline"
        rows.append([_ts(e.get("crowned_at")), "owner baseline" if base else _hk(e.get("hotkey")),
                     _num(e.get("delta_s")), _bundle(e.get("bundle"))])  # fmt: skip
    return pd.DataFrame(rows, columns=["Crowned (UTC)", "Hotkey", "ΔS at coronation", "Bundle"])


def refresh():
    """(status html, summary html, payouts, results, kings); never raises."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    empty = pd.DataFrame()
    try:
        info, lb, res, kings = _fetch()
        dets = list(info.get("crown_detectors") or [])
        status = f"<div>Last updated: {now} (UTC)</div>"
        if (info.get("limits") or {}).get("submissions_open") is False:
            status = CLOSED_BANNER + status
        elif info.get("mode") == "test":
            status = TEST_BANNER + status
        return status, _summary(info, lb), _payouts(lb), _results(res, dets), _kings(kings)
    except Exception as e:  # the evaluator is down or not configured: this section only
        msg = html.escape(f"{type(e).__name__}: {e}")[:300]
        return f"<div>Humanizer data unavailable: {msg}</div>", "", empty, empty, empty


def add_section(demo) -> None:
    """Call inside ``with demo:`` (gr.Blocks): adds the section and its own load event."""
    with gr.Accordion("Humanizer competition", open=True):
        gr.HTML(INTRO)
        status = gr.HTML()
        summary = gr.HTML()
        gr.Markdown("**Payouts now**")
        payouts = gr.DataFrame(interactive=False)
        gr.Markdown("**This period: finished submissions**")
        results = gr.DataFrame(interactive=False)
        gr.Markdown("**Kings**")
        kings = gr.DataFrame(interactive=False, datatype=["str", "str", "number", "markdown"])
        button = gr.Button("Refresh humanizer data", size="sm")
    outputs = [status, summary, payouts, results, kings]
    button.click(refresh, outputs=outputs)
    demo.load(refresh, outputs=outputs)
