"""Reveal of a closed period (D-56) and its independent check.

``write_reveal`` puts everything needed to recompute every decision of the period into one public folder:

    manifest.json        as built (commitment, seed block and its hash, Merkle root)
    secret.hex           the committed secret: H(secret) = commitment, seed = H(secret || blockhash(B))
    dataset.json         the hidden inputs with their facts and slots; the human reference ids and strata
    human_scores.json    crown detectors' scores of the human reference texts
    king_run.json        the king's run on this period's inputs (the last one, if the king changed)
    schedule.json        when the commitment was published and the two notary remarks (period/notary.py)
    submissions/<id>.json  per submission: the public status, the statement and the full record (outputs,
                         per-text levels, judges' verdicts and reasons, detector scores, S, the crown test)
    files.json           sha256 of every file above

``verify_reveal`` recomputes from that folder alone: the commitment, the Merkle root, every S and every
crown decision (the bootstrap seed is derived from the period seed). With a chain node it also checks
the seed block's hash and the notary remarks: the commitment was on chain before the seed block, the
Merkle root before the first evaluation. Nothing it checks depends on us.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from humanizer_eval.period.crypto import decrypt_at_rest
from humanizer_eval.period.merkle import leaf_hash, merkle_root
from humanizer_eval.period.seed import derive_seed, subseed, verify_commitment

#: pod details kept in the public record (no addresses, no executor ids)
RUN_META_KEEP = ("executor", "price_per_hour", "replay", "gpu", "seconds")


def _stratum(slot: Mapping[str, Any]) -> str:
    return f"{slot['genre']}|{slot['length_bin']}"


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def public_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    run = dict(out.get("run") or {})
    meta = run.get("meta") or {}
    run["meta"] = {k: meta[k] for k in RUN_META_KEEP if k in meta}
    out["run"] = run
    return out


def write_reveal(
    pid: str,
    *,
    period_dir: Path,
    submissions_dir: Path,  # the period's SubmissionStore root: <id>/status.json, statement.json, record.enc
    king_runs: list[Path],  # king_run.<detectors>.enc of this period
    human_scores: Mapping[str, Any],  # {"ids": [...], "scores": {detector: [...]}}
    key: bytes,
    out_dir: Path,
    schedule_record: Mapping[str, Any] | None = None,  # lifecycle: commitment time, notary remarks
) -> dict[str, Any]:
    """Write the reveal of ``pid`` to ``out_dir`` (replaced if it exists); returns a short summary."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    manifest = json.loads((period_dir / "manifest.json").read_text())
    data = json.loads(decrypt_at_rest(key, (period_dir / "dataset.enc").read_bytes()))
    _dump(tmp / "manifest.json", manifest)
    (tmp / "secret.hex").write_text((period_dir / "secret.hex").read_text().strip() + "\n")
    _dump(tmp / "dataset.json", {"period_id": pid, "items": data["items"], "human_ref": data["human_ref"]})
    _dump(tmp / "human_scores.json", dict(human_scores))
    if schedule_record is not None:
        _dump(tmp / "schedule.json", {"period_id": pid, **schedule_record})
    for kp in sorted(king_runs):
        tag = kp.name.removeprefix("king_run.").removesuffix(".enc")
        _dump(tmp / f"king_run.{tag}.json", json.loads(decrypt_at_rest(key, kp.read_bytes())))
    n = 0
    subs = submissions_dir
    for d in sorted(subs.iterdir()) if subs.exists() else ():
        if not d.is_dir():
            continue
        entry: dict[str, Any] = {"submission_id": d.name}
        if (d / "status.json").exists():
            entry["status"] = json.loads((d / "status.json").read_text())
        if (d / "statement.json").exists():
            entry["statement"] = json.loads((d / "statement.json").read_text())
        if (d / "record.enc").exists():
            entry["record"] = public_record(json.loads(decrypt_at_rest(key, (d / "record.enc").read_bytes())))
        _dump(tmp / "submissions" / f"{d.name}.json", entry)
        n += 1
    files = {str(p.relative_to(tmp)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(tmp.rglob("*")) if p.is_file()}  # fmt: skip
    _dump(tmp / "files.json", files)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.replace(out_dir)
    return {"submissions": n, "files": len(files) + 1, "items": len(data["items"])}


# --- the check -----------------------------------------------------------------------------------
def _close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= tol


def verify_reveal(root: Path, chain: Any = None, notary_address: str | None = None) -> dict[str, Any]:
    """Recompute what the reveal in ``root`` claims; returns {"ok", "checks": [...], "failures": [...]}."""
    import numpy as np

    from humanizer_eval.scoring.crown import crown_decision
    from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, submission_score

    checks: list[str] = []
    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        (checks if ok else fails).append(what)

    files = json.loads((root / "files.json").read_text())
    for rel, sha in files.items():
        p = root / rel
        check(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha, f"sha256 of {rel}")
    m = json.loads((root / "manifest.json").read_text())
    secret = bytes.fromhex((root / "secret.hex").read_text().strip())
    check(verify_commitment(m["commitment"], secret), "H(secret) equals the published commitment")
    if chain is not None and m.get("block") is not None:
        check(chain.block_hash(int(m["block"])) == m["block_hash"].removeprefix("0x"),
              f"block {m['block']} has the hash in the manifest")  # fmt: skip
    seed = derive_seed(secret, m["block_hash"])
    data = json.loads((root / "dataset.json").read_text())
    items, human_ref = data["items"], data["human_ref"]
    root_hex = merkle_root(
        [leaf_hash(it) for it in items] + [leaf_hash({"human_id": h["id"]}) for h in human_ref]
    )
    check(root_hex.hex() == m["merkle_root"], "Merkle root of the inputs and human ids equals the manifest")

    hs = json.loads((root / "human_scores.json").read_text())
    by_id = {h["id"]: h for h in human_ref}
    check(set(hs["ids"]) <= set(by_id), "scored human texts belong to the period's reference set")
    human = HumanReference({d: np.asarray(v, float) for d, v in hs["scores"].items()},
                           [_stratum(by_id[i]) for i in hs["ids"]])  # fmt: skip
    strata = [_stratum(it["slot"]) for it in items]
    ids = [it["id"] for it in items]

    runs: dict[str, Any] = {}
    subs: dict[str, dict[str, Any]] = {}
    for p in sorted((root / "submissions").glob("*.json")) if (root / "submissions").exists() else ():
        e = json.loads(p.read_text())
        rec = e.get("record") or {}
        if rec.get("scores") is None or rec.get("levels") is None:
            continue
        sc = {
            d: np.asarray([math.inf if x is None else x for x in v], float) for d, v in rec["scores"].items()
        }
        runs[e["submission_id"]] = SubmissionRun(sc, np.asarray(rec["levels"], float), strata, ids)
        subs[e["submission_id"]] = rec
    for sid, rec in subs.items():
        got = submission_score(runs[sid], human).s
        check(_close(got, rec["score"]["S"]), f"{sid}: S = {rec['score']['S']:.6f} recomputed")
        cr = rec.get("crown")
        if not cr:
            continue
        king_id = cr.get("king")
        king = runs.get(f"{king_id}-kingrun", runs.get(king_id)) if king_id else None
        if king is None:
            checks.append(f"{sid}: crown test not recomputed (the king's run is not in this period)")
            continue
        dec = crown_decision(king, runs[sid], human, seed=subseed(seed, f"crown:{sid}"), n_boot=cr["n_boot"],
                             alpha=cr["alpha"], delta_min=cr["delta_min"], min_gate_pass=cr["min_gate_pass"])  # fmt: skip
        check(dec.crown == cr["crown"] and _close(dec.lower, cr["lower"]),
              f"{sid}: crown test against {king_id} recomputed (lower bound {cr['lower']:+.4f}, "
              f"crown {cr['crown']})")  # fmt: skip
    sched = root / "schedule.json"
    if sched.exists() and chain is not None:
        from humanizer_eval.period.notary import verify_notary

        starts = [float(r["started_at"]) for r in subs.values() if r.get("started_at")]
        ok, bad = verify_notary(json.loads(sched.read_text()), m, chain, starts=starts,
                                notary_address=notary_address)  # fmt: skip
        checks.extend(ok)
        fails.extend(bad)
    elif chain is not None:
        fails.append("no schedule.json: the publication times cannot be checked")
    return {"ok": not fails, "period_id": m.get("period_id"), "checks": checks, "failures": fails}
