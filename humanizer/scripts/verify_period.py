"""Check a revealed period yourself (D-56): commitment, seed block, Merkle root, every S and every crown test.

    python scripts/verify_period.py --url https://<evaluator>/v1/periods/p1      # download, then check
    python scripts/verify_period.py --dir var/public/periods/p1 [--no-chain]

Needs only this repository (pip install -e .): the same metric code recomputes every number from the
published per-text scores and levels. Against a public finney archive node it also checks the seed
block's hash and the notary remarks: the commitment was on chain before the seed block, the Merkle root
before the first evaluation. Pass --notary <address> to check who signed them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import httpx

from humanizer_eval.period.chain import FINNEY_ARCHIVE_RPC, Chain
from humanizer_eval.period.reveal import verify_reveal


def download(url: str, out: Path) -> Path:
    url = url.rstrip("/")
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        files = http.get(f"{url}/files.json")
        files.raise_for_status()
        (out / "files.json").write_bytes(files.content)
        for rel in json.loads(files.content):
            p = (out / rel).resolve()
            if not p.is_relative_to(out.resolve()):
                raise SystemExit(f"refusing a path outside the folder: {rel}")
            p.parent.mkdir(parents=True, exist_ok=True)
            r = http.get(f"{url}/{rel}")
            r.raise_for_status()
            p.write_bytes(r.content)
    return out


def same_code(root: Path) -> str | None:
    """A note when this checkout's scoring code differs from the one that scored the period (the numbers
    are recomputed with this checkout: check out the published commit if they do not match)."""
    p = root / "evaluator.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text())
    try:
        from types import SimpleNamespace

        from humanizer_eval.detectors.vendors import GPTZero, Pangram, ZeroGPT
        from humanizer_eval.evaluation.conditions import scoring_fingerprint

        gates = json.loads((Path(__file__).resolve().parents[1] / "configs" / "gates.json").read_text())
        dets = {"gptzero": SimpleNamespace(config=GPTZero.config), "pangram": SimpleNamespace(config=Pangram.config),
                "zerogpt": SimpleNamespace(config=ZeroGPT.config)}  # fmt: skip
        mine = scoring_fingerprint(gates, dets, {"n_boot": 2000})
    except Exception as e:
        return f"note: could not compute this checkout's scoring fingerprint ({e})"
    used = ev.get("scoring_fingerprints") or []
    if used and mine not in used:
        return (
            f"note: this checkout scores under fingerprint {mine}, the period was scored under {used}; "
            f"check out commit {ev.get('code_commit')} to recompute with the same code"
        )
    return f"scoring code: same fingerprint as the period ({mine}), evaluator commit {ev.get('code_commit')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="https://<evaluator>/v1/periods/<id>")
    src.add_argument("--dir", type=Path, help="a downloaded reveal folder")
    ap.add_argument("--rpc", default=FINNEY_ARCHIVE_RPC, help="a node that keeps old block bodies")
    ap.add_argument("--notary", default=None, help="the notary's published SS58 address (docs/periods)")
    ap.add_argument("--no-chain", action="store_true", help="skip the block-hash check (offline)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every passed check")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = args.dir or download(args.url, Path(tmp))
        rep = verify_reveal(
            root, chain=None if args.no_chain else Chain(args.rpc), notary_address=args.notary
        )
        code_note = same_code(root)
    if code_note:
        print(code_note)
    for c in rep["checks"] if args.verbose else ():
        print("ok  ", c)
    for f in rep["failures"]:
        print("FAIL", f)
    print(f"period {rep['period_id']}: {len(rep['checks'])} checks passed, {len(rep['failures'])} failed")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
