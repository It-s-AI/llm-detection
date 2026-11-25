"""Fill the built-in leaderboard URL(s) and public key into the validator code (the prod switch, D-92):
the `v4` branch of It-s-AI/llm-detection, where they switch on the humanizer's 10% of the weights.

    python scripts/release_validator.py --repo ../llm-detection \
        --url https://<evaluator>/v1/leaderboard [--url https://<mirror>/...] [--netuid 32] [--commit]

Before editing anything, every URL is fetched and its envelope verified with the key (the same rules the
validator applies), so a release can only point at a live, correctly signed leaderboard. The key defaults
to the one this evaluator signs with (var/keys/leaderboard_ed25519.key). With --commit the change is
committed on the checked-out branch of --repo (never pushed: the owner merges it into llm-detection).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import httpx

from humanizer_eval.paths import var_dir
from humanizer_eval.publish.leaderboard import LeaderboardError, load_or_create_signing_key, verify

TARGET = "detection/validator/humanizer_mechanism.py"


def patch(src: str, key_hex: str, urls: list[str]) -> str:
    out, n1 = re.subn(r'^DEFAULT_PUBLIC_KEY = ".*"$', f'DEFAULT_PUBLIC_KEY = "{key_hex}"', src, flags=re.M)
    tup = "(" + "".join(f'"{u}", ' for u in urls).rstrip(" ") + ")"
    out, n2 = re.subn(r"^DEFAULT_URLS: Tuple\[str, \.\.\.\] = .*$", f"DEFAULT_URLS: Tuple[str, ...] = {tup}", out,
                      flags=re.M)  # fmt: skip
    if n1 != 1 or n2 != 1:
        raise SystemExit(f"{TARGET}: DEFAULT_PUBLIC_KEY / DEFAULT_URLS not found exactly once")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="llm-detection checkout (branch v4)")
    ap.add_argument("--url", action="append", required=True, help="leaderboard URL (repeatable, first wins)")
    ap.add_argument("--public-key", default=None, help="hex ed25519 key (default: this evaluator's)")
    ap.add_argument("--netuid", type=int, default=32)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--pre-launch", action="store_true",
                    help="the server still signs for another netuid (test mode): check the signature and the key only")  # fmt: skip
    args = ap.parse_args()
    key = args.public_key
    if key is None:
        kp = var_dir() / "keys" / "leaderboard_ed25519.key"
        if not kp.exists():
            raise SystemExit(f"{kp} not found; give --public-key")
        key = load_or_create_signing_key(kp).verify_key.encode().hex()
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise SystemExit("--public-key: 32-byte hex")
    for u in args.url:
        if not u.startswith("https://"):
            raise SystemExit(f"{u}: validators fetch over https only")
        try:
            env = httpx.get(u, timeout=20).json()
            netuid = args.netuid
            if args.pre_launch:  # the payload's own netuid: the signature and the key are what is checked
                netuid = json.loads(env["payload"])["netuid"]
                if netuid != args.netuid:
                    print(f"note {u}: signs for netuid {netuid} now; must be {args.netuid} at launch")
            p = verify(env, key, netuid=netuid)
        except (httpx.HTTPError, ValueError, LeaderboardError) as e:
            raise SystemExit(
                f"{u}: not a valid leaderboard for this key and netuid {args.netuid}: {e}"
            ) from e
        print(f"ok  {u}: seq {p['seq']}, kings {p['kings']}")
    f = args.repo / TARGET
    before = f.read_text()
    after = patch(before, key, args.url)
    if after == before:
        print("already up to date")
        return 0
    f.write_text(after)
    subprocess.run(["git", "-C", str(args.repo), "--no-pager", "diff", "--", TARGET], check=True)
    if args.commit:
        subprocess.run(["git", "-C", str(args.repo), "commit", "-q", "-m",
                        "Mechanism 1: built-in leaderboard URL and signing key", "--", TARGET], check=True)  # fmt: skip
        print("committed (not pushed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
