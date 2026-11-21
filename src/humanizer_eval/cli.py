"""`humeval` command line entry point."""

from __future__ import annotations

import argparse
import json

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.paths import ledger_path


def _cmd_budget(_: argparse.Namespace) -> int:
    ledger = SpendLedger(ledger_path(), get_settings().caps())
    print(json.dumps(ledger.summary(), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="humeval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget", help="show spend ledger vs caps").set_defaults(func=_cmd_budget)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
