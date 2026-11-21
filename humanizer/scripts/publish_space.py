"""Add the humanizer section (space/humanizer_tab.py) to the SN32 leaderboard Space, next to the detection
leaderboard and without touching it: the module is a separate file, and app.py only gets a guarded import
and one guarded call (if the module fails, the detection leaderboard runs as before).

    python scripts/publish_space.py --space sergak0/ai-detection-leaderboard            # opens a PR
    python scripts/publish_space.py --space ... --direct                                # commits to main
    python scripts/publish_space.py --space ... --api https://<evaluator> --set-api     # Space variable

By default the change is a pull request on the Space: the live page changes only when it is merged.
HF_MIRROR_TOKEN must be able to write to the Space.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from humanizer_eval.config import get_settings

MODULE = Path(__file__).resolve().parents[1] / "space" / "humanizer_tab.py"
IMPORT_ANCHOR = "load_dotenv()\n"
IMPORT_HOOK = """
try:  # the humanizer section (mechanism 1) lives in its own module; the detection leaderboard never depends on it
    import humanizer_tab
except Exception as _e:
    humanizer_tab = None
    print("humanizer section disabled:", _e)
"""
CALL_ANCHOR = "        last_updated_html = gr.HTML()\n"
CALL_HOOK = """
        if humanizer_tab is not None:
            try:
                humanizer_tab.add_section(demo)
            except Exception as _e:
                print("humanizer section disabled:", _e)
"""


def hooked(app: str) -> str:
    """app.py with the two guarded hooks (unchanged if they are already there)."""
    if "import humanizer_tab" in app:
        return app
    if app.count(IMPORT_ANCHOR) != 1 or app.count(CALL_ANCHOR) != 1:
        raise SystemExit("app.py changed shape; add the two hooks by hand (see IMPORT_HOOK, CALL_HOOK)")
    return app.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_HOOK, 1).replace(
        CALL_ANCHOR, CALL_ANCHOR + CALL_HOOK, 1
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--direct", action="store_true", help="commit to main instead of opening a PR")
    ap.add_argument("--api", default=None, help="https://... the evaluator's public address")
    ap.add_argument("--set-api", action="store_true", help="only set the Space variable HUMANIZER_API")
    args = ap.parse_args()
    token = get_settings().hf_mirror_token
    if not token:
        raise SystemExit("HF_MIRROR_TOKEN is not set")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=token)
    if args.set_api:
        if not args.api or not re.fullmatch(r"https://[A-Za-z0-9.-]+(:\d+)?", args.api.rstrip("/")):
            raise SystemExit("--api: https://host[:port]")
        api.add_space_variable(args.space, "HUMANIZER_API", args.api.rstrip("/"))
        print(json.dumps({"space": args.space, "HUMANIZER_API": args.api.rstrip("/")}))
        return 0
    with tempfile.TemporaryDirectory() as d:
        app = Path(
            hf_hub_download(args.space, "app.py", repo_type="space", token=token, local_dir=d)
        ).read_text()
    new_app = hooked(app)
    ops = [CommitOperationAdd(path_in_repo="humanizer_tab.py", path_or_fileobj=MODULE.read_bytes())]
    if new_app != app:
        ops.append(CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=new_app.encode()))
    info = api.create_commit(
        repo_id=args.space,
        repo_type="space",
        operations=ops,
        commit_message="Humanizer competition (mechanism 1): a separate section",
        commit_description=(
            "Adds humanizer_tab.py (reads the evaluator's public API from the Space variable HUMANIZER_API) "
            "and two guarded hooks in app.py. The detection leaderboard is untouched: if the module fails to "
            "import or build, it is skipped."
        ),
        create_pr=not args.direct,
    )
    print(json.dumps({"space": args.space, "pr_url": getattr(info, "pr_url", None),
                      "commit": getattr(info, "oid", None), "files": [o.path_in_repo for o in ops]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
