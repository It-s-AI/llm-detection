"""King bundles in public (D-70).

At the coronation the king's bundle is copied into our archive (``intake/archive.py``); the crown counts
only then. 48 hours after the coronation the bundle is public:

* **the miner's own repo** at the pinned commit, ``https://huggingface.co/<repo>/tree/<commit>``, once we
  have checked that anyone can download that commit without a token (not private, not gated, no branch or
  tag shadowing the commit) and that its sha256 is still the one we evaluated;
* **otherwise our copy** from the archive, uploaded to our public mirror repo (one repo, created in
  advance, so the write token can be limited to it): ``kings/<submission id>/bundle.tar`` with a README
  naming the original commit and sha256.

The miner's link is checked again every few hours; once it stops working our copy replaces it for good.
A failure on Hugging Face's side (network, 5xx) changes nothing and is retried. Without a mirror
configured, an unreachable king is listed as ``unavailable`` (the copy stays in the archive) and logged.

State: one JSON file ``{submission_id: record}``, rewritten atomically; ``KingPublisher.public()`` is what
``GET /v1/kings`` shows.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from humanizer_eval.intake.hf import BUNDLE_PATH, HubClient, HubError
from humanizer_eval.scoring.crown import KingEntry, KingState

log = logging.getLogger(__name__)

PUBLISH_DELAY_S = 48 * 3600  # D-70 / Q-6
RECHECK_S = 6 * 3600  # the miner's link, once published
RETRY_S = 3600  # an unavailable king (e.g. the mirror failed)
HF = "https://huggingface.co"

MIRROR_README = """---
license: other
tags: [sn32, humanizer, king]
---
# SN32 humanizer king `{sid}`

Copy of the bundle that won the SN32 humanizer crown, published by the subnet owner from its archive
because the original commit was not publicly downloadable when the bundle became public (rule D-70).

| | |
|---|---|
| hotkey | `{hotkey}` |
| crowned at | {crowned_at} |
| original | {original} |
| `bundle.tar` sha256 | `{sha256}` |
| reason | {reason} |
"""


class MirrorError(RuntimeError):
    pass


class HFMirror:
    """Public copies of king bundles in one public model repo of ours, created in advance
    (``huggingface_hub`` and a token that can write to that repo only)."""

    def __init__(self, repo: str, token: str, api: Any = None, prefix: str = "kings"):
        if repo.count("/") != 1 or not (token or api):
            raise ValueError("the mirror needs '<org>/<repo>' and a write token")
        self.repo, self.token, self._api, self.prefix = repo, token, api, prefix.strip("/")

    def path_for(self, sid: str) -> str:
        return f"{self.prefix}/{sid}/{BUNDLE_PATH}"

    def publish(self, sid: str, bundle: Path, rec: dict[str, Any], reason: str) -> tuple[str, str, str]:
        """Upload ``bundle`` and its README in one commit; returns ``(repo, commit sha, bundle path)``."""
        from huggingface_hub import CommitOperationAdd, HfApi

        api = self._api or HfApi(token=self.token)
        path = self.path_for(sid)
        original = f"`{rec['repo']}@{rec['commit']}`" if rec.get("repo") else "not on Hugging Face"
        readme = MIRROR_README.format(sid=sid, hotkey=rec["hotkey"], crowned_at=rec["crowned_at"],
                                      original=original, sha256=rec["sha256"], reason=reason)  # fmt: skip
        info = api.create_commit(
            repo_id=self.repo,
            repo_type="model",
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(bundle)),
                CommitOperationAdd(
                    path_in_repo=f"{self.prefix}/{sid}/README.md", path_or_fileobj=readme.encode()
                ),
            ],
            commit_message=f"SN32 humanizer king {sid} (sha256 {rec['sha256']})",
        )
        return self.repo, str(info.oid), path


class _Transient(Exception):
    """Hugging Face did not answer; try again later, change nothing."""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


class KingPublisher:
    def __init__(
        self,
        path: Path | str,
        king_state_path: Path | str,
        rows: Callable[[str], Any],
        hub: HubClient,
        archive: Any,
        mirror: HFMirror | None = None,
        *,
        delay_s: float = PUBLISH_DELAY_S,
        recheck_s: float = RECHECK_S,
        retry_s: float = RETRY_S,
        now: Callable[[], float] = time.time,
        alerter: Any = None,
    ):
        self.path, self.king_state_path = Path(path), Path(king_state_path)
        self.rows, self.hub, self.archive, self.mirror = rows, hub, archive, mirror
        self.delay_s, self.recheck_s, self.retry_s, self.now = delay_s, recheck_s, retry_s, now
        self.alerter = alerter

    # --- state ----------------------------------------------------------------------------------
    def _load(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, recs: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(recs, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def _kings(self) -> list[KingEntry]:
        if not self.king_state_path.exists():
            return []
        return KingState.from_json(self.king_state_path.read_text()).history

    def _new(self, e: KingEntry) -> dict[str, Any]:
        row = self.rows(e.submission_id)
        return {
            "submission_id": e.submission_id,
            "hotkey": e.hotkey,
            "crowned_at": e.crowned_at.isoformat(),
            "due_ts": e.crowned_at.timestamp() + self.delay_s,
            "repo": getattr(row, "hf_repo", None),
            "commit": getattr(row, "hf_revision", None),
            "sha256": getattr(row, "bundle_sha256", None),
            "bytes": getattr(row, "bundle_bytes", None),
            "source": "pending",
            "url": None,
            "file_url": None,
            "checked_ts": None,
            "published_ts": None,
            "note": "",
        }

    # --- one pass -------------------------------------------------------------------------------
    def run_once(self) -> list[str]:
        """Publish what is due, re-check the miners' links; returns the kings whose link changed."""
        recs, changed = self._load(), []
        for e in self._kings():
            if e.is_baseline:
                continue
            rec = recs.get(e.submission_id) or self._new(e)
            before = (rec["source"], rec["url"])
            try:
                self._step(rec)
            except _Transient as why:
                log.warning("%s: Hugging Face unreachable (%s); retrying later", e.submission_id, why)
            recs[e.submission_id] = rec
            if (rec["source"], rec["url"]) != before:
                changed.append(e.submission_id)
                log.info("%s: bundle %s %s", e.submission_id, rec["source"], rec["url"] or rec["note"])
            self._save(recs)  # after every king: a mirror upload can take hours
        return changed

    def _step(self, rec: dict[str, Any]) -> None:
        now = self.now()
        if now < rec["due_ts"] or rec["source"] == "mirror":
            return
        wait = self.recheck_s if rec["source"] == "miner" else self.retry_s
        if rec["checked_ts"] is not None and rec["source"] != "pending" and now - rec["checked_ts"] < wait:
            return
        why = self._miner_unavailable(rec)
        rec["checked_ts"] = now
        if why is None:
            url = f"{HF}/{rec['repo']}/tree/{rec['commit']}"
            if rec["source"] != "miner":
                rec.update(source="miner", url=url, file_url=self.hub.resolve_url(rec["repo"], rec["commit"]),
                           published_ts=rec["published_ts"] or now, note="")  # fmt: skip
            return
        sid = rec["submission_id"]
        if self.mirror is None or not self.archive.has(sid):
            missing = "no mirror is configured" if self.mirror is None else "our copy is missing"
            rec.update(source="unavailable", url=None, file_url=None, note=f"{why}; {missing}")
            log.error("%s: the king's bundle is not public: %s", sid, rec["note"])
            if self.alerter is not None:
                self.alerter.send(f"king {sid}: bundle not public: {rec['note']}", dedupe=f"publish-{sid}")
            return
        self._mirror(rec, why)

    def _mirror(self, rec: dict[str, Any], why: str) -> bool:
        """Upload our archived copy to the mirror repo and check it; True when it is public."""
        sid, now = rec["submission_id"], self.now()
        assert self.mirror is not None
        try:
            repo, commit, path = self.mirror.publish(sid, self.archive.path(sid), rec, why)
            got = self.hub.bundle_info(repo, commit, path=path)
            if got.sha256 != rec["sha256"] or not self.hub.public_download_ok(repo, commit, path):
                raise MirrorError("our public copy does not match the archived bundle")
        except Exception as e:
            rec.update(source="unavailable", url=None, file_url=None,
                       note=f"{why}; our copy could not be published ({type(e).__name__}: {e})")  # fmt: skip
            log.error("%s: mirror failed: %s", sid, rec["note"])
            return False
        folder = path.rsplit("/", 1)[0]
        rec.update(source="mirror", url=f"{HF}/{repo}/tree/{commit}/{folder}", mirror_repo=repo,
                   mirror_commit=commit, file_url=self.hub.resolve_url(repo, commit, path),
                   published_ts=rec["published_ts"] or now, note=why)  # fmt: skip
        return True

    # --- archive space ---------------------------------------------------------------------------
    def prune(self, is_registered: Any = None) -> list[str]:
        """Free the archive (bundles up to 50 GB each). A king's bundle is kept while it can become the king
        again: every king that still holds the registration it won from (the crown passes back along them,
        D-66). A king whose registration is gone never returns; its bundle is deleted once a public copy of
        ours exists: if the only public copy is the miner's repo, it is mirrored first (the miner may delete
        it later). Without a mirror nothing is deleted. Returns the deleted submission ids."""
        if not self.king_state_path.exists():
            return []
        st = KingState.from_json(self.king_state_path.read_text())
        keep = {e.submission_id for e in st.active_history(is_registered)}
        recs, removed = self._load(), []
        for sid in self.archive.ids():
            rec = recs.get(sid)
            if sid in keep or rec is None or self.now() < rec["due_ts"]:
                continue
            why = "the king's registration ended; archive freed"
            if rec["source"] != "mirror" and (self.mirror is None or not self._mirror(rec, why)):
                continue
            freed = self.archive.remove(sid)
            rec["archive_removed_ts"] = self.now()
            recs[sid] = rec
            self._save(recs)
            removed.append(sid)
            log.info("%s: archived bundle deleted (%.1f GB); public copy %s", sid, freed / 2**30, rec["url"])
        if removed and self.alerter is not None:
            self.alerter.send(f"archive: deleted {len(removed)} bundle(s) of past kings (public copies kept)")
        return removed

    def _miner_unavailable(self, rec: dict[str, Any]) -> str | None:
        """Why the miner's pinned commit is not a valid public link, or None if it is."""
        if not rec.get("repo") or not rec.get("commit"):
            return "the submission has no Hugging Face repo"
        try:
            b = self.hub.bundle_info(rec["repo"], rec["commit"])  # no token: what anyone sees
            if b.private or b.gated:
                return "the miner's repo is private or gated"
            if b.sha256 != rec["sha256"]:
                return "the bundle at the pinned commit changed"
            if not self.hub.public_download_ok(rec["repo"], rec["commit"]):
                return "the bundle cannot be downloaded without a token"
        except HubError as e:
            return f"the miner's commit is not public ({e})"
        except httpx.HTTPError as e:
            raise _Transient(type(e).__name__) from e
        return None

    # --- what GET /v1/kings shows ---------------------------------------------------------------
    def public(self) -> dict[str, Any]:
        recs, out = self._load(), []
        for e in reversed(self._kings()):
            item: dict[str, Any] = {"submission_id": e.submission_id, "hotkey": e.hotkey,
                                    "crowned_at": e.crowned_at.isoformat(), "delta_s": e.delta_s}  # fmt: skip
            rec = recs.get(e.submission_id)
            if e.is_baseline:
                item["bundle"] = {"source": "baseline"}
            elif rec is None or rec["source"] == "pending":
                item["bundle"] = {"source": "pending",
                                  "public_at": _iso(e.crowned_at.timestamp() + self.delay_s)}  # fmt: skip
            else:
                b = {k: rec.get(k) for k in ("source", "url", "file_url", "sha256", "bytes")}
                b["public_at"] = _iso(rec["due_ts"])
                if rec.get("repo"):
                    b["original"] = f"{rec['repo']}@{rec['commit']}"
                if rec["source"] != "miner":
                    b["note"] = rec["note"]
                if rec.get("checked_ts"):
                    b["checked_at"] = _iso(rec["checked_ts"])
                item["bundle"] = b
            out.append(item)
        return {"kings": out}
