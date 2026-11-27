"""Bundles on the miner's Hugging Face repo, pinned by an on-chain commitment (D-60, D-66).

The miner uploads ``bundle.tar`` (code + weights, the bundle format of runner/bundle.py) to a model repo
of their own and commits on chain, from the hotkey that submits::

    <namespace>/<name>@<40-hex commit sha>          (at most 128 bytes, the chain's limit)

A branch or tag name (``main``, ``v1``) is never accepted: a branch moves, a commit does not. The commit
pins the file through git (the LFS pointer in the commit holds the file's sha256), and that sha256 is
recorded at submission and checked again on the pod and when the king's bundle is archived. Changing
the repo afterwards changes nothing for an evaluation; deleting it before the archive voids a crown.

Private repos: the miner may give a fine-grained token that can only read that repo. It stays on our
server (encrypted), turns into a short-lived signed CDN link for the pod, and is deleted after the
evaluation (the king's bundle lives on in our archive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

BUNDLE_PATH = "bundle.tar"
COMMITMENT_MAX_BYTES = 128
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SMALL_FILE_BYTES = 50 * 1024**2  # a non-LFS bundle (tiny) is hashed on our side


class HubError(ValueError):
    """The repo, the commit or the bundle file is not acceptable (HTTP 422)."""


def parse_commitment(data: str) -> tuple[str, str]:
    """``(repo_id, commit)`` from ``<repo_id>@<commit>``; refuses branch and tag names."""
    if not isinstance(data, str) or len(data.encode()) > COMMITMENT_MAX_BYTES or data.count("@") != 1:
        raise HubError("commitment must be '<namespace>/<name>@<40-hex commit sha>'")
    repo, rev = data.strip().split("@")
    if not _REPO.match(repo) or ".." in repo:
        raise HubError("commitment: not a Hugging Face repo id")
    if not _COMMIT.match(rev):
        raise HubError("commitment: the revision must be a full 40-hex commit sha, not a branch or tag")
    return repo, rev


def commitment_for(repo: str, commit: str) -> str:
    c = f"{repo}@{commit}"
    parse_commitment(c)
    return c


@dataclass(frozen=True)
class HFBundle:
    repo: str
    revision: str
    sha256: str
    size: int
    private: bool
    lfs: bool
    gated: bool = False  # downloads need the owner's approval (or "auto" acceptance of terms)


class HubNotReadable(HubError):
    """The repo or commit cannot be read without a token: private (or not there)."""


class HubClient:
    """The few Hub API calls we need, over plain HTTPS (no huggingface_hub dependency)."""

    def __init__(self, endpoint: str = "https://huggingface.co", http: httpx.Client | None = None,
                 timeout: float = 30.0):  # fmt: skip
        self.endpoint = endpoint.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout, headers={"User-Agent": "sn32-humanizer-eval"})

    def _get(self, path: str, token: str | None) -> Any:
        h = {"Authorization": f"Bearer {token}"} if token else {}
        r = self.http.get(self.endpoint + path, headers=h)
        if r.status_code in (401, 403, 404):
            raise HubNotReadable(f"repo or commit not found, or the repo is private ({r.status_code})")
        r.raise_for_status()
        return r.json()

    def resolve_url(self, repo: str, commit: str, path: str = BUNDLE_PATH) -> str:
        return f"{self.endpoint}/{repo}/resolve/{commit}/{quote(path)}"

    def bundle_info(
        self, repo: str, commit: str, token: str | None = None, path: str = BUNDLE_PATH
    ) -> HFBundle:
        """Check the commit and read the bundle's sha256 and size from the Hub's metadata."""
        info = self._get(f"/api/models/{repo}/revision/{commit}?blobs=true", token)
        if info.get("sha") != commit:
            raise HubError("the revision does not resolve to that exact commit")
        refs = self._get(f"/api/models/{repo}/refs", token)
        names = {r.get("name") for k in ("branches", "tags", "converts") for r in refs.get(k) or []}
        if commit in names:
            raise HubError("a branch or tag is named like the commit; refusing an ambiguous revision")
        sib = next((s for s in info.get("siblings") or [] if s.get("rfilename") == path), None)
        if sib is None:
            raise HubError(f"no {path} at that commit")
        lfs = sib.get("lfs") or {}
        size = int(lfs.get("size") or sib.get("size") or 0)
        sha = lfs.get("sha256")
        if not sha:
            if size > SMALL_FILE_BYTES:
                raise HubError(f"{path} must be stored with LFS/Xet")
            sha = self._sha256_small(repo, commit, token, path)
        if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
            raise HubError("bundle sha256 not available")
        return HFBundle(
            repo, commit, sha, size, bool(info.get("private")), bool(lfs), bool(info.get("gated"))
        )

    def _sha256_small(self, repo: str, commit: str, token: str | None, path: str = BUNDLE_PATH) -> str:
        import hashlib

        h = {"Authorization": f"Bearer {token}"} if token else {}
        url = self.resolve_url(repo, commit, path)
        with self.http.stream("GET", url, headers=h, follow_redirects=True) as r:
            r.raise_for_status()
            d, n = hashlib.sha256(), 0
            for chunk in r.iter_bytes():
                n += len(chunk)
                if n > SMALL_FILE_BYTES:
                    raise HubError("bundle larger than its metadata says")
                d.update(chunk)
        return d.hexdigest()

    def public_download_ok(self, repo: str, commit: str, path: str = BUNDLE_PATH) -> bool:
        """Can anyone download the bundle at this commit, with no token? (HEAD, redirects not followed;
        a server error raises: that is not an answer.)"""
        r = self.http.head(self.resolve_url(repo, commit, path), follow_redirects=False)
        if r.status_code >= 500:
            r.raise_for_status()
        if r.status_code in (301, 302, 303, 307, 308):
            return urlsplit(r.headers.get("location", "")).scheme == "https"
        return r.status_code == 200

    def download_url(self, repo: str, commit: str, token: str | None = None) -> str:
        """What the pod downloads: the pinned resolve URL, or for a private repo the short-lived signed
        link the Hub redirects to (so no token ever reaches the pod)."""
        url = self.resolve_url(repo, commit)
        if not token:
            return url
        r = self.http.head(url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=False)
        loc = r.headers.get("location", "")
        if r.status_code not in (301, 302, 303, 307, 308) or urlsplit(loc).scheme != "https":
            raise HubError(f"no signed download link for the private bundle ({r.status_code})")
        return loc
