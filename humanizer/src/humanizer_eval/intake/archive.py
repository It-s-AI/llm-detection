"""Our copy of every king's bundle (D-70): a crown counts only once the bundle is here.

The bundle is downloaded again from the pinned commit after the evaluation that won, streamed to disk,
checked against the sha256 recorded at submission and kept under ``<root>/<submission_id>/bundle.tar``.
King re-runs on later periods use this copy, and the code is published from it 48 h after the coronation,
so a miner who wins and then deletes or closes the repo changes nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx


class ArchiveError(RuntimeError):
    pass


class Archive:
    def __init__(self, root: Path | str, http: httpx.Client | None = None, timeout: float = 120.0):
        self.root = Path(root)
        self.http = http or httpx.Client(timeout=timeout, headers={"User-Agent": "sn32-humanizer-eval"})

    def path(self, sid: str) -> Path:
        return self.root / sid / "bundle.tar"

    def has(self, sid: str) -> bool:
        return self.path(sid).exists()

    def ids(self) -> list[str]:
        return sorted(p.parent.name for p in self.root.glob("*/bundle.tar")) if self.root.exists() else []

    def remove(self, sid: str) -> int:
        """Delete the archived bundle of ``sid``; returns the bytes freed."""
        p = self.path(sid)
        n = p.stat().st_size if p.exists() else 0
        p.unlink(missing_ok=True)
        with contextlib.suppress(OSError):  # not empty: leave it
            p.parent.rmdir()
        return n

    def store(self, sid: str, url: str, sha256: str, max_bytes: int) -> Path:
        """Download ``url`` (https, redirects to https only) and keep it if it matches ``sha256``."""
        dest = self.path(sid)
        if dest.exists():
            return dest
        if urlsplit(url).scheme != "https":
            raise ArchiveError("archive downloads are https only")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        h, n = hashlib.sha256(), 0
        try:
            with self._stream(url) as r, open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    n += len(chunk)
                    if n > max_bytes:
                        raise ArchiveError("bundle larger than allowed")
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != sha256:
                raise ArchiveError("bundle at the pinned commit no longer matches its sha256")
            os.replace(tmp, dest)
        except (httpx.HTTPError, OSError) as e:
            raise ArchiveError(f"bundle could not be downloaded: {type(e).__name__}") from e
        finally:
            tmp.unlink(missing_ok=True)
        return dest

    def _stream(self, url: str):
        # follow redirects by hand so that none of them leaves https
        for _ in range(5):
            req = self.http.build_request("GET", url)
            r = self.http.send(req, stream=True, follow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location", "")
                r.close()
                url = str(httpx.URL(url).join(loc))
                if urlsplit(url).scheme != "https":
                    raise ArchiveError("redirect away from https")
                continue
            if r.status_code != 200:
                r.close()
                raise ArchiveError(f"bundle download failed ({r.status_code})")
            return _Closing(r)
        raise ArchiveError("too many redirects")


class _Closing:
    def __init__(self, r: httpx.Response):
        self.r = r

    def __enter__(self) -> httpx.Response:
        return self.r

    def __exit__(self, *a) -> None:
        self.r.close()
