"""Submission bundle format and safe inspection (D-60, D-61).

A bundle is a gzip/zstd-free plain `.tar` or `.tar.gz` with:

    submission.json      manifest (see `Manifest`)
    src/                 miner code; the entrypoint module lives here
    wheels/              optional: all Python dependencies as wheels (no network at run time)
    requirements.txt     optional: what to install from wheels/
    weights/             optional: model files

The orchestrator NEVER extracts a bundle; it only reads tar headers and the small
manifest member. Extraction happens inside the isolated evaluation pod.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-3-Clause", "OWNER-GRANT"}
MAX_BUNDLE_BYTES = 50 * 1024**3  # D-61 (100 GB until 2026-11-17)
MAX_HEADER_RECORD = 64 * 1024  # one pax/long-name record
MAX_HEADER_BYTES = 512 * 2**20  # all headers of the archive
MAX_MEMBERS = 200_000
MAX_MANIFEST_BYTES = 64 * 1024


class BundleError(ValueError):
    pass


@dataclass
class Manifest:
    name: str
    entrypoint: str = "humanize:humanize"  # module:function inside src/
    license: str = "MIT"
    batch_size: int = 16
    extra: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: bytes) -> Manifest:
        if len(raw) > MAX_MANIFEST_BYTES:
            raise BundleError("manifest too large")
        try:
            d = json.loads(raw)
        except ValueError as e:
            raise BundleError("manifest is not valid JSON") from e
        if not isinstance(d, dict) or not isinstance(d.get("name"), str):
            raise BundleError("manifest must be an object with a string 'name'")
        m = cls(
            name=d["name"][:64],
            entrypoint=str(d.get("entrypoint", "humanize:humanize")),
            license=str(d.get("license", "")),
            batch_size=int(d.get("batch_size", 16)),
        )
        mod, _, fn = m.entrypoint.partition(":")
        if not mod or not fn or not all(p.isidentifier() for p in [*mod.split("."), fn]):
            raise BundleError(f"bad entrypoint {m.entrypoint!r}")
        if m.license not in ALLOWED_LICENSES:
            raise BundleError(f"license {m.license!r} not in {sorted(ALLOWED_LICENSES)}")
        if not 1 <= m.batch_size <= 512:
            raise BundleError("batch_size out of range")
        return m


@dataclass
class BundleInfo:
    sha256: str
    size: int
    members: int
    manifest: Manifest


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return (
        not p.is_absolute() and ".." not in p.parts and not name.startswith(("/", "\\")) and "\\" not in name
    )


def inspect_bundle(path: Path | str, max_bytes: int = MAX_BUNDLE_BYTES) -> BundleInfo:
    """Validate tar headers and the manifest without extracting anything."""
    path = Path(path)
    size = path.stat().st_size
    if size > max_bytes:
        raise BundleError(f"bundle is {size} bytes > {max_bytes}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    manifest_raw = None
    n = 0
    total = 0
    with path.open("rb") as f:  # plain tar or gzip only: bz2/xz decompress slowly enough to stall the pod
        magic = f.read(512)
    gz = magic[:2] == b"\x1f\x8b"
    if not gz and magic[257:262] != b"ustar":
        raise BundleError("the bundle must be a plain .tar or a .tar.gz")
    with tarfile.open(path, "r:gz" if gz else "r:") as tf:
        for m in tf:
            n += 1
            if n > MAX_MEMBERS:
                raise BundleError("too many files in bundle")
            if not _safe_name(m.name):
                raise BundleError(f"unsafe path in bundle: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise BundleError(f"only regular files and directories are allowed: {m.name!r}")
            total += m.size
            if total > max_bytes * 2:
                raise BundleError("uncompressed size too large (possible archive bomb)")
            # header records (pax, GNU long names) count too: inflated and parsed, never in m.size
            if any(len(str(v)) > MAX_HEADER_RECORD for v in (m.pax_headers or {}).values()) or (
                tf.offset - total > MAX_HEADER_BYTES
            ):
                raise BundleError("archive headers too large (possible archive bomb)")
            if m.name.lstrip("./") == "submission.json" and m.isfile():
                fobj = tf.extractfile(m)
                manifest_raw = fobj.read(MAX_MANIFEST_BYTES + 1) if fobj else None
    if manifest_raw is None:
        raise BundleError("submission.json missing")
    return BundleInfo(h.hexdigest(), size, n, Manifest.parse(manifest_raw))


def pack_bundle(src_dir: Path | str, out_path: Path | str) -> Path:
    """Helper for miners/tests: pack a directory into a bundle (regular files only)."""
    src_dir, out_path = Path(src_dir), Path(out_path)
    with tarfile.open(out_path, "w:gz") as tf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_symlink():
                raise BundleError(f"symlinks are not allowed: {p}")
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(src_dir)))
    return out_path


if __name__ == "__main__":  # on the evaluation pod: `python3 inspect_bundle.py bundle.tar` (stdlib only)
    import sys

    try:
        info = inspect_bundle(sys.argv[1])
    except (BundleError, OSError, tarfile.TarError) as e:
        print(f"invalid bundle: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps({"sha256": info.sha256, "size": info.size, "members": info.members,
                      "name": info.manifest.name}))  # fmt: skip
