import io
import json
import tarfile

import pytest

from humanizer_eval.runner.bundle import BundleError, inspect_bundle, pack_bundle

MANIFEST = {"name": "t", "entrypoint": "humanize:humanize", "license": "MIT"}


def make_tar(tmp_path, members, name="b.tar.gz"):
    p = tmp_path / name
    with tarfile.open(p, "w:gz") as tf:
        for m in members:
            tf.addfile(*m) if isinstance(m, tuple) else tf.addfile(m)
    return p


def file_member(name, data: bytes):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    return ti, io.BytesIO(data)


def test_good_bundle(tmp_path):
    p = make_tar(
        tmp_path,
        [
            file_member("submission.json", json.dumps(MANIFEST).encode()),
            file_member("src/humanize.py", b"def humanize(x): return x"),
        ],
    )
    info = inspect_bundle(p)
    assert info.manifest.name == "t" and info.members == 2 and len(info.sha256) == 64


@pytest.mark.parametrize("bad", ["../evil.py", "/etc/passwd", "src/../../x", "a\\..\\b"])
def test_rejects_unsafe_paths(tmp_path, bad):
    p = make_tar(
        tmp_path, [file_member("submission.json", json.dumps(MANIFEST).encode()), file_member(bad, b"x")]
    )
    with pytest.raises(BundleError, match="unsafe path"):
        inspect_bundle(p)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_rejects_links_and_devices(tmp_path, kind):
    ti = tarfile.TarInfo("src/link")
    ti.type = kind
    ti.linkname = "/etc/shadow"
    p = make_tar(tmp_path, [file_member("submission.json", json.dumps(MANIFEST).encode()), ti])
    with pytest.raises(BundleError, match="only regular files"):
        inspect_bundle(p)


@pytest.mark.parametrize(
    "manifest,err",
    [
        ({"entrypoint": "x:y", "license": "MIT"}, "name"),
        ({**MANIFEST, "license": "GPL-3.0"}, "license"),
        ({**MANIFEST, "entrypoint": "os.system;rm:x"}, "entrypoint"),
        ({**MANIFEST, "batch_size": 10_000}, "batch_size"),
    ],
)
def test_manifest_validation(tmp_path, manifest, err):
    p = make_tar(tmp_path, [file_member("submission.json", json.dumps(manifest).encode())])
    with pytest.raises(BundleError, match=err):
        inspect_bundle(p)


def test_missing_manifest_and_size_limit(tmp_path):
    p = make_tar(tmp_path, [file_member("src/a.py", b"x")])
    with pytest.raises(BundleError, match="missing"):
        inspect_bundle(p)
    p2 = make_tar(tmp_path, [file_member("submission.json", json.dumps(MANIFEST).encode())], "c.tar.gz")
    with pytest.raises(BundleError, match="bytes"):
        inspect_bundle(p2, max_bytes=10)


def test_pack_bundle_refuses_symlinks(tmp_path):
    d = tmp_path / "sub"
    (d / "src").mkdir(parents=True)
    (d / "submission.json").write_text(json.dumps(MANIFEST))
    (d / "src" / "humanize.py").write_text("x")
    out = pack_bundle(d, tmp_path / "ok.tar.gz")
    assert inspect_bundle(out).manifest.name == "t"
    (d / "src" / "ln").symlink_to("/etc/passwd")
    with pytest.raises(BundleError, match="symlink"):
        pack_bundle(d, tmp_path / "bad.tar.gz")
