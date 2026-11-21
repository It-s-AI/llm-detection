import hashlib
import json

import httpx
import pytest

from humanizer_eval.intake.archive import Archive, ArchiveError
from humanizer_eval.intake.hf import HubClient, HubError, commitment_for, parse_commitment

REPO = "miner1/humanizer"
COMMIT = "a" * 40
BUNDLE = b"tar bytes of the bundle" * 1000
SHA = hashlib.sha256(BUNDLE).hexdigest()


def test_commitment_pins_a_commit_never_a_branch():
    assert parse_commitment(f"{REPO}@{COMMIT}") == (REPO, COMMIT)
    for bad in (
        f"{REPO}@main",
        f"{REPO}@v1.0",
        f"{REPO}@{COMMIT[:12]}",
        f"{REPO}@{COMMIT.upper()}",
        f"{REPO}",
        f"{REPO}@{COMMIT}@x",
        f"../etc@{COMMIT}",
        f"miner1/{'x' * 90}@{COMMIT}",  # over the chain's 128 bytes
        f"miner1/a b@{COMMIT}",
    ):
        with pytest.raises(HubError):
            parse_commitment(bad)
    assert commitment_for(REPO, COMMIT) == f"{REPO}@{COMMIT}"


class FakeHub:
    """Hub API + resolve + CDN for one repo."""

    def __init__(self, *, private=False, lfs=True, branch_named_like_commit=False, served=BUNDLE):
        self.private, self.lfs, self.ambiguous, self.served = private, lfs, branch_named_like_commit, served
        self.token = "hf_" + "t" * 30
        self.calls = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        p = req.url.path
        self.calls.append((req.method, p, req.headers.get("authorization")))
        authed = req.headers.get("authorization") == f"Bearer {self.token}"
        if req.url.host == "cdn.example.org":
            return httpx.Response(200, content=self.served)
        if self.private and not authed:
            return httpx.Response(401)
        if p == f"/api/models/{REPO}/revision/{COMMIT}":
            sib = {"rfilename": "bundle.tar", "size": len(BUNDLE)}
            if self.lfs:
                sib["lfs"] = {"sha256": SHA, "size": len(BUNDLE), "pointerSize": 130}
            return httpx.Response(200, json={"sha": COMMIT, "private": self.private, "siblings": [sib]})
        if p.startswith(f"/api/models/{REPO}/revision/"):
            return httpx.Response(404)
        if p == f"/api/models/{REPO}/refs":
            names = [{"name": "main"}] + ([{"name": COMMIT}] if self.ambiguous else [])
            return httpx.Response(200, json={"branches": names, "tags": [], "converts": []})
        if p == f"/{REPO}/resolve/{COMMIT}/bundle.tar":
            if req.method == "HEAD" or not self.lfs:
                if not self.lfs and req.method == "GET":
                    return httpx.Response(200, content=self.served)
                return httpx.Response(302, headers={"location": "https://cdn.example.org/b?sig=x"})
            return httpx.Response(302, headers={"location": "https://cdn.example.org/b?sig=x"})
        return httpx.Response(404)

    def client(self) -> HubClient:
        return HubClient(http=httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_bundle_info_from_lfs_metadata():
    hub = FakeHub()
    info = hub.client().bundle_info(REPO, COMMIT)
    assert (info.sha256, info.size, info.private, info.lfs) == (SHA, len(BUNDLE), False, True)
    assert (
        hub.client().download_url(REPO, COMMIT)
        == f"https://huggingface.co/{REPO}/resolve/{COMMIT}/bundle.tar"
    )


def test_small_non_lfs_bundle_is_hashed_here():
    hub = FakeHub(lfs=False)
    assert hub.client().bundle_info(REPO, COMMIT).sha256 == SHA


def test_ambiguous_or_missing_revision_refused():
    with pytest.raises(HubError, match="ambiguous"):
        FakeHub(branch_named_like_commit=True).client().bundle_info(REPO, COMMIT)
    with pytest.raises(HubError, match="not found"):
        FakeHub().client().bundle_info(REPO, "b" * 40)


def test_private_repo_needs_the_token_and_gives_a_signed_link():
    hub = FakeHub(private=True)
    with pytest.raises(HubError):
        hub.client().bundle_info(REPO, COMMIT)
    assert hub.client().bundle_info(REPO, COMMIT, hub.token).private
    assert hub.client().download_url(REPO, COMMIT, hub.token) == "https://cdn.example.org/b?sig=x"


def test_archive_checks_sha_and_refuses_plain_http(tmp_path):
    hub = FakeHub()
    a = Archive(tmp_path, http=httpx.Client(transport=httpx.MockTransport(hub.handler)))
    url = hub.client().resolve_url(REPO, COMMIT)
    p = a.store("s1", url, SHA, 10**9)
    assert p.read_bytes() == BUNDLE and a.has("s1")
    with pytest.raises(ArchiveError, match="no longer matches"):
        a.store("s2", url, "0" * 64, 10**9)
    assert not a.has("s2") and not list((tmp_path / "s2").glob("*.part"))
    with pytest.raises(ArchiveError, match="larger"):
        a.store("s3", url, SHA, 100)
    with pytest.raises(ArchiveError, match="https"):
        a.store("s4", "http://cdn.example.org/b", SHA, 10**9)

    def to_http(req):
        return httpx.Response(302, headers={"location": "http://evil.example.org/b"})

    a2 = Archive(tmp_path, http=httpx.Client(transport=httpx.MockTransport(to_http)))
    with pytest.raises(ArchiveError, match="https"):
        a2.store("s5", url, SHA, 10**9)


# --- submissions pinned by the commitment, crowns after the archive ------------------------------------
def _world(tmp_path, hub, commitments):
    from humanizer_eval.intake import IntakeLimits

    from .test_intake import World

    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=0))
    w.svc.hub = w.env.hub = hub.client()
    w.svc.commitment = lambda hk: commitments.get(hk)
    w.svc.secret_key = w.env.secret_key = bytes(range(32))
    w.env.archive = Archive(
        tmp_path / "archive", http=httpx.Client(transport=httpx.MockTransport(hub.handler))
    )
    pinned = hub.client().resolve_url(REPO, COMMIT)
    w.bundles[pinned] = w.bundles[next(iter(w.bundles))]
    w.bundles["https://cdn.example.org/b?sig=x"] = w.bundles[pinned]
    runner_for = w.env.runner_for

    def runner(sub):  # the king's re-run from the archive: a local bundle path
        if sub.bundle is not None:
            w.from_archive = str(sub.bundle)
            sub = type(sub)(sub.submission_id, sub.hotkey, bundle_url=pinned, bundle_sha256=SHA)
        return runner_for(sub)

    w.env.runner_for = runner
    return w


def test_submission_pinned_by_commitment_and_crown_after_archive(tmp_path):
    from humanizer_eval.intake.client import SeedSigner
    from humanizer_eval.scoring.crown import KingState

    hub = FakeHub()
    m = SeedSigner(b"\x05" * 32).ss58_address
    commitments = {}
    w = _world(tmp_path, hub, commitments)
    w.pay(m)
    with pytest.raises(Exception, match="no commitment"):
        w.svc.submit_hf(m)
    commitments[m] = (f"{REPO}@{COMMIT}", 3)  # before the registration (block 7)
    with pytest.raises(Exception, match="predates this registration"):
        w.svc.submit_hf(m)
    commitments[m] = (f"{REPO}@{COMMIT}", 50)
    with pytest.raises(Exception, match="the chain shows"):
        w.svc.submit_hf(m, commitment=f"{REPO}@{'b' * 40}")
    with pytest.raises(ValueError, match="development only"):
        w.svc.submit(m, bundle_url="https://x.example.org/b.tar", bundle_sha256=SHA, bundle_bytes=10)
    out = w.svc.submit_hf(m, commitment=f"{REPO}@{COMMIT}")
    assert out["state"] == "queued" and out["bundle"] == {"repo": REPO, "commit": COMMIT, "sha256": SHA,
                                                         "bytes": len(BUNDLE)}  # fmt: skip
    sid = out["submission_id"]
    res = w.worker.run_once()
    assert res["crowned"] is True
    st = w.svc.status(m, sid)
    assert st["evaluation"]["crowned"] is True and st["evaluation"]["archived"] is True
    assert st["bundle"]["archived"] and w.env.archive.path(sid).read_bytes() == BUNDLE
    assert KingState.from_json((tmp_path / "king_state.json").read_text()).current.submission_id == sid
    # the next period re-runs the king from our archive, not from the miner's repo
    from dataclasses import replace

    w.ctx = replace(w.ctx, period_id="p-next")
    w.worker.run_once()
    assert w.from_archive == str(w.env.archive.path(sid))


def test_crown_not_awarded_when_the_bundle_is_gone(tmp_path):
    from humanizer_eval.intake.client import SeedSigner
    from humanizer_eval.scoring.crown import KingState

    hub = FakeHub()
    m = SeedSigner(b"\x05" * 32).ss58_address
    w = _world(tmp_path, hub, {m: (f"{REPO}@{COMMIT}", 50)})
    w.pay(m)
    sid = w.svc.submit_hf(m)["submission_id"]
    w.worker.ensure_king(w.ctx)
    hub.served = b"changed after the evaluation"
    assert w.worker.run_once()["crowned"] is False
    ev = w.svc.status(m, sid)["evaluation"]
    assert ev["crowned"] is False and "crown_revoked" in ev and ev["stage"] == "done"
    st = (
        KingState.from_json((tmp_path / "king_state.json").read_text())
        if (tmp_path / "king_state.json").exists()
        else None
    )
    assert st is None or st.current.is_baseline


def test_private_repo_token_is_encrypted_and_dropped_after_the_run(tmp_path):
    from humanizer_eval.intake.client import SeedSigner

    hub = FakeHub(private=True)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w = _world(tmp_path, hub, {m: (f"{REPO}@{COMMIT}", 50)})
    w.pay(m)
    with pytest.raises(ValueError, match="private"):
        w.svc.submit_hf(m)  # public metadata unreadable without the token
    sid = w.svc.submit_hf(m, hf_token=hub.token)["submission_id"]
    raw = (tmp_path / "q.sqlite").read_bytes()
    assert hub.token.encode() not in raw and w.queue.get(sid).secret
    w.worker.run_once()
    assert w.queue.get(sid).secret is None
    # the pod got the signed CDN link, never the token
    assert any(
        c == ("HEAD", f"/{REPO}/resolve/{COMMIT}/bundle.tar", f"Bearer {hub.token}") for c in hub.calls
    )
    assert json.dumps(w.svc.status(m, sid)).find(hub.token) == -1
