import hashlib

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
            if req.method == "HEAD" and getattr(self, "head_down", False):
                return httpx.Response(503)
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

    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=0))
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


def test_an_identical_bundle_belongs_to_the_earliest_commitment_not_the_first_api_call(tmp_path):
    """A copier commits the author's public 'repo@sha' and calls the API first. It must neither block the
    author (even unpaid) nor be evaluated in the author's place."""
    from humanizer_eval.intake.client import SeedSigner

    hub = FakeHub()
    author = SeedSigner(b"\x05" * 32).ss58_address
    thief = SeedSigner(b"\x06" * 32).ss58_address
    w = _world(tmp_path, hub, {author: (f"{REPO}@{COMMIT}", 50), thief: (f"{REPO}@{COMMIT}", 60)})
    t_sid = w.svc.submit_hf(thief)["submission_id"]  # first at the API, committed later
    a_sid = w.svc.submit_hf(author)["submission_id"]  # not refused as a duplicate
    for hk in (thief, author):
        w.pay(hk)
    w.converge()
    outs = [w.worker.run_once() for _ in range(3)]
    states = {o["submission_id"]: o["state"] for o in outs if o}
    assert states[t_sid] == "failed" and states[a_sid] == "done"
    assert w.billing.accounts.hold(t_sid).status == "released"  # the copy pays nothing
    # and a later copier is refused at once
    late = SeedSigner(b"\x07" * 32).ss58_address
    w.svc.commitment = lambda hk: {late: (f"{REPO}@{COMMIT}", 70)}.get(hk)
    with pytest.raises(Exception, match="counts once"):
        w.svc.submit_hf(late)


def test_a_hotkey_with_an_active_submission_is_refused_before_any_hugging_face_call(tmp_path):
    import sqlite3

    from humanizer_eval.intake.client import SeedSigner

    def quotes():
        with sqlite3.connect(tmp_path / "b.sqlite") as c:
            return c.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]

    hub = FakeHub()
    m = SeedSigner(b"\x05" * 32).ss58_address
    w = _world(tmp_path, hub, {m: (f"{REPO}@{COMMIT}", 50)})
    w.svc.submit_hf(m)
    n_calls, n_quotes = len(hub.calls), quotes()
    for _ in range(5):
        with pytest.raises(Exception, match="active submission"):
            w.svc.submit_hf(m)
    assert len(hub.calls) == n_calls and quotes() == n_quotes


def test_a_private_repo_may_be_paid_but_must_be_public_when_the_evaluation_starts(tmp_path):
    """Owner decision 2026-11-26: no check at submission (the miner can keep the repo private while paying,
    a few minutes); when the evaluation starts it must be public, else refused at once (nothing charged,
    the registration used). Tokens are never taken."""
    from humanizer_eval.intake.client import SeedSigner

    hub = FakeHub(private=True)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w = _world(tmp_path, hub, {m: (f"{REPO}@{COMMIT}", 50)})
    with pytest.raises(ValueError, match="no tokens"):
        w.svc.submit_hf(m, hf_token="hf_" + "a" * 30)
    sid = w.svc.submit_hf(m)["submission_id"]  # accepted while private
    w.pay(m)
    w.converge()
    assert w.worker.run_once()["state"] == "paused"  # just paid: it waits for the miner to open it
    w.t[0] += 16 * 60  # past the 15 min grace, still private: refused, no retry
    assert w.worker.run_once()["state"] == "failed"
    assert w.billing.accounts.hold(sid).status == "released" and w.queue.get(sid).state == "done"
    with pytest.raises(Exception, match="already made its submission"):
        w.svc.submit_hf(m)


def test_a_repo_made_public_after_paying_is_evaluated(tmp_path):
    from humanizer_eval.intake.client import SeedSigner

    hub = FakeHub(private=True)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w = _world(tmp_path, hub, {m: (f"{REPO}@{COMMIT}", 50)})
    sid = w.svc.submit_hf(m)["submission_id"]
    assert w.queue.get(sid).bundle_sha256 == ""
    w.pay(m)
    w.converge()
    hub.private = False  # the miner opens the repo
    assert w.worker.run_once()["state"] == "done" and w.queue.get(sid).bundle_sha256 == SHA


def test_a_copier_of_a_private_commitment_is_dropped_even_before_the_sha_is_known(tmp_path):
    """The author keeps the repo private while paying (sha unknown); a copier commits the same repo@commit
    later and is claimed first. It must not be evaluated with the author's bundle (audit 2, A2)."""
    from humanizer_eval.intake.client import SeedSigner

    hub = FakeHub(private=True)
    author = SeedSigner(b"\x05" * 32).ss58_address
    thief = SeedSigner(b"\x06" * 32).ss58_address
    w = _world(tmp_path, hub, {author: (f"{REPO}@{COMMIT}", 50), thief: (f"{REPO}@{COMMIT}", 60)})
    t_sid = w.svc.submit_hf(thief)["submission_id"]  # the copier reaches the API first (sha unknown)
    a_sid = w.svc.submit_hf(author)["submission_id"]  # the author, committed earlier: not refused
    w.pay(thief)
    w.converge()
    hub.private = False
    out = w.worker.run_once()  # the author has not paid yet: the copy waits (nothing run, nothing charged)
    assert out["submission_id"] == t_sid and out["state"] == "paused"
    assert w.queue.get(a_sid).state == "awaiting_payment" and w.billing.accounts.hold(t_sid).status == "open"
    w.pay(author)  # the author pays: the copy is dropped, its hold given back
    w.converge()
    w.t[0] += 3600
    for _ in range(3):
        w.worker.run_once()
    assert w.queue.get(t_sid).state == "failed"
    assert w.billing.accounts.hold(t_sid).status == "released"
