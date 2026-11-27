import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from humanizer_eval.intake.archive import Archive
from humanizer_eval.intake.hf import HubClient
from humanizer_eval.publish.kings import HFMirror, KingPublisher
from humanizer_eval.scoring.crown import KingState

BUNDLE = b"king bundle bytes" * 500
SHA = hashlib.sha256(BUNDLE).hexdigest()
REPO, COMMIT = "miner1/humanizer", "c" * 40
T0 = datetime(2026, 11, 1, tzinfo=UTC)
H = 3600.0


class Hub:
    """Hub API + resolve for several repos; each repo: public / private / gated / gone / down."""

    def __init__(self):
        self.repos = {REPO: {"commit": COMMIT, "state": "public", "sha": SHA, "branches": ["main"]}}

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert "authorization" not in req.headers  # publication checks what anyone sees
        p = req.url.path
        for name, r in self.repos.items():
            if r["state"] == "down":
                return httpx.Response(503)
            closed = r["state"] in ("private", "gone")
            path = r.get("path", "bundle.tar")
            if p == f"/api/models/{name}/revision/{r['commit']}":
                if closed:
                    return httpx.Response(401 if r["state"] == "private" else 404)
                sib = {"rfilename": path, "lfs": {"sha256": r["sha"], "size": len(BUNDLE)}}
                gated = "manual" if r["state"] == "gated" else False
                return httpx.Response(200, json={"sha": r["commit"], "private": False, "gated": gated,
                                                 "siblings": [sib]})  # fmt: skip
            if p == f"/api/models/{name}/refs":
                if closed:
                    return httpx.Response(404)
                return httpx.Response(200, json={"branches": [{"name": b} for b in r["branches"]]})
            if p == f"/{name}/resolve/{r['commit']}/{path}":
                if closed or r["state"] == "gated":
                    return httpx.Response(401)
                return httpx.Response(302, headers={"location": "https://cdn.example.org/x"})
        return httpx.Response(404)

    def client(self) -> HubClient:
        return HubClient(http=httpx.Client(transport=httpx.MockTransport(self.handler)))


class FakeApi:
    """The two huggingface_hub calls HFMirror makes; the upload lands in the fake Hub."""

    def __init__(self, hub: Hub, sha_of_upload: str = SHA):
        self.hub, self.sha, self.commits = hub, sha_of_upload, []

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        files = {op.path_in_repo: op for op in operations}
        assert repo_type == "model" and set(files) == {"kings/sub-1/bundle.tar", "kings/sub-1/README.md"}
        commit = hashlib.sha1(f"{repo_id}{len(self.commits)}".encode()).hexdigest()
        self.commits.append((repo_id, commit, files))
        self.hub.repos[repo_id] = {"commit": commit, "state": "public", "sha": self.sha, "branches": ["main"],
                                   "path": "kings/sub-1/bundle.tar"}  # fmt: skip
        return SimpleNamespace(oid=commit)


def _world(tmp_path, *, mirror=True, archived=True, sha_of_upload=SHA):
    tmp_path.mkdir(parents=True, exist_ok=True)
    hub = Hub()
    ks = KingState.initial("baseline", "owner-hk", T0)
    ks.apply_crown("sub-1", "hk-1", T0 + timedelta(hours=1), 0.03, registered_at=5)
    kp = tmp_path / "king_state.json"
    kp.write_text(ks.to_json())
    rows = {"sub-1": SimpleNamespace(hf_repo=REPO, hf_revision=COMMIT, bundle_sha256=SHA,
                                     bundle_bytes=len(BUNDLE))}  # fmt: skip
    archive = Archive(tmp_path / "archive")
    if archived:
        archive.path("sub-1").parent.mkdir(parents=True)
        archive.path("sub-1").write_bytes(BUNDLE)
    if mirror:
        pytest.importorskip("huggingface_hub")
    api = FakeApi(hub, sha_of_upload)
    m = HFMirror("sn32-owner/sn32-kings", "", api=api) if mirror else None
    clock = {"t": (T0 + timedelta(hours=1)).timestamp()}
    pub = KingPublisher(
        tmp_path / "pubs.json", kp, rows.get, hub.client(), archive, m, now=lambda: clock["t"]
    )
    return hub, api, pub, clock


def _bundle(pub, sid="sub-1"):
    return next(k["bundle"] for k in pub.public()["kings"] if k["submission_id"] == sid)


def test_miner_repo_is_the_public_link_48h_after_the_crown(tmp_path):
    _hub, _, pub, clock = _world(tmp_path, mirror=False)
    clock["t"] += 47 * H
    assert pub.run_once() == []
    b = _bundle(pub)
    assert b["source"] == "pending" and "url" not in b and "original" not in b
    assert b["public_at"] == (T0 + timedelta(hours=49)).isoformat()
    assert _bundle(pub, "baseline") == {"source": "baseline"}
    clock["t"] += 1 * H
    assert pub.run_once() == ["sub-1"]
    b = _bundle(pub)
    assert b["source"] == "miner" and b["url"] == f"https://huggingface.co/{REPO}/tree/{COMMIT}"
    assert b["file_url"] == f"https://huggingface.co/{REPO}/resolve/{COMMIT}/bundle.tar"
    assert b["original"] == f"{REPO}@{COMMIT}" and b["sha256"] == SHA


@pytest.mark.parametrize("state", ["private", "gated", "gone"])
def test_closed_repo_gets_our_copy(tmp_path, state):
    hub, api, pub, clock = _world(tmp_path)
    hub.repos[REPO]["state"] = state
    clock["t"] += 48 * H
    assert pub.run_once() == ["sub-1"]
    b = _bundle(pub)
    repo, commit, files = api.commits[0]
    assert repo == "sn32-owner/sn32-kings" and b["source"] == "mirror"
    assert b["url"] == f"https://huggingface.co/{repo}/tree/{commit}/kings/sub-1"
    assert b["file_url"] == f"https://huggingface.co/{repo}/resolve/{commit}/kings/sub-1/bundle.tar"
    bundle = files["kings/sub-1/bundle.tar"].path_or_fileobj
    assert bundle == str(tmp_path / "archive" / "sub-1" / "bundle.tar")
    readme = files["kings/sub-1/README.md"].path_or_fileobj.decode()
    assert f"{REPO}@{COMMIT}" in readme and SHA in readme
    assert b["original"] == f"{REPO}@{COMMIT}" and b["note"]
    clock["t"] += 100 * H
    hub.repos[REPO]["state"] = "public"
    assert pub.run_once() == [] and len(api.commits) == 1  # our copy stays for good


def test_a_branch_shadowing_the_commit_is_not_a_valid_link(tmp_path):
    hub, _api, pub, clock = _world(tmp_path)
    hub.repos[REPO]["branches"].append(COMMIT)
    clock["t"] += 48 * H
    pub.run_once()
    assert _bundle(pub)["source"] == "mirror" and "ambiguous" in _bundle(pub)["note"]


def test_link_rechecked_and_replaced_when_the_miner_closes_the_repo(tmp_path):
    hub, _api, pub, clock = _world(tmp_path)
    clock["t"] += 48 * H
    pub.run_once()
    assert _bundle(pub)["source"] == "miner"
    hub.repos[REPO]["state"] = "private"
    clock["t"] += 5 * H
    assert pub.run_once() == [] and _bundle(pub)["source"] == "miner"  # not re-checked yet
    clock["t"] += 1 * H
    assert pub.run_once() == ["sub-1"] and _bundle(pub)["source"] == "mirror"


def test_hub_outage_changes_nothing(tmp_path):
    hub, api, pub, clock = _world(tmp_path)
    hub.repos[REPO]["state"] = "down"
    clock["t"] += 48 * H
    assert pub.run_once() == [] and _bundle(pub)["source"] == "pending" and not api.commits
    hub.repos[REPO]["state"] = "public"
    assert pub.run_once() == ["sub-1"] and _bundle(pub)["source"] == "miner"


def test_without_mirror_or_copy_the_king_is_unavailable_and_retried(tmp_path):
    hub, _, pub, clock = _world(tmp_path, mirror=False)
    hub.repos[REPO]["state"] = "gone"
    clock["t"] += 48 * H
    pub.run_once()
    b = _bundle(pub)
    assert b["source"] == "unavailable" and b["url"] is None and "no mirror" in b["note"]
    hub.repos[REPO]["state"] = "public"
    clock["t"] += 0.5 * H
    assert pub.run_once() == []  # retried hourly
    clock["t"] += 0.5 * H
    assert pub.run_once() == ["sub-1"] and _bundle(pub)["source"] == "miner"

    hub2, api2, pub2, clock2 = _world(tmp_path / "b", archived=False)
    hub2.repos[REPO]["state"] = "gone"
    clock2["t"] += 48 * H
    pub2.run_once()
    assert _bundle(pub2)["source"] == "unavailable" and "copy is missing" in _bundle(pub2)["note"]
    assert not api2.commits


def test_mirror_that_does_not_match_is_not_published(tmp_path):
    hub, _api, pub, clock = _world(tmp_path, sha_of_upload="0" * 64)
    hub.repos[REPO]["state"] = "gone"
    clock["t"] += 48 * H
    pub.run_once()
    b = _bundle(pub)
    assert b["source"] == "unavailable" and b["url"] is None and "does not match" in b["note"]


def test_state_file_and_api_route(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from humanizer_eval.intake.api import create_app

    _hub, _, pub, clock = _world(tmp_path, mirror=False)
    clock["t"] += 48 * H
    pub.run_once()
    saved = json.loads((tmp_path / "pubs.json").read_text())
    assert saved["sub-1"]["source"] == "miner" and "baseline" not in saved
    app = create_app(SimpleNamespace(), SimpleNamespace(), kings=pub.public)
    r = TestClient(app).get("/v1/kings")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    kings = r.json()["kings"]
    assert [k["submission_id"] for k in kings] == ["sub-1", "baseline"]
    assert kings[0]["bundle"]["source"] == "miner"


def test_prune_keeps_kings_that_can_return_and_mirrors_before_deleting(tmp_path):
    _, api, pub, clock = _world(tmp_path)
    registered = {"hk-1": True}

    def is_reg(hk, reg):
        return registered.get(hk, False)

    clock["t"] += 49 * H
    pub.run_once()
    assert _bundle(pub)["source"] == "miner"
    assert pub.prune(is_reg) == [] and pub.archive.has("sub-1")  # still registered: may be king again
    registered["hk-1"] = False  # the registration ended: this king never returns
    assert pub.prune(is_reg) == ["sub-1"]
    assert not pub.archive.has("sub-1") and len(api.commits) == 1  # mirrored first, then deleted
    b = _bundle(pub)
    assert b["source"] == "mirror" and b["url"].startswith("https://huggingface.co/sn32-owner/sn32-kings")


def test_prune_without_a_mirror_deletes_nothing(tmp_path):
    *_, pub, clock = _world(tmp_path, mirror=False)
    clock["t"] += 49 * H
    pub.run_once()
    assert pub.prune(lambda hk, reg: False) == [] and pub.archive.has("sub-1")


def test_prune_waits_for_the_publication_delay(tmp_path):
    _, _, pub, _ = _world(tmp_path)
    assert pub.prune(lambda hk, reg: False) == [] and pub.archive.has("sub-1")
