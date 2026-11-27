import json
import shutil
from dataclasses import replace

from evaluation.test_pipeline import KEY, _outputs
from humanizer_eval.intake.client import SeedSigner
from humanizer_eval.period.crypto import encrypt_at_rest
from humanizer_eval.period.merkle import leaf_hash, merkle_root
from humanizer_eval.period.reveal import verify_reveal, write_reveal
from humanizer_eval.period.seed import commitment, derive_seed
from intake.test_intake import URL, World, _sha

SECRET = bytes(range(1, 33))
BH = "56" * 32


def _period_world(tmp_path):
    """A crowned challenger, a loser and the baseline king's run in one period, as the server leaves them."""
    w = World(tmp_path)
    w.ctx = replace(w.ctx, seed=derive_seed(SECRET, BH))
    human_ref = [{"id": hid, "genre": s.split("|")[0], "length_bin": s.split("|")[1]}
                 for hid, s in zip(w.ctx.human_ids, w.ctx.human_strata, strict=True)]  # fmt: skip
    pdir = tmp_path / "pdir"
    pdir.mkdir()
    root = merkle_root(
        [leaf_hash(it) for it in w.ctx.items] + [leaf_hash({"human_id": h["id"]}) for h in human_ref]
    )
    (pdir / "manifest.json").write_text(json.dumps({"period_id": w.ctx.period_id, "commitment": commitment(SECRET),
                                                    "block": 9526464, "block_hash": BH,
                                                    "merkle_root": root.hex(), "n_items": len(w.ctx.items)}))  # fmt: skip
    (pdir / "dataset.enc").write_bytes(
        encrypt_at_rest(KEY, json.dumps({"items": w.ctx.items, "human_ref": human_ref}).encode())
    )
    (pdir / "secret.hex").write_text(SECRET.hex())
    a, b = SeedSigner(b"\x05" * 32).ss58_address, SeedSigner(b"\x06" * 32).ss58_address
    w.pay(a)
    w.pay(b)
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    w.bundles["https://huggingface.co/m/b2/resolve/main/b.tar"] = _outputs(w.ctx)
    sb = w.svc.submit(b, bundle_url="https://huggingface.co/m/b2/resolve/main/b.tar", bundle_sha256=_sha(b),
                      bundle_bytes=10)["submission_id"]  # fmt: skip
    assert not w.worker.run_once()["crowned"]  # same outputs as the king: no crown
    store = w.store_for(w.ctx.period_id)
    kr = tmp_path / "king_run.t.enc"
    shutil.copy(tmp_path / f"king_run.{w.ctx.period_id}.enc", kr)
    hs = {"ids": w.ctx.human_ids, "scores": w.ctx.human_scores}
    return w, pdir, {"submissions_dir": store.root, "king_runs": [kr]}, hs, (sa, sb)


SIGNER = "b853864e1d747a09c664e15b5b3a9a30a1298e4af270c985122a2fefeab5ec7f"
SIGNER_SS58 = "5GEPUnnRQu2h4EURMQBchrgsiFYTm62mXcV2URaX5xC99kE7"
TS_EXTRINSIC = "0x280502000b800880f9a101"  # testnet block 8487085, the timestamp inherent


class FakeChain:
    """Block bodies with the notary's remarks where the schedule says they are."""

    def __init__(self, remarks):
        self.remarks = remarks  # block -> remark text

    def block_hash(self, n):
        return BH

    def block_extrinsics(self, n):
        ex = [TS_EXTRINSIC]
        if n in self.remarks:
            ex.append("0x" + "45028400" + SIGNER + "01" * 64 + self.remarks[n].encode().hex())
        return ex


def _schedule(manifest, cblock=9526400, mblock=9526500):
    from humanizer_eval.period.notary import commitment_text, merkle_text

    ct = commitment_text(32, manifest["period_id"], manifest["commitment"], manifest["block"])
    mt = merkle_text(
        32, manifest["period_id"], manifest["merkle_root"], manifest["n_items"], manifest["block_hash"]
    )
    rec = {"commitment": manifest["commitment"], "seed_block": manifest["block"],
           "notary_commitment": {"block": cblock, "signer_pubkey": SIGNER, "text": ct},
           "notary_merkle": {"block": mblock, "signer_pubkey": SIGNER, "text": mt}}  # fmt: skip
    return rec, FakeChain({cblock: ct, mblock: mt})


def test_reveal_round_trip_recomputes_every_decision(tmp_path):
    w, pdir, sdir, hs, (sa, sb) = _period_world(tmp_path)
    out = tmp_path / "reveal" / w.ctx.period_id
    rec, chain = _schedule(json.loads((pdir / "manifest.json").read_text()))
    info = write_reveal(w.ctx.period_id, period_dir=pdir, **sdir, human_scores=hs, key=KEY, out_dir=out,
                        schedule_record=rec)  # fmt: skip
    assert info["submissions"] == 3  # two miners and the owner's baseline run
    rec = json.loads((out / "submissions" / f"{sa}.json").read_text())["record"]
    assert rec["outputs"] and rec["crown"]["king"] == "baseline-identity"
    assert (out / "secret.hex").read_text().strip() == SECRET.hex() and (out / "king_run.t.json").exists()

    rep = verify_reveal(out, chain=chain, notary_address=SIGNER_SS58)
    assert rep["ok"], rep["failures"]
    text = "\n".join(rep["checks"])
    assert (
        f"remarks signed by {SIGNER_SS58}" in text and "commitment block 9526400 < seed block 9526464" in text
    )
    assert "H(secret) equals" in text and "Merkle root" in text
    assert f"{sa}: crown test against baseline-identity recomputed" in text
    assert f"{sb}: crown test against {sa} recomputed" in text


def test_tampering_is_caught(tmp_path):
    w, pdir, sdir, hs, (_, sb) = _period_world(tmp_path)
    out = tmp_path / "reveal" / w.ctx.period_id
    write_reveal(w.ctx.period_id, period_dir=pdir, **sdir, human_scores=hs, key=KEY, out_dir=out)
    p = out / "submissions" / f"{sb}.json"
    e = json.loads(p.read_text())
    e["record"]["score"]["S"] += 0.01
    p.write_text(json.dumps(e))
    rep = verify_reveal(out)
    assert not rep["ok"]
    assert any("sha256 of submissions" in f for f in rep["failures"])
    assert any(f"{sb}: S =" in f for f in rep["failures"])


def test_pod_details_are_not_published(tmp_path):
    from humanizer_eval.period.reveal import public_record

    rec = {
        "run": {"status": "ok", "meta": {"pod": {"ssh": "root@1.2.3.4"}, "attempts": [1], "executor": "H100"}}
    }
    assert public_record(rec)["run"]["meta"] == {"executor": "H100"}


def test_notary_remarks_that_are_missing_or_late_fail(tmp_path):
    from humanizer_eval.period.notary import verify_notary

    _, pdir, *_ = _period_world(tmp_path)
    m = json.loads((pdir / "manifest.json").read_text())
    rec, chain = _schedule(m, cblock=9526470)  # after the seed block: the secret could follow B's hash
    _, bad = verify_notary(rec, m, chain)
    assert bad == ["commitment block 9526470 < seed block 9526464"]
    rec, chain = _schedule(m)
    chain.remarks.pop(9526500)  # the Merkle-root remark is not where the schedule says
    assert verify_notary(rec, m, chain)[1] == ["Merkle root on chain in block 9526500"]
    assert verify_notary({}, m, chain)[1][0].endswith("were not notarized on chain")
    _, chain = _schedule(m)
    assert verify_notary(rec, m, chain, notary_address="5FX72T8QELk9HbUKDXk9Kj9MYpAToHHKWegZDcGFiQ344dYR")[1] == [
        "remarks signed by 5FX72T8QELk9HbUKDXk9Kj9MYpAToHHKWegZDcGFiQ344dYR"]  # fmt: skip


def test_ss58_and_block_time_decoding():
    from humanizer_eval.period.notary import block_time, ss58_pubkey

    assert ss58_pubkey(SIGNER_SS58) == SIGNER
    t = block_time(FakeChain({}), 8487085)
    assert 1.79e9 < t < 1.80e9  # 2026-11


def test_evaluations_before_the_notarized_merkle_root_are_caught(tmp_path):
    import hashlib

    w, pdir, sdir, hs, _ = _period_world(tmp_path)
    out = tmp_path / "reveal" / w.ctx.period_id
    rec, chain = _schedule(json.loads((pdir / "manifest.json").read_text()))
    write_reveal(w.ctx.period_id, period_dir=pdir, **sdir, human_scores=hs, key=KEY, out_dir=out,
                 schedule_record=rec)  # fmt: skip
    for p in (out / "submissions").glob("*.json"):  # claim the evaluations ran in 2023
        e = json.loads(p.read_text())
        e["record"]["started_at"] = 1.7e9
        p.write_text(json.dumps(e))
    files = {
        k: hashlib.sha256((out / k).read_bytes()).hexdigest()
        for k in json.loads((out / "files.json").read_text())
    }
    (out / "files.json").write_text(json.dumps(files))
    rep = verify_reveal(out, chain=chain)
    assert rep["failures"] == [f"Merkle root on chain before the first evaluation of {w.ctx.period_id}"]


def test_a_period_without_a_seed_block_fails_cleanly(tmp_path):
    from humanizer_eval.period.notary import verify_notary

    _, pdir, *_ = _period_world(tmp_path)
    m = json.loads((pdir / "manifest.json").read_text())
    rec, chain = _schedule(m)
    assert verify_notary(rec, {**m, "block": None}, chain)[1] == [
        f"{m['period_id']}: the manifest names no seed block"
    ]
    assert verify_notary({}, {"period_id": "pilot-1"}, chain)[1][0].endswith("were not notarized on chain")
