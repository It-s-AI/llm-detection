import json
import random

from nacl.public import PrivateKey

from humanizer_eval.llm.openrouter import ChatResult
from humanizer_eval.period.build import build_period, save_period
from humanizer_eval.period.crypto import (
    decrypt_at_rest,
    encrypt_at_rest,
    load_or_create_key,
    open_sealed,
    seal_to,
)
from humanizer_eval.period.generate import GenConfig, clean_generation, generate_input
from humanizer_eval.period.merkle import leaf_hash, merkle_proof, merkle_root, verify_proof
from humanizer_eval.period.seed import commitment, derive_seed, make_secret, subseed, verify_commitment

BLOCK = "0x" + "ab" * 32


def test_commit_reveal_and_subseeds():
    s = make_secret()
    c = commitment(s)
    assert verify_commitment(c, s) and not verify_commitment(c, make_secret())
    seed = derive_seed(s, BLOCK)
    assert seed == derive_seed(s, BLOCK) and seed != derive_seed(s, "0x" + "cd" * 32)
    assert subseed(seed, "a") != subseed(seed, "b") and subseed(seed, "a") == subseed(seed, "a")


def test_merkle_proofs():
    leaves = [leaf_hash({"i": i}) for i in range(7)]
    root = merkle_root(leaves)
    for i in range(7):
        assert verify_proof(leaves[i], merkle_proof(leaves, i), root)
    assert not verify_proof(leaf_hash({"i": 99}), merkle_proof(leaves, 0), root)


def test_crypto_roundtrips(tmp_path):
    key = load_or_create_key(tmp_path / "k")
    assert load_or_create_key(tmp_path / "k") == key
    assert decrypt_at_rest(key, encrypt_at_rest(key, b"secret")) == b"secret"
    sk = PrivateKey.generate()
    assert open_sealed(sk, seal_to(sk.public_key.encode().hex(), b"inputs")) == b"inputs"


def test_clean_generation_strips_wrappers():
    raw = "Sure! Here's your essay:\n\n# Why Cities Need Trees\n\nTrees cool streets. They help.\n\nLet me know if you need edits!"
    assert clean_generation(raw) == "Trees cool streets. They help."


class FakeLLM:
    """Topic proposals and generations are deterministic functions of the prompt."""

    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, **kw):
        user = messages[-1]["content"]
        self.prompts.append(user)
        purpose = kw.get("purpose", "")
        if purpose == "period:topics":
            out = {
                "topics": [
                    {
                        "topic": f"topic {i} about {user.splitlines()[1]}",
                        "audience": "students",
                        "stance": "it matters",
                    }
                    for i in range(30)
                ]
            }
            return ChatResult(json.dumps(out), model, "fake", 1, 1, 0.001, None, json=out)
        if purpose == "period:generate":
            words = int(__import__("re").search(r"(\d+)[ -]?words?\b", user).group(1))
            text = " ".join(["word"] * words) + "."
            return ChatResult(text, model, "fake", 1, 1, 0.002, None)
        if purpose.endswith(":facts"):
            out = {
                "facts": [{"fact": "A fact.", "sentences": [0]}, {"fact": "Another fact.", "sentences": [0]}]
            }
            return ChatResult(json.dumps(out), model, "fake", 1, 1, 0.001, None, json=out)
        if purpose.endswith(":validate"):
            ids = __import__("re").findall(r"^(\d+)\. FACT:", user, __import__("re").M)
            out = {"verdicts": [{"id": int(i), "verdict": "supported", "evidence": ""} for i in ids]}
            return ChatResult(json.dumps(out), model, "fake", 1, 1, 0.001, None, json=out)
        raise AssertionError(purpose)


class FakePool:
    def __init__(self):
        self.texts = [
            {
                "id": f"h{i}",
                "text": f"POOLTEXT secret human text {i}",
                "genre": g,
                "cluster_id": 0,
                "length_bin": "200-250",
                "register": "formal",
            }
            for i, g in enumerate(["essay", "news"] * 50)
        ]

    def sample_slots(self, n, genre_proportions, seed):
        rng = random.Random(seed)
        return [
            {
                "genre": rng.choice(["essay", "news"]),
                "cluster_id": 0,
                "cluster_label": "school life",
                "length_bin": "200-250",
                "register": "formal",
            }
            for _ in range(n)
        ]

    def sample_reference(self, slot_counts, seed):
        rng = random.Random(seed)
        out = []
        for (genre, *_rest), k in slot_counts.items():
            cands = [t for t in self.texts if t["genre"] == genre]
            out += rng.sample(cands, min(k, len(cands)))
        return out


CFG = GenConfig(
    generators=[{"model": "m/a", "weight": 1}, {"model": "m/b", "weight": 1}],
    genre_proportions={"essay": 1, "news": 1},
    length_words={"min": 150, "max": 300, "mean": 220},
    topic_model="m/t",
    facts_model="m/f",
    fact_validators=["m/v1", "m/v2"],
)


def test_generate_input_respects_length_bin():
    g = generate_input(
        FakeLLM(),
        random.Random(1),
        CFG,
        {"genre": "essay", "length_bin": "200-250", "register": "informal"},
        {"topic": "x", "stance": "y"},
    )
    assert g.ok and 200 <= g.target_words <= 250


def test_build_period_is_deterministic_and_never_sends_pool_text(tmp_path):
    llm, pool = FakeLLM(), FakePool()
    secret = b"s" * 32
    r1 = build_period("p1", secret, BLOCK, pool, llm, CFG, n=20, workers=4)
    r2 = build_period("p1", secret, BLOCK, FakePool(), FakeLLM(), CFG, n=20, workers=4)
    assert r1.manifest["merkle_root"] == r2.manifest["merkle_root"]
    assert r1.manifest["n_items"] == 20 and r1.manifest["n_human_ref"] >= 2
    # D-51: no pool text ever reaches any LLM prompt
    assert not any("POOLTEXT" in p for p in llm.prompts)
    # human reference matches the slot strata counts
    assert {h["genre"] for h in r1.human_ref} <= {it["slot"]["genre"] for it in r1.items}
    out = save_period(r1, tmp_path / "p1", b"k" * 32)
    assert "POOLTEXT" not in (out / "manifest.json").read_text()
    assert b"POOLTEXT" not in (out / "dataset.enc").read_bytes()


def test_every_genre_template_formats():
    from humanizer_eval.period.generate import GENRE_TEMPLATES

    for genre, templates in GENRE_TEMPLATES.items():
        assert templates, genre
        for t in templates:
            out = t.format(words=200, topic="T", audience="A", stance="S")
            assert "200" in out and "T" in out, (genre, t)
    # production mix (Q-2): mostly sections of papers, mostly non-argumentative essays
    assert sum("abstract" in t for t in GENRE_TEMPLATES["academic"]) == 1
    assert len(GENRE_TEMPLATES["academic"]) >= 6
