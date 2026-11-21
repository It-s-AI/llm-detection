from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ChatResult

SCHEMA = {"title": "x", "type": "object"}


class Counting:
    def __init__(self, json_out=None):
        self.n = 0
        self.json_out = {"ok": True} if json_out is None else json_out

    def chat(self, model, messages, **kw):
        self.n += 1
        js = self.json_out if kw.get("json_schema") else None
        return ChatResult("t", model, "p", 1, 1, 0.01, "id", json=js)


def test_hit_is_free_and_identical(tmp_path):
    inner = Counting()
    llm = CachedLLM(inner, tmp_path / "c.sqlite", key=b"k" * 32)
    m = [{"role": "user", "content": "hi"}]
    a = llm.chat("m", m, json_schema=SCHEMA, temperature=0.0)
    b = llm.chat("m", m, json_schema=SCHEMA, temperature=0.0)
    assert inner.n == 1 and a.json == b.json == {"ok": True}
    assert b.cost_usd == 0.0 and b.finish_reason == "cached"
    # persists across instances
    again = CachedLLM(inner, tmp_path / "c.sqlite", key=b"k" * 32).chat("m", m, json_schema=SCHEMA)
    assert inner.n == 1 and again.json == {"ok": True}


def test_key_includes_model_and_messages(tmp_path):
    inner = Counting()
    llm = CachedLLM(inner, tmp_path / "c.sqlite")
    llm.chat("m1", [{"role": "user", "content": "a"}], json_schema=SCHEMA)
    llm.chat("m2", [{"role": "user", "content": "a"}], json_schema=SCHEMA)
    llm.chat("m1", [{"role": "user", "content": "b"}], json_schema=SCHEMA)
    assert inner.n == 3


def test_sampling_and_failed_parses_not_cached(tmp_path):
    inner = Counting()
    llm = CachedLLM(inner, tmp_path / "c.sqlite")
    m = [{"role": "user", "content": "a"}]
    llm.chat("m", m, temperature=0.9)
    llm.chat("m", m, temperature=0.9)
    assert inner.n == 2
    bad = Counting(json_out=False)
    bad.json_out = None  # schema requested but nothing parsed
    llm2 = CachedLLM(bad, tmp_path / "d.sqlite")
    llm2.chat("m", m, json_schema=SCHEMA)
    llm2.chat("m", m, json_schema=SCHEMA)
    assert bad.n == 2


def test_values_encrypted_at_rest(tmp_path):
    llm = CachedLLM(Counting(json_out={"secret": "hidden text"}), tmp_path / "c.sqlite", key=b"k" * 32)
    llm.chat("m", [{"role": "user", "content": "a"}], json_schema=SCHEMA)
    assert b"hidden text" not in (tmp_path / "c.sqlite").read_bytes()
