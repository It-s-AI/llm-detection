from humanizer_eval.config import Settings


def test_repr_masks_secrets(monkeypatch):
    monkeypatch.setenv("PANGRAM_API_KEY", "super-secret-value")
    s = Settings(_env_file=None)
    assert "super-secret-value" not in repr(s)
    assert s.pangram_api_key == "super-secret-value"


def test_caps_include_paid_services():
    s = Settings(_env_file=None)
    caps = s.caps()
    for svc in ("pangram", "gptzero", "zerogpt", "openrouter", "lium"):
        assert caps[svc] > 0
