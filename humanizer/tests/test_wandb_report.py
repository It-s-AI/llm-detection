import importlib.util
from pathlib import Path


def _mod():
    spec = importlib.util.spec_from_file_location(
        "wr", Path(__file__).parents[1] / "scripts" / "wandb_report.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_metrics_come_only_from_public_fields():
    m = _mod()
    s = {
        "info": {
            "period_id": "p1",
            "n_inputs": 484,
            "quote_total_usdc": "150.12",
            "price_card": {"version": "v"},
            "queue": {"waiting": 2, "running": 1},
            "limits": {"submissions_open": True, "evals_last_24h": 3},
        },
        "periods": {"current": "p1", "periods": {"p1": {"commitment": "c" * 64, "merkle_root": "m" * 64}}},
        "results": {"period_id": "p1", "results": [{"submission_id": "s1", "S": 0.3, "crowned": True}]},
        "board": {"seq": 9, "issued_at": 100.0, "kings": ["hkA"], "king": {"hotkey": "hkA", "delta_s": 0.1}},
    }
    nums, text = m.metrics(s, now=160.0)
    assert (
        nums["intake/queue_waiting"] == 2 and nums["crown/n_kings"] == 1 and nums["leaderboard/age_s"] == 60.0
    )
    assert nums["results/n_finished"] == 1 and nums["intake/open"] == 1
    assert text["period/commitment"] == "c" * 64 and text["crown/king_hotkey"] == "hkA"


def test_an_empty_evaluator_still_reports():
    m = _mod()
    nums, text = m.metrics({"info": {"period_id": "p1"}, "periods": {}, "results": {}, "board": {}}, now=0.0)
    assert nums["crown/n_kings"] == 0 and text == {"period/id": "p1", "crown/kings": "[]"}


def test_events_follow_each_submission_by_uid():
    m = _mod()
    q = lambda *xs: {"queue": [{"submission_id": i, "uid": u, "hotkey": h, "name": "", "state": st}  # noqa: E731
                               for i, u, h, st in xs]}  # fmt: skip
    s = {
        "info": {"period_id": "p2"},
        "results": {"results": []},
        "board": {},
        "queue": q(("s1", 5, "hkA", "waiting")),
    }
    evs, mem = m.events(s, {}, set())
    assert [(e["event"], e["uid"]) for e in evs] == [("queued", 5)]
    s["queue"] = q(("s1", 5, "hkA", "running"), ("s2", 9, "hkB", "waiting"))
    evs, mem = m.events(s, mem, set())
    assert [(e["event"], e["submission_id"]) for e in evs] == [("started", "s1"), ("queued", "s2")]
    s["queue"] = q()
    s["results"] = {
        "results": [{"submission_id": "s1", "uid": 5, "hotkey": "hkA", "S": 0.4, "crowned": True}]
    }
    s["board"] = {"king": {"hotkey": "hkA"}, "kings": ["hkA"]}
    evs, _ = m.events(s, mem, set())
    assert [e["event"] for e in evs] == ["left_queue", "finished", "king"]
    assert evs[0]["submission_id"] == "s2" and evs[1]["S"] == 0.4 and evs[2]["to"] == "hkA"
