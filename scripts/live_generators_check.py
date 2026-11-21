"""Generate one AI input per configured generator model under HIDDEN routing (cents)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.generate import GenConfig, generate_input


def main() -> int:
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    cfg_d = json.loads(Path("configs/period.json").read_text())
    out = {}
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    for g in cfg_d["generators"]:
        if only and g["model"] not in only:
            continue
        cfg = GenConfig.from_dict({**cfg_d, "generators": [{**g, "weight": 1}]})
        slot = {"genre": "essay", "length_bin": "200-250", "register": "formal"}
        topic = {
            "topic": "whether high schools should start classes later in the morning",
            "audience": "school board",
            "stance": "later start times improve learning",
        }
        r = generate_input(llm, random.Random(0), cfg, slot, topic)
        out[g["model"]] = {
            "ok": r.ok,
            "provider": r.provider,
            "words": len(r.text.split()),
            "target": r.target_words,
            "cost": round(r.cost_usd, 5),
            "notes": r.notes,
            "head": r.text[:90],
        }
        print(g["model"], {k: v for k, v in out[g["model"]].items() if k != "head"})
    (var_dir() / "live_generators_check.json").write_text(json.dumps(out, indent=1))
    print("openrouter:", led.summary().get("openrouter"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
