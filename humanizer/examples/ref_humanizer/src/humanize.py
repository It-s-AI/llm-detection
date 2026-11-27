"""Reference humanizer v2 (owner's example submission): Qwen3-4B-Instruct-2507 + LoRA, merged into weights/.

Contract (spec D-60): ``humanize(items) -> list[str]`` with items ``{"text", "lang", "genre"}``, one string per
item, same order. Runs with no network: the weights are read from ``<bundle>/weights``, the Python
dependencies are installed by the harness from ``<bundle>/wheels`` (requirements.txt). The harness seeds
``random``/``numpy``/``torch`` from ``HUMEVAL_SEED`` before the first call.
"""

from __future__ import annotations

import os

from prompt import build_prompt

SUB = os.environ.get("HUMEVAL_SUB", "/sub")
WEIGHTS = os.environ.get("REF_WEIGHTS", os.path.join(SUB, "weights"))
MAX_NEW_TOKENS = 1400  # inputs are 150-350 words; outputs past 1.3x the input length are cut by the scorer

_model = None
_tok = None


def _load():
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(WEIGHTS)
        _tok.padding_side = "left"
        _model = AutoModelForCausalLM.from_pretrained(
            WEIGHTS, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        _model.to("cuda").eval()
    return _model, _tok


def humanize(items):
    import torch

    model, tok = _load()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(it["text"], it.get("genre"))}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for it in items
    ]
    # longest first inside the batch keeps padding small; results are put back in input order
    order = sorted(range(len(items)), key=lambda i: -len(prompts[i]))
    out = [""] * len(items)
    for start in range(0, len(order), 16):
        idx = order[start : start + 16]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        longest = max(len(items[i]["text"].split()) for i in idx)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=min(MAX_NEW_TOKENS, int(longest * 2.2) + 64),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        end_think = tok.convert_tokens_to_ids("</think>")  # drop a reasoning block if a model emits one
        for row, i in enumerate(idx):
            ids = gen[row, enc["input_ids"].shape[1] :].tolist()
            if isinstance(end_think, int) and end_think in ids:
                ids = ids[ids.index(end_think) + 1 :]
            text = tok.decode(ids, skip_special_tokens=True).strip()
            out[i] = text or items[i]["text"]  # never return an empty string
    return out
