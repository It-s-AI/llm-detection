"""LoRA fine-tune for the reference humanizer: AI rewrite -> original human text (inverse paraphrase).

Runs on one GPU with internet (to fetch the base model). Writes the merged model (bf16 safetensors) to
``--out/weights`` and the adapter to ``--out/adapter``; ``train_log.json`` holds the loss curve.

    pip install transformers==4.57.1 peft==0.17.1
    python train_lora.py --pairs pairs.jsonl --out /work/bundle [--base Qwen/Qwen3-4B-Instruct-2507]

pairs.jsonl rows: {"genre", "ai", "human"}. The prompt is ``src/prompt.py`` (the same one used at run time).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompt import build_prompt


def encode(tok, row: dict, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(row["ai"], row.get("genre"))}],
        tokenize=False,
        add_generation_prompt=True,
    )
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(row["human"].strip() + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) > max_len:
        return None
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids}


def batches(rows: list[dict], tokens_per_batch: int, rng: random.Random):
    """Length-bucketed batches with about ``tokens_per_batch`` tokens each (after padding)."""
    rows = sorted(rows, key=lambda r: len(r["input_ids"]))
    out, cur = [], []
    for r in rows:
        if cur and (len(cur) + 1) * len(r["input_ids"]) > tokens_per_batch:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(batch: list[dict], pad_id: int, device: str) -> dict:
    n = max(len(r["input_ids"]) for r in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, r in enumerate(batch):
        k = len(r["input_ids"])
        ids[i, :k] = torch.tensor(r["input_ids"])
        lab[i, :k] = torch.tensor(r["labels"])
        att[i, :k] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    raw = [json.loads(x) for x in Path(a.pairs).read_text().splitlines() if x.strip()]
    rows = [e for e in (encode(tok, r, a.max_len) for r in raw) if e]
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * a.eval_frac))
    ev, tr = rows[:n_eval], rows[n_eval:]

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(
        r=a.rank,
        lora_alpha=2 * a.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    per_epoch = len(batches(tr, a.tokens_per_batch, random.Random(0)))
    total = math.ceil(per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, max(1, total // 20), total)

    def eval_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for b in batches(ev, a.tokens_per_batch, random.Random(1)):
                losses.append(model(**collate(b, tok.pad_token_id, "cuda")).loss.item())
        model.train()
        return sum(losses) / len(losses)

    log = {"base": a.base, "pairs": len(raw), "train": len(tr), "eval": len(ev), "steps": total,
           "eval_loss": [(0, eval_loss())], "train_loss": []}  # fmt: skip
    model.train()
    step, t0 = 0, time.time()
    while step < total:
        for b in batches(tr, a.tokens_per_batch, rng):
            loss = model(**collate(b, tok.pad_token_id, "cuda")).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                log["train_loss"].append((step, round(loss.item(), 4)))
                print(f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step % max(1, total // 4) == 0 or step == total:
                log["eval_loss"].append((step, eval_loss()))
                print(f"eval {log['eval_loss'][-1]}", flush=True)
            if step >= total:
                break
    log["train_seconds"] = round(time.time() - t0, 1)
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True  # gradient checkpointing turned the KV cache off; inference needs it
    merged.save_pretrained(out / "weights", safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(out / "weights")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != "train_loss"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
