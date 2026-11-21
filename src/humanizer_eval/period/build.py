"""Build one period dataset D_t (D-51..D-57).

Flow: seed -> slots (metadata only, from the pool) -> topics (topic model sees only
genre + cluster label) -> AI inputs (generator mix, user-like prompts) -> facts ->
human reference H (same stratum counts) -> Merkle root -> encrypted dataset +
public manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from humanizer_eval.gates.judges import Judge
from humanizer_eval.gates.meaning import extract_facts_v2, prepare_facts
from humanizer_eval.period.crypto import encrypt_at_rest
from humanizer_eval.period.generate import GenConfig, generate_input, propose_topics
from humanizer_eval.period.merkle import leaf_hash, merkle_root
from humanizer_eval.period.seed import commitment, derive_seed, subseed


class PoolLike(Protocol):
    def sample_slots(
        self, n: int, genre_proportions: dict[str, float], seed: int
    ) -> list[dict[str, Any]]: ...

    def sample_reference(self, slot_counts: dict[tuple, int], seed: int) -> list[dict[str, Any]]: ...


@dataclass
class PeriodResult:
    manifest: dict[str, Any]
    items: list[dict[str, Any]]
    human_ref: list[dict[str, Any]]
    cost_usd: float


SLOT_FIELDS = ("genre", "cluster_id", "cluster_label", "length_bin", "register")


def _slot_key(slot: dict[str, Any]) -> tuple:
    return tuple(slot.get(f) for f in SLOT_FIELDS)


def build_period(
    period_id: str,
    secret: bytes,
    block_hash: str,
    pool: PoolLike,
    llm,
    cfg: GenConfig,
    n: int,
    *,
    human_factor: float = 1.0,
    workers: int = 8,
) -> PeriodResult:
    seed = derive_seed(secret, block_hash)
    slots = pool.sample_slots(n, cfg.genre_proportions, subseed(seed, "slots"))
    cost = 0.0
    # topics: one batched request per (genre, cluster) group
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(slots):
        groups[(s["genre"], s.get("cluster_label", s["genre"]))].append(i)
    topics: dict[int, dict[str, Any]] = {}
    for (genre, label), idx in sorted(groups.items()):
        need = len(idx)
        got: list[dict] = []
        tries = 0
        while len(got) < need and tries < 3:
            batch, c = propose_topics(
                llm,
                cfg.topic_model,
                genre,
                label,
                min(25, need - len(got) + 2),
                subseed(seed, f"topics:{genre}:{label}:{tries}"),
            )
            cost += c
            got.extend(batch)
            tries += 1
        for k, i in enumerate(idx):
            topics[i] = (
                got[k % len(got)] if got else {"topic": label, "audience": "general readers", "stance": ""}
            )

    # generation (parallel; per-item RNG keeps results independent of scheduling)
    def _gen(i: int):
        item_rng = random.Random(subseed(seed, f"item:{i}"))
        return i, generate_input(llm, item_rng, cfg, slots[i], topics[i])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        gens = dict(ex.map(_gen, range(len(slots))))
    facts_judge = Judge(cfg.facts_model)
    validators = tuple(Judge(m) for m in cfg.fact_validators)
    if validators and len(validators) != 2:
        raise ValueError("fact_validators must name exactly two models (two judge families)")

    def _facts(i: int):
        g = gens[i]
        if not g.ok:
            return i, None, 0.0
        if validators:
            f, rep = prepare_facts(llm, facts_judge, validators, g.text)
            c = rep.cost_usd
        else:
            f, c = extract_facts_v2(llm, facts_judge, g.text)
        return i, [x.to_dict() for x in f] if f else None, c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        facts = {i: (f, c) for i, f, c in ex.map(_facts, range(len(slots)))}
    items = []
    for i, s in enumerate(slots):
        g = gens[i]
        f, c = facts[i]
        cost += g.cost_usd + c
        if not g.ok or not f:
            continue
        items.append(
            {
                "id": hashlib.sha256(g.text.encode()).hexdigest()[:16],
                "slot": {
                    k: s.get(k) for k in ("genre", "cluster_id", "cluster_label", "length_bin", "register")
                },
                "topic": topics[i],
                "generator": {"model": g.model, "provider": g.provider, "template": g.template},
                "target_words": g.target_words,
                "text": g.text,
                "facts": f,
            }
        )
    counts = Counter(_slot_key(it["slot"]) for it in items)
    want = {k: max(1, round(v * human_factor)) for k, v in counts.items()}
    human_ref = pool.sample_reference(want, subseed(seed, "reference"))
    root = merkle_root(
        [leaf_hash(it) for it in items] + [leaf_hash({"human_id": h["id"]}) for h in human_ref]
    )
    manifest = {
        "period_id": period_id,
        "commitment": commitment(secret),
        "block_hash": block_hash,
        "merkle_root": root.hex(),
        "n_items": len(items),
        "n_requested": n,
        "n_human_ref": len(human_ref),
        "genre_counts": dict(Counter(it["slot"]["genre"] for it in items)),
        "generator_counts": dict(Counter(it["generator"]["model"] for it in items)),
        "config_sha256": hashlib.sha256(json.dumps(cfg.__dict__, sort_keys=True).encode()).hexdigest(),
        "created_at": time.time(),
        "cost_usd": round(cost, 4),
    }
    return PeriodResult(manifest, items, human_ref, cost)


def save_period(result: PeriodResult, out_dir: Path | str, key: bytes) -> Path:
    """Hidden data is written ONLY encrypted; the manifest (public commitments) in clear."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(
        {
            "items": result.items,
            "human_ref": [
                {"id": h["id"], **{k: h.get(k) for k in ("genre", "length_bin", "register", "cluster_id")}}
                for h in result.human_ref
            ],
        }
    ).encode()
    (out / "dataset.enc").write_bytes(encrypt_at_rest(key, blob))
    (out / "manifest.json").write_text(json.dumps(result.manifest, indent=1))
    return out
