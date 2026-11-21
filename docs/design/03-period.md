# 03. Period builder
Status: draft, implemented autonomously
Implements decisions: D-51..D-58, D-90

## Goal
Build the period dataset D_t: AI inputs from the same distribution as the human reference set, but without
being tied to specific human texts; prepare everything needed for an honest reveal.

## Flow
1. `seed_t = sha256(secret_t ‖ blockhash)`; the commitment `sha256(secret_t)` is published in advance (D-57).
2. Slots from the pool: metadata only (genre, cluster, cluster label, length, register) — no texts (D-51).
3. Topics: the model sees only the genre and the cluster label and invents a specific topic, audience, and stance.
4. AI inputs: the generator is chosen by weight from `configs/period.json` (12 models, ~75% closed / 25% open),
   the request is written in a user's style, wrappers are cleaned up ("Sure, here's…", headings, "Let me know…"), length control.
5. Facts of each input (for gate G1).
6. Human reference set H: a sample from the pool with the same counts per group (D-54).
7. Merkle root of all items → on-chain (later); the dataset is written **only in encrypted form**, the manifest in the clear.
All LLM calls use HIDDEN routing (trusted providers only).

## Not doing yet
Fresh content/It's AI logs as a topic source (D-52, needs access/a decision); writing commitments to the chain
(needs a testnet wallet); publishing the release.

## Tests and acceptance
Offline: commit-reveal, Merkle proofs, encryption, cleanup of generations, length per slot, build determinism
from the seed, **no pool text ends up in any prompt** (D-51), the human reference set matches by group.
Live pilot: 80 inputs, detector admission test (AUROC ≥ 0.9 on raw AI vs. humans).
