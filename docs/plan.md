# Development plan

Workflow for each phase: design note (`docs/design/NN-*.md`) → approval →
code + tests → report (`docs/reports/NN-*.md`, which also serves as the PR description). Decisions are in [`spec.md`](spec.md).

Test strategy: almost everything runs on the fake detector and recorded responses (free);
simulations run on the It's AI proxy; paid APIs only on small samples (~$150 before launch).

| Phase | What | Paid | Needed from the owner |
|---|---|---|---|
| 0 | Repository, CI, specification, spend log | no | repository access ✅ |
| 1 | Detector gateway: adapters, rate limiter, retries, cache, cost accounting, canaries, fake detector, record/replay; contract tests, determinism | ~$15 | detector keys ✅ |
| 2 | Human pool: pre-2022 corpora, cleaning, labeling, embeddings, clusters | embeddings (OpenRouter) | HF ✅, storage (this machine) |
| 3 | Period builder: slots, topics, generation, facts, human reference set H, commit-reveal, Merkle, encryption; admission test | generation, ~$40 detectors | OpenRouter ✅, topic source |
| 4 | Gates and judges: normalization, G0–G3, injections, labeling tool | LLM | labeling of ~200 texts |
| 5 | Scorer and statistics: AUROC by group, soft-min, bootstrap, crown 85/10/5, baselines, simulations on the proxy | little | — |
| 6 | Execution on Lium CVM: client with budget guard, harness, inner container, red-team | Lium ≤ $50 | Lium ✅ |
| 7 | Submission intake and billing (Stripe) | no | Stripe test keys |
| 8 | Publication, signed leaderboard, verifier | no | HF write, signing key storage |
| 9 | Integration into the llm-detection validator (two mechanisms) | no | testnet, test TAO |
| 10 | Miner kit and documentation | no | — |
| 11 | Overall tests, testnet, soft launch 10% → 25% → 50% | launch | owner key, live Stripe |
