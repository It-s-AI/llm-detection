# humanizer_eval

The evaluator of the **SN32 (It's AI) humanizer competition** on Bittensor: subnet 32 pays for a model that rewrites AI-generated text so that commercial AI detectors take it for human
writing, with its meaning and quality kept.

Miners publish a humanizer (code + weights) in their own Hugging Face repo and pin it on chain. The
evaluator runs it offline on a fresh GPU pod over the period's hidden texts. It checks every output with
LLM judges (meaning, style, naturalness), sends the outputs to GPTZero, Pangram and Winston AI (each at a pinned model version), and decides
with a paired statistical test against the current king whether the submission takes the crown. The king
and the two previous kings share the competition's emission 85/10/5. Validators set one weight vector,
detection 80% and humanizer 20%; the split and the 85/10/5 shares are fixed in the validator code, and the
only thing they take from a leaderboard signed by the subnet owner is who the kings are.

This folder is part of the subnet repository [It-s-AI/llm-detection](https://github.com/It-s-AI/llm-detection)
(branch `v4`, folder `humanizer/`): the subnet owner runs it; validators and miners do not need it to run
their neurons, but it is the exact code every submission is judged by.

## Start here

| You are | Read |
|---|---|
| a miner | [the rules](docs/rules.md), then [how to build and submit](docs/miners.md) and the [reference humanizer](examples/ref_humanizer/README.md) |
| a validator | [what changes for validators](docs/validators.md) |
| anyone | [the money: prices, paying in TAO, balances in USDC, holds, refunds](docs/billing-policy.md) |

**Test launch:** the code, the rules and the reference humanizer are public; submissions open at the production launch.
See the last section of [the rules](docs/rules.md#test-launch-before-production). The leaderboard is the
"Humanizer competition" section of the [SN32 leaderboard](https://huggingface.co/spaces/sergak0/ai-detection-leaderboard).

## What is in this repository

* `src/humanizer_eval/`: everything the evaluator does.
  - `period/`: building the hidden texts of a period.
  - `gates/`: the quality judges.
  - `detectors/`: the detector clients.
  - `scoring/`: the metric and the crown test.
  - `runner/` and `harness/`: the bundle and the GPU pod.
  - `billing/`: quotes, holds and refunds; TAO payments kept as USDC on Base.
  - `intake/`: the miners' API and the queue.
  - `publish/`: the signed leaderboard and the kings' bundles.
* `scripts/miner_submit.py`: the miner's command-line tool: upload, commit on chain, submit, pay, follow.
* `examples/`: the reference humanizer (Qwen3-4B + LoRA) and a trivial identity bundle.
* `docs/spec.md`: every design decision, numbered `D-xx` (in Russian). The English summary is `docs/rules.md`.
  `docs/reports/` holds the measurements behind them.
* `space/`: the humanizer section of the leaderboard Space. `deploy/`: how the evaluator server is set up.

Everything is open except the current period's texts, seeds and keys, which are revealed with the period.

## Development

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
uv run pytest            # offline tests (live tests are marked `-m live` and excluded by default)
uv run ruff check . && uv run ruff format --check .
```

Secrets are read from environment variables or a `.env` file and are never committed or logged. Every
paid API call goes through a local spend ledger with hard per-service caps.
