# Announcement: SN32 v4 test launch (Discord)

**SN32 v4: the humanizer competition**

Subnet 32 gets a second competition. Next to detection, **half of the miners' emission** will pay for a
**humanizer**: a model that rewrites AI-generated English text so that GPTZero, ZeroGPT and Pangram take
it for human writing, while LLM judges check that the meaning and the quality are kept.

**Now: code and rules only.** Everything is in the `v4` branch of It-s-AI/llm-detection:
https://github.com/It-s-AI/llm-detection/tree/v4. The evaluator, the rules, the miner guide and a complete
reference humanizer are in `humanizer/`. Submissions are not open yet.

**Monday, 23 November, 12:00 UTC**
• Submissions open.
• Validators on v4 switch to 50% detection / 50% humanizer automatically, all at the same moment.

**Validators: please update now.**
`git fetch origin && git checkout v4 && pip install -e .`, then restart your validator (`run.sh` follows
the branch you are on). Until Monday 12:00 UTC v4 sets exactly the v3 weights. No new compute and no new
keys: the humanizer half comes from a leaderboard signed by the subnet owner and checked by your validator.
Details: `humanizer/docs/validators.md`.

**Miners**
• One king. Your submission runs on the same ~500 hidden texts as the king (a new set every 7 days); beat
  it with statistical confidence and you are the new king.
• Payouts: 85% to the king, 10% and 5% to the two previous kings. For the first 3 days after the first
  crown, the kings get a quarter of that.
• Your bundle (code + weights, up to 50 GB) lives in your own Hugging Face repo; you commit
  `<repo>@<commit>` on chain. It runs offline on one H100.
• Each evaluation is paid at cost (detectors, judges, GPU), about $100–150 for a period; the unused part
  of the hold is refunded.
• Read first: `humanizer/docs/rules.md`, `humanizer/docs/miners.md`, `humanizer/examples/ref_humanizer`.

Leaderboard: the "Humanizer competition" section of https://huggingface.co/spaces/sergak0/ai-detection-leaderboard
