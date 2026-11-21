# Announcement draft: test launch (for Discord; fill in EVALUATOR and the date)

**SN32 humanizer competition: test launch**

Subnet 32 gets a second competition. Next to detection, half of the miners' emission pays for a **humanizer**: a model
that rewrites AI-generated English text so that GPTZero, ZeroGPT and Pangram take it for human writing,
while LLM judges check that the meaning and the quality are kept.

**How it works**
• One king. Your submission runs on the same hidden texts as the king (~500 per 4-day period), and if
  it beats the king with statistical confidence you are the new king at once.
• Payouts: 85% to the king, 10% and 5% to the two previous kings (different hotkeys).
• You publish your bundle (code + weights, up to 50 GB) in your own Hugging Face repo and commit
  `<repo>@<commit sha>` on chain. Only that exact commit is evaluated, on one H100 with no network.
• Each evaluation is paid by the submitter at cost (detectors, judges, pod); the unused hold is refunded.
• Validators: no new compute. SN32 v4 adds the humanizer kings (half of the weights) from a leaderboard
  signed by the subnet owner to the usual detection weights.

**Test launch: code and rules first**
• The evaluator's code, the rules, the miner guide and a complete reference humanizer are public now, so
  you can build and check your bundle locally.
• No submissions during the test launch: an evaluation costs real money (judges, a GPU pod) and would
  not count. They open at the production launch.
• Leaderboard: the "Humanizer competition" section of
  https://huggingface.co/spaces/sergak0/ai-detection-leaderboard

**Read first:** rules, the miner guide and a complete reference humanizer (Qwen3-4B + LoRA) are in
https://github.com/It-s-AI/llm-detection/tree/v4/humanizer: `docs/rules.md`, `docs/miners.md`,
`examples/ref_humanizer`. Validators: `docs/validators.md` (the validator code is the same `v4` branch).

**Production launch:** in about half a week, on mainnet subnet 32, with the crown detectors (GPTZero,
ZeroGPT, Pangram) and real payments: submissions open then.
