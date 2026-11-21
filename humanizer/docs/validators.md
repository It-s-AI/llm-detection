# Validators: the humanizer competition in SN32 v4

From v4, SN32 runs two competitions on **one set of weights**, **half each**; there is no second mechanism.

* **Detection** (unchanged): your validator scores the detection miners exactly as before. They get
  **50%** of your weights, in the same proportions as in v3.
* **Humanizer competition** ([`rules.md`](rules.md)): the other **50%**, split 85% / 10% / 5%
  between the current king and the two previous kings.

**Validators run no evaluation for the humanizer competition.** The subnet owner's evaluator runs the
miners' humanizers and publishes the payouts as a **signed leaderboard**. Your validator fetches it,
verifies it and adds it to the weights it sets anyway. There is no extra compute, no new API key and no
new cost.

## What changes in your validator

The code is the `v4` branch of [It-s-AI/llm-detection](https://github.com/It-s-AI/llm-detection/tree/v4)
(version 4.0.0, on top of v3.15.1); the evaluator's code is in its `humanizer/` folder (run by the subnet
owner, not by validators). It adds `detection/validator/humanizer_mechanism.py` and one step in
`set_weights`. The validator:

1. **Fetches** the leaderboard from the configured URLs. The first valid one wins.
2. **Verifies** it:
   - the ed25519 signature against the public key of the owner's leaderboard key, built into the release;
   - the schema, `netuid` 32 and the competition id;
   - the expiry (an envelope is valid for 3 hours);
   - that the sequence number never goes back, so an older leaderboard replayed later is refused.
3. **Maps hotkeys to UIDs.** Shares of hotkeys that are not registered, and the `__burn__` share, go to
   the subnet owner's UID (burn).
4. **Blends** the two competitions: `weights = 0.5 × detection + 0.5 × humanizer`. The detection part is
   your normalized scores as in v3. The result then goes through the usual weight processing and
   commit-reveal.
5. **Keeps the last valid leaderboard on disk** (it survives restarts). When no fresh one can be had (the
   owner's server is down, a bad signature), the kings of the last valid leaderboard keep their share,
   also past its expiry. Only a validator that has never received a valid leaderboard sends the humanizer
   half to the subnet owner's UID (burn).

The split is a constant of the code, so every validator running v4 sets the same split; it changes only
with another release.

Until the release that builds in the leaderboard's public key and URL, the humanizer share is not applied
and v4 sets exactly the v3 weights, so it is safe to deploy early. A failure while computing the share
(for example an RPC error) reuses the previous round's humanizer weights.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--humanizer.urls` | built-in | comma-separated leaderboard URLs, overrides the built-in list |
| `--humanizer.public_key` | built-in | hex ed25519 public key of the signer, overrides the built-in key |
| `--humanizer.disable` | off | leave the humanizer competition out (not recommended: your weights then differ from the other validators') |

## Timeline

1. **Now**: the code and the rules are public in the `v4` branch (version 4.0.1, with the leaderboard's
   URL and public key built in); no submissions yet. **Update now:** `git fetch origin && git checkout v4
   && pip install -e .`, then restart your validator; `run.sh` follows the branch you are on. Until the
   start v4 sets exactly the v3 weights and does not even fetch the leaderboard.
2. **Monday, 2026-11-23, 12:00 UTC**: submissions open, and every validator on v4 switches to 50/50 at
   that moment by itself. A validator still on v3 keeps setting the v3 weights.

## Check it yourself

The leaderboard is public: `GET <evaluator>/v1/leaderboard` returns the envelope
`{"payload": "<json>", "signature": "<hex>", "public_key": "<hex>"}`. The payload's `weights` map hotkeys
to shares of the humanizer competition (they sum to 1). The same verification code is in the evaluator
(`humanizer/src/humanizer_eval/publish/leaderboard.py`) and is covered by tests on both sides.
