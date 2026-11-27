# Validators: the humanizer competition in SN32 v4

From v4, SN32 runs two competitions on **one set of weights**; there is no second mechanism.

* **Detection** (unchanged): your validator scores the detection miners exactly as before. They get
  **80%** of your weights, in the same proportions as in v3.
* **Humanizer competition** ([`rules.md`](rules.md)): the other **20%**, split 85% / 10% / 5%
  between the current king and the two previous kings.

The split and the 85/10/5 shares are constants of the validator code
(`HUMANIZER_SHARE = 0.2`, `KING_SHARES = (0.85, 0.10, 0.05)`), so every validator on the same release
sets the same split; they change only with another release.

**Validators run no evaluation for the humanizer competition.** The subnet owner's evaluator runs the
miners' humanizers and publishes **who the kings are** in a **signed leaderboard**. Your validator
fetches it, verifies it and adds the kings' shares to the weights it sets anyway. There is no extra
compute, no new API key and no new cost.

## What your validator does

The code is the `v4` branch of [It-s-AI/llm-detection](https://github.com/It-s-AI/llm-detection/tree/v4);
the evaluator's code is in its `humanizer/` folder (run by the subnet owner, not by validators). The
humanizer part is `detection/validator/humanizer_mechanism.py` plus one step in `set_weights`. The
validator:

1. **Fetches** the leaderboard from the configured URLs. The first valid one wins.
2. **Verifies** it:
   - the ed25519 signature against the public key of the owner's leaderboard key, built into the release;
   - the schema (`sn32-humanizer-leaderboard/2`), `netuid` 32 and the competition id;
   - the expiry (an envelope is valid for 3 hours);
   - that the sequence number never goes back, so an older leaderboard replayed later is refused;
   - that `kings` is a list of at most 3 distinct hotkeys.
3. **Pays the kings.** Kings whose hotkeys are not in the metagraph, or registered again since they were
   crowned (the signed `king_registered_at` against the metagraph's registration block), are dropped and
   the rest close ranks; they get 85% / 10% / 5% in their order, and the shares of missing slots go to the current king.
4. **Blends** the two competitions: `weights = 0.8 × detection + 0.2 × humanizer`. The detection part is
   your normalized scores as in v3. The result then goes through the usual weight processing and
   commit-reveal.
5. **Keeps the last valid leaderboard on disk** (it survives restarts). When no fresh one can be had (the
   owner's server is down, a bad signature), the kings of the last valid leaderboard keep their share,
   also past its expiry.

When there is no king to pay (the owner's baseline is king, none of the kings is registered, or the
validator has never received a valid leaderboard), the validator sets the detection weights alone: the
humanizer 20% goes to detection. Nothing is burned. A failure while computing the humanizer part (for
example an RPC error) reuses the previous round's humanizer weights.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--humanizer.urls` | built-in | comma-separated leaderboard URLs, overrides the built-in list |
| `--humanizer.public_key` | built-in | hex ed25519 public key of the signer, overrides the built-in key |
| `--humanizer.disable` | off | leave the humanizer competition out (not recommended: your weights then differ from the other validators') |

## Updating

Update with `git fetch origin && git checkout v4 && pip install -e .`, then restart your
validator; `run.sh` follows the branch you are on. A validator still on v3 keeps setting the v3 weights.
An older v4 validator (leaderboard schema `sn32-humanizer-leaderboard/1`) refuses the new leaderboard and
keeps paying the kings of the last `/1` leaderboard it had, with the old split, until it is updated.

## Check it yourself

The leaderboard is public: `GET <evaluator>/v1/leaderboard` returns the envelope
`{"payload": "<json>", "signature": "<hex>", "public_key": "<hex>"}`. The payload's `kings` lists the
kings' hotkeys, newest first: the current king, the previous distinct king and the one before (0 to 3
hotkeys; empty while the owner's baseline is king). The same verification code is in the evaluator
(`humanizer/src/humanizer_eval/publish/leaderboard.py`) and is covered by tests on both sides.
