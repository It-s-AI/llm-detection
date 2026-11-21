# 11. Mechanism 1 weights in the validator: signed leaderboard — report

## What was done
- **Owner side** (`src/humanizer_eval/publish/leaderboard.py`).
  - The crown state is turned into 85/10/5 payout weights (D-12, D-13) and signed with a separate ed25519 key.
  - The key is created once: `var/keys/leaderboard_ed25519.key`, permissions 600.
  - The signature covers: subnet, mechanism, version number (increases over time), expiry (3 h), the current king and recent coronations.
  - The intake server serves the signed leaderboard at `GET /v1/leaderboard` and re-signs it once a minute.
- **Validator side**: `llm-detection` code with changes, branch `llm-detection-mechanism1` in the humanizer_eval repository (commits 898ca30 and 528e4f5). We push nothing to It-s-AI/llm-detection.
  - Module `detection/validator/humanizer_mechanism.py`:
    1. downloads the leaderboard from one of the URLs;
    2. checks the signature with the pinned public key, the subnet, the mechanism, the expiry, and that the version number has not decreased;
    3. maps hotkeys to UIDs;
    4. sets mechanism 1 weights (`set_weights(..., mechid=1)`).
  - The share of an unregistered hotkey and the burn share go to the subnet owner's UID. In SN32 this is currently UID 129.
  - Nothing valid (the owner's server has been down for longer than the expiry, the signature is invalid) → everything to burn (D-92).
  - The last valid leaderboard is stored on the validator's disk. A short outage of the owner's server changes nothing, and a validator restart does not lose state.
  - At the start of `BaseValidatorNeuron.set_weights`, before mechanism 0, `set_humanizer_weights()` is called. The order matters; see localnet. While the subnet has one mechanism, the call does nothing. Mechanism 1 errors do not affect mechanism 0.
  - Flags: `--humanizer.disable`, `--humanizer.urls`, `--humanizer.public_key`. The default values (URL and key) will be filled in at launch.

## Mapping to decisions
| Decision | Where in code | Verified by | Status |
|---|---|---|---|
| D-12 85/10/5 to distinct hotkeys | `KingState.payout_weights` → `build_payload` | test | ✅ |
| D-13 the baseline share is burned | `__burn__` weight → owner UID | test; UID 129 in the live SN32 metagraph | ✅ |
| D-92 weights from the signed leaderboard, otherwise burn | `humanizer_mechanism.py` | 9 tests in llm-detection, cross-check of the owner's signature against the validator's verification | ✅ |
| D-14 soft launch (10% of emission) | the mechanism share is set by the subnet owner on chain | — | ⏳ at launch |

## What is known about the SN32 network (finney, today)
| Parameter | Value | What it means |
|---|---|---|
| mechanisms | 1 | at launch the owner sets 2 (`sudo_set_mechanism_count`) and the share (`sudo_set_mechanism_emission_split`); with 256 UIDs the chain will refuse, see below |
| `max_allowed_uids` | 256 | two mechanisms require ≤ 128 (localnet) |
| `min_allowed_weights` | 1 | weights can be set on 1–3 UIDs, without "smearing" |
| commit-reveal | enabled, period 1 | bittensor 10.5 handles it itself in `set_weights`, for `mechid=1` too |
| `weights_rate_limit` | 100 blocks | on chain each mechanism has its own; the SDK checks against mechanism 0, so mechanism 1 is set first |
| owner UID | 129 | burn goes here |

## Deviations and new rules
- **The signing key is a separate ed25519 key, not the owner's hotkey.** If the evaluation server is compromised, this does not give access to the hotkey. The cost: changing the key requires a validator release.
- **The share of a deregistered king goes to burn**, not to the next king. This is simpler and gives no benefit from deregistration. If you want it otherwise, it is one line.
- **Leaderboard expiry is 3 h.** Shorter, and an owner server outage quickly sends the weights to burn. Longer, and after a coronation the old weights can circulate in the network for longer (the validator will still take the new version as soon as it downloads it).

## Test results
- `tests/publish/` — 11 tests:
  - signing and verification;
  - 85/10/5 shares;
  - baseline → everything to burn;
  - forgeries: modified weights, foreign key, different subnet or mechanism, expired leaderboard, version rollback;
  - sum of weights.
- `tests/intake/` — the `/v1/leaderboard` endpoint serves a signature that verifies.
- `llm-detection/tests/test_humanizer_mechanism.py` — 9 tests:
  - UID mapping and burn;
  - second URL when the first fails;
  - server outage before and after expiry;
  - rejecting an old version after a restart;
  - calling `set_weights` with `mechid=1` only when there are two mechanisms.
- Cross-repository check: a signature from `humanizer_eval` is accepted by the validator module.

## Localnet test (subtensor `devnet-ready`, Lium pod, 7 runs, $0.21)
A subnet with two mechanisms, split 90/10, a validator and two miners on a different coldkey. Mechanism 1 weights are set by the validator code from `llm-detection` with a signed leaderboard.

| What was tested | Result |
|---|---|
| Enable the second mechanism with 256 UIDs | **chain refusal** `TooManyUIDsPerMechanism`: `max_uids × mechanisms ≤ 256` |
| The same with `max_allowed_uids = 128` | ✅ enabled |
| Mechanism 1, then immediately mechanism 0 (new order in the validator) | ✅ both weight sets on chain; mechanism 1 exactly per the leaderboard (miner 0.85, owner UID 0.15) |
| Mechanism 0, then immediately mechanism 1 (as before) | ❌ the SDK silently sends nothing |
| Commit-reveal enabled (as in SN32) | ✅ mechanism 1 weights were revealed after 6 s (fast blocks) |
| Earnings over 105 blocks: mechanism 0 miner / mechanism 1 miner | 36.90 / 3.485, ratio 0.0945 vs. 0.0944 expected |
| Burn to the owner UID | the owner's hotkey received exactly the owner's share, 18.0; the burn share was not credited to it, it was burned |

### Issues found
1. **Integration bug (fixed).**
   - `set_weights` in bittensor 10.5, before sending, checks the rate limit against mechanism 0's `LastUpdate`, whatever `mechid` is passed. The chain itself counts the limit for each mechanism separately.
   - Therefore, with the order "mechanism 0, then 1", the second set of weights would never be sent, and mechanism 1 would go to burn for all validators.
   - Now the validator sets mechanism 1 first. Commit 528e4f5 in the `llm-detection-mechanism1` branch.
2. **UID limit for two mechanisms — a launch decision.**
   - With the current subtensor code (`main` and `devnet-ready` branches), a second mechanism cannot be enabled in SN32 with 256 UIDs. `max_allowed_uids` must be lowered to 128, and the chain will trim half of mechanism 0's UIDs.
   - On finney, five subnets with 256 UIDs already have two mechanisms (44, 87, 89, 93, 113). They were probably enabled before this check existed.
   - Options:
     - (a) lower SN32 to 128 UIDs;
     - (b) ask OTF whether there is an exception or a plan to raise the budget;
     - (c) run the humanizer competition on a separate subnet.
   - Must be decided before launch.
3. **Miners on the owner's coldkey receive nothing:** the chain burns their emission. This confirms the burn scheme via the owner UID. The owner must keep their own test miners on a different key.

### Steps to launch two mechanisms (SN32 owner; tested on localnet: owner is not sudo, epoch 40, window 10 blocks)
Owner's decision: 128 UIDs × 2 mechanisms (Q-13).
1. **Validators first** switch to the code from the `llm-detection-mechanism1` branch; it has the leaderboard URL and the public key built in.
2. All calls are made with the owner's coldkey **directly, without the sudo wrapper**, and outside the admin-action freeze window. On finney the window is the last 10 blocks of a 360-block epoch. To check the window: `subtensor.is_in_admin_freeze_window(32)`.
   ```python
   from bittensor.core.extrinsics.utils import sudo_call_extrinsic
   from bittensor.utils.weight_utils import convert_maybe_split_to_u16
   # root_call=True in bittensor 10.5 means "send as is", without Sudo.sudo
   sudo_call_extrinsic(st, owner, "sudo_set_max_allowed_uids", {"netuid": 32, "max_allowed_uids": 128}, root_call=True)
   sudo_call_extrinsic(st, owner, "sudo_set_mechanism_count", {"netuid": 32, "mechanism_count": 2}, root_call=True)
   sudo_call_extrinsic(st, owner, "sudo_set_mechanism_emission_split",
                       {"netuid": 32, "maybe_split": convert_maybe_split_to_u16([9, 1])}, root_call=True)
   ```
3. **Do not use** `sudo_set_mechanism_count_extrinsic` and `sudo_set_mechanism_emission_split_extrinsic` from bittensor as is. They wrap the call in `Sudo.sudo`, and for an owner who is not sudo the chain responds "Invalid signing address" (verified).
4. Step 1 of item 2 trims the subnet to 128 UIDs: the chain itself removes the excess non-immune UIDs. This is the owner's decision.
5. On finney the owner has a rate limit on hyperparameter changes (`OwnerHyperparamRateLimit = 2`). On localnet three calls in a row went through.

Data: `docs/reports/data/11-localnet/` (mechanism 1) and `docs/reports/data/11-localnet-owner/` (owner steps) (result and log of the last run). Scripts: `scripts/localnet_mechanism_test.py` (pod rental) and `scripts/localnet/mech1_test.py` (the test itself, on the pod).

## What we need from you
- Do you agree with the rule "deregistered king's share → burn"?
