# 14. Dress rehearsal on the test server (2026-11-20)

## What we ran
Everything that had previously been checked only by tests — on the test server, testnet 87, It's AI detector, test Stripe. A 45-minute period instead of 4 days, 50 and 20 texts instead of 500. The rotation cycle ran every ~90 s, chain writes were made from the notary wallet `5D76hDg…`, public copies went to `sergak0/sn32-kings`, into the `rehearsal/` folder.

## Results
| Step | Outcome |
|---|---|
| pilot-2 commitment on-chain | block 8487924, before seed block 8487970 |
| pilot-2 build after finalization of B | 49 texts, $0.56; Merkle root on-chain in block 8488022 |
| The server loaded pilot-2 on its own: H scored once, king run in advance | king's S 0.0063, READY 20 minutes before the switchover |
| Switchover pilot-1 → pilot-2, immediately followed by the pilot-3 commitment, reveal of pilot-1 | a single rotation pass |
| Two submissions at once (t1, t2; different bundles on HF, commitments on-chain, payment by a Stripe test payment) | two pods in parallel; both S = 0.0063 = king's S, no crown; 49/49 outputs at level 1; charged 17¢ and 52¢, the rest of the hold released |
| Build and preparation of pilot-3 in parallel with these evaluations | 20 texts, $0.29; king on pilot-3 S = 0.0093 |
| Reveal of pilot-2 after the evaluations finished, verification from the public API | `verify_period.py --url …/v1/periods/pilot-2 --notary 5D76hDg…`: **25 checks, 0 errors** — commitment, seed block, Merkle root, all S values, both crown tests, both on-chain records, signature and order |
| Reveal of pilot-1 (old pilot, without a notary) | 14 checks passed; as expected, flagged "not recorded on-chain" |
| Copy of the king to our HF mirror (the miner's repository is unavailable) | `rehearsal/kings/…`, sha256 matched, download without a token works |
| Copies of the schedule and reveals on HF | `rehearsal/periods/schedule.json`, `pilot-1/`, `pilot-2/` |

Costs: OpenRouter ~$1.1 (two builds and judges; repeated outputs were taken from the judge cache), Lium a few cents, testnet TAO ~τ0.002. Stripe — test mode.

## What we found and fixed
- **An identical bundle from two hotkeys** is rejected (409: "this exact bundle was already submitted") — the rule works. Two different bundles were made for the rehearsal.
- **`verify_period.py` crashed** on a period without a seed block in the manifest (the old pilot-1). Now this is a regular failed check (`8b4714e`).
- **Log watching**: `cut` in the pipeline was buffering output. This does not affect the server.

## After the rehearsal
The server has been returned to test mode: pilot-1, submissions closed, rotation timer off. The rehearsal schedule is in `var/rehearsal-2026-11-20/`. The pilot-4 secret is there as well: its commitment was published on the last pass, but the period was not built.

## Not verified live
Live Stripe, the three paid detectors on the server, crown with archive and publication after 48 h (the rehearsal bundles are no better than the king), archive cleanup.
