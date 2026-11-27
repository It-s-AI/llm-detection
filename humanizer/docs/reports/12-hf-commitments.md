# 12. Bundles on the miner's HF with an on-chain commitment, king archive, registration rules — report

## What was done (owner's decisions 2026-11-13)
- **The bundle lives in the miner's own HF repository, the version is pinned on chain** (`intake/hf.py`).
  - The miner puts `bundle.tar` into their model repository.
  - From the submission hotkey they commit `<namespace>/<repo>@<40-hex commit>` to the chain. The chain limit is 128 bytes; SN32 does not currently use commitments.
  - **Only this commit** is used:
    - a branch or tag name (`main`, `v1`) is rejected;
    - so are a shortened sha and an upper-case sha;
    - if the repository has a branch named like the commit, the submission is rejected as ambiguous.
  - The file's sha256 and size are read from the LFS metadata of this commit and recorded at submission. The pod downloads `resolve/<commit>/bundle.tar` and verifies the sha256. So a later change to the repository does not affect the evaluation.
  - The commitment must be made after the hotkey's current registration.
  - The server takes chain data from the metagraph snapshot: `refresh_metagraph.py` now writes commitments together with their block.
- **Private repositories.**
  - The miner can provide a read-only token. We store it encrypted.
  - For the pod, the server uses this token to obtain a short-lived signed CDN link; the token itself never reaches the pod.
  - After the evaluation the token is deleted.
- **The crown counts only after the bundle is copied into our archive** (`intake/archive.py`).
  - After a win the server downloads the bundle once more from the pinned commit: https only, with a size limit and sha256 verification.
  - If the file is missing or has changed, the crown is not awarded, and the evaluation is still paid for.
  - King reruns in subsequent periods run from the archive.
  - If the process crashed during copying, the step resumes after a restart.
- **One bundle counts once:** the same sha256 from another hotkey is rejected.
- **One submission per hotkey registration.** These do not use up the registration:
  - a submission cancelled before launch;
  - an unpaid quote that expired;
  - a failure on our side.
- **A deregistered king drops out.**
  - The others shift in the 85/10/5 payouts: the fourth-from-last king returns to third place.
  - If the current king is deregistered, the crown returns to the previous one. If that king has already been evaluated in this period, its run is taken from the saved outputs; no new pod is needed.
  - The signed leaderboard is built by the same rules.
- **Aggregates open to the miner:**
  - the distribution of levels 1 / 0.5 / 0, overall and per gate;
  - the contribution of each detector: E, AUROC, P_human;
  - the number of judge errors.

  We do not show the texts of the judges' reasons: they quote hidden inputs.
- **Miner CLI:** `miner_submit.py publish` uploads the bundle to HF, commits to the chain and files the submission in one command.

## Mapping to decisions
| Decision | Where in code | How verified | Status |
|---|---|---|---|
| D-60 bundle on the miner's HF, version by commit only | `intake/hf.py`, `service.submit_hf` | tests: branches, tags, short sha, ambiguous ref, missing file; live HF API (response shape) | ✅ |
| D-66 one submission per registration, rollback when a king is deregistered | `queue.create`, `KingState.active_history`, `worker.ensure_king` | tests: repeat from the same registration, new registration, our failure, 85/10/5 shift, crown return without a pod | ✅ |
| D-70 crown after the copy | `intake/archive.py`, `worker._finish_crown` | tests: successful copy → crown; file changed → crown not awarded, previous king stays | ✅ |
| D-70 publication after 48 h: the miner's repository, otherwise our copy | `publish/kings.py`, `GET /v1/kings` | tests: link to the miner's commit; private / gated / deleted / branch named like the commit → our copy; recheck every 6 h; an HF failure changes nothing; copy did not match → not published | ✅ (live upload to our HF not verified) |
| D-58 aggregates before reveal | `pipeline` (fields `levels`, `gates`, `detectors`) | test: only allowed fields, no "reasons" | ✅ |

## Test results
- `tests/intake/test_hf.py` — 9 tests:
  - commitment;
  - LFS metadata, small file without LFS;
  - ambiguous and missing revisions;
  - private repository and signed link;
  - archive: sha256, limit, redirect to http;
  - crown after archiving and refusal without it;
  - token encrypted and deleted.
- `tests/intake/test_intake.py`: one submission per registration, our failure, crown return, payout shift, registration blocks in the snapshot.
- Full suite: 446 tests, green.

## Risks
- **A public repository is visible to everyone before evaluation.** An exact copy of the bundle from another hotkey is rejected. A modified copy is not: it will pass if it is paid for and evaluated before the original. Closing this risk requires a private repository with a token.
- **Snapshot delay.** The server sees commitments through the metagraph snapshot, which must be refreshed every 2–5 minutes. Right after the commit the API responds "commitment not visible yet, retry in a few minutes".
- **HF signed links live for a limited time.** We obtain the link right before renting the pod, so the download starts within its validity period.

## King publication after 48 h (owner's decision 2026-11-17)
- **Link to the miner's repository.** 48 h after coronation, `GET /v1/kings` shows `huggingface.co/<repo>/tree/<commit>`. Conditions:
  - anyone can download the commit without a token: the repository is neither private nor gated;
  - there is no branch or tag named like the commit;
  - the sha256 is the same as at evaluation.

  The server checks the link every 6 h.
- **Our copy.** If a condition is violated, the server uploads the bundle from the archive into our public copies repository (a single one, created in advance), into the folder `kings/<id>/`. The folder's README states the source commit and sha256. After the upload we verify the sha256 of the copy, and from then on the link points to it permanently.
- **An HF failure** (network, 5xx) changes nothing; the check is repeated.
- **Needed for the copy:** a public repository `<organization>/sn32-kings` and a token with write access to it only (`--mirror-repo`, `HF_MIRROR_TOKEN`). Without them such a king is marked `unavailable` and an error is logged, while the copy stays in the archive.

## What next
- A live check of uploading the copy to our HF: needs an organization and a token.
- Delete archives of kings that are no longer needed either for payouts or for publication.
