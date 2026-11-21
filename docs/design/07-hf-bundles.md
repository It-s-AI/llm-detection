# 07. Miner bundles via our Hugging Face — design note

> **Rejected by the owner on 2026-11-13.** The note is kept as an option that was considered. The current scheme is in Q-14 (`docs/spec.md`).

## Problem
A miner sends a bundle (code + weights, up to 100 GB). We need to:
- **control the copy**: the miner cannot change or delete it after submission;
- **hide other miners' submissions**: a miner cannot see others' unevaluated submissions (D-70);
- **keep secrets off the pod**: Lium pods hold none of our tokens (D-62);
- **not route bundles through our server.**

## Why not "miners create repositories in our organization"
| HF option | What it gives | Why it does not fit |
|---|---|---|
| Organization, `contributor` role (free) | creates repositories and edits only their own, cannot delete others' | sees **all** of the organization's repositories, including private ones, i.e. other miners' submissions |
| Team plan + Resource Groups | each miner sees only their own group | an external member is a paid seat, $20 per month per miner; external members can only be invited with read/write rights in a group |

## Proposed scheme (free for miners, miners have no rights in our organization)
1. **The miner uploads the bundle to their own HF repository** and turns on manual access approval. Then they grant access to our bot:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.update_repo_settings("<miner>/<repo>", gated="manual")
   api.grant_access("<miner>/<repo>", user="<our bot>")
   ```
   No one but us can download the files; only the file list is visible.
2. **Submission:** repository, commit (full sha), file path, sha256 and size. Signed with the hotkey, as now.
3. **Copy on the HF side.** The server immediately makes a copy in our private organization: `HfApi().duplicate_repo("<miner>/<repo>", "<our org>/sub-<id>", private=True)`. The files are not downloaded in the process.
4. **Verification without downloading.** sha256 and size are checked against the copy's LFS metadata (`get_paths_info(..., expand=True)`; the LFS oid is the file's sha256). On a mismatch, the submission is rejected before payment.
5. **From then on, only our copy.** The miner may delete or rewrite their repository; this does not affect the evaluation.
6. **Pod without tokens.** The server obtains a short-lived signed URL for our file: it requests `resolve` with our token and takes the CDN redirect address. This URL is passed to the pod; the pod downloads the file and checks sha256, as now. The token never reaches the pod.
7. **After evaluation:**
   - copies without the crown stay private and are deleted after the period reveal;
   - the king's copy becomes public after 48 h (D-70) and is used to re-run the king in each period.

## What is needed from the owner
- **An HF organization** for submission copies and **a bot account**:
  - a fine-grained bot token with write access to this organization;
  - this token is stored only on the intake server.
- **Storage.** A free organization has 100 GB of private storage, while a single bundle can weigh up to 100 GB. Options:
  - **(a)** a Team plan with one seat (the bot only): $20 per month, 1 TB of private storage, $18 per TB per month beyond that;
  - **(b)** lower the bundle limit, e.g. to 20 GB, and keep only the current period's copies.
- **No miner involvement.** We do not invite miners, and there are no paid seats for them.

## What to check before enabling (one free test with the bot token)
- copying a gated repository to which the bot was granted access via `grant_access`;
- the lifetime of the signed CDN URL: it must last long enough to start a 100 GB download;
- `files_copy_pending`: how long to wait until the LFS files are copied.

## Code changes (after the decision)
- `intake/service.py`: submission fields `hf_repo`, `hf_revision`, `hf_path` instead of an arbitrary https URL; copying and verification before the quote.
- `intake/worker.py`: before renting a pod — a signed URL for our copy.
- King publication and deletion of copies without the crown — on the period schedule.
- The arbitrary https URL remains for development only.
