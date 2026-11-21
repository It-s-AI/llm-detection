# Evaluation server deployment

One server: the API for miners, the queue and the worker (`scripts/intake_server.py`), a snapshot of the SN32 metagraph every
3 minutes, backups every hour. Lium pods are rented from the same server; keys live only on this server.

## Machine
- **OS:** Ubuntu 22.04 or 24.04.
- **Resources:** 4 vCPU, 8–16 GB of memory.
- **Disk: 300 GB SSD.**
  - Archive of king bundles: the last 4 kings at ≤ 50 GB each plus room to download a new one, up to 250 GB in total in the worst case. Bundles of ordinary submissions never land on the server: the pod downloads them directly from HF.
  - Everything else (system, environments, pools, periods, caches, backups): about 30 GB.
  - In practice, 4–8B models produce bundles of 10–20 GB, and the archive will take ~100 GB.
- **Network:** a public IP and a domain, ports 80 and 443 open. All other ports closed.

## Steps
1. **User and directories.** Create the user `humeval` and the directory `/srv/humeval/{humanizer_eval,var,backups}`, owned by `humeval`.
2. **Code.** Clone the `integration` branch of the `ChervovNikita/humanizer_eval` repository (the working one; the public copy is the `humanizer/` folder of the `v4` branch in It-s-AI/llm-detection) into `/srv/humeval/humanizer_eval`, then:
   ```
   uv venv .venv && uv pip install -e '.[api,pool]'
   ```
3. **bittensor** is needed only for the metagraph snapshot. We install it into a separate environment:
   ```
   uv venv /srv/humeval/btenv
   VIRTUAL_ENV=/srv/humeval/btenv uv pip install bittensor==10.5.0 -e /srv/humeval/humanizer_eval
   ```
4. **Secrets.**
   - Copy `deploy/env.example` to `/etc/humeval/humeval.env` (mode 600, owner `humeval`) and fill it in.
   - Move `var/keys/` from the development machine: `period.key`, `leaderboard_ed25519.key`, `intake_secrets.key`.
   - Write the public key and the leaderboard URL into the validator (the `v4` branch in It-s-AI/llm-detection) with the `release_validator.py` script.
5. **Period data.** You need `var/periods/<id>/` and the file `var/periods/CURRENT` with the id of the current period, as well as the pools `var/pool-full-v2` for human texts.
6. **systemd.** Copy `deploy/systemd/*` to `/etc/systemd/system/`, set your domain in `humeval-intake.service`, then:
   ```
   systemctl daemon-reload
   systemctl enable --now humeval-metagraph.timer humeval-backup.timer humeval-intake.service
   ```
7. **Caddy.** Install Caddy, put `deploy/Caddyfile` with your domain into `/etc/caddy/Caddyfile` and restart it: `systemctl reload caddy`. Caddy obtains the Let's Encrypt certificate by itself.
8. **Stripe.**
   - The webhook is optional: the miner confirms payment with the `paid` command, and the server also queries Stripe on its own.
   - If you want a webhook for speed, create an endpoint `https://<domain>/v1/stripe/webhook` in the dashboard for the `checkout.session.completed` event and put the signing secret into `STRIPE_WEBHOOK_SECRET`.
   - Set the live key `sk_live_` only at launch; the server refuses to work with it without the `--live-stripe` flag.
9. **Check.**
   - `curl https://<domain>/v1/info` shows the period and the queue.
   - `curl https://<domain>/v1/leaderboard` returns signed weights.
   - `journalctl -u humeval-intake -f` shows the logs.

## Leaderboard (HF Space `sergak0/ai-detection-leaderboard`)
- The humanizer is a separate section next to the detector leaderboard: the module `space/humanizer_tab.py` and two hooks in `app.py` under try/except. If the module fails, the detector part works as before.
- Rollout: `python scripts/publish_space.py --space sergak0/ai-detection-leaderboard` opens a PR in the Space (`--direct` pushes straight to main).
- After the server is up, set its address: `python scripts/publish_space.py --space sergak0/ai-detection-leaderboard --api https://<domain> --set-api` (the Space variable `HUMANIZER_API`).
- The section reads only the public `/v1/info`, `/v1/leaderboard`, `/v1/results`, `/v1/kings` and caches them for a minute.

## Test launch server (2026-11-18)
- **Machine:** `188.245.168.78`, shared with other projects. We do not touch other people's services and files. The root disk is 100% full, so everything of ours lives on the volume `/mnt/HC_Volume_107517418/humeval`. On `/` there are only the systemd units and the system user `humeval`. We do not use apt.
- **Layout on the volume:**
  - `bin/` — uv and caddy;
  - `python/` — Python 3.12 from uv;
  - `humanizer_eval/` — the code, `integration` branch, with its own `.venv`;
  - `btenv/` — the environment with bittensor;
  - `var/` — data and state;
  - `etc/humeval.env` — keys, mode 640;
  - `caddy/` — Caddyfile and certificates;
  - `logs/` — logs, instead of journald;
  - `backups/`, `tmp/`.
- **Services:**
  - `humeval-intake` — API and worker on `127.0.0.1:8032`;
  - `humeval-caddy` — HTTPS by IP, a Let's Encrypt certificate for ~6 days, renewed automatically;
  - timers `humeval-metagraph` (testnet 87 snapshot every 3 minutes) and `humeval-backup` (every hour).
- **Mode:** `test`. Network `test`, subnet 87, period `pilot-1`, It's AI only, test Stripe only, 10 evaluations per day. The `HUMEVAL_*` variables are in `etc/humeval.env`.
- **Code update** without a token on the server: `git bundle create he.bundle integration`, then `scp` into `tmp/` and on the server `git -C humanizer_eval pull tmp/he.bundle integration`. After that, `systemctl restart humeval-intake`.
- **Check:**
  - `curl https://188.245.168.78/v1/info`;
  - logs: `tail -f /mnt/HC_Volume_107517418/humeval/logs/intake.log`.

## Period rotation (prod)
- **Units:** `humeval-rotate.service` and `.timer`, every 15 minutes. Each pass does whatever is due (`period/lifecycle.py`):
  - publishes the commitment of the next period and block B;
  - builds the period after B is finalized (~$5 on OpenRouter for 500 texts);
  - switches `CURRENT` once the server has written `READY`;
  - reveals the closed period into `var/public/periods/<id>/`.
- **What the server does:** the `prepare_loop` thread loads the period from `NEXT` by itself (H is scored here, once, and cached in `human_scores.<detectors>.enc`) and runs the king in advance. Lium allows one pod more than `--parallel`.
- **The first period** is built manually (`build_period.py --current`). The timer picks it up on its first pass.
- **On the test server the timer is not enabled.** Every period costs money (generation, detectors on H, the king's run), and there are no submissions. It is enabled when moving to prod: `systemctl enable --now humeval-rotate.timer`.
- **Check the schedule:** `python scripts/rotate_period.py --status` (changes nothing).
- **Public copy on HF:** `HUMEVAL_ROTATE_ARGS="--hf-repo <org>/<repo>"` in env. The same token `HF_MIRROR_TOKEN` is used; it needs write access to that repository.
- **On-chain notary.** The wallet `humeval-notary` (`5D76hDgEK6VT3y9XChyWTQUVqSmJEftiCGtxfGBrnnYTjkwH`) was created on the test server on 2026-11-20 and is stored in `var/keys/wallets`, coldkey without a password. It is included in backups and moves together with `var/keys`. We keep ~τ0.1 on it: enough for hundreds of periods. On finney it is still empty.
  - Create: `btenv/bin/python scripts/notary_post.py --wallet humeval-notary --path var/keys/wallets --create`.
  - Balance: the same script with `--balance --network finney`.
  - In env: `HUMEVAL_ROTATE_ARGS="--notary-wallet humeval-notary --notary-python <volume>/btenv/bin/python"`.
  - The watchdog monitors the balance: `--notary <address>`, threshold τ0.01.
  - Without the commitment written on chain the period is not built; without the Merkle root written the server does not prepare the period and does not switch to it.
  - If the commitment did not make it before block B: `rotate_period.py --recommit "<reason>"`. A cancelled commitment stays in the schedule.
- **The bundle archive** is cleaned in the publication loop. Bundles of kings with a live registration are kept: they can regain the crown. The bundle of a king without a registration is deleted only after a public copy exists in the mirror. Without `--mirror-repo`, nothing is deleted.

## Moving to the production server
- **What to move:** `var/keys/` (the period key, the submission secrets key, the leaderboard signing key — its public part is hard-coded in validators), `var/intake/` (queue, balances, king, archive), `var/periods/`, `var/detector_cache.sqlite`, `var/calibration/llm_cache.sqlite`, `spend_ledger.sqlite`. The easiest way is from a fresh backup (`scripts/backup_state.py`).
- **Validators:** `python scripts/release_validator.py --repo ../llm-detection --url https://<prod>/v1/leaderboard --commit` checks the live leaderboard with the key and writes the URL and the key into the validator branch. No push: the owner merges into llm-detection.

## What is not automated
- **HF repository for king copies.** After 48 h the server shows the miner's repository. If it has been closed, our copy is needed: a public repository `<organization>/sn32-kings`, the flag `--mirror-repo <organization>/sn32-kings` in `humeval-intake.service`, and `HF_MIRROR_TOKEN` (write access only to that repository) in env. Without them such a king is marked `unavailable`.
- **Off-machine copy of backups.** `/srv/humeval/backups` must be copied to encrypted storage: it holds keys and miner balances.
