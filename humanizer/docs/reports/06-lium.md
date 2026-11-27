# 06. Execution on Lium, part 1 — report

## What was done
- `LiumClient` with safeguards: **other projects' pods are inaccessible by design** (deleting/scheduling deletion is possible only for pods
  from our registry whose live name starts with `humeval-`); top-ups, payments, credit transfers and
  any endpoints not listed are never called (allowlist); $50 limit via the spend log; minimum
  remaining balance = current spend rate of other projects' pods × 48 h; no more than one pod of our own; price ceiling; auto-deletion
  at rental time (`termination_hours`) + scheduled deletion.
- Environment probe script `scripts/lium_probe.py` (rental → checks over SSH → deletion in `finally`).

## Probe results (4 RTX 3090 rentals, $0.20–0.22/h, **$0.035** in total)
| Check | Result |
|---|---|
| Kernel / privileges in the pod | 6.8.0-149, root |
| GPU in the pod | ✅ |
| Docker inside the pod (DinD) | ✅ Docker 27.3.1 |
| Internet access of the pod itself | yes (as expected) |
| Inner container `--network none` | ✅ only `lo`, no outbound access |
| Hardened container (no capabilities, `no-new-privileges`, read-only, user nobody, process limit) | ✅ works |
| GPU in the inner container with `--network none` | ✅ works |
| **gVisor (runsc 20261102)** | ⛔ **does not start** — neither via docker (network namespace error), nor directly (`runsc do` — the sandbox crashes on startup), nor with GPU (nvproxy) — limitations of the Sysbox pod |
| Other projects' pods | untouched (balance and pods checked before/after) |

## Two important findings
1. **There are currently no CVM nodes with passed TDX attestation in Lium's public listing**: all 158 nodes have
   `tdx_attestation_passed = false` (47 have hardware TDX support). For production (D-62: CVM only) this is a
   blocker — we need to ask Lium when attested CVM nodes will become available for rent. For debugging on dummy
   data, regular nodes are fine.
2. **gVisor is not available on Lium.** What remains is isolation with a regular hardened container (`runc`). Per our
   agreement: "if gVisor can't be made to work, we enable the two-stage crown" (D-63, the mode is already built in as a setting).
   **Your decision is needed:** enable the two-stage crown or keep the instant one (code review of the candidate +
   early dataset rotation in case of an incident).

## Tests
11 safeguard tests (kill switch, other project's pod, spoofed name, minimum balance, limit, price/hours,
concurrency, deletion on scheduling failure, endpoint allowlist, absence of payment endpoints in the code).

## What we need from you
- Decision on the two-stage crown (see above).
- Question to Lium: (a) when attested CVM nodes will be available; (b) whether a pod's outbound traffic can be blocked at the runner level.
