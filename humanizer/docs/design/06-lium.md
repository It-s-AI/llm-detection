# 06. Execution on Lium (part 1: client and environment check)
Status: draft, implemented autonomously
Implements decisions: D-62, D-65

## Goal
Safely rent a disposable pod for a single submission without breaking anything for other projects on the shared account,
and find out which pod capabilities the harness can rely on.

## Interfaces
- `LiumClient`: `executors()`, `me()`, `pods()`, `check_can_rent()`, `rent()`, `delete()`, `statement()`.
- Registry of our own pods (`var/lium/registry.json`), name prefix `humeval-`.
- All restrictions are in code, before a request is sent: endpoint allowlist, only our own pods, kill switch,
  ≤ 1 pod of our own, price ceiling, $50 limit (spend log), minimum balance reserve for other projects' pods,
  `termination_hours` + scheduled deletion (if that fails, the pod is deleted immediately).

## What we are not doing in this part
The submission run harness itself (bundle upload, launch, output collection) — part 2.
