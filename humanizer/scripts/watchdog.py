"""Watchdog for the evaluator server (systemd timer, every 10 minutes): checks what can go wrong while nobody
is looking and alerts the owner through @alarmer_bot (ops/alerts.py). An outage is re-sent at most once an
hour, a low balance once a day; a daily summary goes out at --summary-hour (UTC) so a silent channel means "all fine".

    python scripts/watchdog.py --state /srv/humeval/var/intake [--public-url https://...] [--dry-run]

Checks: the API answers (locally and on the public address), the TLS certificate is not about to expire,
the metagraph snapshot is fresh, the disk has room, no evaluation is stuck, backups are recent, the spend
caps of the local ledger have room (if any are set), the soft limit holds, the OpenRouter credit and the
Lium balance (and no forgotten pod).

Soft limit (SOFT_LIMIT_SINCE, SOFT_LIMIT_GAP_USD): since the launch, what we spent on the paid services
should stay under what miners paid in (net of refunds) plus a gap for the owner's own runs (the kings').
Past it the owner is alerted; nothing is stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import socket
import ssl
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.ops.alerts import from_settings
from humanizer_eval.paths import ledger_path, var_dir

SERVICES = ("pangram", "gptzero", "zerogpt", "openrouter", "lium")


def soft_limit(ledger_db: Path, billing_db: Path, since: float, gap_usd: float, weekly_usd: float = 0.0
               ) -> tuple[float, float]:  # fmt: skip
    """(spent, allowed) in USD since ``since``: the ledger's reserved and committed amounts on the paid
    services, against the miners' net payments minus refunds plus ``gap_usd`` plus ``weekly_usd`` per
    period begun (the same limit as budget.PoolLimit)."""
    import sqlite3

    from humanizer_eval.budget import pool_allowance

    marks = ",".join("?" * len(SERVICES))
    with sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True) as c:
        spent = c.execute(
            f"SELECT COALESCE(SUM(COALESCE(actual_usd, estimate_usd)), 0) FROM ledger "
            f"WHERE ts >= ? AND status != 'released' AND service IN ({marks})",
            (since, *SERVICES),
        ).fetchone()[0]
    paid = 0
    if billing_db.exists():
        with sqlite3.connect(f"file:{billing_db}?mode=ro", uri=True) as c:
            paid = c.execute(
                "SELECT COALESCE(SUM(net_cents - refunded_cents), 0) FROM payments WHERE ts >= ?", (since,)
            ).fetchone()[0]
    return float(spent), paid / 100 + pool_allowance(since, gap_usd, weekly_usd)


def check_api(url: str) -> str | None:
    try:
        r = httpx.get(url.rstrip("/") + "/v1/info", timeout=15)
        return None if r.status_code == 200 else f"API {url} answers HTTP {r.status_code}"
    except httpx.HTTPError as e:
        return f"API {url} does not answer ({type(e).__name__})"


def cert_days_left(url: str) -> float | None:
    u = urlsplit(url)
    if u.scheme != "https" or not u.hostname:
        return None
    ctx = ssl.create_default_context()
    with (
        socket.create_connection((u.hostname, u.port or 443), timeout=15) as sock,
        ctx.wrap_socket(sock, server_hostname=u.hostname) as s,
    ):
        not_after = s.getpeercert()["notAfter"]
    end = dt.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=dt.UTC)
    return (end - dt.datetime.now(dt.UTC)).total_seconds() / 86400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=None, help="the intake state dir (queue, metagraph)")
    ap.add_argument("--local-url", default="http://127.0.0.1:8032")
    ap.add_argument("--public-url", default=None)
    ap.add_argument("--backups", type=Path, default=None)
    ap.add_argument("--summary-hour", type=int, default=9, help="UTC hour of the daily summary")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--stuck-hours", type=float, default=3.0)
    ap.add_argument("--notary", default=None, help="SS58 address of the notary wallet (fees for remarks)")
    ap.add_argument("--notary-rpc", default=None, help="its chain (default: finney)")
    ap.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = ap.parse_args()
    s = get_settings()
    state = args.state or (var_dir() / "intake")
    alerter = from_settings(state_path=var_dir() / "ops" / "alerts_state.json")
    problems: list[tuple[str, str]] = []  # (dedupe key, text)
    facts: list[str] = []

    for name, url in (("api-local", args.local_url), ("api-public", args.public_url)):
        if url and (p := check_api(url)):
            problems.append((name, p))
    if args.public_url:
        try:
            days = cert_days_left(args.public_url)
            if days is not None and days < 2:
                problems.append(("tls", f"TLS certificate of {args.public_url} expires in {days:.1f} days"))
        except (OSError, ssl.SSLError) as e:
            problems.append(("tls", f"TLS check of {args.public_url} failed: {type(e).__name__}: {e}"))

    snap = state / "metagraph.json"
    if snap.exists():
        age = (time.time() - json.loads(snap.read_text()).get("ts", 0)) / 60
        if age > 30:
            problems.append(("metagraph", f"metagraph snapshot is {age:.0f} min old (registrations unseen)"))
    free = shutil.disk_usage(var_dir())
    free_gb, free_frac = free.free / 2**30, free.free / free.total
    if free_gb < args.min_free_gb or free_frac < 0.1:
        problems.append(("disk", f"disk: {free_gb:.0f} GB free ({free_frac:.0%})"))
    facts.append(f"disk {free_gb:.0f} GB free")
    root = shutil.disk_usage("/")  # nothing of ours is there, but a full system disk breaks ssh and journald
    if root.free < 2 * 2**30:
        problems.append(("root-disk", f"system disk /: {root.free / 2**30:.1f} GB free (everything of ours is on "
                         "the volume; free the system disk)"))  # fmt: skip

    q = state / "queue.sqlite"
    if q.exists():
        import sqlite3

        with sqlite3.connect(q) as c:
            counts = dict(c.execute("SELECT state, COUNT(*) FROM submissions GROUP BY state").fetchall())
            stuck = c.execute(
                "SELECT id, started_ts FROM submissions WHERE state = 'running' AND started_ts < ?",
                (time.time() - args.stuck_hours * 3600,),
            ).fetchall()
            failed = c.execute(
                "SELECT COUNT(*) FROM submissions WHERE state = 'failed' AND finished_ts > ?",
                (time.time() - 86400,),
            ).fetchone()[0]
        for sid, started in stuck:
            problems.append(
                (f"stuck-{sid}", f"evaluation {sid} running for {(time.time() - started) / 3600:.1f} h")
            )
        facts.append(f"queue {counts}; failed in 24h: {failed}")

    if args.backups and args.backups.exists():
        runs = sorted(p for p in args.backups.iterdir() if p.is_dir())
        if not runs or time.time() - runs[-1].stat().st_mtime > 3 * 3600:
            problems.append(("backup", "no state backup in the last 3 hours"))

    led = SpendLedger(ledger_path(), s.caps())
    for svc in SERVICES:
        cap = s.caps().get(svc)
        if not cap or cap == float("inf"):
            continue
        left = led.remaining(svc)
        facts.append(f"{svc} ${left:.0f} of cap left")
        if left < max(10.0, 0.1 * cap):
            problems.append((f"cap-{svc}", f"{svc}: only ${left:.2f} left under the spend cap ${cap:.0f}"))

    if not s.soft_limit_since:
        problems.append(("soft-limit", "SOFT_LIMIT_SINCE is not set: no spend alert at all"))
    spent = allowed = 0.0
    try:  # a broken check must not silence the others (the alerts are sent at the end)
        if s.soft_limit_since:
            since = dt.datetime.fromisoformat(s.soft_limit_since).replace(tzinfo=dt.UTC).timestamp()
            spent, allowed = soft_limit(ledger_path(), state / "billing.sqlite", since, s.soft_limit_gap_usd,
                                        s.soft_limit_weekly_usd)  # fmt: skip
            facts.append(
                f"spent ${spent:.0f} of ${allowed:.0f} (miners' money + gap + weekly) since {s.soft_limit_since}"
            )
    except Exception as e:
        problems.append(("soft-limit", f"soft-limit check failed: {type(e).__name__}: {e}"))
    if s.soft_limit_since and spent > allowed:
        problems.append(("soft-limit", f"soft limit: spent ${spent:.2f} since {s.soft_limit_since}, "
                         f"allowed ${allowed:.2f} (miners' money + gap + weekly allowance)"))  # fmt: skip

    if s.openrouter_api_key:
        try:
            r = httpx.get("https://openrouter.ai/api/v1/credits",
                          headers={"Authorization": f"Bearer {s.openrouter_api_key}"}, timeout=15)  # fmt: skip
            d = r.json().get("data") or {}
            credit = float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
            facts.append(f"OpenRouter credit ${credit:.0f}")
            if credit < 20:
                problems.append(("openrouter-credit", f"OpenRouter credit is ${credit:.2f}"))
        except (httpx.HTTPError, ValueError) as e:
            problems.append(("openrouter-credit", f"OpenRouter credit check failed: {type(e).__name__}"))

    if s.lium_api_key:
        try:
            from humanizer_eval.runner.lium import LiumClient, Registry

            lc = LiumClient(
                s.lium_api_key, led, Registry(var_dir() / "lium" / "registry.json"), enabled=False
            )
            me = lc.me()
            balance, burn = float(me.get("balance") or 0), float(me.get("burn_per_hour") or 0)
            facts.append(f"Lium ${balance:.0f} (burn ${burn:.2f}/h)")
            if burn and balance / burn < 24:
                problems.append(
                    ("lium-runway", f"Lium: ${balance:.2f} left, {balance / burn:.0f} h at ${burn:.2f}/h")
                )
            reg = lc.registry.load()
            for p in lc.pods():  # live pods with our prefix, aged from our registry
                name = str(p.get("pod_name") or p.get("name") or "")
                if not name.startswith("humeval-"):
                    continue
                own = reg.get(p.get("id"))
                age_h = (time.time() - own.created_at) / 3600 if own else None
                if age_h is None or age_h > args.stuck_hours:
                    how = "not in our registry" if age_h is None else f"{age_h:.1f} h old"
                    problems.append((f"pod-{name}", f"our Lium pod {name} is {how} (forgotten?)"))
        except Exception as e:  # a monitoring check must not crash the watchdog
            problems.append(("lium-api", f"Lium check failed: {type(e).__name__}"))

    if args.notary:
        try:
            from humanizer_eval.period.chain import FINNEY_RPC, Chain
            from humanizer_eval.period.notary import ss58_pubkey

            tao = Chain(args.notary_rpc or FINNEY_RPC).free_balance(ss58_pubkey(args.notary))
            facts.append(f"notary τ{tao:.4f}")
            if tao < 0.01:  # ~100 remarks; two per period
                problems.append(
                    ("notary-balance", f"notary wallet {args.notary}: τ{tao:.4f} left for remark fees")
                )
        except Exception as e:
            problems.append(("notary-balance", f"notary balance check failed: {type(e).__name__}"))

    money = (
        "cap-",
        "openrouter-credit",
        "lium-runway",
        "notary-balance",
    )  # balances: once a day; outages: once an hour
    for key, text in problems:
        if args.dry_run:
            print("ALERT", key, text)
        else:
            alerter.send(text, dedupe=key, min_interval_s=86400 if key.startswith(money) else 3600)
    now = dt.datetime.now(dt.UTC)
    if now.hour == args.summary_hour:
        summary = ("daily: " + ("all checks pass" if not problems else f"{len(problems)} problem(s)") + "; "
                   + "; ".join(facts))  # fmt: skip
        if args.dry_run:
            print("SUMMARY", summary)
        else:
            alerter.send(summary, dedupe=f"summary-{now.date()}", min_interval_s=86400)
    print(json.dumps({"problems": [t for _, t in problems], "facts": facts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
