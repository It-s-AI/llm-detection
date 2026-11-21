"""Simulate the crown test on synthetic periods: A/A false-crown rate and power curves.

    python scripts/simulate_crown.py              # full run (~2 min on 16 cores)
    python scripts/simulate_crown.py --quick      # smoke run (seconds)
    python scripts/simulate_crown.py --csv var/sim/crown.csv
    python scripts/simulate_crown.py --decompose  # also: where the bootstrap se comes from
    python scripts/simulate_crown.py --mde 300,500,750 --gate-fail 0.12 --half-rate 0.23
                                                  # smallest dS crowned with 80% power, per period size

Prints markdown tables to stdout. No network, no paid APIs; see humanizer_eval.scoring.simulate
for the generative model.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from humanizer_eval.scoring.simulate import (
    SimConfig,
    TrialSummary,
    advantage_for_delta,
    default_jobs,
    expected_score,
    king_skill,
    measured_delta,
    power_grid,
    se_decomposition,
)


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="few periods and B=500: a smoke run")
    ap.add_argument("--aa-periods", type=int, default=None, help="A/A periods per phi (default 2000)")
    ap.add_argument("--power-periods", type=int, default=None, help="periods per power cell (default 300)")
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default 2000, D-23)")
    ap.add_argument("--deltas", type=_floats, default=None, help="true dS grid, e.g. 0.01,0.02,0.04")
    ap.add_argument("--phis", type=_floats, default=[0.0, 0.5, 0.8])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta-min", type=float, default=0.0)
    ap.add_argument("--n-inputs", type=int, default=500)
    ap.add_argument("--n-human", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--csv", type=Path, default=None, help="also write all cells to this CSV file")
    ap.add_argument("--decompose", action="store_true", help="A/A se: inputs-only vs H-only vs both")
    ap.add_argument("--decompose-periods", type=int, default=None, help="periods per phi (default 1000)")
    ap.add_argument("--w-prob", type=float, default=None, help="probability-term weight (default D-20)")
    ap.add_argument("--gate-fail", type=float, default=None, help="share of outputs at quality level 0")
    ap.add_argument("--half-rate", type=float, default=None, help="share of outputs at quality level 0.5")
    ap.add_argument("--king-aurocs", type=_floats, default=None, help="king's raw AUROC per detector")
    ap.add_argument("--mde", type=lambda x: [int(v) for v in x.split(",")], default=None,
                    help="period sizes n (H = n): print the dS crowned with 80%% power instead")  # fmt: skip
    ap.add_argument("--power-target", type=float, default=0.8)
    a = ap.parse_args(argv)

    aa_periods = a.aa_periods or (200 if a.quick else 2000)
    power_periods = a.power_periods or (60 if a.quick else 300)
    n_boot = a.n_boot or (500 if a.quick else 2000)
    deltas = a.deltas or ([0.02, 0.04] if a.quick else [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08])
    cfg = replace(SimConfig(), n_inputs=a.n_inputs, n_human=a.n_human)
    over = {"w_prob": a.w_prob, "gate_fail_rate": a.gate_fail, "half_rate": a.half_rate}
    cfg = replace(cfg, **{k: v for k, v in over.items() if v is not None})
    if a.king_aurocs:
        if len(a.king_aurocs) != len(cfg.detectors):
            raise SystemExit(f"--king-aurocs needs {len(cfg.detectors)} values")
        dets = tuple(replace(d, king_auroc=v) for d, v in zip(cfg.detectors, a.king_aurocs, strict=True))
        cfg = replace(cfg, detectors=dets)
    kw = {"alpha": a.alpha, "delta_min": a.delta_min}
    if a.mde:
        return _mde(cfg, a, deltas, power_periods, n_boot, kw)

    t0 = time.perf_counter()
    s_king = expected_score(cfg, king_skill(cfg))
    print("# Crown test simulation (synthetic)\n")
    print(
        f"n = {cfg.n_inputs} inputs, m = {cfg.n_human} human texts, {len(cfg.strata)} strata, "
        f"gate fail rate {cfg.gate_fail_rate:.0%}, B = {n_boot}, alpha = {a.alpha}, "
        f"delta_min = {a.delta_min}, seed = {a.seed}"
    )
    dets = ", ".join(
        f"{d.name} (king AUROC {d.king_auroc:.2f}, noise {d.noise_sd}, {d.decimals} dp)"
        for d in cfg.detectors
    )
    print(f"detectors: {dets}")
    print(f"king population S = {s_king:.4f}\n")

    aa = power_grid(cfg, [0.0], a.phis, n_periods=aa_periods, n_boot=n_boot, seed=a.seed, jobs=a.jobs, **kw)
    print("## A/A: challenger identical in distribution to the king\n")
    print("| phi | periods | false crowns | rate | MC se | 95% CI (Wilson) | sd(dS hat) | mean boot se |")
    print("|---|---|---|---|---|---|---|---|")
    for r in aa:
        lo, hi = r.wilson()
        print(
            f"| {r.phi:.1f} | {r.n_periods} | {r.crowns} | {r.rate:.4f} | {r.mc_se:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] | {r.sd_delta_hat:.4f} | {r.mean_boot_se:.4f} |"
        )
    print()

    pw = power_grid(
        cfg, deltas, a.phis, n_periods=power_periods, n_boot=n_boot, seed=a.seed, jobs=a.jobs, **kw
    )
    cells: dict[tuple[float, float], TrialSummary] = {(r.target_delta, r.phi): r for r in pw}
    print(f"## Power: P(crown) by true dS ({power_periods} periods per cell; dS = 0 row is the A/A above)\n")
    head = " | ".join(f"phi={phi:g}" for phi in a.phis)
    print(f"| true dS | latent advantage | dS check (n=2e5) | {head} |")
    print("|---" * (3 + len(a.phis)) + "|")
    print("| 0 | 0 | 0 | " + " | ".join(f"{r.rate:.3f}" for r in aa) + " |")
    for delta in deltas:
        adv = advantage_for_delta(cfg, delta)
        check = measured_delta(cfg, adv)
        rates = " | ".join(f"{cells[(delta, phi)].rate:.3f}" for phi in a.phis)
        print(f"| {delta:g} | {adv:.4f} | {check:.4f} | {rates} |")

    if a.decompose:
        dp = a.decompose_periods or (100 if a.quick else 1000)
        print(f"\n## A/A bootstrap se by resampling scheme ({dp} periods per phi)\n")
        print("| phi | true sd(dS hat) | se: inputs + H (D-23) | se: inputs only | se: H only |")
        print("|---|---|---|---|---|")
        for phi in a.phis:
            r = se_decomposition(cfg, phi=phi, n_periods=dp, n_boot=n_boot, seed=a.seed, jobs=a.jobs)
            print(
                f"| {phi:g} | {r.sd_delta_hat:.4f} | {r.se_both:.4f} | {r.se_inputs:.4f} | {r.se_human:.4f} |"
            )

    print(f"\nelapsed {time.perf_counter() - t0:.1f} s with {a.jobs} jobs")

    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(
                ["kind", "phi", "target_delta", "advantage", "periods", "crowns", "rate", "mc_se",
                 "ci_lo", "ci_hi", "mean_delta_hat", "sd_delta_hat", "mean_boot_se"]
            )  # fmt: skip
            for kind, rows in (("aa", aa), ("power", pw)):
                for r in rows:
                    lo, hi = r.wilson()
                    wr.writerow(
                        [kind, r.phi, r.target_delta, f"{r.advantage:.6f}", r.n_periods, r.crowns,
                         f"{r.rate:.6f}", f"{r.mc_se:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                         f"{r.mean_delta_hat:.6f}", f"{r.sd_delta_hat:.6f}", f"{r.mean_boot_se:.6f}"]
                    )  # fmt: skip
        print(f"wrote {a.csv}")
    return 0


def _mde(cfg: SimConfig, a, deltas, power_periods: int, n_boot: int, kw) -> int:
    """Smallest true dS crowned with probability >= power_target (linear interpolation), per n and phi."""
    t0 = time.perf_counter()
    s_king = expected_score(cfg, king_skill(cfg))
    print("# Detectable improvement by period size (synthetic)\n")
    print(
        f"w_prob = {cfg.w_prob}, quality levels: 0 {cfg.gate_fail_rate:.0%}, 0.5 {cfg.half_rate:.0%}; "
        f"king AUROC {', '.join(f'{d.king_auroc:.3f}' for d in cfg.detectors)}; king S = {s_king:.4f}; "
        f"B = {n_boot}, {power_periods} periods per cell, alpha = {a.alpha}\n"
    )
    head = " | ".join(f"phi={phi:g}" for phi in a.phis)
    print(f"| n (= H) | A/A crowns, phi={a.phis[0]:g} | {head} |")
    print("|---" * (2 + len(a.phis)) + "|")
    reach = [d for d in deltas if s_king + d < 1.0]
    for n in a.mde:
        c = replace(cfg, n_inputs=n, n_human=n)
        aa = power_grid(c, [0.0], a.phis[:1], n_periods=power_periods, n_boot=n_boot, seed=a.seed,
                        jobs=a.jobs, **kw)[0]  # fmt: skip
        pw = power_grid(
            c, reach, a.phis, n_periods=power_periods, n_boot=n_boot, seed=a.seed, jobs=a.jobs, **kw
        )
        cells = {(r.target_delta, r.phi): r.rate for r in pw}
        out = []
        for phi in a.phis:
            prev_d, prev_r, hit = 0.0, aa.rate, None
            for d in reach:
                r = cells[(d, phi)]
                if r >= a.power_target:
                    hit = prev_d + (d - prev_d) * (a.power_target - prev_r) / max(r - prev_r, 1e-9)
                    break
                prev_d, prev_r = d, r
            out.append(
                f"{hit:.3f} ({hit / max(s_king, 1e-9):.0%})" if hit is not None else f"> {reach[-1]:g}"
            )
        print(f"| {n} | {aa.rate:.3f} | " + " | ".join(out) + " |")
    print(f"\ncells: dS (relative to king S); elapsed {time.perf_counter() - t0:.0f} s with {a.jobs} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
