#!/usr/bin/env python3
"""Build the comparison tables for the report from the run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs")
PRIMARY = "collision_area_fraction"
SECONDARY = "outside_area_fraction"


def load(run: str) -> dict:
    path = RUNS / run / "results.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {r["shoe"]: r for r in payload["results"] if "after" in r}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[
        "expB_translation", "expC_pose", "expD_full",
        "ablE_no_support", "ablF_no_beta_prior",
        "ablG_no_ankle_exempt", "ablH_no_length",
    ])
    args = parser.parse_args()

    baseline = json.loads((RUNS / "baseline_numpy.json").read_text())
    shoes = sorted(baseline)
    runs = {name: load(name) for name in args.runs}
    runs = {name: value for name, value in runs.items() if value}

    def column(values, key):
        present = [v for v in values if v is not None]
        return sum(present) if present else float("nan")

    print("\n## Aggregate over", len(shoes), "shoes (lower is better)\n")
    header = f"{'method':<24}{'SUM coll':>10}{'SUM out':>10}{'mean|toe-20|':>14}{'mean|beta|':>12}{'clear':>7}{'penetr':>8}"
    print(header); print("-" * len(header))

    def report(label, getter):
        coll = sum(getter(s)[PRIMARY] for s in shoes)
        out = sum(getter(s)[SECONDARY] for s in shoes)
        toe = sum(abs(getter(s)["toe_allowance_mm"] - 20.0) for s in shoes) / len(shoes)
        beta = sum((getter(s)["beta_l2_norm"] or 0.0) for s in shoes) / len(shoes)
        clear = sum(1 for s in shoes if getter(s)["status"] == "clear")
        pen = sum(getter(s)["plantar_excess_penetration_count"] for s in shoes)
        print(f"{label:<24}{coll:>10.4f}{out:>10.4f}{toe:>14.1f}{beta:>12.2f}{clear:>7}{pen:>8}")

    report("C5 start (no fit)", lambda s: baseline[s]["checkpoint5"])
    report("A: NumPy fitter", lambda s: baseline[s]["numpy_fitter"])
    for name, data in runs.items():
        if all(s in data for s in shoes):
            report(f"torch: {name}", lambda s, d=data: d[s]["after"])
        else:
            print(f"torch: {name:<17} INCOMPLETE ({len(data)}/{len(shoes)} shoes)")

    full = runs.get("expD_full")
    if full and all(s in full for s in shoes):
        print("\n## Per shoe: A (NumPy) vs D (torch, full)\n")
        header = (f"{'shoe':<40}{'A coll':>9}{'D coll':>9}{'A out':>9}{'D out':>9}"
                  f"{'D toe':>8}{'D |b|':>7}{'winner':>9}")
        print(header); print("-" * len(header))
        wins = losses = ties = 0
        for s in shoes:
            a = baseline[s]["numpy_fitter"]; d = full[s]["after"]
            delta = d[PRIMARY] - a[PRIMARY]
            if abs(delta) < 1e-6:
                mark, t = "tie", 1; ties += 1
            elif delta < 0:
                mark = "torch"; wins += 1
            else:
                mark = "numpy"; losses += 1
            print(f"{s:<40}{a[PRIMARY]:>9.4f}{d[PRIMARY]:>9.4f}{a[SECONDARY]:>9.4f}"
                  f"{d[SECONDARY]:>9.4f}{d['toe_allowance_mm']:>8.1f}"
                  f"{(d['beta_l2_norm'] or 0):>7.2f}{mark:>9}")
        print(f"\ntorch better on {wins}, worse on {losses}, tied on {ties} "
              f"(primary metric = exact collision area fraction)")
        seconds = [full[s]["fit_seconds"] for s in shoes]
        memory = [full[s]["peak_memory_bytes"] / 1e6 for s in shoes]
        bake = [full[s]["field"]["bake_seconds"] for s in shoes]
        print(f"\nfit  {min(seconds):.1f}-{max(seconds):.1f}s (mean {sum(seconds)/len(seconds):.1f}s)"
              f"   bake {min(bake):.2f}-{max(bake):.2f}s"
              f"   peak GPU {max(memory):.0f} MB")


if __name__ == "__main__":
    main()
