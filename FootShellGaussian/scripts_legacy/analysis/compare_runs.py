"""Side-by-side exact-evaluator comparison of several runs.

Reports the metrics the fit is judged on *and* the two placement quantities
that say whether a low collision number was earned honestly: where the heel
sits relative to the functional heel, and how much toe space went unused.
A fit that floats short and forward can post a flattering collision area.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/foot_fit_outputs_runs"
)


def _heel_x_mm(record: dict) -> float | None:
    """Heel position, preferring the measured value, else the derived one.

    Runs made before ``heel_x_mm`` was reported can still be compared: the
    minimum vertex X follows from the toe allowance and the foot length, which
    both were always stored. That proxy is the minimum over *all* vertices
    rather than the plantar ones, so it reads a little lower.
    """

    if record.get("heel_x_mm") is not None:
        return float(record["heel_x_mm"])
    allowance, length = record.get("toe_allowance_mm"), record.get("foot_length_mm")
    if allowance is None or length is None:
        return None
    return 262.5 - float(allowance) - float(length)


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = [item for item in payload["results"] if "after" in item]
    out: dict = {"n": len(rows), "shoes": {}}
    for key in ("collision_area_fraction", "outside_area_fraction"):
        out[key] = sum(item["after"][key] for item in rows)
    out["clear"] = sum(
        1
        for item in rows
        if item["after"]["collision_area_fraction"] < 1e-9
        and item["after"]["outside_area_fraction"] < 1e-9
    )
    out["status_clear"] = sum(
        1 for item in rows if item["after"].get("status") == "clear"
    )
    for item in rows:
        after = item["after"]
        out["shoes"][item["shoe"]] = {
            "coll": after["collision_area_fraction"],
            "out": after["outside_area_fraction"],
            "toe": after["toe_allowance_mm"],
            "heel": _heel_x_mm(after),
            "pen": after["plantar_excess_penetration_count"],
            "beta": after["beta_l2_norm"],
            "status": after.get("status"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    loaded = {}
    for name in args.runs:
        path = args.root / name / "results.json"
        if not path.is_file():
            print(f"missing: {path}")
            continue
        loaded[name] = summarize(path)

    print(f"{'run':34s} {'n':>3s} {'Scoll':>8s} {'Sout':>8s} {'clear':>5s} "
          f"{'heel<0':>6s} {'toe>25':>6s} {'pen':>4s}")
    for name, data in loaded.items():
        shoes = data["shoes"].values()
        behind = sum(1 for s in shoes if s["heel"] is not None and s["heel"] < -0.05)
        longtoe = sum(1 for s in shoes if s["toe"] is not None and s["toe"] > 25.0)
        pen = sum(s["pen"] for s in shoes)
        print(f"{name:34s} {data['n']:3d} {data['collision_area_fraction']:8.4f} "
              f"{data['outside_area_fraction']:8.4f} {data['clear']:5d} "
              f"{behind:6d} {longtoe:6d} {pen:4d}")

    names = list(loaded)
    if len(names) < 2:
        return
    every = sorted(set.intersection(*(set(loaded[n]["shoes"]) for n in names)))
    print(f"\nper shoe ({len(every)} in common)")
    header = "".join(f"{n[:17]:>19s}" for n in names)
    print(f"{'shoe':38s}{header}")
    for shoe in every:
        cells = ""
        for name in names:
            s = loaded[name]["shoes"][shoe]
            heel = "  n/a" if s["heel"] is None else f"{s['heel']:5.1f}"
            cells += f"  {s['coll']:.3f}/{s['out']:.3f} {heel}"
        print(f"{shoe[:38]:38s}{cells}")
    print("\ncells are  collision/outside area fraction, then heel X in mm "
          "(negative = behind the functional heel, i.e. into the counter)")


if __name__ == "__main__":
    main()
