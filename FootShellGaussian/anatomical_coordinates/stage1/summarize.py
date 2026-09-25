"""Tables from the Stage 1 runs: A vs A+ vs B, per split, per region.

    python -m anatomical_coordinates.stage1.summarize

Numbers are means over the held-out shoes of a split and over seeds; the
per-shoe values are in each run's JSON. With 2-4 test shoes per split, read
the per-shoe values before trusting a difference in the mean.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from . import common
from .train import SPLITS, VARIANTS

HIDDEN = ("interior", "sole", "top")


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def collect() -> dict:
    rows = defaultdict(lambda: defaultdict(list))
    for split in SPLITS:
        for path in sorted((common.STAGE1_OUTPUT / "runs" / split).glob("*.json")):
            run = json.loads(path.read_text())
            rows[(split, run["variant"])]["runs"].append(run)
    return rows


def table(rows, mode: str) -> list[str]:
    lines = []
    header = (f"{'split':8s} {'var':6s} | " + " | ".join(
        f"{r:>8s} fld rec2" for r in ("visible",) + HIDDEN) + " | false")
    lines.append(header)
    for split in SPLITS:
        if mode != "fit" and split == "all":
            continue
        for variant in VARIANTS:
            runs = rows.get((split, variant), {}).get("runs", [])
            if not runs:
                continue
            shoes = [shoe for run in runs for shoe in run[mode].values()]
            cells = []
            for region in ("visible",) + HIDDEN:
                fld = _mean([s[region]["field_mm"] for s in shoes])
                rec = _mean([s[region]["recall_2mm"] for s in shoes])
                cells.append(f"{'':>8s} {fld if fld is not None else float('nan'):4.2f} "
                             f"{(rec if rec is not None else float('nan'))*100:3.0f}%")
            false = _mean([s["false_surface"] for s in shoes])
            lines.append(f"{split:8s} {variant:6s} | " + " | ".join(cells)
                         + f" | {false*100 if false is not None else float('nan'):4.1f}%"
                         + f"   ({len(runs)} seed{'s' if len(runs) > 1 else ''})")
    return lines


def main() -> None:
    rows = collect()
    out = []
    for mode, title in (("fit", "FIT - held-out samples of training shoes"),
                        ("full", "FULL - unseen shoe, code fitted to all its samples"),
                        ("ring", "RING - unseen shoe, code fitted only to what a 0-degree ring observes")):
        out.append(f"\n{title}\n  fld = near-surface distance error (mm), rec2 = surface recall within 2 mm, "
                   f"false = invented surface in empty space")
        out.extend("  " + line for line in table(rows, mode))
    text = "\n".join(out)
    print(text)
    (common.STAGE1_OUTPUT / "runs" / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
