"""Aggregate per-shoe metric JSON files into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "core_eight.json"

GEOMETRY_KEYS = (
    "chamfer_l1_percent",
    "accuracy_percent",
    "completeness_percent",
    "f_score_1_percent",
    "normal_consistency",
    "p95_distance_percent",
)
VIEW_KEYS = (
    "silhouette_iou",
    "boundary_f_score",
    "depth_mae_percent",
    "depth_overlap_coverage",
    "underside_depth_mae_percent",
    "top_view_depth_mae_percent",
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_benchmark(path: Path) -> tuple[str, list[dict[str, str]]]:
    payload = _read_json(path)
    shoes = payload.get("shoes")
    if not isinstance(shoes, list) or not shoes:
        raise ValueError("Benchmark configuration has no shoes")
    records: list[dict[str, str]] = []
    for item in shoes:
        if not isinstance(item, dict) or "name" not in item or "category" not in item:
            raise ValueError("Every benchmark shoe needs name and category")
        records.append({"name": str(item["name"]), "category": str(item["category"])})
    return str(payload.get("name", path.stem)), records


def collect_rows(
    input_root: Path,
    shoes: list[dict[str, str]],
    allow_incomplete: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not methods:
        raise ValueError(f"No method directories found under {input_root}")
    for method_path in methods:
        missing: list[str] = []
        for shoe in shoes:
            result_root = method_path / shoe["name"]
            geometry_path = result_root / "geometry_metrics.json"
            if not geometry_path.is_file():
                missing.append(shoe["name"])
                continue
            geometry = _read_json(geometry_path)
            geometry_headline = geometry.get("headline")
            if not isinstance(geometry_headline, dict):
                raise ValueError(f"Missing geometry headline in {geometry_path}")
            row: dict[str, object] = {
                "method": method_path.name,
                "shoe": shoe["name"],
                "category": shoe["category"],
            }
            for key in GEOMETRY_KEYS:
                row[key] = float(geometry_headline[key])

            view_path = result_root / "view_metrics.json"
            if view_path.is_file():
                view = _read_json(view_path)
                view_headline = view.get("headline")
                if not isinstance(view_headline, dict):
                    raise ValueError(f"Missing view headline in {view_path}")
                for key in VIEW_KEYS:
                    row[key] = (
                        float(view_headline[key]) if key in view_headline else ""
                    )
                protocol = view.get("evaluation_protocol")
                if protocol is None:
                    protocol = (
                        "full_view"
                        if all(
                            key in view_headline
                            for key in (
                                "underside_depth_mae_percent",
                                "top_view_depth_mae_percent",
                            )
                        )
                        else "unknown"
                    )
                row["evaluation_protocol"] = str(protocol)
                row["heldout_eligible"] = bool(view.get("heldout_eligible", False))
            else:
                for key in VIEW_KEYS:
                    row[key] = ""
                row["evaluation_protocol"] = "unknown"
                row["heldout_eligible"] = False
            rows.append(row)
        if missing and not allow_incomplete:
            raise ValueError(
                f"Method {method_path.name} is missing {len(missing)} benchmark shoes: "
                + ", ".join(missing)
            )
    if not rows:
        raise ValueError("No completed geometry evaluations were found")
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {
            "method": method,
            "shoe_count": len(selected),
            "heldout_eligible": bool(selected)
            and all(bool(row["heldout_eligible"]) for row in selected),
        }
        protocols = {
            str(row["evaluation_protocol"])
            for row in selected
            if row["evaluation_protocol"] != "unknown"
        }
        if len(protocols) > 1:
            raise ValueError(f"Method {method} mixes evaluation protocols: {protocols}")
        summary["evaluation_protocol"] = next(iter(protocols), "unknown")
        for key in GEOMETRY_KEYS + VIEW_KEYS:
            values = [float(row[key]) for row in selected if row[key] != ""]
            summary[f"{key}_mean"] = mean(values) if values else ""
            summary[f"{key}_median"] = median(values) if values else ""
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_name(name: str) -> str:
    return name.replace("_", r"\_")


def write_latex(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{Similarity-aligned geometry reconstruction results on the core shoe "
            r"set. Distances are percentages of the ground-truth bounding-box diagonal.}"
        ),
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Method & Chamfer-L1 $\downarrow$ & Accuracy $\downarrow$ & "
            r"Completeness $\downarrow$ & F-score@1\% $\uparrow$ & "
            r"Normal Cons. $\uparrow$ & P95 Dist. $\downarrow$ \\"
        ),
        r"\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{_latex_name(str(row['method']))} & "
            f"{float(row['chamfer_l1_percent_mean']):.4f} & "
            f"{float(row['accuracy_percent_mean']):.4f} & "
            f"{float(row['completeness_percent_mean']):.4f} & "
            f"{float(row['f_score_1_percent_mean']):.4f} & "
            f"{float(row['normal_consistency_mean']):.4f} & "
            f"{float(row['p95_distance_percent_mean']):.4f} \\\\"
        )
    protocols = {
        str(row.get("evaluation_protocol", "unknown"))
        for row in summaries
        if row["silhouette_iou_mean"] != "" and row["heldout_eligible"]
    }
    turntable_only = protocols == {"turntable"}
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\label{tab:geometry_results}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            (
                r"\caption{Held-out view results on the core shoe set. Depth errors are "
                r"percentages of the ground-truth bounding-box diagonal.}"
            ),
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccc}"
            if turntable_only
            else r"\begin{tabular}{lccccc}",
            r"\toprule",
            (
                r"Method & Silhouette IoU $\uparrow$ & Boundary F-score $\uparrow$ & "
                r"Depth MAE $\downarrow$ & Depth Coverage $\uparrow$ \\"
                if turntable_only
                else r"Method & Silhouette IoU $\uparrow$ & Boundary F-score $\uparrow$ & "
                r"Depth MAE $\downarrow$ & Underside MAE $\downarrow$ & "
                r"Top-view MAE $\downarrow$ \\"
            ),
            r"\midrule",
        ]
    )
    for row in summaries:
        if row["silhouette_iou_mean"] == "" or not row["heldout_eligible"]:
            continue
        if turntable_only:
            lines.append(
                f"{_latex_name(str(row['method']))} & "
                f"{float(row['silhouette_iou_mean']):.4f} & "
                f"{float(row['boundary_f_score_mean']):.4f} & "
                f"{float(row['depth_mae_percent_mean']):.4f} & "
                f"{float(row['depth_overlap_coverage_mean']):.4f} \\\\"
            )
            continue
        lines.append(
            f"{_latex_name(str(row['method']))} & "
            f"{float(row['silhouette_iou_mean']):.4f} & "
            f"{float(row['boundary_f_score_mean']):.4f} & "
            f"{float(row['depth_mae_percent_mean']):.4f} & "
            f"{float(row['underside_depth_mae_percent_mean']):.4f} & "
            f"{float(row['top_view_depth_mae_percent_mean']):.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\label{tab:heldout_results}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    benchmark_name, shoes = load_benchmark(args.benchmark.expanduser().resolve())
    rows = collect_rows(input_root, shoes, args.allow_incomplete)
    summaries = summarize(rows)
    _write_csv(output / "per_shoe.csv", rows)
    _write_csv(output / "method_summary.csv", summaries)
    write_latex(output / "tables.tex", summaries)
    (output / "aggregation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": benchmark_name,
                "input_root": str(input_root),
                "shoe_count": len(shoes),
                "methods": [row["method"] for row in summaries],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Aggregated {len(rows)} per-shoe evaluations into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
