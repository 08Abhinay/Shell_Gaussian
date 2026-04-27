#!/usr/bin/env python3
"""Aggregate NeuS2 eval_log.txt files into a TSV metrics summary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


AVERAGE_RE = re.compile(
    r"AVERAGE_TEST\s+"
    r"PSNR=(?P<psnr>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"SSIM=(?P<ssim>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s+LPIPS=(?P<lpips>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?"
    r"\s+views=(?P<views>\d+)"
)

VIEW_RE = re.compile(
    r"camera_view:.*?"
    r"PSNR=(?P<psnr>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"SSIM=(?P<ssim>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"(?:\s+LPIPS=(?P<lpips>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?))?"
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def parse_eval_log(log_path: Path) -> tuple[float, float, float | None, int] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    average_matches = list(AVERAGE_RE.finditer(text))
    if average_matches:
        match = average_matches[-1]
        lpips = match["lpips"]
        return (
            float(match["psnr"]),
            float(match["ssim"]),
            float(lpips) if lpips is not None else None,
            int(match["views"]),
        )

    view_metrics = []
    for match in VIEW_RE.finditer(text):
        lpips = match["lpips"]
        view_metrics.append(
            (
                float(match["psnr"]),
                float(match["ssim"]),
                float(lpips) if lpips is not None else None,
            )
        )
    if not view_metrics:
        return None

    lpips_values = [metric[2] for metric in view_metrics if metric[2] is not None]
    return (
        mean([metric[0] for metric in view_metrics]),
        mean([metric[1] for metric in view_metrics]),
        mean(lpips_values) if len(lpips_values) == len(view_metrics) else None,
        len(view_metrics),
    )


def resolve_shoes(output_dir: Path, shoes_file: Path | None) -> list[str]:
    if shoes_file is None:
        return sorted(
            path.name.rsplit("_neus2_", 1)[0]
            for path in output_dir.glob("*_neus2_*")
            if path.is_dir() and (path / "eval_log.txt").is_file()
        )

    shoes = []
    for line in shoes_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            shoes.append(line)
    return shoes


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "output",
        help="Directory containing *_neus2_* output folders.",
    )
    parser.add_argument(
        "--shoes-file",
        type=Path,
        default=default_root / "bash_scripts" / "shoes.txt",
        help="Optional ordered shoe list. Use an empty string to scan output-dir.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to write summary TSV. Defaults to stdout only.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    shoes_file = args.shoes_file.resolve() if str(args.shoes_file) else None
    shoes = resolve_shoes(output_dir, shoes_file)

    rows = []
    missing = []
    for shoe in shoes:
        log_path = output_dir / f"{shoe}_neus2_10000" / "eval_log.txt"
        metrics = parse_eval_log(log_path) if log_path.is_file() else None
        if metrics is None:
            missing.append(shoe)
            continue
        rows.append((shoe, *metrics))

    if not rows:
        print("No eval metrics found.")
        return 1

    total_views = sum(row[4] for row in rows)
    avg_psnr = mean([row[1] for row in rows])
    avg_ssim = mean([row[2] for row in rows])
    lpips_values = [row[3] for row in rows if row[3] is not None]
    avg_lpips = mean(lpips_values) if len(lpips_values) == len(rows) else None

    lines = ["shoe\tPSNR\tSSIM\tLPIPS\tviews"]
    for shoe, psnr, ssim, lpips, views in rows:
        lpips_text = f"{lpips:.12f}" if lpips is not None else ""
        lines.append(f"{shoe}\t{psnr:.12f}\t{ssim:.12f}\t{lpips_text}\t{views}")

    avg_lpips_text = f"{avg_lpips:.12f}" if avg_lpips is not None else ""
    lines.append(f"AVERAGE\t{avg_psnr:.12f}\t{avg_ssim:.12f}\t{avg_lpips_text}\t{total_views}")
    summary_text = "\n".join(lines) + "\n"

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary_text, encoding="utf-8")

    print(summary_text, end="")
    if missing:
        print("Missing eval metrics for:")
        for shoe in missing:
            print(f"- {shoe}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
