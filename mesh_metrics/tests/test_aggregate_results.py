from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mesh_metrics.aggregate_results import (
    GEOMETRY_KEYS,
    VIEW_KEYS,
    collect_rows,
    summarize,
    write_latex,
)


class AggregateResultsTest(unittest.TestCase):
    def test_collect_and_summarize_partial_development_run(self) -> None:
        shoes = [
            {"name": "shoe_a", "category": "sneaker"},
            {"name": "shoe_b", "category": "sandal"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "method_a" / "shoe_a"
            result.mkdir(parents=True)
            geometry = {
                "headline": {
                    key: float(index + 1)
                    for index, key in enumerate(GEOMETRY_KEYS)
                }
            }
            view = {
                "headline": {key: float(index + 10) for index, key in enumerate(VIEW_KEYS)},
                "heldout_eligible": True,
            }
            (result / "geometry_metrics.json").write_text(json.dumps(geometry))
            (result / "view_metrics.json").write_text(json.dumps(view))

            rows = collect_rows(root, shoes, allow_incomplete=True)
            self.assertEqual(len(rows), 1)
            summaries = summarize(rows)
            self.assertEqual(summaries[0]["shoe_count"], 1)
            self.assertTrue(summaries[0]["heldout_eligible"])
            latex = root / "tables.tex"
            write_latex(latex, summaries)
            self.assertIn("method\\_a", latex.read_text())

    def test_complete_mode_rejects_missing_shoes(self) -> None:
        shoes = [{"name": "missing", "category": "sneaker"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "method_a").mkdir()
            with self.assertRaisesRegex(ValueError, "missing 1 benchmark shoes"):
                collect_rows(root, shoes, allow_incomplete=False)

    def test_turntable_results_omit_elevation_only_columns(self) -> None:
        shoes = [{"name": "shoe_a", "category": "sneaker"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "method_a" / "shoe_a"
            result.mkdir(parents=True)
            (result / "geometry_metrics.json").write_text(
                json.dumps(
                    {
                        "headline": {
                            key: float(index + 1)
                            for index, key in enumerate(GEOMETRY_KEYS)
                        }
                    }
                )
            )
            common_view = {
                key: float(index + 1)
                for index, key in enumerate(VIEW_KEYS[:4])
            }
            (result / "view_metrics.json").write_text(
                json.dumps(
                    {
                        "evaluation_protocol": "turntable",
                        "headline": common_view,
                        "heldout_eligible": True,
                    }
                )
            )

            rows = collect_rows(root, shoes, allow_incomplete=False)
            summaries = summarize(rows)
            self.assertEqual(summaries[0]["evaluation_protocol"], "turntable")
            self.assertEqual(summaries[0]["underside_depth_mae_percent_mean"], "")
            latex = root / "tables.tex"
            write_latex(latex, summaries)
            text = latex.read_text()
            self.assertIn("Depth Coverage", text)
            self.assertNotIn("Underside MAE", text)


if __name__ == "__main__":
    unittest.main()
