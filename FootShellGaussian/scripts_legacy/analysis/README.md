# Analysis and superseded drivers

Nothing here is used by the pipeline. These are kept because they produced
numbers that appear in the reports, and because the comparisons they run are
worth repeating if the pipeline changes.

| file | what it was for | replaced by |
|---|---|---|
| `run_to_11d.py` | the first end-to-end orchestrator | `anatomical_coordinates/pipeline/run_pipeline.py` |
| `run_torch_foot_fit.py` | per-shoe foot fitting driver and experiment harness | `anatomical_coordinates/pipeline/fit_stage.py` |
| `compare_runs.py` | side-by-side exact-evaluator comparison of two runs | — |
| `benchmark_runtime.py` | fitting runtime against the search fitter | — |
| `evaluate_baseline.py` | scores the previous fitter with the same evaluator | — |
| `summarize_results.py` | tables from a run's `results.json` | — |

They import from `anatomical_coordinates`, so run them from the repository root.
