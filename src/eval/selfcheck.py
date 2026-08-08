"""Phase 1 acceptance: score a real dataset's ground truth against itself.

`tests/test_metrics.py` proves this on hand-written fixtures. This proves it on
the actual generated corpus, which is a stronger claim: it exercises every chart
type, every value magnitude, log axes, numeric x values, and the pie special
case, against the same code path the baselines will use.

Anything below 100% is a harness bug, not a model result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.metrics import aggregate, score_sample
from src.eval.schema import ChartData


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def run(manifest: Path, roundtrip_json: bool = True) -> tuple[dict, list[str]]:
    """Score gold against itself. Returns (aggregate, failing sample ids)."""
    rows = load_manifest(manifest)
    samples = []

    for row in rows:
        gold = ChartData.model_validate(row["gold"])
        # Round-trip through JSON text so serialisation bugs surface here rather
        # than during a baseline run.
        pred = ChartData.from_json(gold.to_json()) if roundtrip_json else gold.model_copy(deep=True)
        samples.append(score_sample(gold, pred, row["id"]))

    failures = [
        s.sample_id
        for s in samples
        if not (
            s.valid_json
            and s.exact_structural_match
            and s.value_accuracy.get(0.01, 0.0) == 1.0
            and s.n_missing_points == 0
            and s.n_spurious_points == 0
            and not s.scale_errors
            and not s.used_positional_fallback
        )
    ]
    return aggregate(samples), failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    agg, failures = run(args.manifest)
    o = agg["overall"]

    print(f"self-consistency check over {o['n']} charts\n")
    for key in (
        "valid_json_rate", "schema_conformance_rate", "chart_type_accuracy", "title_accuracy",
        "x_label_accuracy", "y_label_accuracy", "series_name_accuracy",
        "series_count_accuracy", "point_count_accuracy",
        "exact_structural_match", "point_recall",
    ):
        print(f"  {key:26s} {o[key]:.4f}")
    for tol, val in o["value_accuracy"].items():
        print(f"  value_accuracy @{tol:<4s}       {val:.4f}")
    print(f"  mape                       {o['mape']:.6f}")
    print(f"  missing / spurious points  {o['n_missing_points']} / {o['n_spurious_points']}")
    print(f"  positional fallbacks       {o['n_positional_fallback']}")
    print(f"  scale-error flags          {o['n_scale_errors']}")

    print("\nby chart type:")
    for ct, stats in agg["by_chart_type"].items():
        print(f"  {ct:14s} n={stats['n']:4d}  value@1%={stats['value_accuracy']['1%']:.4f}"
              f"  structural={stats['exact_structural_match']:.4f}")

    if failures:
        print(f"\nFAILED on {len(failures)} samples: {failures[:20]}")
        raise SystemExit(1)
    print("\nPASS: ground truth scores 100% against itself")


if __name__ == "__main__":
    main()
