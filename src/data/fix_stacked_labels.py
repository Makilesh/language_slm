"""Relabel single-series `stacked_bar` charts as `bar`.

`synth.render` draws a stacked bar as one `ax.bar(..., bottom=cumulative)` per
series. With a single series that is `bottom=0` — pixel-identical to the plain
`bar` branch. So a single-series chart labelled `stacked_bar` is ground truth no
model can satisfy: the image genuinely is a bar chart.

Left in place it does two things, both bad:

* **Training**: 502 of 10,000 samples taught the model that the bar /
  stacked_bar distinction is a coin flip, against 515 visually identical
  samples labelled `bar`.
* **Evaluation**: 8 of 150 eval charts were unlearnable, so chart-type accuracy
  was capped at 94.7% for a model that is otherwise perfect. Phase 1's reported
  88.7% was measured against that ceiling.

The fix relabels rather than regenerates, because the images are correct — only
the label was wrong. Regenerating would invalidate the committed predictions and
cost a full baseline re-run for an identical result.

`synth.sample_spec` now forces `n_series >= 2` for stacked bars, so new data
does not need this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def needs_fix(gold: dict) -> bool:
    return gold.get("chart_type") == "stacked_bar" and len(gold.get("series", [])) < 2


def fix_manifest(path: Path, dry_run: bool = False) -> int:
    rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    fixed = 0

    for row in rows:
        gold = row.get("gold")
        if gold and needs_fix(gold):
            gold["chart_type"] = "bar"
            fixed += 1
            # `properties` drives the per-chart-type breakdown in reports, and
            # `spec` is the generator record; both must agree with the gold or
            # the results tables disagree with the labels they describe.
            if isinstance(row.get("properties"), dict):
                row["properties"]["chart_type"] = "bar"
            if isinstance(row.get("spec"), dict):
                row["spec"]["chart_type"] = "bar"

    if fixed and not dry_run:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    verb = "would relabel" if dry_run else "relabelled"
    print(f"  {path}: {verb} {fixed}/{len(rows)}")
    return fixed


def fix_dataset(path: Path, dry_run: bool = False) -> int:
    """Training samples store the target as rendered JSON text, not a gold dict."""
    from src.data.format import target_json
    from src.eval.schema import ChartData, parse_prediction

    rows = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    fixed = 0

    for row in rows:
        msg = row["messages"][-1]
        parsed, err, _ = parse_prediction(msg["content"][0]["text"])
        if err or parsed is None:
            continue
        gold = parsed.model_dump(mode="json")
        if needs_fix(gold):
            gold["chart_type"] = "bar"
            msg["content"][0]["text"] = target_json(ChartData.model_validate(gold))
            if isinstance(row.get("properties"), dict):
                row["properties"]["chart_type"] = "bar"
            fixed += 1

    if fixed and not dry_run:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    verb = "would relabel" if dry_run else "relabelled"
    print(f"  {path}: {verb} {fixed}/{len(rows)}")
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, action="append", default=[])
    ap.add_argument("--dataset", type=Path, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    if args.manifest:
        print("manifests:")
        for p in args.manifest:
            if p.exists():
                total += fix_manifest(p, args.dry_run)
    if args.dataset:
        print("training datasets:")
        for p in args.dataset:
            if p.exists():
                total += fix_dataset(p, args.dry_run)

    print(f"\ntotal {'would be ' if args.dry_run else ''}relabelled: {total}")


if __name__ == "__main__":
    main()
