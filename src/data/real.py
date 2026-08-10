"""Convert a real chart-to-table corpus into the project schema.

Source: `ahmed-masry/unichart-table-data` — real charts (largely Statista/Pew
style) paired with their underlying data as a markdown pipe table:

    |Characteristic | Brunei Darussalam | Belarus | Venezuela |
    |---|---|---|---|
    | 1970 | 2.83 |  -  | 3.79 |

What this corpus does *not* carry, and what we therefore refuse to invent:

* **chart_type** — nowhere in the annotation. Guessing "bar" because most
  Statista charts are bars would put fabricated labels into ground truth.
* **title** — not annotated.
* **y_label** — not annotated; the units live in the chart image only.
* **x_label** — the first column header, but it is usually the literal
  placeholder "Characteristic", which is not what the axis says. It is kept only
  when it is something else.

Each converted sample records `scored_fields`, and `metrics.score_sample` honours
it: real charts are scored on values, series names, and structure, and are
excluded from the denominator for the fields they cannot support. That keeps the
real subset useful for the metric that matters most (value accuracy) without
inventing ground truth for the rest.

`chart_type` still has to be *something* to satisfy the schema, so it is set to
"bar" and excluded from scoring. That placeholder never reaches a metric.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

from PIL import Image

from src.eval.schema import ChartData, ChartType, Point, Series

REPO_ID = "ahmed-masry/unichart-table-data"

# Cells the corpus uses for "no value here".
NULL_CELLS = {"-", "--", "", "n/a", "na", "null", "none", "*"}

PLACEHOLDER_X_LABELS = {"characteristic", "category", "unnamed: 0", ""}

_NUM = re.compile(r"-?[\d,]*\.?\d+")


def parse_number(cell: str) -> float | None:
    """Pull a number out of a table cell, tolerating %, currency and commas."""
    text = cell.strip()
    if text.casefold() in NULL_CELLS:
        return None
    match = _NUM.search(text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def parse_pipe_table(table: str) -> tuple[list[str], list[str], list[list[float | None]]] | None:
    """Parse the markdown pipe table into (x_header, series_names, columns).

    Returns None when the table is malformed or degenerate, so bad rows are
    dropped rather than silently producing empty gold.
    """
    lines = [ln.strip() for ln in table.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    if len(header) < 2:
        return None

    # Drop the |---|---| separator row if present.
    body = lines[1:]
    if body and set(body[0].replace("|", "").replace("-", "").strip()) == set():
        body = body[1:]
    if not body:
        return None

    x_header, series_names = header[0], header[1:]
    xs: list[str] = []
    columns: list[list[float | None]] = [[] for _ in series_names]

    for line in body:
        row = cells(line)
        if len(row) != len(header):
            continue  # ragged row; skip rather than guess alignment
        xs.append(row[0])
        for i in range(len(series_names)):
            columns[i].append(parse_number(row[i + 1]))

    if not xs:
        return None
    return x_header, series_names, columns


def to_chart_data(table: str) -> tuple[ChartData, set[str]] | None:
    """Convert one pipe table to ChartData plus the set of scorable fields."""
    parsed = parse_pipe_table(table)
    if parsed is None:
        return None
    x_header, series_names, columns = parsed

    series: list[Series] = []
    for name, column in zip(series_names, columns):
        # A missing cell means the point is not plotted, so it is omitted rather
        # than zero-filled -- zero-filling would invent data points.
        points = [
            Point(x=xs, y=y)
            for xs, y in zip([str(x) for x in range(len(column))], column)
            if y is not None
        ]
        if points:
            series.append(Series(name=name.strip(), data=points))

    if not series:
        return None

    x_label = None if x_header.strip().casefold() in PLACEHOLDER_X_LABELS else x_header.strip()

    scored = {"series_names"}
    if x_label is not None:
        scored.add("x_label")

    return (
        ChartData(
            # Placeholder: required by the schema, excluded from scoring.
            chart_type=ChartType.BAR,
            title=None,
            x_label=x_label,
            y_label=None,
            series=series,
        ),
        scored,
    )


def rebuild_x_values(table: str, data: ChartData) -> ChartData:
    """Replace positional x placeholders with the real first-column labels."""
    parsed = parse_pipe_table(table)
    if parsed is None:
        return data
    _, series_names, columns = parsed

    lines = [ln.strip() for ln in table.strip().splitlines() if ln.strip()]
    body = lines[1:]
    if body and set(body[0].replace("|", "").replace("-", "").strip()) == set():
        body = body[1:]
    header_len = len([c for c in lines[0].strip().strip("|").split("|")])
    xs = [
        [c.strip() for c in ln.strip().strip("|").split("|")][0]
        for ln in body
        if len([c for c in ln.strip().strip("|").split("|")]) == header_len
    ]

    by_name = {name.strip(): col for name, col in zip(series_names, columns)}
    for s in data.series:
        column = by_name.get(s.name)
        if column is None:
            continue
        s.data = [
            Point(x=xs[i], y=y) for i, y in enumerate(column) if y is not None and i < len(xs)
        ]
    return data


def load_excluded_ids(manifests: list[Path]) -> set[str]:
    """Sample ids already used by another split.

    The eval subset and the training subset are drawn from the same corpus, so
    without this they overlap by construction — `contamination_report` would
    catch it, but only after paying for the whole download and conversion.
    Excluding up front is both cheaper and the correct default.
    """
    out: set[str] = set()
    for m in manifests:
        p = Path(m)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                out.add(json.loads(line)["id"])
    return out


def build(
    out_dir: Path,
    n: int,
    shard: str | None,
    seed: int,
    max_series: int,
    max_points: int,
    exclude_ids: set[str] | None = None,
    manifest_name: str = "real.jsonl",
) -> Path:
    """Select and convert `n` real charts.

    Two passes. The first reads only the text columns (cheap) to find every
    eligible row; the second re-reads just the selected rows to decode images.

    Random selection is not a detail. Rows in this corpus arrive in blocks from
    the same source chart family, so taking the first N yields a bimodal series
    distribution — in one run, 73 single-series charts and 127 five-series ones
    with nothing in between. That is a property of the file order, not of real
    charts, and it would quietly bias every number computed on the subset.
    """
    import random

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    if shard is None:
        info = HfApi().dataset_info(REPO_ID, files_metadata=True)
        shards = sorted(
            (f.size or 0, f.rfilename)
            for f in info.siblings
            if f.rfilename.endswith(".parquet")
        )
        shard = shards[0][1]

    print(f"using shard {shard}")
    path = hf_hub_download(REPO_ID, shard, repo_type="dataset")
    pf = pq.ParquetFile(path)

    # ---- pass 1: find eligible rows without touching image bytes ---------- #
    eligible: list[int] = []
    skipped = {"unparseable": 0, "too_many_series": 0, "too_dense": 0, "empty": 0}
    index = 0

    exclude_ids = exclude_ids or set()
    skipped["excluded"] = 0

    for batch in pf.iter_batches(batch_size=1024, columns=["table", "sample_id"]):
        for row in batch.to_pylist():
            converted = to_chart_data(row["table"] or "")
            if f"real_{row['sample_id']}" in exclude_ids:
                skipped["excluded"] += 1
            elif converted is None:
                skipped["unparseable"] += 1
            else:
                gold, _ = converted
                if gold.n_series > max_series:
                    skipped["too_many_series"] += 1
                elif gold.n_points > max_points:
                    # Filtered, never truncated: dropping points from a chart
                    # would corrupt the ground truth it is meant to provide.
                    skipped["too_dense"] += 1
                elif gold.n_points == 0:
                    skipped["empty"] += 1
                else:
                    eligible.append(index)
            index += 1

    print(f"scanned {index} rows, {len(eligible)} eligible, skipped {skipped}")
    if not eligible:
        raise SystemExit("no eligible rows -- loosen --max-series / --max-points")

    rng = random.Random(seed)
    chosen = set(rng.sample(eligible, min(n, len(eligible))))

    # ---- pass 2: materialise only the chosen rows ------------------------- #
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / manifest_name

    written = 0
    index = 0
    with manifest.open("w", encoding="utf-8") as fh:
        for batch in pf.iter_batches(batch_size=256, columns=["image", "table", "sample_id"]):
            for row in batch.to_pylist():
                if index in chosen:
                    converted = to_chart_data(row["table"] or "")
                    if converted is not None:
                        gold, scored = converted
                        gold = rebuild_x_values(row["table"], gold)
                        sample_id = f"real_{row['sample_id']}"
                        img_path = images / f"{sample_id}.png"
                        try:
                            Image.open(io.BytesIO(row["image"])).convert("RGB").save(img_path)
                            fh.write(
                                json.dumps(
                                    {
                                        "id": sample_id,
                                        "image": str(img_path.relative_to(out_dir)).replace(
                                            "\\", "/"
                                        ),
                                        "gold": gold.model_dump(mode="json"),
                                        "scored_fields": sorted(scored),
                                        "source": REPO_ID,
                                        "properties": {
                                            "chart_type": None,
                                            "n_series": gold.n_series,
                                            "n_points": gold.n_points,
                                            "source": "real",
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            written += 1
                        except Exception:  # noqa: BLE001 - a corrupt image is a skip
                            pass
                index += 1

    print(f"wrote {written} real charts -> {manifest}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-series", type=int, default=6)
    ap.add_argument(
        "--max-points",
        type=int,
        default=24,
        help="skip charts denser than this; they are legitimate but their JSON "
        "runs past a practical generation budget",
    )
    ap.add_argument(
        "--exclude-manifest", action="append", default=[], type=Path,
        help="repeatable; skip sample ids already present in these manifests, so "
        "train and eval subsets drawn from the same corpus cannot overlap",
    )
    ap.add_argument("--manifest-name", default="real.jsonl")
    args = ap.parse_args()

    excluded = load_excluded_ids(args.exclude_manifest)
    if excluded:
        print(f"excluding {len(excluded)} ids from {len(args.exclude_manifest)} manifest(s)")

    manifest = build(
        args.out, args.n, args.shard, args.seed, args.max_series, args.max_points,
        exclude_ids=excluded, manifest_name=args.manifest_name,
    )

    rows = [json.loads(x) for x in manifest.open(encoding="utf-8")]
    from collections import Counter

    series_dist = Counter(r["properties"]["n_series"] for r in rows)
    print("\nseries-count distribution:")
    for k in sorted(series_dist):
        print(f"  {k} series: {series_dist[k]}")
    print(f"\nmean points/chart: {sum(r['properties']['n_points'] for r in rows) / len(rows):.1f}")
    print(f"scored fields: {sorted({f for r in rows for f in r['scored_fields']})}")


if __name__ == "__main__":
    main()
