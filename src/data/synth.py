"""Synthetic chart generation with exact ground truth.

Serves two purposes with one generator: Phase 1's hard-case evaluation set, and
Phase 2's training corpus. The only difference is volume and the `hard_rate`
knob.

Why synthetic charts are not padding. Real chart-to-table datasets give you
messiness but noisy labels, and — more importantly — they give you no control
over *which* property is hard. If you want to know whether the model fails on
log axes specifically, you need charts that differ only in having a log axis.
That is only obtainable by construction.

Two decisions worth stating because they are judgement calls, not facts:

* **Values are rounded to what a human could actually read off the chart**
  (2-4 significant figures depending on magnitude). Generating y=121.4837 and
  scoring a model against that measures rendering precision, not chart reading.
  The gold is the rounded value, and the bar is drawn at the rounded value, so
  they agree exactly.

* **Stacked bars record segment values, not cumulative heights.** A stacked bar
  at 30/50/20 is stored as 30, 50, 20 — not 30, 80, 100. This is the reading a
  human would give and the one that reconstructs the chart, but it is genuinely
  ambiguous, so it is stated here and in the model card.

* **Dual-axis charts keep only the primary y label**, because the schema has one
  `y_label` field. This is a real schema limitation; dual-axis charts are
  included anyway as a known-hard case, and the limitation is reported rather
  than designed around.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval.schema import ChartData, ChartType, Point, Series

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #

TITLES = [
    "Quarterly Revenue by Region", "Monthly Active Users", "Energy Mix by Source",
    "Customer Satisfaction Scores", "Unit Sales by Product Line", "Website Traffic by Channel",
    "Annual Rainfall by Station", "Market Share by Vendor", "Defect Rate by Assembly Line",
    "Population Growth by District", "Hospital Admissions by Ward", "Crop Yield by Variety",
    "Server Latency by Datacenter", "Household Spending by Category", "Export Volume by Country",
    "Patent Filings by Sector", "Battery Capacity by Cell Type", "Delivery Times by Carrier",
]

X_LABELS = ["Quarter", "Month", "Year", "Category", "Region", "Product", "Channel",
            "Station", "Vendor", "Department", "Cohort", "Segment", "Site", "Batch"]

Y_LABELS = [
    "Revenue (thousands USD)", "Users", "Percentage (%)", "Score", "Units Sold",
    "Sessions", "Rainfall (mm)", "Market Share (%)", "Defects per 1000", "Population",
    "Admissions", "Yield (tonnes/ha)", "Latency (ms)", "Spending (EUR)", "Volume (kt)",
]

SERIES_NAMES = [
    ["North America", "EMEA", "APAC", "LATAM", "MEA"],
    ["Product A", "Product B", "Product C", "Product D"],
    ["Solar", "Wind", "Hydro", "Nuclear", "Coal", "Gas"],
    ["Baseline", "Treatment", "Control"],
    ["2021", "2022", "2023", "2024"],
    ["Mobile", "Desktop", "Tablet"],
    ["Organic", "Paid", "Referral", "Direct", "Email"],
    ["Team Alpha", "Team Beta", "Team Gamma"],
]

CATEGORY_SETS = [
    ["Q1", "Q2", "Q3", "Q4"],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["2018", "2019", "2020", "2021", "2022", "2023", "2024"],
    ["North", "South", "East", "West", "Central"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Very Low", "Low", "Medium", "High", "Very High"],
    [f"Site {i}" for i in range(1, 13)],
    [f"Batch {i:02d}" for i in range(1, 15)],
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
]

# Single series in a pie chart has no natural name -- the slices carry the
# labels. A fixed, predictable name beats an arbitrary one the model could
# never infer from the image.
PIE_SERIES_NAME = "Share"

# Palettes range from matplotlib defaults to deliberately awkward ones, because
# a model trained only on tab10 learns colour identity rather than shape.
PALETTES = [
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#7d5ba6"],
    ["#000000", "#444444", "#777777", "#aaaaaa", "#cccccc", "#eeeeee"],  # greyscale
    ["#d0e1f9", "#4d648d", "#283655", "#1e1f26", "#96c9dc", "#5a7d9a"],  # low contrast
    ["#ff006e", "#fb5607", "#ffbe0b", "#8338ec", "#3a86ff", "#06d6a0"],  # garish
    ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#606c38"],
]

FONT_FAMILIES = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #


@dataclass
class ChartSpec:
    """Every randomised property of one chart. Fully determines the render."""

    chart_type: str
    n_series: int
    n_points: int
    seed: int

    title: str | None
    x_label: str | None
    y_label: str | None
    series_names: list[str]
    categories: list[str]
    values: list[list[float]]

    palette: list[str]
    font_family: str
    fig_w: float
    fig_h: float
    dpi: int
    grid: bool
    legend_loc: str | None
    label_rotation: int
    log_y: bool
    dual_axis: bool
    numeric_x: bool

    hard_flags: list[str] = field(default_factory=list)

    def properties(self) -> dict:
        """Measurable properties, for correlating failure rate against them."""
        return {
            "chart_type": self.chart_type,
            "n_series": self.n_series,
            "n_points": self.n_points,
            "log_y": self.log_y,
            "dual_axis": self.dual_axis,
            "label_rotation": self.label_rotation,
            "grid": self.grid,
            "legend": self.legend_loc is not None,
            "dpi": self.dpi,
            "has_title": self.title is not None,
            "numeric_x": self.numeric_x,
            "value_magnitude": _magnitude(self.values),
            "hard_flags": self.hard_flags,
        }


def _magnitude(values: list[list[float]]) -> int:
    flat = [abs(v) for row in values for v in row if abs(v) > 0]
    return int(math.floor(math.log10(max(flat)))) if flat else 0


def _readable(value: float) -> float:
    """Round to what a human could plausibly read off an axis."""
    if value == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(value)))
    if magnitude >= 3:
        return float(round(value, -(magnitude - 2)))   # 12,340 not 12,337.2
    if magnitude >= 1:
        return float(round(value, 1))                   # 42.7
    return float(round(value, 3))                       # 0.084


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #


def sample_spec(rng: random.Random, hard_rate: float = 0.35) -> ChartSpec:
    """Draw one randomised chart specification.

    `hard_rate` is the probability of *each* hard property firing independently,
    so a chart can be hard in several ways at once — which is what breaks models
    in practice.
    """
    hard: list[str] = []

    def is_hard(flag: str) -> bool:
        if rng.random() < hard_rate:
            hard.append(flag)
            return True
        return False

    chart_type = rng.choice(
        ["bar", "bar", "line", "line", "pie", "scatter", "stacked_bar", "stacked_bar"]
    )

    if chart_type == "pie":
        n_series = 1
        n_points = rng.randint(3, 7)
    elif chart_type == "scatter":
        n_series = rng.randint(1, 3)
        n_points = rng.randint(8, 25) if is_hard("dense_scatter") else rng.randint(4, 8)
    else:
        n_series = rng.randint(1, 5) if is_hard("many_series") else rng.randint(1, 3)
        n_points = rng.randint(3, 12)

    numeric_x = chart_type == "scatter"

    # Clamp to the pool rather than padding it. Synthesising extra labels ("2018'")
    # produces axis text that no real chart would carry, and the model would be
    # learning an artefact of the generator.
    categories_pool = rng.choice(CATEGORY_SETS)
    if not numeric_x:
        # Scatter uses numeric x, so it is not bound by the category pool and
        # keeps its dense-point hard case.
        n_points = min(n_points, len(categories_pool))
    categories = categories_pool[:n_points]

    names_pool = rng.choice(SERIES_NAMES)
    n_series = min(n_series, len(names_pool))

    # A stacked bar with one series renders as a plain bar chart -- the render
    # branch draws a single ax.bar() with bottom=0, pixel-identical to the
    # "bar" branch. Labelling that image "stacked_bar" is unlearnable ground
    # truth: it teaches the model that the distinction is random, and caps
    # chart-type accuracy on that population at chance. Stacking needs
    # something to stack.
    if chart_type == "stacked_bar":
        n_series = max(2, n_series)

    series_names = (
        [PIE_SERIES_NAME] if chart_type == "pie" else names_pool[:n_series]
    )

    log_y = chart_type in ("bar", "line", "scatter") and is_hard("log_axis")
    dual_axis = chart_type in ("bar", "line") and n_series == 2 and is_hard("dual_axis")

    # Magnitude is randomised aggressively: a model that has only seen values in
    # [0, 100] learns the axis range, not the reading.
    exponent = rng.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6])
    base = 10.0**exponent

    values: list[list[float]] = []
    for s in range(n_series):
        if log_y:
            row = [_readable(base * (10 ** rng.uniform(0, 3))) for _ in range(n_points)]
        elif chart_type in ("line", "scatter"):
            level = base * rng.uniform(0.5, 5)
            trend = rng.uniform(-0.15, 0.25)
            row = [
                _readable(max(0.0, level * (1 + trend) ** i * rng.uniform(0.85, 1.15)))
                for i in range(n_points)
            ]
        else:
            level = base * rng.uniform(0.5, 5) * (1 + 0.3 * s)
            row = [_readable(max(0.0, level * rng.uniform(0.4, 1.6))) for _ in range(n_points)]
        values.append(row)

    if chart_type == "pie":  # pie slices must be positive to render
        values = [[max(v, base * 0.05) for v in values[0]]]

    return ChartSpec(
        chart_type=chart_type,
        n_series=n_series,
        n_points=n_points,
        seed=rng.randint(0, 2**31 - 1),
        title=None if is_hard("no_title") else rng.choice(TITLES),
        x_label=None if chart_type == "pie" else rng.choice(X_LABELS),
        y_label=None if chart_type == "pie" else rng.choice(Y_LABELS),
        series_names=series_names,
        categories=categories,
        values=values,
        palette=rng.choice(PALETTES[2:]) if is_hard("odd_palette") else rng.choice(PALETTES[:2]),
        font_family=rng.choice(FONT_FAMILIES),
        fig_w=rng.uniform(5.0, 11.0),
        fig_h=rng.uniform(3.5, 7.0),
        dpi=rng.choice([72, 96, 100, 130, 160]),
        grid=not is_hard("no_gridlines"),
        legend_loc=(
            None
            if n_series == 1 and chart_type != "pie"
            else rng.choice(["best", "upper left", "upper right", "lower left", "center right"])
        ),
        label_rotation=rng.choice([45, 60, 90]) if is_hard("rotated_labels") else 0,
        log_y=log_y,
        dual_axis=dual_axis,
        numeric_x=numeric_x,
        hard_flags=hard,
    )


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #


def spec_to_chart_data(spec: ChartSpec) -> ChartData:
    """Exact ground truth, by construction — these are the numbers we drew."""
    x_values: list[str | float] = (
        [float(i + 1) for i in range(spec.n_points)] if spec.numeric_x else list(spec.categories)
    )
    return ChartData(
        chart_type=ChartType(spec.chart_type),
        title=spec.title,
        x_label=spec.x_label,
        y_label=spec.y_label,
        series=[
            Series(
                name=spec.series_names[s],
                data=[Point(x=x_values[i], y=spec.values[s][i]) for i in range(spec.n_points)],
            )
            for s in range(spec.n_series)
        ],
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render(spec: ChartSpec, out_path: Path) -> Path:
    """Render exactly the values recorded in the spec."""
    plt.rcParams["font.family"] = spec.font_family
    fig, ax = plt.subplots(figsize=(spec.fig_w, spec.fig_h), dpi=spec.dpi)
    colors = spec.palette
    x = np.arange(spec.n_points)

    if spec.chart_type == "pie":
        ax.pie(spec.values[0], labels=spec.categories,
               colors=colors[: spec.n_points], autopct="%1.1f%%")
        ax.axis("equal")

    elif spec.chart_type == "stacked_bar":
        bottom = np.zeros(spec.n_points)
        for s in range(spec.n_series):
            row = np.array(spec.values[s])
            ax.bar(x, row, 0.6, bottom=bottom,
                   label=spec.series_names[s], color=colors[s % len(colors)])
            bottom += row

    elif spec.chart_type == "bar":
        width = 0.8 / spec.n_series
        for s in range(spec.n_series):
            offset = (s - (spec.n_series - 1) / 2) * width
            ax.bar(x + offset, spec.values[s], width,
                   label=spec.series_names[s], color=colors[s % len(colors)])

    elif spec.chart_type == "line":
        axes = [ax]
        if spec.dual_axis:
            axes.append(ax.twinx())
        for s in range(spec.n_series):
            target = axes[min(s, len(axes) - 1)]
            target.plot(x, spec.values[s], marker="o",
                        label=spec.series_names[s], color=colors[s % len(colors)])

    elif spec.chart_type == "scatter":
        for s in range(spec.n_series):
            ax.scatter(x + 1, spec.values[s], s=38,
                       label=spec.series_names[s], color=colors[s % len(colors)])

    if spec.chart_type != "pie":
        if spec.log_y:
            ax.set_yscale("log")
        if spec.title:
            ax.set_title(spec.title)
        if spec.x_label:
            ax.set_xlabel(spec.x_label)
        if spec.y_label:
            ax.set_ylabel(spec.y_label)
        if not spec.numeric_x:
            ax.set_xticks(x)
            ax.set_xticklabels(
                spec.categories,
                rotation=spec.label_rotation,
                ha="right" if spec.label_rotation else "center",
            )
        if spec.grid:
            ax.grid(axis="y", alpha=0.3, linestyle="--")
        if spec.legend_loc and spec.n_series > 1:
            ax.legend(loc=spec.legend_loc, fontsize="small")
    elif spec.title:
        ax.set_title(spec.title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# dataset generation
# --------------------------------------------------------------------------- #


def generate(
    out_dir: Path,
    n: int,
    seed: int = 0,
    hard_rate: float = 0.35,
    prefix: str = "synth",
) -> Path:
    """Render `n` charts and write a JSONL manifest beside them.

    Each manifest line carries the image path, the gold ChartData, and the
    chart's measurable properties, so Phase 4's "what makes this model fail"
    table is a groupby rather than a re-derivation.
    """
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"{prefix}.jsonl"

    rng = random.Random(seed)
    with manifest.open("w", encoding="utf-8") as fh:
        for i in range(n):
            spec = sample_spec(rng, hard_rate=hard_rate)
            sample_id = f"{prefix}_{i:05d}"
            path = render(spec, images / f"{sample_id}.png")
            fh.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "image": str(path.relative_to(out_dir)).replace("\\", "/"),
                        "gold": spec_to_chart_data(spec).model_dump(mode="json"),
                        "properties": spec.properties(),
                        "spec": asdict(spec),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return manifest


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-rate", type=float, default=0.35)
    ap.add_argument("--prefix", default="synth")
    args = ap.parse_args()

    manifest = generate(args.out, args.n, args.seed, args.hard_rate, args.prefix)

    from collections import Counter

    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    types = Counter(r["properties"]["chart_type"] for r in rows)
    flags = Counter(f for r in rows for f in r["properties"]["hard_flags"])

    print(f"wrote {len(rows)} charts -> {manifest}")
    print("\nchart types:")
    for k, v in types.most_common():
        print(f"  {k:14s} {v:5d}  ({v / len(rows):.0%})")
    print("\nhard properties:")
    for k, v in flags.most_common():
        print(f"  {k:16s} {v:5d}  ({v / len(rows):.0%})")


if __name__ == "__main__":
    main()
