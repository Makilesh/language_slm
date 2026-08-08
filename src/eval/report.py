"""Turn scored runs into a committed markdown table.

Every number in results/*.md comes through here from a scores/*.json file that
the harness produced, so nothing in a report can be typed by hand. If a run is
not in this table, it was not measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, agg: dict) -> str:
    o = agg["overall"]
    cfg = agg.get("config", {})
    va = o["value_accuracy"]
    med = "n/a" if o.get("median_ape") is None else f"{o['median_ape']:.0f}%"
    cells = [
        name,
        str(o["n"]),
        pct(o["valid_json_rate"]),
        pct(o.get("schema_conformance_rate")),
        pct(o["chart_type_accuracy"]),
        pct(o["series_name_accuracy"]),
        pct(o["exact_structural_match"]),
        pct(o["point_recall"]),
        pct(va.get("1%")),
        f"**{pct(va.get('5%'))}**",   # the headline metric
        pct(va.get("10%")),
        med,
        f"{cfg.get('long_edge', '?')}px",
        "yes" if cfg.get("constrained") else "no",
    ]
    return "| " + " | ".join(cells) + " |"


HEADER = (
    "| run | n | valid JSON | schema-exact | chart type | series names | structural "
    "| point recall | val@1% | val@5% | val@10% | median APE | res | constrained |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def build(runs: list[tuple[str, Path]], out: Path, title: str, preamble: str) -> Path:
    lines = [f"# {title}", "", preamble, "", HEADER]
    loaded: list[tuple[str, dict]] = []

    for name, path in runs:
        if not path.exists():
            print(f"  skip (missing): {path}")
            continue
        agg = load(path)
        loaded.append((name, agg))
        lines.append(row(name, agg))

    for name, agg in loaded:
        by_type = agg.get("by_chart_type", {})
        if not by_type:
            continue
        lines += [
            "",
            f"## {name} — by chart type",
            "",
            "| chart type | n | val@5% | structural | point recall |",
            "|---|---|---|---|---|",
        ]
        for ct, s in sorted(by_type.items()):
            lines.append(
                f"| {ct} | {s['n']} | {pct(s['value_accuracy'].get('5%'))} "
                f"| {pct(s['exact_structural_match'])} | {pct(s['point_recall'])} |"
            )

        by_series = agg.get("by_series_count", {})
        if by_series:
            lines += [
                "",
                f"## {name} — by series count",
                "",
                "| series | n | val@5% | structural |",
                "|---|---|---|---|",
            ]
            for bucket, s in by_series.items():
                lines.append(
                    f"| {bucket} | {s['n']} | {pct(s['value_accuracy'].get('5%'))} "
                    f"| {pct(s['exact_structural_match'])} |"
                )

    lines += ["", "## Run configs", ""]
    for name, agg in loaded:
        cfg = agg.get("config", {})
        timing = agg.get("timing", {})
        parts = [
            f"`{cfg.get('model')}`",
            f"adapter `{cfg['adapter']}`" if cfg.get("adapter") else "no adapter",
            str(cfg.get("dtype")),
            f"prompt `{cfg.get('prompt')}` (v{cfg.get('prompt_version')})",
            f"constrained={cfg.get('constrained')}",
            f"{cfg.get('long_edge')}px",
            f"max_new_tokens={cfg.get('max_new_tokens')}",
        ]
        if timing.get("median_seconds") is not None:
            parts.append(f"median {timing['median_seconds']:.1f}s/chart")
        if timing.get("mean_tokens") is not None:
            parts.append(f"mean {timing['mean_tokens']:.0f} tokens")
        lines.append(f"- **{name}** — " + ", ".join(parts))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", action="append", default=[], metavar="NAME=PATH",
        help="repeatable, e.g. --run 'base zero-shot=results/scores/base_minimal.json'",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Baselines")
    ap.add_argument("--preamble", default="")
    ap.add_argument(
        "--preamble-file", type=Path, default=None,
        help="markdown prepended to the table; easier than passing prose through argv",
    )
    args = ap.parse_args()

    preamble = args.preamble
    if args.preamble_file:
        preamble = args.preamble_file.read_text(encoding="utf-8")

    runs = []
    for item in args.run:
        name, _, path = item.partition("=")
        runs.append((name.strip(), Path(path.strip())))
    build(runs, args.out, args.title, preamble)


if __name__ == "__main__":
    main()
