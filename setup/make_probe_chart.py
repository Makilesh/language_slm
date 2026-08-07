"""Render the Phase 0 probe chart.

A deliberately non-trivial chart: 3 series, 8 categories, legend, grid, rotated
tick labels, y-axis in thousands. Rendered once at high resolution so that every
downscale in verify_vlm.py is a real downscale, not an upscale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "assets" / "demo" / "probe_chart.png"

# Fixed so the probe image is byte-stable across runs; the VRAM table is only
# comparable across resolutions if the source image never changes.
SEED = 42

CATEGORIES = [
    "Q1 2023",
    "Q2 2023",
    "Q3 2023",
    "Q4 2023",
    "Q1 2024",
    "Q2 2024",
    "Q3 2024",
    "Q4 2024",
]
SERIES = ["North America", "EMEA", "APAC"]


def render(out_path: Path, dpi: int = 200) -> Path:
    rng = np.random.default_rng(SEED)
    base = np.array([120.0, 78.0, 45.0])
    growth = np.array([1.06, 1.09, 1.14])

    values = np.stack(
        [
            base[i] * growth[i] ** np.arange(len(CATEGORIES))
            + rng.normal(0, base[i] * 0.04, len(CATEGORIES))
            for i in range(len(SERIES))
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=dpi)
    width = 0.26
    x = np.arange(len(CATEGORIES))

    for i, name in enumerate(SERIES):
        ax.bar(x + (i - 1) * width, values[i], width, label=name)

    ax.set_title("Quarterly Revenue by Region")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Revenue (thousands USD)")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=45, ha="right")
    ax.legend(title="Region", loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    path = render(args.out, args.dpi)
    from PIL import Image

    with Image.open(path) as im:
        print(f"wrote {path}  {im.width}x{im.height}")


if __name__ == "__main__":
    main()
