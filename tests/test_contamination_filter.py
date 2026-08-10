"""Pre-filtering candidate pools against the eval set.

Regression cover for a real finding: the source corpus stores some charts twice
under consecutive sample ids, so id-based exclusion let the same chart reach
both splits. Only the content hash caught it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import (  # noqa: E402
    contamination_report,
    eval_signatures,
    filter_contaminated,
)


def gold(title="T", values=(1.0, 2.0)):
    return {
        "chart_type": "bar",
        "title": title,
        "x_label": "X",
        "y_label": "Y",
        "series": [{"name": "A", "data": [{"x": f"c{i}", "y": v} for i, v in enumerate(values)]}],
    }


def make_eval(tmp_path: Path, rows: list[dict]) -> Path:
    d = tmp_path / "ev"
    (d / "images").mkdir(parents=True, exist_ok=True)
    p = d / "ev.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def train(tmp_path: Path, sid: str, g: dict) -> dict:
    return {"id": sid, "image": "images/x.png", "gold": g, "_manifest_dir": str(tmp_path)}


def test_duplicate_under_a_different_id_is_still_caught(tmp_path):
    """The exact real-corpus failure: same chart, consecutive ids."""
    g = gold("Shared")
    ev = make_eval(tmp_path, [{"id": "real_31795", "image": "images/e.png", "gold": g}])
    exact, near, imgs = eval_signatures([ev])

    pool = [train(tmp_path, "real_31796", dict(g)), train(tmp_path, "real_99999", gold("Other"))]
    kept, dropped = filter_contaminated(pool, exact, near, imgs)

    assert [r["id"] for r in kept] == ["real_99999"]
    assert dropped["exact_gold"] == 1


def test_filtered_pool_reports_clean(tmp_path):
    """Filter and report must agree -- they share the same signatures."""
    g = gold("Shared")
    ev = make_eval(tmp_path, [{"id": "e0", "image": "images/e.png", "gold": g}])
    exact, near, imgs = eval_signatures([ev])

    pool = [train(tmp_path, "t0", dict(g))] + [
        train(tmp_path, f"t{i}", gold(f"Unique{i}")) for i in range(1, 6)
    ]
    kept, _ = filter_contaminated(pool, exact, near, imgs)
    assert contamination_report(kept, [ev])["total_contaminated"] == 0


def test_near_duplicates_are_dropped_too(tmp_path):
    ev = make_eval(
        tmp_path, [{"id": "e0", "image": "images/e.png", "gold": gold(values=(1.234, 5.678))}]
    )
    exact, near, imgs = eval_signatures([ev])
    kept, dropped = filter_contaminated(
        [train(tmp_path, "t0", gold(values=(1.236, 5.681)))], exact, near, imgs
    )
    assert kept == [] and dropped["near_duplicate"] == 1


def test_clean_pool_is_untouched(tmp_path):
    ev = make_eval(tmp_path, [{"id": "e0", "image": "images/e.png", "gold": gold("Eval")}])
    exact, near, imgs = eval_signatures([ev])
    pool = [train(tmp_path, f"t{i}", gold(f"Train{i}")) for i in range(5)]
    kept, dropped = filter_contaminated(pool, exact, near, imgs)
    assert len(kept) == 5 and sum(dropped.values()) == 0


def test_each_row_is_counted_once(tmp_path):
    """A row that is both an image and a gold match must not be double-dropped."""
    from PIL import Image

    ev_dir = tmp_path / "ev"
    (ev_dir / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (3, 3, 3)).save(ev_dir / "images" / "e.png")
    g = gold("Both")
    ev = ev_dir / "ev.jsonl"
    ev.write_text(
        json.dumps({"id": "e0", "image": "images/e.png", "gold": g}) + "\n", encoding="utf-8"
    )

    tr = tmp_path / "tr"
    (tr / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (3, 3, 3)).save(tr / "images" / "x.png")

    exact, near, imgs = eval_signatures([ev])
    kept, dropped = filter_contaminated(
        [{"id": "t0", "image": "images/x.png", "gold": dict(g), "_manifest_dir": str(tr)}],
        exact, near, imgs,
    )
    assert kept == []
    assert sum(dropped.values()) == 1
