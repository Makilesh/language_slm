"""Contamination detection and token-budget arithmetic.

A contamination check that silently passes is worse than none: it converts an
unknown risk into a false assurance printed in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import (  # noqa: E402
    contamination_report,
    gold_signature,
    image_tokens_for,
    mix,
)


def gold(title="T", values=(1.0, 2.0), name="A", ctype="bar"):
    return {
        "chart_type": ctype,
        "title": title,
        "x_label": "X",
        "y_label": "Y",
        "series": [{"name": name, "data": [{"x": f"c{i}", "y": v} for i, v in enumerate(values)]}],
    }


# --------------------------------------------------------------------------- #
# signatures
# --------------------------------------------------------------------------- #


def test_identical_gold_has_identical_signature():
    assert gold_signature(gold()) == gold_signature(gold())


def test_different_values_differ():
    assert gold_signature(gold(values=(1.0, 2.0))) != gold_signature(gold(values=(1.0, 9.0)))


def test_series_order_is_not_significant():
    """The schema gives series order no meaning, and neither does the metric."""
    a = {**gold(), "series": [
        {"name": "A", "data": [{"x": "c0", "y": 1.0}]},
        {"name": "B", "data": [{"x": "c0", "y": 2.0}]},
    ]}
    b = {**gold(), "series": [
        {"name": "B", "data": [{"x": "c0", "y": 2.0}]},
        {"name": "A", "data": [{"x": "c0", "y": 1.0}]},
    ]}
    assert gold_signature(a) == gold_signature(b)


def test_near_signature_collapses_small_value_drift():
    """Two charts differing in the last digit are near-duplicates, not new data."""
    a, b = gold(values=(1.234, 5.678)), gold(values=(1.236, 5.681))
    assert gold_signature(a, sig=4) != gold_signature(b, sig=4)
    assert gold_signature(a, sig=2) == gold_signature(b, sig=2)


@pytest.mark.parametrize("bad", [0.0, float("inf"), float("nan")])
def test_signature_survives_degenerate_values(bad):
    gold_signature(gold(values=(bad, 1.0)))  # must not raise


# --------------------------------------------------------------------------- #
# contamination
# --------------------------------------------------------------------------- #


def write_manifest(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    d = tmp_path / name
    (d / "images").mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def train_row(tmp_path: Path, sid: str, g: dict) -> dict:
    return {"id": sid, "image": "images/x.png", "gold": g, "_manifest_dir": str(tmp_path)}


def test_clean_split_reports_nothing(tmp_path):
    ev = write_manifest(tmp_path, "ev", [{"id": "e0", "image": "images/e0.png", "gold": gold("Eval")}])
    rep = contamination_report([train_row(tmp_path, "t0", gold("Train"))], [ev])
    assert rep["total_contaminated"] == 0


def test_exact_gold_reuse_is_caught(tmp_path):
    g = gold("Shared")
    ev = write_manifest(tmp_path, "ev", [{"id": "e0", "image": "images/e0.png", "gold": g}])
    rep = contamination_report([train_row(tmp_path, "t0", dict(g))], [ev])
    assert rep["hits"]["exact_gold"] == 1
    assert rep["examples"]["exact_gold"][0]["train"] == "t0"


def test_near_duplicate_is_caught(tmp_path):
    ev = write_manifest(
        tmp_path, "ev",
        [{"id": "e0", "image": "images/e0.png", "gold": gold(values=(1.234, 5.678))}],
    )
    rep = contamination_report([train_row(tmp_path, "t0", gold(values=(1.236, 5.681)))], [ev])
    assert rep["hits"]["near_duplicate"] == 1


def test_exact_hit_is_not_double_counted_as_near(tmp_path):
    g = gold("Shared")
    ev = write_manifest(tmp_path, "ev", [{"id": "e0", "image": "images/e0.png", "gold": g}])
    rep = contamination_report([train_row(tmp_path, "t0", dict(g))], [ev])
    assert rep["hits"]["exact_gold"] == 1
    assert rep["hits"]["near_duplicate"] == 0


def test_identical_image_bytes_are_caught(tmp_path):
    from PIL import Image

    ev_dir = tmp_path / "ev"
    (ev_dir / "images").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (8, 8), (7, 7, 7))
    img.save(ev_dir / "images" / "e0.png")

    tr_dir = tmp_path / "tr"
    (tr_dir / "images").mkdir(parents=True, exist_ok=True)
    img.save(tr_dir / "images" / "t0.png")

    ev = ev_dir / "ev.jsonl"
    ev.write_text(
        json.dumps({"id": "e0", "image": "images/e0.png", "gold": gold("Eval")}) + "\n",
        encoding="utf-8",
    )
    rows = [{"id": "t0", "image": "images/t0.png", "gold": gold("Train"),
             "_manifest_dir": str(tr_dir)}]
    rep = contamination_report(rows, [ev])
    assert rep["hits"]["image_bytes"] == 1


def test_internal_duplicates_are_surfaced(tmp_path):
    ev = write_manifest(tmp_path, "ev", [{"id": "e0", "image": "images/e0.png", "gold": gold("Eval")}])
    g = gold("Repeat")
    rows = [train_row(tmp_path, f"t{i}", dict(g)) for i in range(3)]
    rep = contamination_report(rows, [ev])
    assert rep["internal_duplicate_groups"] == 1
    assert rep["internal_duplicate_rows"] == 2  # 3 rows, 2 redundant


# --------------------------------------------------------------------------- #
# token arithmetic
# --------------------------------------------------------------------------- #


def test_image_tokens_match_phase0_measurements():
    """Phase 0 measured these exactly; the analytic form must reproduce them."""
    assert image_tokens_for(1800, 1100, 448) == 126     # 14x9 grid
    assert image_tokens_for(1800, 1100, 768) == 360     # 24x15
    assert image_tokens_for(1800, 1100, 1280) == 960    # 40x24


def test_image_tokens_scale_with_area_not_edge():
    """Doubling the long edge roughly quadruples tokens -- the Phase 0 lesson."""
    a = image_tokens_for(1000, 1000, 448)
    b = image_tokens_for(1000, 1000, 896)
    assert 3.5 < b / a < 4.5


# --------------------------------------------------------------------------- #
# mixing
# --------------------------------------------------------------------------- #


def test_mix_respects_ratio():
    import random

    synth = [{"id": f"s{i}", "source": "synth"} for i in range(100)]
    real = [{"id": f"r{i}", "source": "real"} for i in range(100)]
    out = mix(synth, real, 0.7, 50, random.Random(0))
    assert len(out) == 50
    assert sum(1 for r in out if r["source"] == "synth") == 35


def test_mix_backfills_when_a_source_is_short(capsys):
    import random

    synth = [{"id": f"s{i}", "source": "synth"} for i in range(100)]
    real = [{"id": f"r{i}", "source": "real"} for i in range(5)]
    out = mix(synth, real, 0.7, 50, random.Random(0))
    assert len(out) == 50                      # count is honoured
    assert sum(1 for r in out if r["source"] == "real") == 5
    assert "WARNING" in capsys.readouterr().out  # and the shortfall is announced
