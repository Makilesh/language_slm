"""Training-sample formatting.

The expensive failure this guards is a training prompt that differs from the
inference prompt. It cannot be caught by watching loss — loss falls either way
— so it has to be asserted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.format import build_sample, stats, target_json  # noqa: E402
from src.eval.prompts import PROMPT_VERSION, build_messages  # noqa: E402
from src.eval.schema import ChartData, parse_prediction  # noqa: E402


def gold_dict():
    return {
        "chart_type": "bar",
        "title": "Quarterly Revenue",
        "x_label": "Quarter",
        "y_label": "Revenue",
        "series": [{"name": "EMEA", "data": [{"x": "Q1", "y": 1.5}, {"x": "Q2", "y": 2.0}]}],
    }


def row(**kw):
    base = {"id": "s0", "image": "images/s0.png", "gold": gold_dict()}
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# target
# --------------------------------------------------------------------------- #


def test_target_round_trips_through_the_parser():
    """What we train on must be what the eval parser accepts, with no repairs."""
    parsed, err, repairs = parse_prediction(target_json(gold_dict()))
    assert err is None
    assert repairs == []
    assert parsed.series[0].data[0].y == 1.5


def test_target_is_compact():
    t = target_json(gold_dict())
    assert ", " not in t and '": ' not in t
    assert "\n" not in t


def test_target_key_order_is_schema_order_not_sorted():
    t = target_json(gold_dict())
    assert t.index('"chart_type"') < t.index('"title"') < t.index('"series"')


def test_target_accepts_a_chartdata_instance():
    assert target_json(ChartData.model_validate(gold_dict())) == target_json(gold_dict())


# --------------------------------------------------------------------------- #
# sample shape
# --------------------------------------------------------------------------- #


def test_sample_user_turn_is_identical_to_inference():
    """Structural guarantee: the user turn comes from the harness's own builder."""
    s = build_sample(row(), "engineered")
    assert s["messages"][:-1] == build_messages("engineered")


def test_sample_has_exactly_one_image_placeholder():
    s = build_sample(row(), "engineered")
    images = [
        c for m in s["messages"] for c in m["content"] if c.get("type") == "image"
    ]
    assert len(images) == 1


def test_assistant_turn_is_last_and_carries_the_target():
    s = build_sample(row(), "engineered")
    last = s["messages"][-1]
    assert last["role"] == "assistant"
    assert last["content"][0]["text"] == target_json(gold_dict())


def test_sample_records_prompt_version():
    """A prompt change must be attributable, since it invalidates trained runs."""
    assert build_sample(row(), "minimal")["prompt_version"] == PROMPT_VERSION


def test_sample_carries_image_path_not_pixels():
    """Resolution ablations must not require rebuilding the dataset."""
    s = build_sample(row(), "engineered")
    assert s["image"] == "images/s0.png"


def test_sample_preserves_scored_fields_for_real_charts():
    s = build_sample(row(scored_fields=["series", "values"], source="real"), "engineered")
    assert s["scored_fields"] == ["series", "values"]
    assert s["source"] == "real"


@pytest.mark.parametrize("style", ["minimal", "engineered"])
def test_both_prompt_styles_build(style):
    s = build_sample(row(), style)
    assert s["messages"][:-1] == build_messages(style)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_stats_report_mix_and_degradation():
    samples = (
        [build_sample(row(source="synth", degradations=["jpeg"]), "engineered") for _ in range(7)]
        + [build_sample(row(source="real"), "engineered") for _ in range(3)]
    )
    st = stats(samples)
    assert st["n"] == 10
    assert st["synthetic_fraction"] == 0.7
    assert st["n_degraded"] == 7
    assert st["degradation_kinds"]["jpeg"] == 7


def test_stats_on_empty_input_does_not_divide_by_zero():
    st = stats([])
    assert st["n"] == 0 and st["synthetic_fraction"] == 0


def test_stats_handle_unannotated_chart_type():
    """Real charts carry chart_type=None; mixing them with synth must not crash.

    `sorted()` cannot compare None to str, and folding None into "bar" would
    invent ground truth the corpus does not have.
    """
    samples = [
        build_sample(row(source="real", properties={"chart_type": None, "n_series": 2}), "engineered"),
        build_sample(row(source="synth", properties={"chart_type": "bar", "n_series": 1}), "engineered"),
    ]
    st = stats(samples)
    assert st["by_chart_type"] == {"bar": 1, "unannotated": 1}


def test_series_counts_sort_numerically():
    """String sorting would order 10 before 2 and make the table unreadable."""
    samples = [
        build_sample(row(properties={"chart_type": "bar", "n_series": n}), "engineered")
        for n in (10, 2, 1)
    ]
    assert list(stats(samples)["by_series_count"]) == ["1", "2", "10"]


# --------------------------------------------------------------------------- #
# the real check -- needs the processor
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_training_prompt_is_byte_identical_to_inference_prompt():
    """The failure this whole module exists to prevent."""
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    from src.data.format import verify_template_matches_inference

    try:
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    except Exception as exc:  # noqa: BLE001 - offline / uncached is a skip
        pytest.skip(f"processor unavailable: {type(exc).__name__}")

    for style in ("minimal", "engineered"):
        verify_template_matches_inference(proc, build_sample(row(), style), style)
