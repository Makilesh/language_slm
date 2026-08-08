"""Metric tests.

The headline one is `test_gold_scores_perfectly_against_itself` — the Phase 1
acceptance criterion. The rest pin the behaviours that separate this harness
from a naive one: order independence, honest handling of count mismatches, and
refusing to silently forgive a scale error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import (  # noqa: E402
    TOLERANCES,
    aggregate,
    detect_scale_error,
    match_points,
    match_series,
    normalize_text,
    relative_error,
    score_sample,
)
from src.eval.schema import ChartData, parse_prediction  # noqa: E402


def chart(**overrides) -> ChartData:
    base = {
        "chart_type": "bar",
        "title": "Quarterly Revenue by Region",
        "x_label": "Quarter",
        "y_label": "Revenue (thousands USD)",
        "series": [
            {
                "name": "North America",
                "data": [{"x": "Q1", "y": 121.5}, {"x": "Q2", "y": 138.2}],
            },
            {
                "name": "EMEA",
                "data": [{"x": "Q1", "y": 78.0}, {"x": "Q2", "y": 95.4}],
            },
        ],
    }
    base.update(overrides)
    return ChartData.model_validate(base)


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #


def test_gold_scores_perfectly_against_itself():
    """Phase 1 acceptance: the harness scores ground truth against itself at 100%."""
    gold = chart()
    m = score_sample(gold, gold.model_copy(deep=True), "self")

    assert m.valid_json
    assert m.chart_type_correct
    assert m.title_correct and m.x_label_correct and m.y_label_correct
    assert m.series_count_correct and m.point_count_correct
    assert m.series_name_accuracy == 1.0
    assert m.exact_structural_match
    assert m.point_recall == 1.0
    assert m.n_missing_points == 0 and m.n_spurious_points == 0
    for tol in TOLERANCES:
        assert m.value_accuracy[tol] == 1.0
    assert m.mape == pytest.approx(0.0)
    assert not m.scale_errors
    assert not m.used_positional_fallback


def test_aggregate_of_perfect_predictions_is_all_ones():
    charts = [
        chart(),
        chart(chart_type="line", title="Users over time"),
        chart(chart_type="pie", series=[{"name": "A", "data": [{"x": "a", "y": 1.0}]}]),
    ]
    samples = [score_sample(c, c.model_copy(deep=True), str(i)) for i, c in enumerate(charts)]
    agg = aggregate(samples)

    o = agg["overall"]
    assert o["valid_json_rate"] == 1.0
    assert o["chart_type_accuracy"] == 1.0
    assert o["exact_structural_match"] == 1.0
    assert o["value_accuracy"]["5%"] == 1.0
    assert o["mape"] == pytest.approx(0.0)
    assert set(agg["by_chart_type"]) == {"bar", "line", "pie"}


# --------------------------------------------------------------------------- #
# the cases naive implementations get wrong
# --------------------------------------------------------------------------- #


def test_series_order_is_not_penalised():
    gold = chart()
    shuffled = chart(series=list(reversed(gold.model_dump(mode="json")["series"])))

    m = score_sample(gold, shuffled, "shuffled")
    assert m.series_name_accuracy == 1.0
    assert m.value_accuracy[0.01] == 1.0
    assert m.exact_structural_match


def test_point_order_within_series_is_not_penalised():
    gold = chart()
    d = gold.model_dump(mode="json")
    for s in d["series"]:
        s["data"] = list(reversed(s["data"]))

    m = score_sample(gold, ChartData.model_validate(d), "pt-order")
    assert m.value_accuracy[0.01] == 1.0
    assert m.n_missing_points == 0 and m.n_spurious_points == 0


def test_missing_points_do_not_inflate_value_accuracy():
    """Dropping the hard points must not look like perfect extraction."""
    gold = chart()
    d = gold.model_dump(mode="json")
    for s in d["series"]:
        s["data"] = s["data"][:1]  # keep only the first point

    m = score_sample(gold, ChartData.model_validate(d), "truncated")

    # The kept points are exactly right, so value accuracy over *matched* pairs
    # is 1.0 -- and that is precisely why the count penalty must be separate
    # and visible.
    assert m.value_accuracy[0.01] == 1.0
    assert m.n_matched_points == 2
    assert m.n_missing_points == 2
    assert m.point_recall == 0.5
    assert not m.point_count_correct
    assert not m.exact_structural_match


def test_spurious_points_are_counted():
    gold = chart()
    d = gold.model_dump(mode="json")
    d["series"][0]["data"].append({"x": "Q3", "y": 150.0})

    m = score_sample(gold, ChartData.model_validate(d), "extra")
    assert m.n_spurious_points == 1
    assert not m.point_count_correct


def test_missing_series_counts_its_points_as_missing():
    gold = chart()
    d = gold.model_dump(mode="json")
    d["series"] = d["series"][:1]

    m = score_sample(gold, ChartData.model_validate(d), "dropped-series")
    assert not m.series_count_correct
    assert m.n_missing_points == 2  # the whole EMEA series
    assert m.point_recall == 0.5


def test_tolerances_are_graded_not_binary():
    gold = chart(series=[{"name": "A", "data": [{"x": "a", "y": 100.0}]}])
    # 3% off: outside 1%, inside 5% and 10%.
    pred = chart(series=[{"name": "A", "data": [{"x": "a", "y": 103.0}]}])

    m = score_sample(gold, pred, "tol")
    assert m.value_accuracy[0.01] == 0.0
    assert m.value_accuracy[0.05] == 1.0
    assert m.value_accuracy[0.10] == 1.0
    assert m.mape == pytest.approx(3.0)


def test_scale_error_is_detected_and_still_scored_wrong():
    """A thousands-vs-raw misread is flagged, never silently forgiven."""
    gold = chart(
        series=[{"name": "A", "data": [{"x": str(i), "y": float(i * 10)} for i in range(1, 6)]}]
    )
    pred = chart(
        series=[{"name": "A", "data": [{"x": str(i), "y": float(i * 10_000)} for i in range(1, 6)]}]
    )

    m = score_sample(gold, pred, "scale")
    assert m.scale_errors == [1e3]
    assert m.value_accuracy[0.10] == 0.0  # flagged, but still wrong

    # Structure really is exact here -- same labels, same series, same point
    # counts -- and only the values are off. The two metrics are supposed to
    # disagree in exactly this case; that separation is why value accuracy is
    # the headline number and structural match is reported beside it.
    assert m.exact_structural_match
    assert m.mape > 1000


def test_numeric_and_string_x_values_match():
    gold = chart(series=[{"name": "A", "data": [{"x": 2023, "y": 5.0}]}])
    pred = chart(series=[{"name": "A", "data": [{"x": "2023", "y": 5.0}]}])

    m = score_sample(gold, pred, "x-coerce")
    assert m.n_matched_points == 1
    assert m.value_accuracy[0.01] == 1.0


def test_positional_fallback_when_x_labels_are_wrong_but_values_right():
    """An axis-labelling error must not masquerade as a value-reading error."""
    gold = chart(
        series=[{"name": "A", "data": [{"x": f"Cat {i}", "y": float(i)} for i in range(1, 6)]}]
    )
    pred = chart(
        series=[{"name": "A", "data": [{"x": str(i), "y": float(i)} for i in range(1, 6)]}]
    )

    m = score_sample(gold, pred, "x-mislabel")
    assert m.used_positional_fallback
    assert m.n_matched_points == 5
    assert m.value_accuracy[0.01] == 1.0


def test_unparseable_output_scores_zero_but_stays_in_denominator():
    gold = chart()
    m = score_sample(gold, None, "broken", parse_error="invalid JSON: Expecting value")

    assert not m.valid_json
    assert m.value_accuracy[0.05] == 0.0
    assert not m.exact_structural_match

    agg = aggregate([m, score_sample(gold, gold.model_copy(deep=True), "ok")])
    assert agg["overall"]["n"] == 2
    assert agg["overall"]["valid_json_rate"] == 0.5


def test_series_matching_is_bijective():
    """One predicted series cannot claim credit against several gold ones."""
    gold = chart()
    pred = chart(series=[{"name": "North America", "data": [{"x": "Q1", "y": 121.5}]}])

    pairs, unmatched_gold, unmatched_pred = match_series(gold.series, pred.series)
    assert len(pairs) == 1
    assert len(unmatched_gold) == 1
    assert not unmatched_pred


def test_series_matched_by_data_when_names_are_degenerate():
    gold = chart()
    d = gold.model_dump(mode="json")
    for s in d["series"]:
        s["name"] = ""
    blank = ChartData.model_validate(d)
    # Reverse so a positional match would be wrong.
    blank.series = list(reversed(blank.series))

    pairs, _, _ = match_series(gold.series, blank.series)
    for gi, pj in pairs:
        assert gold.series[gi].data[0].y == blank.series[pj].data[0].y


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "a,b",
    [
        ("Revenue (USD)", "revenue (usd)"),
        ("  Quarter  ", "Quarter"),
        ("Sales:", "Sales"),
        ("A  B", "a b"),
    ],
)
def test_normalize_forgives_only_formatting(a, b):
    assert normalize_text(a) == normalize_text(b)


def test_normalize_does_not_forgive_word_changes():
    assert normalize_text("Total Revenue") != normalize_text("Revenue")


def test_relative_error_at_zero_gold():
    assert relative_error(0.0, 0.0) == 0.0
    assert relative_error(0.0, 5.0) == float("inf")


def test_empty_prediction_series_matches_nothing():
    pairs, unmatched_gold, unmatched_pred = match_series(chart().series, [])
    assert not pairs and len(unmatched_gold) == 2 and not unmatched_pred


def test_detect_scale_error_needs_consistency():
    from src.eval.schema import Point

    pairs = [(Point(x="a", y=1.0), Point(x="a", y=1000.0)),
             (Point(x="b", y=2.0), Point(x="b", y=2000.0)),
             (Point(x="c", y=3.0), Point(x="c", y=3.5))]
    assert detect_scale_error(pairs) is None  # only 2 of 3 agree


def test_match_points_handles_duplicate_x_values():
    from src.eval.schema import Point

    gold = [Point(x="a", y=1.0), Point(x="a", y=2.0)]
    pred = [Point(x="a", y=1.0), Point(x="a", y=2.0)]
    pairs, missing, spurious, _ = match_points(gold, pred)
    assert len(pairs) == 2 and missing == 0 and spurious == 0


# --------------------------------------------------------------------------- #
# lenient parsing (for un-constrained baselines)
# --------------------------------------------------------------------------- #


def test_parse_plain_json():
    parsed, err, repairs = parse_prediction(chart().to_json())
    assert err is None and parsed is not None
    assert repairs == []  # exact schema match needs no repair


def test_parse_markdown_fenced():
    parsed, err, repairs = parse_prediction(f"```json\n{chart().to_json()}\n```")
    assert err is None and parsed is not None
    assert "stripped markdown fence" in repairs


def test_parse_with_surrounding_prose():
    parsed, err, repairs = parse_prediction(
        f"Here is the data:\n{chart().to_json()}\nHope that helps!"
    )
    assert err is None and parsed is not None
    assert "extracted from prose" in repairs


def test_points_key_is_repaired_and_recorded():
    """The exact failure that scored baseline A at 0% valid JSON.

    The model read the chart correctly and called the list "points". Extraction
    quality is 100%; formatting conformance is 0%. Those must not collapse into
    one number.
    """
    raw = json.dumps(
        {
            "chart_type": "bar",
            "title": "T",
            "x_label": "X",
            "y_label": "Y",
            "series": [{"name": "A", "points": [{"x": "Q1", "y": 1.0}]}],
        }
    )
    parsed, err, repairs = parse_prediction(raw)
    assert err is None
    assert parsed.series[0].data[0].y == 1.0
    assert "data<-points" in repairs

    strict, strict_err, _ = parse_prediction(raw, lenient=False)
    assert strict is None and "schema violation" in strict_err


def test_repairs_are_reported_separately_from_conformance():
    gold = chart(series=[{"name": "A", "data": [{"x": "Q1", "y": 1.0}]}])
    raw = json.dumps(
        {
            "chart_type": "bar",
            "title": gold.title,
            "x_label": gold.x_label,
            "y_label": gold.y_label,
            "series": [{"name": "A", "points": [{"x": "Q1", "y": 1.0}]}],
        }
    )
    parsed, err, repairs = parse_prediction(raw)
    m = score_sample(gold, parsed, "repaired", parse_error=err, repairs=repairs)
    agg = aggregate([m])

    o = agg["overall"]
    assert o["valid_json_rate"] == 1.0          # usable
    assert o["schema_conformance_rate"] == 0.0  # but not conformant
    assert o["n_repaired"] == 1
    assert o["value_accuracy"]["1%"] == 1.0     # extraction was perfect
    assert agg["repairs"]["data<-points"] == 1


@pytest.mark.parametrize(
    "text,fragment",
    [
        ("", "empty"),
        ("I cannot read this chart.", "no JSON object"),
        ('{"chart_type": "bar",}', "invalid JSON"),
        ('{"chart_type": "donut", "series": []}', "schema violation"),
    ],
)
def test_parse_failures_are_explained(text, fragment):
    parsed, err, _ = parse_prediction(text)
    assert parsed is None
    assert fragment in err


def test_selfcheck_gold_needs_no_repairs():
    """Gold round-tripped through JSON must be schema-exact, not merely repairable."""
    parsed, err, repairs = parse_prediction(chart().to_json())
    assert err is None and repairs == []
