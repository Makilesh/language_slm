"""Graded chart-to-data metrics.

Scoring extraction is mostly a matching problem. Before any number can be
computed you have to decide which predicted series corresponds to which gold
series, and which predicted point corresponds to which gold point. Naive
implementations index by position, which punishes a model for emitting the same
data in a different order — so they measure ordering, not extraction.

What this module does about the three cases the brief calls out:

* **Series order differs.** Series are matched by normalised name, then by
  optimal assignment (Hungarian) over a name-similarity cost matrix, with a
  small data-similarity term purely to break ties when names are degenerate
  (all empty, all identical). Order is never penalised.

* **Point counts differ.** Points are matched on their x value, not their
  index. Unmatched gold points are `missing`, unmatched predictions are
  `spurious`, both reported explicitly and separately from value accuracy. Value
  accuracy is computed only over matched pairs, so a model cannot inflate it by
  emitting fewer points — the count penalty is a distinct number you have to
  read alongside it.

* **Value scale ambiguity.** A series read in thousands when the gold is raw is
  detected and *logged*, never silently rescaled. It stays wrong in the score.

Everything is per-sample first, then aggregated, so breakdowns by chart type and
series count come for free.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import mean

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from .schema import ChartData, Point, Series

# Default relative tolerances for value accuracy. The brief asks for all three
# to be reported; 5% is the headline.
TOLERANCES: tuple[float, ...] = (0.01, 0.05, 0.10)

# Below this magnitude a relative tolerance is meaningless (a gold value of 0.0
# makes every relative error infinite), so comparisons fall back to absolute.
ZERO_EPS = 1e-9

# Ratios that indicate a unit misread rather than a misread bar height.
SCALE_FACTORS: tuple[float, ...] = (1e-6, 1e-3, 1e-2, 1e2, 1e3, 1e6)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"^[\s\W_]+|[\s\W_]+$")


# --------------------------------------------------------------------------- #
# normalisation and comparison primitives
# --------------------------------------------------------------------------- #


def normalize_text(value: str | None) -> str:
    """Casefold, collapse whitespace, strip surrounding punctuation.

    Deliberately conservative: it forgives capitalisation, stray whitespace and
    trailing colons, but not word changes. Aggressive normalisation (dropping
    all punctuation, stemming) inflates label accuracy into meaninglessness.
    """
    if value is None:
        return ""
    text = _WS.sub(" ", str(value)).strip().casefold()
    return _PUNCT.sub("", text)


def as_float(value: object) -> float | None:
    """Coerce to float if the value is numeric, else None.

    Handles the common case of a gold x of `2023` meeting a predicted x of
    `"2023"` — those should match, and string comparison alone would miss it.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and not math.isfinite(value) else float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip().replace(",", ""))
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def x_matches(gold: str | float, pred: str | float) -> bool:
    """Compare x values numerically when both are numeric, textually otherwise."""
    g_num, p_num = as_float(gold), as_float(pred)
    if g_num is not None and p_num is not None:
        scale = max(abs(g_num), abs(p_num), 1.0)
        return abs(g_num - p_num) <= 1e-6 * scale
    return normalize_text(str(gold)) == normalize_text(str(pred))


def relative_error(gold: float, pred: float) -> float:
    """|pred - gold| / |gold|, with an absolute fallback near zero.

    Returns inf when gold is ~0 and pred is not, which is the honest answer: the
    prediction is wrong by an unbounded factor.
    """
    if abs(gold) < ZERO_EPS:
        return 0.0 if abs(pred) < ZERO_EPS else math.inf
    return abs(pred - gold) / abs(gold)


def within(gold: float, pred: float, tol: float) -> bool:
    return relative_error(gold, pred) <= tol


def text_correct(gold: str | None, pred: str | None) -> bool:
    """Both absent counts as correct; one absent does not."""
    return normalize_text(gold) == normalize_text(pred)


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #


def _name_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 0.0  # two blanks tell us nothing; let the data term decide
    return fuzz.ratio(na, nb) / 100.0


def _data_similarity(a: Series, b: Series) -> float:
    """Rough agreement between two series' y values, for tie-breaking only.

    Never strong enough to override a clear name match; it exists so that
    matching stays deterministic and sensible when every series is called
    "series 1" or nothing at all.
    """
    if not a.data or not b.data:
        return 0.0
    n = min(len(a.data), len(b.data))
    hits = sum(1 for i in range(n) if within(a.data[i].y, b.data[i].y, 0.10))
    return hits / max(len(a.data), len(b.data))


def match_series(
    gold: list[Series],
    pred: list[Series],
    name_weight: float = 0.85,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimally pair gold and predicted series.

    Returns (pairs, unmatched_gold_idx, unmatched_pred_idx). Pairing is
    bijective: a model that emits one series cannot claim credit against three
    gold ones.
    """
    if not gold or not pred:
        return [], list(range(len(gold))), list(range(len(pred)))

    score = np.zeros((len(gold), len(pred)), dtype=float)
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            score[i, j] = (
                name_weight * _name_similarity(g.name, p.name)
                + (1 - name_weight) * _data_similarity(g, p)
            )

    rows, cols = linear_sum_assignment(-score)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    matched_g = {i for i, _ in pairs}
    matched_p = {j for _, j in pairs}
    return (
        pairs,
        [i for i in range(len(gold)) if i not in matched_g],
        [j for j in range(len(pred)) if j not in matched_p],
    )


def match_points(
    gold: list[Point], pred: list[Point]
) -> tuple[list[tuple[Point, Point]], int, int, bool]:
    """Pair points by x value, falling back to position when x values are useless.

    Returns (pairs, n_missing, n_spurious, used_positional_fallback).

    The fallback matters: a model can read every bar height correctly while
    labelling the x axis with indices instead of category names. Scoring that as
    a total miss would confuse an axis-labelling error with a value-reading
    error, and those need different fixes.
    """
    pairs: list[tuple[Point, Point]] = []
    remaining = list(range(len(pred)))

    for g in gold:
        for slot, j in enumerate(remaining):
            if x_matches(g.x, pred[j].x):
                pairs.append((g, pred[j]))
                remaining.pop(slot)
                break

    # If x-based matching found almost nothing but the shapes line up, the x
    # values are mislabelled rather than the points being absent.
    if gold and pred and len(pairs) < 0.5 * min(len(gold), len(pred)):
        n = min(len(gold), len(pred))
        return (
            [(gold[i], pred[i]) for i in range(n)],
            len(gold) - n,
            len(pred) - n,
            True,
        )

    return pairs, len(gold) - len(pairs), len(pred) - len(pairs), False


def detect_scale_error(pairs: list[tuple[Point, Point]]) -> float | None:
    """Return the scale factor if a series looks uniformly mis-scaled.

    Requires the same factor to explain at least 80% of the points, so a couple
    of coincidental ratios cannot trigger it.
    """
    usable = [(g.y, p.y) for g, p in pairs if abs(g.y) > ZERO_EPS and abs(p.y) > ZERO_EPS]
    if len(usable) < 3:
        return None

    for factor in SCALE_FACTORS:
        hits = sum(1 for g, p in usable if within(g * factor, p, 0.05))
        if hits >= 0.8 * len(usable):
            return factor
    return None


# --------------------------------------------------------------------------- #
# per-sample scoring
# --------------------------------------------------------------------------- #


# Fields a sample can be scored on. Real chart-to-table corpora ship the
# underlying values but not the chart type, title, or axis labels — their first
# column header is a generic "Characteristic" placeholder. Scoring a model
# against ground truth that does not exist would be fabrication, so each sample
# declares which fields it actually carries and the rest are excluded from both
# the per-sample verdict and the aggregate denominator.
ALL_FIELDS = frozenset({"chart_type", "title", "x_label", "y_label", "series_names"})


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


@dataclass
class SampleMetrics:
    sample_id: str
    valid_json: bool
    parse_error: str | None = None

    scored_fields: frozenset[str] = ALL_FIELDS

    # Key substitutions needed to fit the output onto the schema. Empty means
    # the model matched the schema exactly.
    repairs: list[str] = field(default_factory=list)

    chart_type_correct: bool = False
    gold_chart_type: str | None = None

    title_correct: bool = False
    x_label_correct: bool = False
    y_label_correct: bool = False

    gold_n_series: int = 0
    pred_n_series: int = 0
    series_count_correct: bool = False
    series_name_accuracy: float = 0.0

    gold_n_points: int = 0
    pred_n_points: int = 0
    n_matched_points: int = 0
    n_missing_points: int = 0
    n_spurious_points: int = 0
    point_count_correct: bool = False

    # tolerance -> fraction of matched points within it
    value_accuracy: dict[float, float] = field(default_factory=dict)

    # MAPE is unbounded: one point read as 36000 against a gold of 0.036 puts
    # the mean in the millions and drowns out every other sample. It is kept
    # because a scale blow-up is real information, but the median is what you
    # read to compare two runs.
    mape: float | None = None
    median_ape: float | None = None

    exact_structural_match: bool = False
    used_positional_fallback: bool = False
    scale_errors: list[float] = field(default_factory=list)

    @property
    def point_recall(self) -> float:
        """Share of gold points the prediction actually accounted for.

        Read alongside value accuracy: value accuracy over 3 of 30 matched
        points is not the same claim as over 30 of 30.
        """
        return self.n_matched_points / self.gold_n_points if self.gold_n_points else 0.0


def score_sample(
    gold: ChartData,
    pred: ChartData | None,
    sample_id: str = "",
    parse_error: str | None = None,
    tolerances: tuple[float, ...] = TOLERANCES,
    scored_fields: frozenset[str] | set[str] | None = None,
    repairs: list[str] | None = None,
) -> SampleMetrics:
    """Score one prediction against gold.

    `pred=None` means the output could not be parsed; the sample still appears
    in the aggregate with zeros so that unparseable output is penalised rather
    than quietly dropped from the denominator.

    `scored_fields` restricts which label fields count, for corpora whose ground
    truth does not include them. Values and structure are always scored.
    """
    fields = frozenset(scored_fields) if scored_fields is not None else ALL_FIELDS
    m = SampleMetrics(
        sample_id=sample_id,
        valid_json=pred is not None,
        parse_error=parse_error,
        scored_fields=fields,
        repairs=list(repairs or []),
        gold_chart_type=gold.chart_type.value,
        gold_n_series=gold.n_series,
        gold_n_points=gold.n_points,
        value_accuracy={t: 0.0 for t in tolerances},
    )
    if pred is None:
        return m

    m.chart_type_correct = gold.chart_type == pred.chart_type
    m.title_correct = text_correct(gold.title, pred.title)
    m.x_label_correct = text_correct(gold.x_label, pred.x_label)
    m.y_label_correct = text_correct(gold.y_label, pred.y_label)

    m.pred_n_series = pred.n_series
    m.pred_n_points = pred.n_points
    m.series_count_correct = gold.n_series == pred.n_series

    pairs, unmatched_gold, _ = match_series(gold.series, pred.series)

    name_hits = sum(1 for i, j in pairs if text_correct(gold.series[i].name, pred.series[j].name))
    m.series_name_accuracy = name_hits / gold.n_series if gold.n_series else 0.0

    errors: list[float] = []
    within_counts = {t: 0 for t in tolerances}
    per_series_counts_ok = True

    for gi, pj in pairs:
        g_series, p_series = gold.series[gi], pred.series[pj]
        pt_pairs, missing, spurious, fallback = match_points(g_series.data, p_series.data)

        m.n_matched_points += len(pt_pairs)
        m.n_missing_points += missing
        m.n_spurious_points += spurious
        m.used_positional_fallback |= fallback
        per_series_counts_ok &= len(g_series.data) == len(p_series.data)

        factor = detect_scale_error(pt_pairs)
        if factor is not None:
            m.scale_errors.append(factor)

        for g_pt, p_pt in pt_pairs:
            err = relative_error(g_pt.y, p_pt.y)
            if math.isfinite(err):
                errors.append(err)
            for tol in tolerances:
                if err <= tol:
                    within_counts[tol] += 1

    # Gold points in series the model never produced are missing, not absent
    # from the problem.
    m.n_missing_points += sum(len(gold.series[i].data) for i in unmatched_gold)

    if m.n_matched_points:
        m.value_accuracy = {t: within_counts[t] / m.n_matched_points for t in tolerances}
    if errors:
        m.mape = 100.0 * mean(errors)
        m.median_ape = 100.0 * _median(errors)

    m.point_count_correct = per_series_counts_ok and m.series_count_correct

    # Only fields this sample actually carries ground truth for can veto the
    # structural verdict.
    checks = {
        "chart_type": m.chart_type_correct,
        "title": m.title_correct,
        "x_label": m.x_label_correct,
        "y_label": m.y_label_correct,
        "series_names": math.isclose(m.series_name_accuracy, 1.0),
    }
    m.exact_structural_match = bool(
        m.series_count_correct
        and m.point_count_correct
        and all(ok for field_name, ok in checks.items() if field_name in fields)
    )
    return m


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _series_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


def _summarise(samples: list[SampleMetrics], tolerances: tuple[float, ...]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0}

    parsed = [s for s in samples if s.valid_json]

    def avg(fn, over=None) -> float:
        rows = samples if over is None else over
        return mean([fn(s) for s in rows]) if rows else 0.0

    def avg_field(field_name: str, fn) -> float | None:
        """Average over only the samples that carry ground truth for this field.

        Returns None when no sample does, so the report shows "n/a" rather than
        a zero that looks like a model failure.
        """
        rows = [s for s in samples if field_name in s.scored_fields]
        return mean([fn(s) for s in rows]) if rows else None

    # Value accuracy is micro-averaged over points, not macro-averaged over
    # charts: a 40-point chart should not weigh the same as a 3-point one.
    total_matched = sum(s.n_matched_points for s in parsed)
    value_acc = {}
    for tol in tolerances:
        hits = sum(s.value_accuracy.get(tol, 0.0) * s.n_matched_points for s in parsed)
        value_acc[f"{tol:.0%}"] = hits / total_matched if total_matched else 0.0

    mapes = [s.mape for s in parsed if s.mape is not None]
    median_apes = [s.median_ape for s in parsed if s.median_ape is not None]

    # Two different failures, reported separately:
    #   schema_conformance_rate -- matched the schema with no key substitutions
    #   valid_json_rate         -- usable after well-known key aliases
    # The gap between them is pure formatting deviation, which constrained
    # decoding is supposed to close (Phase 3, B8).
    strict = [s for s in parsed if not s.repairs]

    return {
        "n": n,
        "valid_json_rate": len(parsed) / n,
        "schema_conformance_rate": len(strict) / n,
        "n_repaired": len(parsed) - len(strict),
        "chart_type_accuracy": avg_field("chart_type", lambda s: s.chart_type_correct),
        "title_accuracy": avg_field("title", lambda s: s.title_correct),
        "x_label_accuracy": avg_field("x_label", lambda s: s.x_label_correct),
        "y_label_accuracy": avg_field("y_label", lambda s: s.y_label_correct),
        "series_name_accuracy": avg_field("series_names", lambda s: s.series_name_accuracy),
        "n_fully_annotated": sum(1 for s in samples if s.scored_fields == ALL_FIELDS),
        "series_count_accuracy": avg(lambda s: s.series_count_correct),
        "point_count_accuracy": avg(lambda s: s.point_count_correct),
        "exact_structural_match": avg(lambda s: s.exact_structural_match),
        "value_accuracy": value_acc,
        "mape": mean(mapes) if mapes else None,
        "median_ape": _median(median_apes),
        "point_recall": avg(lambda s: s.point_recall),
        "n_matched_points": total_matched,
        "n_missing_points": sum(s.n_missing_points for s in parsed),
        "n_spurious_points": sum(s.n_spurious_points for s in parsed),
        "n_positional_fallback": sum(1 for s in parsed if s.used_positional_fallback),
        "n_scale_errors": sum(1 for s in parsed if s.scale_errors),
    }


def aggregate(samples: list[SampleMetrics], tolerances: tuple[float, ...] = TOLERANCES) -> dict:
    """Dataset-level metrics plus breakdowns by chart type and series count."""
    overall = _summarise(samples, tolerances)

    by_type: dict[str, dict] = {}
    for ct in sorted({s.gold_chart_type for s in samples if s.gold_chart_type}):
        by_type[ct] = _summarise([s for s in samples if s.gold_chart_type == ct], tolerances)

    by_series: dict[str, dict] = {}
    for bucket in ("1", "2-3", "4-5", "6+"):
        rows = [s for s in samples if _series_bucket(s.gold_n_series) == bucket]
        if rows:
            by_series[bucket] = _summarise(rows, tolerances)

    return {
        "overall": overall,
        "by_chart_type": by_type,
        "by_series_count": by_series,
        "parse_errors": sorted(
            {s.parse_error for s in samples if s.parse_error is not None}
        ),
        "repairs": _repair_counts(samples),
    }


def _repair_counts(samples: list[SampleMetrics]) -> dict[str, int]:
    """How often each key substitution was needed, most common first."""
    counts: dict[str, int] = {}
    for s in samples:
        for r in set(s.repairs):  # once per sample, not once per point
            counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
