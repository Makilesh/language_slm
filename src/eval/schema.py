"""The canonical chart-to-data output format.

One schema, versioned, used by every producer and consumer in the project:
synthetic data generation, training targets, constrained decoding at inference,
and metrics. If this changes, bump SCHEMA_VERSION and say so in the results
files, because scores computed under different schema versions are not
comparable.

Design notes that are not obvious:

* `x` is `str | float`. Categorical axes ("Q1 2023", "Denmark") and numeric axes
  (scatter, time series) are both real, and forcing one into the other loses
  information the metric needs. Metrics compare x values by normalised string
  when either side is a string.

* `y` is always `float`. A chart value that cannot be read as a number is not an
  extraction we can score, so producers must resolve it or omit the point.

* Nothing is `Optional` for convenience. `title`, `x_label`, `y_label` are
  nullable because charts genuinely lack them; `series` and `chart_type` are not,
  because a chart always has both.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    STACKED_BAR = "stacked_bar"


class Point(BaseModel):
    model_config = {"extra": "forbid"}

    x: str | float
    y: float

    @field_validator("y", mode="before")
    @classmethod
    def _reject_nonfinite(cls, v: Any) -> Any:
        # NaN/inf serialise to bare `NaN`/`Infinity`, which is invalid JSON and
        # would silently poison every downstream tolerance comparison.
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            raise ValueError("y must be finite")
        return v


class Series(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    data: list[Point]


class ChartData(BaseModel):
    """One chart's full underlying data."""

    model_config = {"extra": "forbid"}

    chart_type: ChartType
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    series: list[Series] = Field(default_factory=list)

    # -- convenience ------------------------------------------------------- #

    @property
    def n_series(self) -> int:
        return len(self.series)

    @property
    def n_points(self) -> int:
        return sum(len(s.data) for s in self.series)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ChartData":
        return cls.model_validate_json(text)


def json_schema() -> dict:
    """JSON Schema for constrained decoding.

    Fed to XGrammar/Outlines so malformed JSON is structurally impossible at
    inference, which is the whole point of standing rule: metrics should measure
    extraction quality, not JSON syntax.
    """
    return ChartData.model_json_schema()


# Key names models reach for instead of ours. Accepting these is not leniency
# for its own sake: the point of the metrics is extraction quality, and a model
# that reads every bar correctly but calls the list "points" has an extraction
# score of 100% and a formatting score of 0%. Conflating those into "0% valid
# JSON" measures the prompt, not the model. Every substitution is recorded so
# the formatting deviation stays visible.
SERIES_DATA_ALIASES = ("data", "points", "values", "series_data", "datapoints")
X_ALIASES = ("x", "label", "category", "name")
Y_ALIASES = ("y", "value", "val")
SERIES_ALIASES = ("series", "data", "datasets")


def _repair(obj: dict, repairs: list[str]) -> dict:
    """Rename well-known key variants onto the canonical schema, in place."""
    if not isinstance(obj, dict):
        return obj

    for alias in SERIES_ALIASES:
        if alias in obj and isinstance(obj[alias], list):
            if alias != "series":
                obj["series"] = obj.pop(alias)
                repairs.append(f"series<-{alias}")
            break

    for s in obj.get("series") or []:
        if not isinstance(s, dict):
            continue
        for alias in SERIES_DATA_ALIASES:
            if alias in s and isinstance(s[alias], list):
                if alias != "data":
                    s["data"] = s.pop(alias)
                    repairs.append(f"data<-{alias}")
                break
        for pt in s.get("data") or []:
            if not isinstance(pt, dict):
                continue
            for alias in X_ALIASES:
                if alias in pt:
                    if alias != "x":
                        pt["x"] = pt.pop(alias)
                        repairs.append(f"x<-{alias}")
                    break
            for alias in Y_ALIASES:
                if alias in pt:
                    if alias != "y":
                        pt["y"] = pt.pop(alias)
                        repairs.append(f"y<-{alias}")
                    break
            # Drop unknown extras rather than failing the whole chart on them.
            for extra in [k for k in pt if k not in ("x", "y")]:
                pt.pop(extra)
                repairs.append(f"dropped point.{extra}")
        for extra in [k for k in s if k not in ("name", "data")]:
            s.pop(extra)
            repairs.append(f"dropped series.{extra}")

    for extra in [
        k for k in obj if k not in ("chart_type", "title", "x_label", "y_label", "series")
    ]:
        obj.pop(extra)
        repairs.append(f"dropped {extra}")
    return obj


def parse_prediction(
    text: str, lenient: bool = True
) -> tuple["ChartData | None", str | None, list[str]]:
    """Parse model output, reporting why it failed rather than raising.

    Returns (parsed, error, repairs). `repairs` lists every key substitution
    that was needed; an empty list means the output matched the schema exactly.
    Aggregation reports both the strict conformance rate and the repaired rate,
    so leniency never hides a formatting failure — it just stops that failure
    from destroying the extraction measurement.

    `lenient=False` gives strict schema conformance, which is what the
    valid-JSON target in the brief refers to.
    """
    repairs: list[str] = []
    if text is None:
        return None, "empty output", repairs

    candidate = text.strip()
    if not candidate:
        return None, "empty output", repairs

    # Strip a ```json ... ``` fence if present.
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
        candidate = candidate.strip()
        repairs.append("stripped markdown fence")

    # Fall back to the outermost brace pair if there is surrounding prose.
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None, "no JSON object found", repairs
        candidate = candidate[start : end + 1]
        repairs.append("extracted from prose")

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}", repairs

    if lenient and isinstance(obj, dict):
        obj = _repair(obj, repairs)

    try:
        return ChartData.model_validate(obj), None, repairs
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return None, f"schema violation: {first}", repairs
