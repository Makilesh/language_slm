"""Prompts for chart-to-data extraction.

Two variants, because the brief wants the gap between them measured: a naive
instruction and a carefully engineered one. Reporting only the engineered
baseline overstates how much fine-tuning added; reporting only the naive one
overstates it further.

Keep these stable once baselines are run. Changing a prompt invalidates every
number produced with it.
"""

from __future__ import annotations

PROMPT_VERSION = "1.1"

# v1.1: the original said "a list of {x, y} points", which never named the key
# and led the model to emit "points" instead of "data" on 150/150 charts. That
# measured the prompt's ambiguity, not the model's extraction ability. Minimal
# means terse, not under-specified — the target schema still has to be stated
# unambiguously or the baseline is meaningless.
MINIMAL = (
    "Extract the underlying data from this chart as JSON with keys: "
    'chart_type, title, x_label, y_label, series. Each series is '
    '{"name": ..., "data": [{"x": ..., "y": ...}]}. Respond with JSON only.'
)

ENGINEERED = """\
You are a precise chart data extractor. Read the chart image and output the \
underlying data as a single JSON object. Output JSON only — no prose, no \
markdown fences, no explanation.

Schema:
{
  "chart_type": one of "bar", "line", "pie", "scatter", "stacked_bar",
  "title": the chart title as written, or null if absent,
  "x_label": the x-axis label as written, or null if absent,
  "y_label": the y-axis label as written, or null if absent,
  "series": [ { "name": series name from the legend, "data": [ {"x": ..., "y": number}, ... ] } ]
}

Rules:
- Include every series in the legend, and every point in each series.
- "x" is the category label as written on the axis (a string), or the numeric \
x value for scatter plots.
- "y" is always a number. Read it against the y-axis gridlines and tick values.
- Report values in the units shown on the axis. If the axis says "thousands", \
report 45 for a bar at 45, not 45000.
- For stacked bars, report each segment's own value, not the cumulative height.
- If the y-axis is logarithmic, read the value from the log scale.
- For a chart with one unlabelled series, use a short descriptive name.
- Do not invent points that are not plotted, and do not omit plotted points.
"""


def build_messages(prompt_style: str) -> list[dict]:
    """Chat messages with a single image placeholder.

    The image goes first: Qwen-VL's template expects the visual content before
    the instruction, and reversing them measurably degrades grounding.
    """
    text = {"minimal": MINIMAL, "engineered": ENGINEERED}[prompt_style]
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        }
    ]
