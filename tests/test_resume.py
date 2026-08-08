"""Resume semantics for the prediction cache.

Both directions have bitten this project already: `--resume` once treated 150
failed generations as complete, and the naive fix would have dropped failures
out of the scored denominator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.run_eval import load_predictions  # noqa: E402


def write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def ok(sid: str, raw: str = "{}") -> dict:
    return {"id": sid, "raw": raw, "n_tokens": 3, "seconds": 1.0, "gen_error": None}


def bad(sid: str, err: str = "AttributeError: boom") -> dict:
    return {"id": sid, "raw": "", "n_tokens": 0, "seconds": 0.0, "gen_error": err}


def test_missing_file_is_empty(tmp_path):
    assert load_predictions(tmp_path / "nope.jsonl") == {}


def test_resume_retries_failed_rows(tmp_path):
    p = write(tmp_path, [ok("a"), bad("b")])
    assert set(load_predictions(p, drop_errors=True)) == {"a"}


def test_scoring_keeps_failed_rows(tmp_path):
    """A generation failure must stay in the denominator."""
    p = write(tmp_path, [ok("a"), bad("b")])
    preds = load_predictions(p)
    assert set(preds) == {"a", "b"}
    assert preds["b"]["gen_error"]


def test_retry_row_supersedes_the_failure(tmp_path):
    """Append-only file: the later row wins for both callers."""
    p = write(tmp_path, [bad("a"), ok("a", raw='{"x":1}')])
    for kwargs in ({}, {"drop_errors": True}):
        preds = load_predictions(p, **kwargs)
        assert preds["a"]["gen_error"] is None
        assert preds["a"]["raw"] == '{"x":1}'


def test_failure_after_success_reopens_the_sample(tmp_path):
    """A later failure invalidates the earlier success rather than being ignored."""
    p = write(tmp_path, [ok("a"), bad("a")])
    assert "a" not in load_predictions(p, drop_errors=True)
    assert load_predictions(p)["a"]["gen_error"]


def test_all_failed_still_scores_as_n_rows(tmp_path):
    """The 150/150 wipeout must not shrink to n=0."""
    p = write(tmp_path, [bad(f"s{i}") for i in range(150)])
    assert len(load_predictions(p)) == 150
    assert load_predictions(p, drop_errors=True) == {}


def test_blank_lines_tolerated(tmp_path):
    p = tmp_path / "preds.jsonl"
    p.write_text(json.dumps(ok("a")) + "\n\n" + json.dumps(ok("b")) + "\n", encoding="utf-8")
    assert set(load_predictions(p)) == {"a", "b"}
