"""Dump predictions as a self-contained HTML page for human inspection.

Phase 1 acceptance requires eyeballing real predictions, and a JSONL file is not
something anyone eyeballs honestly. This renders each sample as the chart image
beside its gold and predicted JSON, with the per-field verdict, so a wrong
extraction is obvious at a glance instead of requiring a diff.

Images are embedded as data URIs so the page is one file that opens anywhere.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

from src.eval.metrics import score_sample
from src.eval.schema import ChartData, parse_prediction


def thumb_data_uri(path: Path, max_edge: int = 720) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_edge / max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def verdict_chips(m) -> str:
    chips = []

    def chip(label: str, ok: bool | None) -> str:
        if ok is None:
            return f'<span class="chip na">{label}: n/a</span>'
        cls = "ok" if ok else "bad"
        return f'<span class="chip {cls}">{label}</span>'

    chips.append(chip("JSON", m.valid_json))
    for name, attr in (
        ("chart type", "chart_type_correct"),
        ("title", "title_correct"),
        ("x label", "x_label_correct"),
        ("y label", "y_label_correct"),
    ):
        key = attr.replace("_correct", "")
        chips.append(chip(name, getattr(m, attr) if key in m.scored_fields else None))

    chips.append(chip(f"series {m.pred_n_series}/{m.gold_n_series}", m.series_count_correct))
    chips.append(
        f'<span class="chip {"ok" if m.value_accuracy.get(0.05, 0) >= 0.9 else "bad"}">'
        f"value@5%: {m.value_accuracy.get(0.05, 0.0):.0%}</span>"
    )
    chips.append(
        f'<span class="chip neutral">matched {m.n_matched_points}/{m.gold_n_points}'
        f" · missing {m.n_missing_points} · extra {m.n_spurious_points}</span>"
    )
    if m.scale_errors:
        chips.append(f'<span class="chip bad">scale x{m.scale_errors[0]:g}</span>')
    if m.used_positional_fallback:
        chips.append('<span class="chip warn">positional match</span>')
    if m.repairs:
        shown = ", ".join(sorted(set(m.repairs))[:3])
        chips.append(f'<span class="chip warn">repaired: {html.escape(shown)}</span>')
    return "".join(chips)


CSS = """
:root { color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --muted:#666;
        --card:#f7f7f8; --line:#e3e3e6; --ok:#0a7c42; --bad:#b3261e; --warn:#8a6100; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#e8e8ea; --muted:#a0a0a8; --card:#1e1e22;
          --line:#33333a; --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; }
}
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:var(--muted); margin-bottom:22px; }
.sample { border:1px solid var(--line); border-radius:10px; margin-bottom:20px;
          overflow:hidden; background:var(--card); }
.hdr { padding:10px 14px; border-bottom:1px solid var(--line);
       display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.sid { font-weight:600; margin-right:8px; }
.body { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr) minmax(0,1fr); gap:0; }
@media (max-width:900px) { .body { grid-template-columns:1fr; } }
.cell { padding:14px; border-right:1px solid var(--line); min-width:0; }
.cell:last-child { border-right:none; }
.cell h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase;
           letter-spacing:.06em; color:var(--muted); }
img { max-width:100%; height:auto; border-radius:6px; display:block; }
pre { margin:0; padding:10px; background:var(--bg); border:1px solid var(--line);
      border-radius:6px; overflow-x:auto; font-size:12px; line-height:1.45;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.chip { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
        border:1px solid var(--line); }
.chip.ok { color:var(--ok); border-color:var(--ok); }
.chip.bad { color:var(--bad); border-color:var(--bad); }
.chip.warn { color:var(--warn); border-color:var(--warn); }
.chip.na, .chip.neutral { color:var(--muted); }
.err { color:var(--bad); font-size:13px; }
"""


def build(manifest: Path, preds: Path, out: Path, n: int, only_failures: bool) -> Path:
    gold_rows = {
        r["id"]: r for r in (json.loads(x) for x in manifest.open(encoding="utf-8") if x.strip())
    }
    pred_rows = {
        r["id"]: r for r in (json.loads(x) for x in preds.open(encoding="utf-8") if x.strip())
    }

    cards = []
    for sid, prow in pred_rows.items():
        if sid not in gold_rows:
            continue
        grow = gold_rows[sid]
        gold = ChartData.model_validate(grow["gold"])
        parsed, err, repairs = parse_prediction(prow["raw"])
        m = score_sample(
            gold, parsed, sid, parse_error=err,
            scored_fields=set(grow["scored_fields"]) if "scored_fields" in grow else None,
            repairs=repairs,
        )
        if only_failures and m.value_accuracy.get(0.05, 0.0) >= 0.9 and m.valid_json:
            continue

        img = thumb_data_uri(manifest.parent / grow["image"])
        pred_json = parsed.to_json(indent=2) if parsed else (prow["raw"] or "")
        err_html = f'<div class="err">{html.escape(err)}</div>' if err else ""

        cards.append(f"""
<div class="sample">
  <div class="hdr"><span class="sid">{html.escape(sid)}</span>{verdict_chips(m)}</div>
  <div class="body">
    <div class="cell"><h3>chart</h3><img src="{img}" alt="chart {html.escape(sid)}"></div>
    <div class="cell"><h3>gold</h3><pre>{html.escape(gold.to_json(indent=2))}</pre></div>
    <div class="cell"><h3>prediction</h3>{err_html}
      <pre>{html.escape(pred_json)}</pre></div>
  </div>
</div>""")
        if len(cards) >= n:
            break

    title = f"Predictions — {preds.stem}"
    page = (
        f"<title>{html.escape(title)}</title><style>{CSS}</style>"
        f"<h1>{html.escape(title)}</h1>"
        f'<div class="sub">{len(cards)} samples from <code>{html.escape(str(preds))}</code>'
        f" scored against <code>{html.escape(str(manifest))}</code>."
        f'{" Showing failures only." if only_failures else ""}</div>'
        + "".join(cards)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  ({len(cards)} samples, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only-failures", action="store_true")
    args = ap.parse_args()
    build(args.manifest, args.preds, args.out, args.n, args.only_failures)


if __name__ == "__main__":
    main()
