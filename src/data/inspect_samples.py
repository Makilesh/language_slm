"""Dump fully-rendered training samples for manual inspection.

Phase 2 acceptance asks for five rendered samples to be eyeballed. This writes
the literal strings the model trains on — not a summary, not a reconstruction —
with the masked span marked, plus a side-by-side HTML view pairing each sample
with its actual chart image.

It also runs `verify_template_matches_inference` on every dumped sample, so the
inspection artifact cannot itself be produced from a mismatched template.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from pathlib import Path

from PIL import Image

from src.data.format import render_text, verify_template_matches_inference

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def pick(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Choose a spread rather than the first n, which are all one source.

    Deliberately biased toward variety: a degraded sample, a real sample, and a
    multi-series synthetic one are each worth more to look at than three more
    clean bar charts.
    """
    rng = random.Random(seed)
    buckets = {
        "degraded": [r for r in rows if r.get("degradations")],
        "real": [r for r in rows if r.get("source") == "real"],
        "multi": [
            r for r in rows
            if (r.get("properties") or {}).get("n_series", 0) and
            (r.get("properties") or {}).get("n_series", 0) >= 3
        ],
        "any": rows,
    }
    out, seen = [], set()
    for key in ("degraded", "real", "multi", "any", "any"):
        pool = [r for r in buckets[key] if r["id"] not in seen] or [
            r for r in rows if r["id"] not in seen
        ]
        if not pool:
            break
        chosen = rng.choice(pool)
        out.append(chosen)
        seen.add(chosen["id"])
        if len(out) >= n:
            break
    return out


def data_uri(path: Path, max_edge: int = 448) -> str:
    """Embed the image at the resolution the model will actually see it.

    Showing a 1800px render would let a reviewer sign off on legibility the
    model never gets. 448 is the training resolution
    (configs/RESOLUTION_POLICY.md).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max_edge / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(samples: list[tuple[dict, str, str]], out: Path) -> Path:
    parts = [
        "<style>",
        "body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:2rem;",
        "background:#fbfbfd;color:#111}",
        "@media(prefers-color-scheme:dark){body{background:#14141a;color:#e8e8ee}",
        ".card{background:#1d1d26!important;border-color:#33333f!important}",
        ".masked{background:#2a2118!important}.trained{background:#152a1a!important}}",
        ".card{background:#fff;border:1px solid #e2e2ea;border-radius:10px;",
        "padding:1rem;margin-bottom:2rem}",
        ".row{display:flex;gap:1.25rem;flex-wrap:wrap}.col{flex:1;min-width:320px}",
        "img{max-width:100%;border:1px solid #ccc;border-radius:6px}",
        "pre{white-space:pre-wrap;word-break:break-word;padding:.7rem;border-radius:6px;",
        "margin:.3rem 0;overflow-x:auto}",
        ".masked{background:#fdf3e7}.trained{background:#e9f7ec}",
        "h2{font-size:15px;margin:.2rem 0 .8rem}",
        ".tag{display:inline-block;background:#ececf4;border-radius:4px;padding:1px 7px;",
        "margin-right:5px;font-size:11px}",
        "@media(prefers-color-scheme:dark){.tag{background:#2c2c38}}",
        "</style>",
        "<h1>Training samples — exactly what the model sees</h1>",
        "<p>Image shown at <b>448 px</b>, the training resolution. "
        "Orange = masked to -100. Green = the only span contributing loss.</p>",
    ]
    for i, (s, prompt_text, answer) in enumerate(samples, 1):
        tags = "".join(
            f'<span class="tag">{html.escape(str(t))}</span>'
            for t in [
                s["id"],
                s.get("source", "?"),
                *(s.get("degradations") or ["clean"]),
            ]
        )
        parts += [
            '<div class="card">',
            f"<h2>Sample {i} {tags}</h2>",
            '<div class="row">',
            f'<div class="col"><img src="{data_uri(Path(s["image_root"]) / s["image"])}"></div>',
            '<div class="col">',
            "<b>MASKED (labels = -100)</b>",
            f'<pre class="masked">{html.escape(prompt_text)}</pre>',
            "<b>TRAINED ON</b>",
            f'<pre class="trained">{html.escape(answer)}</pre>',
            "</div></div></div>",
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--html", type=Path, default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.dataset.open(encoding="utf-8") if x.strip()]
    chosen = pick(rows, args.n, args.seed)

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)

    rendered = []
    for s in chosen:
        # The inspection artifact must not be able to show a template the
        # trainer would not use.
        verify_template_matches_inference(proc, s, s["prompt_style"])
        prompt_text, full_text = render_text(proc, s)
        answer = full_text[len(prompt_text):]
        rendered.append((s, prompt_text, answer))

    chunks = []
    for i, (s, prompt_text, answer) in enumerate(rendered, 1):
        chunks.append(
            f"{'=' * 78}\nSAMPLE {i}  id={s['id']}  source={s.get('source')}  "
            f"degradations={s.get('degradations')}\n"
            f"image: {Path(s['image_root']) / s['image']}\n{'=' * 78}\n\n"
            f"--- MASKED (labels = -100) ---\n{prompt_text}\n"
            f"--- TRAINED ON (labels = token ids) ---\n{answer}\n"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chunks), encoding="utf-8")
    print(f"template check passed on {len(rendered)} samples")
    print(f"wrote {args.out}")

    html_out = args.html or args.out.with_suffix(".html")
    write_html(rendered, html_out)
    print(f"wrote {html_out}")


if __name__ == "__main__":
    main()
