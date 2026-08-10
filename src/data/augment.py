"""Realism degradation for a subset of training charts.

Models trained only on clean matplotlib renders fail on the thing people
actually upload to a demo: a phone photo of a monitor, a JPEG that has been
through three chat apps, a cropped screenshot with browser chrome still
attached. The renders in `synth.py` are pristine in a way no real input is.

Design decision worth stating: degraded images are written as **separate files**
alongside the clean ones rather than replacing them. That costs disk but makes
the B7 ablation (degradation on vs off) a manifest-level switch instead of a
re-render, so both arms train on pixel-identical charts differing only in
degradation. Re-rendering would confound the ablation with generator RNG.

Every degradation is recorded per-sample in the manifest, so Phase 4 can ask
"does JPEG hurt more than blur?" as a groupby rather than a new experiment.

The gold data is never touched. Degradation changes how the chart *looks*, not
what it says, so a correct reading of a degraded chart is the same reading.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

# Kept mild on purpose. The goal is "this went through a screenshot and a chat
# app", not "this is unreadable" -- a chart whose values a human cannot recover
# is a mislabelled training sample, because the gold still asserts them.
JPEG_QUALITY = (30, 75)
ROTATION_DEG = (-1.5, 1.5)
BLUR_RADIUS = (0.3, 1.0)
RESCALE_FACTOR = (0.6, 0.85)
BRIGHTNESS = (0.85, 1.15)
CONTRAST = (0.85, 1.15)
BORDER_PX = (8, 28)

# Screenshot chrome: a light grey band on top like a window title bar, plus a
# thin border. Crude, but it reproduces the framing artifact that matters --
# the chart no longer starts at pixel zero.
CHROME_FILL = (240, 240, 242)
CHROME_LINE = (200, 200, 205)


def jpeg_recompress(img: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=rng.randint(*JPEG_QUALITY))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(*ROTATION_DEG)
    # expand=True so corners are not clipped; white fill matches chart background.
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(*BLUR_RADIUS)))


def rescale(img: Image.Image, rng: random.Random) -> Image.Image:
    """Downscale then back up, to reproduce resampling mush.

    This is the one that most resembles a chart pasted into a document and
    scaled to fit, which is a very common real input.
    """
    f = rng.uniform(*RESCALE_FACTOR)
    small = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.BILINEAR)
    return small.resize(img.size, Image.BICUBIC)


def photometric(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(*BRIGHTNESS))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(*CONTRAST))


def screenshot_border(img: Image.Image, rng: random.Random) -> Image.Image:
    pad = rng.randint(*BORDER_PX)
    chrome = rng.random() < 0.5
    top = pad + (rng.randint(18, 34) if chrome else 0)

    canvas = Image.new("RGB", (img.width + 2 * pad, img.height + top + pad), CHROME_FILL)
    if chrome:
        # thin separator under the fake title bar
        for x in range(canvas.width):
            canvas.putpixel((x, top - 1), CHROME_LINE)
    canvas.paste(img, (pad, top))
    return canvas


# Order matters: geometry first, then optics, then compression -- the same
# order a real screenshot pipeline applies them. JPEG last, because compressing
# and *then* blurring would hide exactly the artifact we are trying to teach.
PIPELINE = [
    ("rescale", rescale, 0.45),
    ("rotate", rotate, 0.35),
    ("border", screenshot_border, 0.40),
    ("blur", blur, 0.40),
    ("photometric", photometric, 0.35),
    ("jpeg", jpeg_recompress, 0.65),
]


def degrade(img: Image.Image, rng: random.Random) -> tuple[Image.Image, list[str]]:
    """Apply a random subset of degradations. Returns (image, names applied).

    At least one is always applied -- a "degraded" sample that came out
    pristine would mislabel the B7 ablation.
    """
    applied: list[str] = []
    for name, fn, p in PIPELINE:
        if rng.random() < p:
            img = fn(img, rng)
            applied.append(name)

    if not applied:
        img = jpeg_recompress(img, rng)
        applied.append("jpeg")
    return img, applied


def build(
    manifest: Path,
    out_dir: Path,
    rate: float,
    seed: int,
    suffix: str = "_deg",
) -> Path:
    """Write degraded variants for `rate` of the manifest's samples.

    Emits a new manifest where degraded samples point at the new image and
    carry a `degradations` list; untouched samples are copied through
    unchanged with `degradations: []`.
    """
    manifest = Path(manifest)
    src_dir = manifest.parent
    out_dir = Path(out_dir)
    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in manifest.open(encoding="utf-8") if x.strip()]
    rng = random.Random(seed)

    # Choose the degraded subset up front from a shuffled copy, so the count is
    # exact rather than binomial around `rate`.
    n_deg = round(len(rows) * rate)
    chosen = set(rng.sample([r["id"] for r in rows], n_deg))

    out_manifest = out_dir / manifest.name
    n_written = 0
    with out_manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            row = dict(row)
            if row["id"] in chosen:
                with Image.open(src_dir / row["image"]) as im:
                    degraded, applied = degrade(im.convert("RGB"), rng)
                rel = f"images/{row['id']}{suffix}.png"
                degraded.save(out_dir / rel)
                row["image"] = rel
                row["degradations"] = applied
                n_written += 1
            else:
                # Point at the original, via a path relative to the new manifest.
                row["image"] = _relative(src_dir / row["image"], out_dir)
                row["degradations"] = []
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"degraded {n_written}/{len(rows)} ({n_written / max(1, len(rows)):.1%})")
    print(f"wrote {out_manifest}")
    return out_manifest


def _relative(target: Path, start: Path) -> str:
    """Path from `start` to `target`, POSIX-style, tolerating different drives."""
    import os

    try:
        return os.path.relpath(target, start).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return str(target.resolve()).replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build(args.manifest, args.out, args.rate, args.seed)


if __name__ == "__main__":
    main()
