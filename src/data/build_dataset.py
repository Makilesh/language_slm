"""Assemble the training set: mix sources, check contamination, report stats.

Three jobs, in order of how badly getting them wrong would hurt:

1. **Contamination.** Training on the eval set inflates every number in the
   repo and is undetectable from the results alone. Checked three ways, because
   the synthetic and real subsets leak differently:

   * identical image bytes — catches a file copied into both splits
   * exact gold signature — catches the same *data* rendered twice, which is
     the realistic synthetic collision: independent RNG draws can land on the
     same title/labels/series/values
   * near-duplicate signature — values rounded harder, so a chart differing
     only in the last digit still trips it

   Random generation genuinely collides. With ~9 title strings x ~14 x-labels
   x a handful of category sets, the birthday problem is not on our side at
   10k samples, so this is a real check and not ceremony.

2. **Mixing.** Synthetic charts have exact labels and controllable difficulty;
   real charts have messiness that no generator reproduces. The default is 70/30
   synthetic/real, and the ratio is a CLI flag so B4 can ablate it without
   editing code.

3. **Stats and token budget.** Image tokens are computed analytically from the
   Phase 0 geometry (one token per 32x32 px region after rounding each side up
   to a multiple of 32) rather than by running the processor over 10k images.
   Same number, seconds instead of minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from src.data.format import build_sample, stats, target_json

# Phase 0: Qwen3-VL patch_size 16 x spatial_merge 2 => one image token per
# 32x32 pixel region. setup/VRAM_BUDGET.md has the measured confirmation.
PX_PER_TOKEN = 32


# --------------------------------------------------------------------------- #
# contamination
# --------------------------------------------------------------------------- #


def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def gold_signature(gold: dict, sig: int = 4) -> str:
    """Stable hash of a chart's *content*, independent of rendering.

    Series are sorted by name so a permutation is not treated as a new chart —
    the schema does not give series order any meaning, and neither does the
    metric.
    """
    series = sorted(
        (
            str(s.get("name")),
            tuple((str(p.get("x")), _round_sig(float(p.get("y", 0.0)), sig)) for p in s["data"]),
        )
        for s in gold.get("series", [])
    )
    payload = json.dumps(
        [
            gold.get("chart_type"),
            gold.get("title"),
            gold.get("x_label"),
            gold.get("y_label"),
            series,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def image_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def contamination_report(train_rows: list[dict], eval_manifests: list[Path]) -> dict:
    """Compare the assembled training rows against every eval manifest."""
    eval_exact: dict[str, str] = {}
    eval_near: dict[str, str] = {}
    eval_imgs: dict[str, str] = {}

    for man in eval_manifests:
        base = Path(man).parent
        for line in Path(man).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            eval_exact.setdefault(gold_signature(r["gold"], sig=4), f"{man.name}:{r['id']}")
            eval_near.setdefault(gold_signature(r["gold"], sig=2), f"{man.name}:{r['id']}")
            h = image_hash(base / r["image"])
            if h:
                eval_imgs.setdefault(h, f"{man.name}:{r['id']}")

    hits = {"image_bytes": [], "exact_gold": [], "near_duplicate": []}
    for row in train_rows:
        src = Path(row["_manifest_dir"]) / row["image"]
        h = image_hash(src)
        if h and h in eval_imgs:
            hits["image_bytes"].append({"train": row["id"], "eval": eval_imgs[h]})

        s4 = gold_signature(row["gold"], sig=4)
        if s4 in eval_exact:
            hits["exact_gold"].append({"train": row["id"], "eval": eval_exact[s4]})
            continue  # already the stronger finding; do not double-count as near
        s2 = gold_signature(row["gold"], sig=2)
        if s2 in eval_near:
            hits["near_duplicate"].append({"train": row["id"], "eval": eval_near[s2]})

    # Internal duplication is not contamination, but a training set that is 5%
    # the same chart is wasting capacity, so it is worth surfacing.
    internal = Counter(gold_signature(r["gold"], sig=4) for r in train_rows)
    dup_groups = {k: v for k, v in internal.items() if v > 1}

    return {
        "n_train": len(train_rows),
        "n_eval_signatures": len(eval_exact),
        "hits": {k: len(v) for k, v in hits.items()},
        "total_contaminated": sum(len(v) for v in hits.values()),
        "examples": {k: v[:10] for k, v in hits.items()},
        "internal_duplicate_groups": len(dup_groups),
        "internal_duplicate_rows": sum(dup_groups.values()) - len(dup_groups),
    }


# --------------------------------------------------------------------------- #
# token budget
# --------------------------------------------------------------------------- #


def image_tokens_for(width: int, height: int, long_edge: int) -> int:
    """Image tokens after resizing the long edge and snapping to the token grid."""
    scale = long_edge / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    gw = max(1, round(w / PX_PER_TOKEN))
    gh = max(1, round(h / PX_PER_TOKEN))
    return gw * gh


def token_stats(rows: list[dict], long_edge: int, tokenizer=None) -> dict:
    """Distribution of image / target / total tokens across the corpus."""
    from PIL import Image

    img_toks, tgt_toks = [], []
    for row in rows:
        p = Path(row["_manifest_dir"]) / row["image"]
        try:
            with Image.open(p) as im:
                img_toks.append(image_tokens_for(im.width, im.height, long_edge))
        except OSError:
            continue
        if tokenizer is not None:
            tgt_toks.append(len(tokenizer(target_json(row["gold"]))["input_ids"]))

    def dist(v: list[int]) -> dict:
        if not v:
            return {}
        s = sorted(v)
        return {
            "min": s[0],
            "p50": s[len(s) // 2],
            "p90": s[int(0.9 * (len(s) - 1))],
            "p99": s[int(0.99 * (len(s) - 1))],
            "max": s[-1],
            "mean": round(sum(s) / len(s), 1),
        }

    out = {"long_edge": long_edge, "image_tokens": dist(img_toks)}
    if tgt_toks:
        out["target_tokens"] = dist(tgt_toks)
        combined = sorted(a + b for a, b in zip(img_toks, tgt_toks))
        out["image_plus_target"] = dist(combined)
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def eval_signatures(eval_manifests: list[Path]) -> tuple[set[str], set[str], set[str]]:
    """(exact gold, near gold, image byte) hashes for every eval sample."""
    exact, near, imgs = set(), set(), set()
    for man in eval_manifests:
        base = Path(man).parent
        for line in Path(man).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            exact.add(gold_signature(r["gold"], sig=4))
            near.add(gold_signature(r["gold"], sig=2))
            h = image_hash(base / r["image"])
            if h:
                imgs.add(h)
    return exact, near, imgs


def filter_contaminated(
    rows: list[dict], exact: set[str], near: set[str], imgs: set[str]
) -> tuple[list[dict], dict]:
    """Drop rows overlapping the eval set, before mixing rather than after.

    Filtering the *pools* keeps the requested count and mix ratio intact;
    filtering the selection would silently shrink the dataset and skew the
    ratio by however many rows happened to be contaminated.

    Id-based exclusion upstream (`real.py --exclude-manifest`) is not
    sufficient on its own: the real corpus stores some charts twice under
    consecutive sample ids, so the same chart reaches both splits with two
    different ids. Only a content hash catches that.
    """
    kept, dropped = [], {"image_bytes": 0, "exact_gold": 0, "near_duplicate": 0}
    for row in rows:
        h = image_hash(Path(row["_manifest_dir"]) / row["image"])
        if h and h in imgs:
            dropped["image_bytes"] += 1
            continue
        if gold_signature(row["gold"], sig=4) in exact:
            dropped["exact_gold"] += 1
            continue
        if gold_signature(row["gold"], sig=2) in near:
            dropped["near_duplicate"] += 1
            continue
        kept.append(row)
    return kept, dropped


def load_manifest(path: Path, source: str) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            r["source"] = source
            r["_manifest_dir"] = str(path.parent)
            rows.append(r)
    return rows


def mix(synth: list[dict], real: list[dict], ratio: float, n: int, rng: random.Random) -> list[dict]:
    """Sample `n` rows at the requested synthetic fraction.

    If a source cannot cover its quota the shortfall is taken from the other and
    reported, rather than silently returning a smaller or differently-mixed set.
    """
    want_s = round(n * ratio)
    want_r = n - want_s

    take_s = min(want_s, len(synth))
    take_r = min(want_r, len(real))

    if take_s < want_s:
        print(f"  WARNING: wanted {want_s} synthetic, have {len(synth)}")
    if take_r < want_r:
        print(f"  WARNING: wanted {want_r} real, have {len(real)}")

    short = (want_s - take_s) + (want_r - take_r)
    if short:
        extra_s = min(len(synth) - take_s, short)
        take_s += extra_s
        short -= extra_s
        take_r += min(len(real) - take_r, short)
        print(f"  backfilled shortfall -> synth={take_s}, real={take_r}")

    out = rng.sample(synth, take_s) + rng.sample(real, take_r)
    rng.shuffle(out)
    return out


def build(args) -> None:
    rng = random.Random(args.seed)

    synth = load_manifest(args.synth, "synth") if args.synth else []
    real = load_manifest(args.real, "real") if args.real else []
    print(f"available: {len(synth)} synthetic, {len(real)} real")

    eval_paths = [Path(p) for p in args.eval_manifest]
    print("\npre-filtering candidate pools against the eval set...")
    exact, near, imgs = eval_signatures(eval_paths)
    print(f"  eval signatures: {len(exact)} gold, {len(imgs)} images")

    synth, dropped_s = filter_contaminated(synth, exact, near, imgs)
    real, dropped_r = filter_contaminated(real, exact, near, imgs)
    for label, d in (("synthetic", dropped_s), ("real", dropped_r)):
        total = sum(d.values())
        print(f"  {label:10s} dropped {total}" + (f"  {d}" if total else ""))
    prefilter_dropped = {
        "synthetic": dropped_s,
        "real": dropped_r,
        "total": sum(dropped_s.values()) + sum(dropped_r.values()),
    }
    print(f"  clean pools: {len(synth)} synthetic, {len(real)} real")

    n = args.n or (len(synth) + len(real))
    rows = mix(synth, real, args.synth_ratio, n, rng)
    print(f"\nselected: {len(rows)} rows at synth_ratio={args.synth_ratio}")

    # Re-run the full check on the actual selection. After pre-filtering this
    # must come back clean; if it does not, the filter and the report disagree
    # and the dataset is not trustworthy.
    print("\nverifying selection...")
    contam = contamination_report(rows, eval_paths)
    contam["prefilter_dropped"] = prefilter_dropped
    for k, v in contam["hits"].items():
        print(f"  {k:16s} {v}")
    print(f"  internal dup rows {contam['internal_duplicate_rows']}")

    if contam["total_contaminated"] and not args.allow_contamination:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(contam, indent=2), encoding="utf-8")
        raise SystemExit(
            f"\nABORT: {contam['total_contaminated']} rows survived pre-filtering.\n"
            f"That is a bug in the filter, not a data problem -- both use the same\n"
            f"signatures, so they cannot legitimately disagree. Details in {args.report}."
        )

    print("\ntoken budget...")
    tok = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
    tstats = token_stats(rows, args.long_edge, tok)
    for key in ("image_tokens", "target_tokens", "image_plus_target"):
        if key in tstats:
            print(f"  {key:20s} {tstats[key]}")

    samples = [build_sample(r, args.prompt) for r in rows]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r, s in zip(rows, samples):
            # Keep the manifest dir so the collator can resolve images that live
            # under a different tree than the dataset file.
            s["image_root"] = r["_manifest_dir"]
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    dstats = stats(samples)
    report = {
        "config": {
            "synth_manifest": str(args.synth) if args.synth else None,
            "real_manifest": str(args.real) if args.real else None,
            "n_requested": args.n,
            "synth_ratio": args.synth_ratio,
            "seed": args.seed,
            "prompt": args.prompt,
            "long_edge": args.long_edge,
        },
        "stats": dstats,
        "tokens": tstats,
        "contamination": contam,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"n={dstats['n']}  synthetic={dstats['synthetic_fraction']:.1%}  "
          f"degraded={dstats['degraded_fraction']:.1%}")
    print(f"chart types: {dstats['by_chart_type']}")
    print(f"series counts: {dstats['by_series_count']}")
    print(f"wrote {out}")
    print(f"wrote {args.report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", type=Path, default=None)
    ap.add_argument("--real", type=Path, default=None)
    ap.add_argument("--eval-manifest", action="append", default=[], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--synth-ratio", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="engineered", choices=["minimal", "engineered"])
    ap.add_argument("--long-edge", type=int, default=448)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--allow-contamination", action="store_true")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
