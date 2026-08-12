"""B1 train-mode targeting.

The ablation is only meaningful if each arm trains exactly what its name says.
The specific hazard: the vision blocks and the merger both expose `linear_fc1` /
`linear_fc2`, so a loosely-anchored pattern attaches LoRA to the vision tower
while the config claims the tower is frozen — producing three runs that differ
by less than their labels do, with no error anywhere.

These tests match the regexes against the real Qwen3-VL module names (captured
from a meta-device build, so no weights are loaded).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.sft import (  # noqa: E402
    TRAIN_MODES,
    Config,
    assert_mode_matches_params,
    load_config,
)

N_LAYERS, N_VISION_BLOCKS, N_DEEPSTACK = 36, 24, 3

MODULES: list[str] = (
    [
        f"model.language_model.layers.{i}.{sub}"
        for i in range(N_LAYERS)
        for sub in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    ]
    + [
        f"model.visual.blocks.{i}.{sub}"
        for i in range(N_VISION_BLOCKS)
        for sub in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
    ]
    + [f"model.visual.merger.linear_fc{j}" for j in (1, 2)]
    + [
        f"model.visual.deepstack_merger_list.{i}.linear_fc{j}"
        for i in range(N_DEEPSTACK)
        for j in (1, 2)
    ]
    + ["lm_head", "model.visual.patch_embed.proj"]
)


def matched(mode: str) -> list[str]:
    pat = re.compile(TRAIN_MODES[mode])
    return [m for m in MODULES if pat.search(m)]


def buckets(names: list[str]) -> dict[str, int]:
    out = {"vision_tower": 0, "projector": 0, "llm": 0, "other": 0}
    for n in names:
        if "visual.blocks" in n:
            out["vision_tower"] += 1
        elif "merger" in n:
            out["projector"] += 1
        elif "language_model" in n:
            out["llm"] += 1
        else:
            out["other"] += 1
    return out


# --------------------------------------------------------------------------- #
# targeting
# --------------------------------------------------------------------------- #


def test_decoder_only_touches_nothing_visual():
    b = buckets(matched("decoder"))
    assert b["llm"] == N_LAYERS * 7 == 252
    assert b["vision_tower"] == 0
    assert b["projector"] == 0
    assert b["other"] == 0


def test_decoder_projector_adds_mergers_but_not_the_tower():
    """The trap: merger and vision blocks share `linear_fc*` naming."""
    b = buckets(matched("decoder_projector"))
    assert b["llm"] == 252
    assert b["projector"] == 2 + N_DEEPSTACK * 2 == 8
    assert b["vision_tower"] == 0, "pattern leaked into the vision tower"


def test_full_mode_adds_the_vision_blocks():
    b = buckets(matched("decoder_projector_vision"))
    assert b["llm"] == 252
    assert b["projector"] == 8
    assert b["vision_tower"] == N_VISION_BLOCKS * 4 == 96


@pytest.mark.parametrize("mode", sorted(TRAIN_MODES))
def test_no_mode_touches_lm_head_or_patch_embed(mode):
    """lm_head is unquantized and huge; patch_embed is the raw pixel projection."""
    assert buckets(matched(mode))["other"] == 0


def test_modes_are_strictly_nested():
    """Each arm must be a superset of the previous, or the deltas are not attributable."""
    a, b, c = (set(matched(m)) for m in
               ("decoder", "decoder_projector", "decoder_projector_vision"))
    assert a < b < c


# --------------------------------------------------------------------------- #
# the runtime guard
# --------------------------------------------------------------------------- #


def test_guard_rejects_a_frozen_decoder():
    with pytest.raises(SystemExit, match="no decoder params"):
        assert_mode_matches_params("decoder", {"llm": 0, "projector": 0, "vision_tower": 0})


def test_guard_rejects_unrequested_vision_training():
    """Standing rule 2, enforced at runtime rather than trusted to the regex."""
    with pytest.raises(SystemExit, match="vision tower must be frozen"):
        assert_mode_matches_params("decoder", {"llm": 252, "projector": 0, "vision_tower": 96})


def test_guard_rejects_unrequested_projector_training():
    with pytest.raises(SystemExit, match="projector must be frozen"):
        assert_mode_matches_params("decoder", {"llm": 252, "projector": 8, "vision_tower": 0})


def test_guard_rejects_a_silently_empty_match():
    with pytest.raises(SystemExit, match="projector requested but nothing matched"):
        assert_mode_matches_params(
            "decoder_projector", {"llm": 252, "projector": 0, "vision_tower": 0}
        )


@pytest.mark.parametrize(
    "mode,b",
    [
        ("decoder", {"llm": 252, "projector": 0, "vision_tower": 0}),
        ("decoder_projector", {"llm": 252, "projector": 8, "vision_tower": 0}),
        ("decoder_projector_vision", {"llm": 252, "projector": 8, "vision_tower": 96}),
    ],
)
def test_guard_accepts_correct_configurations(mode, b):
    assert_mode_matches_params(mode, b)


# --------------------------------------------------------------------------- #
# configs on disk
# --------------------------------------------------------------------------- #

CONFIGS = {
    "decoder": "configs/b1a_decoder.yaml",
    "decoder_projector": "configs/b1b_decoder_projector.yaml",
    "decoder_projector_vision": "configs/b1c_decoder_projector_vision.yaml",
}
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,path", CONFIGS.items())
def test_config_declares_the_expected_mode(mode, path):
    assert yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))["train_mode"] == mode


def test_b1_configs_differ_only_in_the_ablated_variable():
    """One variable per experiment (standing rule 5), checked mechanically."""
    loaded = {
        m: yaml.safe_load((ROOT / p).read_text(encoding="utf-8")) for m, p in CONFIGS.items()
    }
    allowed = {"train_mode", "output_dir", "run_name"}
    base = loaded["decoder"]
    for mode, cfg in loaded.items():
        differing = {k for k in base if base[k] != cfg.get(k)}
        assert differing <= allowed, f"{mode} also differs in {differing - allowed}"


def test_configs_use_the_phase0_resolution_and_measured_seq_len():
    for path in CONFIGS.values():
        cfg = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        assert cfg["long_edge"] == 448, "only 448px trains on this card (Phase 0)"
        assert cfg["max_seq_len"] >= 1928, "must clear the measured max target length"


def test_config_hash_ignores_paths_but_tracks_the_ablation():
    a = Config(train_mode="decoder", output_dir="x", run_name="x")
    b = Config(train_mode="decoder", output_dir="y", run_name="y")
    c = Config(train_mode="decoder_projector", output_dir="x", run_name="x")
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_override_parsing_coerces_types(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lora_r: 32\nlr: 1.0e-4\nquantize_vision: false\n", encoding="utf-8")
    cfg = load_config(p, ["lora_r=64", "lr=2e-4", "quantize_vision=true"])
    assert cfg.lora_r == 64 and isinstance(cfg.lora_r, int)
    assert cfg.lr == pytest.approx(2e-4)
    assert cfg.quantize_vision is True


def test_override_on_a_none_default_is_typed(tmp_path):
    """`limit` defaults to None, so its type must be inferred from the value.

    Assigning the string "64" instead of int 64 fails much later, in a slice.
    """
    p = tmp_path / "c.yaml"
    p.write_text("lora_r: 32\n", encoding="utf-8")
    cfg = load_config(p, ["limit=64"])
    assert cfg.limit == 64 and isinstance(cfg.limit, int)

    assert load_config(p, ["limit=none"]).limit is None


def test_unknown_override_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("lora_r: 32\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown override"):
        load_config(p, ["loraa_r=64"])
