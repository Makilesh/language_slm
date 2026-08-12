# B1 — the flagship ablation: which part of the model to train.
#
#   (a) LLM decoder only, vision frozen      <- the recommended default
#   (b) decoder + vision-language projector
#   (c) decoder + projector + vision LoRA
#
# The three configs differ in exactly one field (`train_mode`), enforced by
# tests/test_train_modes.py::test_b1_configs_differ_only_in_the_ablated_variable.
#
# Every arm reports accuracy AND peak VRAM AND the general-capability delta.
# A run that gains on charts while losing on DocVQA is not a success, so the
# forgetting probe is part of the loop, not an afterthought.
#
# Base model reference points, for the deltas:
#   value@5%   23.9%   (results/baselines.md, arm C)
#   DocVQA     ANLS 0.9237 / EM 0.86   (results/scores/docvqa_base.json)

$ErrorActionPreference = "Stop"
Set-Location "D:\GEN AI\language_slm"
$env:PYTHONPATH = "D:\GEN AI\language_slm"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"

$py = ".\.venv\Scripts\python.exe"

# Phase 1 found max_new_tokens=1024 truncates 6/150 dense charts. Evaluate the
# fine-tunes at 2048 so the ceiling is the model, not the budget.
$maxNewTokens = 2048
$evalManifest = "data/eval/synth/synth.jsonl"

$arms = @(
    @{ name = "b1a_decoder";                  config = "configs/b1a_decoder.yaml" },
    @{ name = "b1b_decoder_projector";        config = "configs/b1b_decoder_projector.yaml" },
    @{ name = "b1c_decoder_projector_vision"; config = "configs/b1c_decoder_projector_vision.yaml" }
)

foreach ($arm in $arms) {
    $name = $arm.name
    $adapter = "outputs/$name/adapter"

    Write-Output "`n$('#' * 20) TRAIN $name $('#' * 20)"
    if (Test-Path "$adapter/adapter_model.safetensors") {
        Write-Output "  adapter exists, skipping training"
    } else {
        & $py -m src.train.sft --config $arm.config
        if ($LASTEXITCODE -ne 0) { throw "$name training failed" }
    }

    Write-Output "`n$('#' * 20) EVAL $name $('#' * 20)"
    & $py -m src.eval.run_eval `
        --manifest $evalManifest `
        --adapter $adapter `
        --out "results/preds/$name.jsonl" `
        --report "results/scores/$name.json" `
        --prompt engineered --long-edge 448 `
        --max-new-tokens $maxNewTokens --resume
    if ($LASTEXITCODE -ne 0) { throw "$name eval failed" }

    Write-Output "`n$('#' * 20) FORGETTING PROBE $name $('#' * 20)"
    & $py -m src.eval.forgetting_probe `
        --adapter $adapter `
        --out "results/preds/docvqa_$name.jsonl" `
        --report "results/scores/docvqa_$name.json"
    if ($LASTEXITCODE -ne 0) { throw "$name forgetting probe failed" }
}

Write-Output "`n$('#' * 20) REPORT $('#' * 20)"
& $py -m src.eval.report `
    --run "base (prompted)=results/scores/base_constrained.json" `
    --run "B1a decoder only=results/scores/b1a_decoder.json" `
    --run "B1b + projector=results/scores/b1b_decoder_projector.json" `
    --run "B1c + vision LoRA=results/scores/b1c_decoder_projector_vision.json" `
    --title "B1 - what to train" `
    --out results/ablations.md

Write-Output "`nB1 complete"
