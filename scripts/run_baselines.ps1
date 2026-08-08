# Phase 1 baseline sweep.
#
# Each arm is one variable off the previous. All arms use --resume, so a killed
# run picks up where it stopped rather than re-spending GPU hours.
#
# Baseline D (frontier VLM ceiling) is deliberately absent -- it needs an
# external API and was descoped for this pass. results/baselines.md says so
# rather than leaving a silent gap.

$ErrorActionPreference = "Stop"
Set-Location "D:\GEN AI\language_slm"
$env:PYTHONPATH = "D:\GEN AI\language_slm"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_OFFLINE = "1"

$py = ".\.venv\Scripts\python.exe"
$manifest = "data/eval/synth/synth.jsonl"

# The committed baselines ran at the default 1024, which truncates 6/150 dense
# charts and understates valid-JSON. 2048 covers all but the two degenerate
# repetition loops (results/baselines.md). Raise this before Phase 3 -- it
# changes every arm equally, so cross-arm comparisons stay valid.
$maxNewTokens = 1024

function Run-Arm {
    param([string]$Name, [string[]]$Extra)
    Write-Output ""
    Write-Output ("#" * 10 + " $Name " + "#" * 10)
    & $py -m src.eval.run_eval `
        --manifest $manifest `
        --out "results/preds/$Name.jsonl" `
        --report "results/scores/$Name.json" `
        --max-new-tokens $maxNewTokens `
        --resume @Extra
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit $LASTEXITCODE" }
}

# A: zero-shot, minimal prompt (prompt v1.1 -- v1.0 never named the `data` key
#    and scored 0% schema-exact; see results/baselines.md)
Run-Arm "base_minimal" @("--prompt", "minimal")

# B: same model, engineered prompt. Isolates prompt engineering.
Run-Arm "base_engineered" @("--prompt", "engineered")

# C: engineered prompt + constrained decoding. Isolates grammar enforcement,
#    which is the B8 question in Phase 3.
Run-Arm "base_constrained" @("--prompt", "engineered", "--constrained")

Write-Output ""
Write-Output ("#" * 10 + " forgetting probe (base model) " + "#" * 10)
& $py -m src.eval.forgetting_probe `
    --out results/preds/docvqa_base.jsonl `
    --report results/scores/docvqa_base.json
if ($LASTEXITCODE -ne 0) { throw "forgetting probe failed with exit $LASTEXITCODE" }

Write-Output ""
Write-Output ("#" * 10 + " report " + "#" * 10)
& $py -m src.eval.report `
    --run "A. zero-shot, minimal=results/scores/base_minimal.json" `
    --run "B. engineered prompt=results/scores/base_engineered.json" `
    --run "C. engineered + constrained=results/scores/base_constrained.json" `
    --title "Phase 1 baselines — Qwen3-VL-4B-Instruct, no training" `
    --preamble-file results/baselines_preamble.md `
    --out results/baselines.md
& $py -m src.eval.dump_predictions --manifest $manifest `
    --preds results/preds/base_engineered.jsonl `
    --out results/figures/predictions_base_engineered.html --n 20

Write-Output "`nsweep complete"
