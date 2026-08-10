# Phase 2: assemble the training corpus.
#
# Order matters. Degradation runs on the rendered synthetic charts, and the
# dataset build runs on the degraded manifest, so a re-run of the generator
# invalidates everything downstream of it.
#
# Real charts are fetched with --exclude-manifest against the eval split. The
# two subsets come from the same corpus, so without that they would overlap by
# construction and the contamination check would (correctly) abort the build.

$ErrorActionPreference = "Stop"
Set-Location "D:\GEN AI\language_slm"
$env:PYTHONPATH = "D:\GEN AI\language_slm"
$env:PYTHONUNBUFFERED = "1"

$py = ".\.venv\Scripts\python.exe"

# Seeds are offset from the eval set's (0) so the two never draw the same specs.
$synthSeed = 1000
$realSeed  = 2000
$mixSeed   = 3000

$nSynth = 10000
$nReal  = 3500
$nTotal = 10000
$synthRatio = 0.70
$degradeRate = 0.20
$longEdge = 448   # configs/RESOLUTION_POLICY.md

if (-not (Test-Path "data/train/synth/trsynth.jsonl")) {
    Write-Output "`n=== synthetic generation ==="
    & $py -m src.data.synth --out data/train/synth --n $nSynth --seed $synthSeed `
        --hard-rate 0.35 --prefix trsynth
    if ($LASTEXITCODE -ne 0) { throw "synthetic generation failed" }
}

if (-not (Test-Path "data/train/real/real.jsonl")) {
    Write-Output "`n=== real charts ==="
    & $py -m src.data.real --out data/train/real --n $nReal --seed $realSeed `
        --exclude-manifest data/eval/real/real.jsonl
    if ($LASTEXITCODE -ne 0) { throw "real chart fetch failed" }
}

Write-Output "`n=== degradation ($degradeRate of synthetic) ==="
& $py -m src.data.augment --manifest data/train/synth/trsynth.jsonl `
    --out data/train/synth_deg --rate $degradeRate --seed $synthSeed
if ($LASTEXITCODE -ne 0) { throw "degradation failed" }

Write-Output "`n=== assemble + contamination check ==="
& $py -m src.data.build_dataset `
    --synth data/train/synth_deg/trsynth.jsonl `
    --real data/train/real/real.jsonl `
    --eval-manifest data/eval/synth/synth.jsonl `
    --eval-manifest data/eval/real/real.jsonl `
    --out data/train/train.jsonl `
    --report results/dataset_stats.json `
    --n $nTotal --synth-ratio $synthRatio --seed $mixSeed --long-edge $longEdge
if ($LASTEXITCODE -ne 0) { throw "dataset build failed (contamination aborts here)" }

Write-Output "`n=== rendered samples for manual inspection ==="
& $py -m src.data.inspect_samples --dataset data/train/train.jsonl `
    --out results/training_samples.txt --n 5
if ($LASTEXITCODE -ne 0) { throw "sample dump failed" }

Write-Output "`nPhase 2 data build complete"
