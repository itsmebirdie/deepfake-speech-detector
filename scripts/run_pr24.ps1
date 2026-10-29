<#
.SYNOPSIS
    PR-24: pipeline validation (Windows / PowerShell version of scripts/run_pr24.sh). Owner: Member C.
    Takes the released baseline through OUR evaluator and OUR trainer and checks that nothing in the
    pipeline changes the result.

      1. parity    released weights -> scripts/evaluate.py; must match PR-06 (same weights, new code path)
      2. frozen    train only the heads on the balanced subset, evaluate on the full test split
      3. finetune  train the whole baseline on the balanced subset, evaluate on the full test split
      4. compare   table of all runs vs PR-06; the trained runs must not be worse (they should be better)

    The script STOPS at the first step that fails (including a failed parity or no-regression check)
    and exits with a non-zero code.

.DESCRIPTION
    Run from anywhere; it moves to the repo root. Extra arguments are config overrides applied to every step.
      .\scripts\run_pr24.ps1
      .\scripts\run_pr24.ps1 data.num_workers=8 train.batch_size=16
      $env:STEPS = "parity"; .\scripts\run_pr24.ps1     # only the parity check (evaluation only, no training)
    Environment variables: STEPS (default "parity frozen finetune"), PREFIX (default baseline_pipeline),
    OUT_DIR (default experiments), REFERENCE (default pr06, or a results folder), PYTHON (default python).

    Settings that mirror the authors' recipe (main.py): loss = 0.5*CE_bin + 0.5*CE_voc, Adam lr 1e-4 on all
    weights, weight decay 1e-4. Ours is L_bin + beta*L_voc, so beta=1 doubles the loss and weight_decay=2e-4
    cancels that under Adam's L2 decay. mu=0 because baseline_only has no phase branch.
    Audio is forced to the baseline's native 24 kHz / 96,000 samples.
#>

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$OUT = if ($env:OUT_DIR) { $env:OUT_DIR } else { "experiments" }
$PREFIX = if ($env:PREFIX) { $env:PREFIX } else { "baseline_pipeline" }
$REFERENCE = if ($env:REFERENCE) { $env:REFERENCE } else { "pr06" }
$STEPS = if ($env:STEPS) { $env:STEPS } else { "parity frozen finetune" }

$MODEL_CFG = "configs/model/baseline_only.yaml"
$EVAL_CFG = "configs/eval/in_distribution.yaml"
$RESULTS = "$OUT/results"

[string[]]$COMMON = @("audio.sample_rate=24000", "audio.num_samples=96000", "output_dir=$OUT") + $args
[string[]]$TRAIN = $COMMON + @(
    "data.subset.enabled=true", "train.loss.beta=1.0", "train.loss.mu=0.0",
    "train.optimizer.weight_decay=2e-4", "train.optimizer.lr_encoder=1e-4", "train.optimizer.lr_new=1e-4"
)

function Test-Step([string]$step) { return (" $STEPS " -like "* $step *") }
function Say([string]$message) { Write-Host "`n==== $message ====" }

# PowerShell does not stop when an external program fails, so every call goes through here.
function Invoke-Py([string[]]$PyArgs) {
    & $PY @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED (exit code $LASTEXITCODE): $PY $($PyArgs -join ' ')"
    }
}

# The most common first-run problem: the project is not installed in this Python environment.
& $PY -c "import phasecon" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Python ('$PY') cannot import 'phasecon'. Activate your virtual environment and run: pip install -r requirements.txt; pip install -e ."
}

$toCompare = @()

if (Test-Step "parity") {
    Say "1/3 parity: released weights through scripts/evaluate.py"
    Invoke-Py (@("scripts/evaluate.py", "--config", $MODEL_CFG, $EVAL_CFG, "--run-name", "${PREFIX}_released", "--override") + $COMMON)
    Invoke-Py @("scripts/compare_results.py", "--reference", $REFERENCE, "--parity", "$RESULTS/${PREFIX}_released")
}

foreach ($mode in @("frozen", "finetune")) {
    if (-not (Test-Step $mode)) { continue }
    $freeze = if ($mode -eq "frozen") { "true" } else { "false" }

    Say "${mode}: train on the balanced subset (freeze_encoder=$freeze)"
    Invoke-Py (@("scripts/train.py", "--config", $MODEL_CFG, "--override") + $TRAIN + @("model.baseline.freeze_encoder=$freeze", "run_name=${PREFIX}_${mode}_train"))

    Say "${mode}: evaluate best checkpoint on the full test split"
    Invoke-Py (@("scripts/evaluate.py", "--config", $MODEL_CFG, $EVAL_CFG, "--checkpoint", "$OUT/checkpoints/${PREFIX}_${mode}_train/best.pt", "--run-name", "${PREFIX}_${mode}", "--override") + $COMMON + @("model.baseline.weights=null"))
    $toCompare += "$RESULTS/${PREFIX}_${mode}"
}

if ($toCompare.Count -gt 0) {
    Say "compare against the reference ($REFERENCE)"
    Invoke-Py (@("scripts/compare_results.py", "--reference", $REFERENCE, "--no-regress", "--out", "$RESULTS/${PREFIX}_comparison.md") + $toCompare)
}

Say "PR-24 steps finished: $STEPS"
