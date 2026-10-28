<#
.SYNOPSIS
    PR-24: pipeline validation. Takes the released baseline through OUR evaluator and OUR trainer and
    checks that nothing in the pipeline changes the result. Owner: Member C.

    1. parity    released weights -> scripts/evaluate.py; must match PR-06 (same weights, new code path)
    2. frozen    train only the heads on the balanced subset, evaluate on the full test split
    3. finetune  train the whole baseline on the balanced subset, evaluate on the full test split
    4. compare   table of all runs vs PR-06; the trained runs must not be worse (they should be better)

.DESCRIPTION
    Usage (from the repo root; extra arguments are config overrides applied to every step):
      powershell -File scripts/run_pr24.ps1
      powershell -File scripts/run_pr24.ps1 data.num_workers=8 train.batch_size=16
      $env:STEPS="parity"; powershell -File scripts/run_pr24.ps1    # only the parity check
#>

$ErrorActionPreference = "Stop"

# Navigate to repo root (parent of the scripts directory)
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ".."

# Ensure Python can find local packages (like phasecon) from the root directory
$env:PYTHONPATH = "."

$PY = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$OUT = if ($env:OUT_DIR) { $env:OUT_DIR } else { "experiments" }
$PREFIX = if ($env:PREFIX) { $env:PREFIX } else { "baseline_pipeline" }
$REFERENCE = if ($env:REFERENCE) { $env:REFERENCE } else { "pr06" }
$STEPS = if ($env:STEPS) { $env:STEPS } else { "parity frozen finetune" }

$MODEL_CFG = @("configs/model/baseline_only.yaml")

# $args contains any extra positional arguments passed to the script
$COMMON = @("audio.sample_rate=24000", "audio.num_samples=96000", "output_dir=$OUT") + $args
$TRAIN = $COMMON + @(
    "data.subset.enabled=true",
    "train.loss.beta=1.0",
    "train.loss.mu=0.0",
    "train.optimizer.weight_decay=2e-4",
    "train.optimizer.lr_encoder=1e-4",
    "train.optimizer.lr_new=1e-4"
)

$RESULTS = "$OUT/results"
$TO_COMPARE = @()

function Test-Step {
    param([string]$step)
    return (" $STEPS " -like "* $step *")
}

function Say {
    param([string]$message)
    Write-Host "`n==== $message ===="
}

if (Test-Step "parity") {
    Say "1/3 parity: released weights through scripts/evaluate.py"
    & $PY scripts/evaluate.py --config $MODEL_CFG configs/eval/in_distribution.yaml --run-name "${PREFIX}_released" --override $COMMON
    & $PY scripts/compare_results.py --reference $REFERENCE --parity "$RESULTS/${PREFIX}_released"
}

foreach ($mode in @("frozen", "finetune")) {
    if (-not (Test-Step $mode)) { continue }
    $freeze = ($mode -eq "frozen").ToString().ToLower()
    
    Say ("{0}: train on the balanced subset (freeze_encoder={1})" -f $mode, $freeze)
    & $PY scripts/train.py --config $MODEL_CFG --override $TRAIN "model.baseline.freeze_encoder=$freeze" "run_name=${PREFIX}_${mode}_train"
    
    Say ("{0}: evaluate best checkpoint on the full test split" -f $mode)
    & $PY scripts/evaluate.py --config $MODEL_CFG configs/eval/in_distribution.yaml --checkpoint "$OUT/checkpoints/${PREFIX}_${mode}_train/best.pt" --run-name "${PREFIX}_${mode}" --override $COMMON "model.baseline.weights=null"
    
    $TO_COMPARE += "$RESULTS/${PREFIX}_${mode}"
}

if ($TO_COMPARE.Count -gt 0) {
    Say "compare against the reference ($REFERENCE)"
    & $PY scripts/compare_results.py --reference $REFERENCE --no-regress --out "$RESULTS/${PREFIX}_comparison.md" $TO_COMPARE
}

Say "PR-24 steps finished: $STEPS"