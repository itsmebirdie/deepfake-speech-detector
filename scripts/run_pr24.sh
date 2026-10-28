#!/usr/bin/env bash
# PR-24: pipeline validation. Takes the released baseline through OUR evaluator and OUR trainer and
# checks that nothing in the pipeline changes the result. Owner: Member C.
#
#   1. parity    released weights -> scripts/evaluate.py; must match PR-06 (same weights, new code path)
#   2. frozen    train only the heads on the balanced subset, evaluate on the full test split
#   3. finetune  train the whole baseline on the balanced subset, evaluate on the full test split
#   4. compare   table of all runs vs PR-06; the trained runs must not be worse (they should be better)
#
# Usage (from the repo root; extra arguments are config overrides applied to every step):
#   bash scripts/run_pr24.sh
#   bash scripts/run_pr24.sh data.num_workers=8 train.batch_size=16
#   STEPS="parity" bash scripts/run_pr24.sh          # only the parity check (evaluation only, no training)
# Environment: STEPS (default "parity frozen finetune"), PREFIX (default baseline_pipeline), OUT_DIR
# (default experiments), REFERENCE (default pr06; or a results folder), PYTHON (default python).
#
# Settings that mirror the authors' recipe (main.py): loss = 0.5*CE_bin + 0.5*CE_voc, Adam lr 1e-4 on all
# weights, weight decay 1e-4. Ours is L_bin + beta*L_voc, so beta=1 doubles the loss, and weight_decay=2e-4
# cancels that under Adam's L2 decay. mu=0 because there is no phase branch in baseline_only.
# Audio is forced to the baseline's native 24 kHz / 96,000 samples until the repo default changes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=(${PYTHON:-python})
OUT="${OUT_DIR:-experiments}"
PREFIX="${PREFIX:-baseline_pipeline}"
REFERENCE="${REFERENCE:-pr06}"
STEPS="${STEPS:-parity frozen finetune}"
MODEL_CFG=(configs/model/baseline_only.yaml)

COMMON=(audio.sample_rate=24000 audio.num_samples=96000 "output_dir=$OUT" ${@+"$@"})
TRAIN=("${COMMON[@]}" data.subset.enabled=true train.loss.beta=1.0 train.loss.mu=0.0
       train.optimizer.weight_decay=2e-4 train.optimizer.lr_encoder=1e-4 train.optimizer.lr_new=1e-4)
RESULTS="$OUT/results"
TO_COMPARE=()

want() { [[ " $STEPS " == *" $1 "* ]]; }
say() { printf '\n==== %s ====\n' "$*"; }

if want parity; then
  say "1/3 parity: released weights through scripts/evaluate.py"
  "${PY[@]}" scripts/evaluate.py --config "${MODEL_CFG[@]}" configs/eval/in_distribution.yaml \
      --run-name "${PREFIX}_released" --override "${COMMON[@]}"
  "${PY[@]}" scripts/compare_results.py --reference "$REFERENCE" --parity "$RESULTS/${PREFIX}_released"
fi

for mode in frozen finetune; do
  want "$mode" || continue
  freeze=false; [[ $mode == frozen ]] && freeze=true
  say "$mode: train on the balanced subset (freeze_encoder=$freeze)"
  "${PY[@]}" scripts/train.py --config "${MODEL_CFG[@]}" \
      --override "${TRAIN[@]}" "model.baseline.freeze_encoder=$freeze" "run_name=${PREFIX}_${mode}_train"
  say "$mode: evaluate best checkpoint on the full test split"
  "${PY[@]}" scripts/evaluate.py --config "${MODEL_CFG[@]}" configs/eval/in_distribution.yaml \
      --checkpoint "$OUT/checkpoints/${PREFIX}_${mode}_train/best.pt" --run-name "${PREFIX}_${mode}" \
      --override "${COMMON[@]}" model.baseline.weights=null
  TO_COMPARE+=("$RESULTS/${PREFIX}_${mode}")
done

if [[ ${#TO_COMPARE[@]} -gt 0 ]]; then
  say "compare against the reference ($REFERENCE)"
  "${PY[@]}" scripts/compare_results.py --reference "$REFERENCE" --no-regress \
      --out "$RESULTS/${PREFIX}_comparison.md" "${TO_COMPARE[@]}"
fi
say "PR-24 steps finished: $STEPS"
