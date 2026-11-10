#!/usr/bin/env bash
# Full pipeline: GRPO -> compute-matched SFT -> evaluate all four conditions -> report.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
STEPS=${STEPS:-16}; N=${N:-100}
$PY scripts/train_grpo.py --steps "$STEPS" > logs/train_grpo.log 2>&1
$PY scripts/train_sft.py > logs/train_sft.log 2>&1
$PY scripts/evaluate.py --name base --n "$N" > logs/eval_base.log 2>&1
$PY scripts/evaluate.py --name sft --adapter checkpoints/sft/adapter --n "$N" > logs/eval_sft.log 2>&1
$PY scripts/evaluate.py --name grpo --adapter checkpoints/grpo/adapter --n "$N" > logs/eval_grpo.log 2>&1
$PY scripts/evaluate.py --name harness --n "$N" > logs/eval_harness.log 2>&1
$PY scripts/make_report.py
