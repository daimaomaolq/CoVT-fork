#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/train:${REPO_ROOT}/train/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CANDIDATE_TRACE:?set CANDIDATE_TRACE to the completed v4.1 JSONL}"
: "${DVBENCH_INDEX:?set DVBENCH_INDEX to the aligned 873-row evaluation index}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${ADAPTER_PATH:?set ADAPTER_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_REVISION="${CODE_REVISION:-unknown}"
ANCHOR_MODEL_ID="${ANCHOR_MODEL_ID:-sam,dino}"
ANCHOR_PROMPT_MODE="${ANCHOR_PROMPT_MODE:-query_tail}"
VERIFIER_JSONL="${OUTPUT_DIR}/counterfactual_verifier.jsonl"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" train/src/tools/verify_dvgbench_counterfactual_candidates.py \
  --candidate-trace "${CANDIDATE_TRACE}" \
  --output "${VERIFIER_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --device cuda \
  --torch-dtype bfloat16 \
  --attn-implementation sdpa \
  --anchor-model-id "${ANCHOR_MODEL_ID}" \
  --anchor-prompt-mode "${ANCHOR_PROMPT_MODE}" \
  --max-alternatives 1 \
  --duplicate-iou-threshold 0.92 \
  --maximum-initial-overlap 0.85 \
  --minimum-independent-gain 0.0 \
  --first-crop-scale 3.0 \
  --second-crop-scale 4.0 \
  --code-revision "${CODE_REVISION}" \
  --resume

"${PYTHON_BIN}" train/src/tools/eval_dvgbench_counterfactual_selection.py \
  --candidate-trace "${CANDIDATE_TRACE}" \
  --index "${DVBENCH_INDEX}" \
  --verifier-evidence "${VERIFIER_JSONL}" \
  --output "${OUTPUT_DIR}/counterfactual_v4_3.jsonl" \
  --summary-output "${OUTPUT_DIR}/counterfactual_v4_3.summary.json" \
  --max-alternatives 1 \
  --duplicate-iou-threshold 0.92 \
  --maximum-initial-overlap 0.85 \
  --minimum-independent-gain 0.0 \
  --first-crop-scale 3.0 \
  --second-crop-scale 4.0 \
  --code-revision "${CODE_REVISION}"
