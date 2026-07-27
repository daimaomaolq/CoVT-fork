#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/train:${REPO_ROOT}/train/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

: "${CANDIDATE_TRACE:?set CANDIDATE_TRACE to the completed v4.1 JSONL}"
: "${DVBENCH_INDEX:?set DVBENCH_INDEX to the 873-row evaluation index}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${ADAPTER_PATH:?set ADAPTER_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_REVISION="${CODE_REVISION:-unknown}"
ANCHOR_MODEL_ID="${ANCHOR_MODEL_ID:-['sam','dino']}"
ANCHOR_PROMPT_MODE="${ANCHOR_PROMPT_MODE:-query_tail}"
VERIFIER_JSONL="${OUTPUT_DIR}/frozen_candidate_verifier.jsonl"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" train/src/tools/verify_dvgbench_frozen_candidates.py \
  --candidate-trace "${CANDIDATE_TRACE}" \
  --output "${VERIFIER_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "${ADAPTER_PATH}" \
  --anchor-model-id "${ANCHOR_MODEL_ID}" \
  --anchor-prompt-mode "${ANCHOR_PROMPT_MODE}" \
  --max-hypotheses 4 \
  --code-revision "${CODE_REVISION}" \
  --resume

for selector in initial stored_fusion visual_only conservative_visual; do
  "${PYTHON_BIN}" train/src/tools/eval_dvgbench_frozen_candidate_selection.py \
    --candidate-trace "${CANDIDATE_TRACE}" \
    --index "${DVBENCH_INDEX}" \
    --verifier-evidence "${VERIFIER_JSONL}" \
    --selector "${selector}" \
    --output "${OUTPUT_DIR}/${selector}.jsonl" \
    --summary-output "${OUTPUT_DIR}/${selector}.summary.json" \
    --code-revision "${CODE_REVISION}"
done

"${PYTHON_BIN}" train/src/tools/summarize_dvgbench_frozen_selectors.py \
  --input-dir "${OUTPUT_DIR}" \
  --json-output "${OUTPUT_DIR}/selector_comparison.json" \
  --csv-output "${OUTPUT_DIR}/selector_comparison.csv"
