#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 INDEX MODEL_PATH OUTPUT_DIR [ADAPTER_PATH] [INITIAL_PREDICTIONS]" >&2
  exit 2
fi

INDEX=$1
MODEL_PATH=$2
OUTPUT_DIR=$3
ADAPTER_PATH=${4:-}
INITIAL_PREDICTIONS=${5:-}
PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-auto}
TORCH_DTYPE=${TORCH_DTYPE:-auto}
FEEDBACK_MODE=${FEEDBACK_MODE:-template}
LIMIT=${LIMIT:-0}
ANCHOR_MODEL_ID=${ANCHOR_MODEL_ID:-"['sam','dino']"}
ANCHOR_PROMPT_MODE=${ANCHOR_PROMPT_MODE:-query_tail}
ANCHOR_TOKEN_COUNTS=${ANCHOR_TOKEN_COUNTS:-}
RESUME=${RESUME:-1}
EXPERIMENT_FLAGS=()
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
EVAL_SCRIPT="$REPO_ROOT/train/src/tools/eval_dvgbench_agentic_v3.py"
SUMMARY_SCRIPT="$REPO_ROOT/train/src/tools/summarize_dvgbench_agentic_v3_matrix.py"
mkdir -p "$OUTPUT_DIR"

run_experiment() {
  local name=$1
  local method=$2
  local budget=$3
  local feedback=$4
  shift 4
  local disabled=("$@")
  if [[ "$RESUME" == "1" && -s "$OUTPUT_DIR/$name.jsonl" && -s "$OUTPUT_DIR/$name.summary.json" ]]; then
    echo "[agentic-matrix] skip completed $name"
    return
  fi
  local command=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    --index "$INDEX"
    --output "$OUTPUT_DIR/$name.jsonl"
    --summary-output "$OUTPUT_DIR/$name.summary.json"
    --query-field question
    --method "$method"
    --model-path "$MODEL_PATH"
    --device "$DEVICE"
    --torch-dtype "$TORCH_DTYPE"
    --anchor-model-id "$ANCHOR_MODEL_ID"
    --anchor-prompt-mode "$ANCHOR_PROMPT_MODE"
    --max-specialized-unit-calls "$budget"
    --feedback-mode "$feedback"
  )
  command+=("${EXPERIMENT_FLAGS[@]}")
  if [[ "$RESUME" == "1" ]]; then
    command+=(--resume)
  fi
  if [[ -n "$ADAPTER_PATH" ]]; then
    command+=(--adapter-path "$ADAPTER_PATH")
  fi
  if [[ -n "$INITIAL_PREDICTIONS" ]]; then
    command+=(--initial-predictions "$INITIAL_PREDICTIONS")
    command+=(--require-initial-confidence)
  fi
  if [[ "$LIMIT" -gt 0 ]]; then
    command+=(--limit "$LIMIT")
  fi
  local agent
  if [[ -n "$ANCHOR_TOKEN_COUNTS" ]]; then
    command+=(--anchor-token-counts "$ANCHOR_TOKEN_COUNTS")
  fi
  for agent in "${disabled[@]}"; do
    command+=(--disable-unit "$agent")
  done
  echo "[agentic-matrix] $name"
  "${command[@]}"
}

if [[ -z "$INITIAL_PREDICTIONS" ]]; then
  run_experiment main_one_pass one_pass 0 off
  INITIAL_PREDICTIONS="$OUTPUT_DIR/main_one_pass.jsonl"
else
  run_experiment main_one_pass one_pass 0 off
fi
run_experiment main_confidence_gated confidence_gated 1 off
run_experiment main_parent_only parent_only 1 off
run_experiment main_static_all static_all 3 off
run_experiment main_hierarchical hierarchical 3 "$FEEDBACK_MODE"

# Component ablations. There is deliberately no Query Rewrite agent.
run_experiment ablation_without_target hierarchical 3 "$FEEDBACK_MODE" target
run_experiment ablation_without_context hierarchical 3 "$FEEDBACK_MODE" context
run_experiment ablation_without_relation hierarchical 3 "$FEEDBACK_MODE" relation
run_experiment ablation_without_zoom hierarchical 3 "$FEEDBACK_MODE" zoom

# Mechanism ablations for routing constraints and crop-sensitive coordinates.
EXPERIMENT_FLAGS=(--no-constraint-graph)
run_experiment ablation_without_constraint_graph hierarchical 3 "$FEEDBACK_MODE"
EXPERIMENT_FLAGS=(--no-semantic-frame-protection)
run_experiment ablation_without_semantic_frame_protection hierarchical 3 "$FEEDBACK_MODE"
EXPERIMENT_FLAGS=(--no-false-repair-guard)
run_experiment ablation_without_false_repair_guard hierarchical 3 "$FEEDBACK_MODE"
EXPERIMENT_FLAGS=()

for budget in 0 1 2; do
  run_experiment "budget_k$budget" hierarchical "$budget" "$FEEDBACK_MODE"
done

"$PYTHON_BIN" "$SUMMARY_SCRIPT" \
  --input-dir "$OUTPUT_DIR" \
  --csv-output "$OUTPUT_DIR/agentic_matrix.csv" \
  --json-output "$OUTPUT_DIR/agentic_matrix.json"
