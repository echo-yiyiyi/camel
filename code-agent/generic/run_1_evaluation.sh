#!/bin/bash
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Single task evaluation script - evaluates only task 2

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMEL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Paths
GROUND_TRUTH_DIR="$(dirname "$CAMEL_ROOT")/ground_truth/single_agent"
TASK_LIST="${CAMEL_ROOT}/code-agent/task_list.json"
MODEL="gpt-4-1"

# Output directory
OUTPUT_BASE="${CAMEL_ROOT}/evaluation_reports"
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Single Task Evaluation - Task 2"
echo "=============================================="
echo "CAMEL Root: $CAMEL_ROOT"
echo "Ground Truth: $GROUND_TRUTH_DIR"
echo "Model: $MODEL"
echo "=============================================="

# Evaluate task 2 only
LOG_DIR="${CAMEL_ROOT}/logsw-context-refactored"
SCRIPT_DIR_EVAL="${CAMEL_ROOT}/task-script-w-context-refactored/single_agent"
OUTPUT_DIR="${OUTPUT_BASE}/task_2_only"

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"
python analysis_agent.py \
    --log-dir "$LOG_DIR" \
    --script-dir "$SCRIPT_DIR_EVAL" \
    --ground-truth-dir "$GROUND_TRUTH_DIR" \
    --task-list "$TASK_LIST" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL" \
    --task-index 2

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo "Report saved to: $OUTPUT_DIR"
