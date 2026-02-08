#!/bin/bash
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Batch evaluation script for code agent generated scripts
# Compares generated scripts against ground truth using LLM-as-Judge

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMEL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Paths (ground_truth is in parent directory of CAMEL_ROOT)
GROUND_TRUTH_DIR="$(dirname "$CAMEL_ROOT")/ground_truth/single_agent"
TASK_LIST="${CAMEL_ROOT}/code-agent/task_list.json"
MODEL="gpt-4-1"  # Use gpt-4-1 for evaluation

# Output directory
OUTPUT_BASE="${CAMEL_ROOT}/evaluation_reports"
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Code Agent Evaluation Script"
echo "=============================================="
echo "CAMEL Root: $CAMEL_ROOT"
echo "Ground Truth: $GROUND_TRUTH_DIR"
echo "Task List: $TASK_LIST"
echo "Model: $MODEL"
echo "Output: $OUTPUT_BASE"
echo "=============================================="

# Function to run evaluation
run_evaluation() {
    local name=$1
    local log_dir=$2
    local script_dir=$3
    local output_dir="${OUTPUT_BASE}/${name}"

    echo ""
    echo "=============================================="
    echo "Evaluating: $name"
    echo "=============================================="
    echo "Log Dir: $log_dir"
    echo "Script Dir: $script_dir"
    echo "Output Dir: $output_dir"
    echo ""

    if [ ! -d "$log_dir" ]; then
        echo "[SKIP] Log directory not found: $log_dir"
        return
    fi

    if [ ! -d "$script_dir" ]; then
        echo "[SKIP] Script directory not found: $script_dir"
        return
    fi

    mkdir -p "$output_dir"

    cd "$SCRIPT_DIR"
    python analysis_agent.py \
        --log-dir "$log_dir" \
        --script-dir "$script_dir" \
        --ground-truth-dir "$GROUND_TRUTH_DIR" \
        --task-list "$TASK_LIST" \
        --output "$output_dir" \
        --model "$MODEL" \
        --max-tasks 10

    echo ""
    echo "Report saved to: $output_dir"
}

# Run evaluation
run_evaluation "w-context-evolve" \
    "${CAMEL_ROOT}/logsw-context-evolve" \
    "${CAMEL_ROOT}/task-script-evolve/single_agent"

# Summary
echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo "Reports saved to: $OUTPUT_BASE"
echo ""
echo "Summary files:"
for dir in "$OUTPUT_BASE"/*/; do
    if [ -f "${dir}evaluation_summary.md" ]; then
        echo "  - ${dir}evaluation_summary.md"
    fi
done
