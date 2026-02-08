#!/bin/bash
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Batch diagnosis script for failed code generation tasks
# Analyzes failures and updates CAMEL.md to improve future generations

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMEL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Paths
GROUND_TRUTH_DIR="$(dirname "$CAMEL_ROOT")/ground_truth/single_agent"
TASK_LIST="${CAMEL_ROOT}/code-agent/task_list.json"
CAMEL_MD="${SCRIPT_DIR}/CAMEL.md"
MODEL="gpt-4-1-mini"

# Input directories (from code agent run)
LOG_DIR="${CAMEL_ROOT}/logsw-context-refactored"
SCRIPT_OUTPUT_DIR="${CAMEL_ROOT}/task-script-w-context-refactored/single_agent"

# Analysis reports (from analysis agent run)
ANALYSIS_DIR="${CAMEL_ROOT}/evaluation_reports/w_context_refactored"

# Output directory for diagnoses
OUTPUT_DIR="${CAMEL_ROOT}/diagnose_reports"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Diagnose Agent - Code Generation Failure Analysis"
echo "=============================================="
echo "CAMEL Root: $CAMEL_ROOT"
echo "Ground Truth: $GROUND_TRUTH_DIR"
echo "Task List: $TASK_LIST"
echo "CAMEL.md: $CAMEL_MD"
echo "Model: $MODEL"
echo ""
echo "Input:"
echo "  Log Dir: $LOG_DIR"
echo "  Script Dir: $SCRIPT_OUTPUT_DIR"
echo "  Analysis Dir: $ANALYSIS_DIR"
echo ""
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Check if required directories exist
if [ ! -d "$LOG_DIR" ]; then
    echo "[ERROR] Log directory not found: $LOG_DIR"
    exit 1
fi

if [ ! -d "$SCRIPT_OUTPUT_DIR" ]; then
    echo "[ERROR] Script directory not found: $SCRIPT_OUTPUT_DIR"
    exit 1
fi

if [ ! -d "$ANALYSIS_DIR" ]; then
    echo "[ERROR] Analysis directory not found: $ANALYSIS_DIR"
    exit 1
fi

# Run diagnosis
cd "$SCRIPT_DIR"
# Single task mode: diagnose task_2
TASK_NUM=2
TASK_DESC=$(python -c "import json; print(json.load(open('$TASK_LIST'))['task_${TASK_NUM}'])")

python diagnose_agent.py \
    --task-name "task_${TASK_NUM}" \
    --task-desc "$TASK_DESC" \
    --log "$(ls ${LOG_DIR}/*task_${TASK_NUM}_*.log | head -1)" \
    --script "$(ls ${SCRIPT_OUTPUT_DIR}/${TASK_NUM}_*.py | head -1)" \
    --ground-truth "$(ls ${GROUND_TRUTH_DIR}/${TASK_NUM}_*.py | head -1)" \
    --analysis-report "$(ls ${ANALYSIS_DIR}/eval_*task_${TASK_NUM}_*.md | head -1)" \
    --camel-md "$CAMEL_MD" \
    --output "$OUTPUT_DIR" \
    --model "$MODEL"

echo ""
echo "=============================================="
echo "Diagnosis Complete!"
echo "=============================================="
echo "Reports saved to: $OUTPUT_DIR"
echo "Updated CAMEL.md: $CAMEL_MD"
echo ""
