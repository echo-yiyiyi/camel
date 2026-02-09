#!/bin/bash
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Evolve script: Extract failed tasks from evaluation reports,
# generate evolve_task_list.json with fix hints, and re-run via generic_code_agent.py
#
# Usage:
#   ./run_evolve.sh [--eval-dir <dir>] [--dry-run] [--model <model>] [--parallel <n>]
#
# Default eval dir: evaluation_reports/w-context-evolve

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMEL_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Defaults
EVAL_DIR="${CAMEL_ROOT}/evaluation_reports/w_context_refactored"
TASK_LIST="${CAMEL_ROOT}/code-agent/task_list.json"
CAMEL_MD="${SCRIPT_DIR}/CAMEL.md"
MODEL="openai/gpt-4.1-mini"
PARALLEL=1
DRY_RUN=false
EXP_ID="evolve-fix"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-dir)
            EVAL_DIR="$2"
            shift 2
            ;;
        --task-list)
            TASK_LIST="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --parallel|-P)
            PARALLEL="$2"
            shift 2
            ;;
        --exp-id)
            EXP_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eval-dir DIR     Evaluation reports directory (default: evaluation_reports/w-context-evolve)"
            echo "  --task-list FILE   Original task list JSON (default: code-agent/task_list.json)"
            echo "  --model MODEL      Model to use (default: openai/gpt-4.1-mini)"
            echo "  --parallel N       Parallel agents (default: 2)"
            echo "  --exp-id ID        Experiment ID (default: evolve-fix)"
            echo "  --dry-run          Generate task list only, don't run agent"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Derived paths
SUMMARY_FILE="${EVAL_DIR}/evaluation_summary.md"
EVOLVE_TASK_LIST="${CAMEL_ROOT}/code-agent/evolve_task_list.json"

echo "=============================================="
echo "Evolve Pipeline - Fix Failed Tasks"
echo "=============================================="
echo "CAMEL Root:      $CAMEL_ROOT"
echo "Eval Dir:        $EVAL_DIR"
echo "Summary File:    $SUMMARY_FILE"
echo "Task List:       $TASK_LIST"
echo "CAMEL.md:        $CAMEL_MD"
echo "Model:           $MODEL"
echo "Parallel:        $PARALLEL"
echo "Exp ID:          $EXP_ID"
echo "Dry Run:         $DRY_RUN"
echo "=============================================="

# Validate inputs
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "[ERROR] Evaluation summary not found: $SUMMARY_FILE"
    exit 1
fi

if [ ! -f "$TASK_LIST" ]; then
    echo "[ERROR] Task list not found: $TASK_LIST"
    exit 1
fi

# =========================================================================
# Step 1: Parse failed tasks and generate evolve_task_list.json
# =========================================================================
echo ""
echo "[Step 1] Parsing evaluation reports for failed tasks..."

python3 "${SCRIPT_DIR}/parse_evolve_tasks.py" \
    "$EVAL_DIR" \
    "$TASK_LIST" \
    "$EVOLVE_TASK_LIST"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to generate evolve task list"
    exit 1
fi

# Check if any tasks were generated
if [ ! -f "$EVOLVE_TASK_LIST" ]; then
    echo "[INFO] No evolve task list generated (no failed tasks). Exiting."
    exit 0
fi

TASK_COUNT=$(python3 -c "import json; print(len(json.load(open('${EVOLVE_TASK_LIST}'))))")
if [ "$TASK_COUNT" -eq 0 ]; then
    echo "[INFO] No failed tasks to re-run. Exiting."
    exit 0
fi

echo ""
echo "[INFO] Generated evolve_task_list.json with $TASK_COUNT task(s)"

# =========================================================================
# Step 2: Run generic_code_agent.py with evolve task list
# =========================================================================
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Skipping agent execution."
    echo "[DRY RUN] To run:"
    echo "  cd $SCRIPT_DIR"
    echo "  python generic_code_agent.py --context $CAMEL_MD --tasks $EVOLVE_TASK_LIST --exp-id $EXP_ID -P $PARALLEL --model $MODEL"
    exit 0
fi

echo ""
echo "[Step 2] Running generic_code_agent.py with evolved tasks..."
echo ""

cd "$SCRIPT_DIR"
python generic_code_agent.py \
    --context "$CAMEL_MD" \
    --tasks "$EVOLVE_TASK_LIST" \
    --exp-id "$EXP_ID" \
    -P "$PARALLEL" \
    --model "$MODEL" \
    --max-tasks "$TASK_COUNT"

# Move output
if [ -d "${CAMEL_ROOT}/task-script" ]; then
    TARGET_DIR="${CAMEL_ROOT}/task-script-${EXP_ID}"
    if [ -d "$TARGET_DIR" ]; then
        echo "[WARN] Target directory exists, merging: $TARGET_DIR"
        cp -r "${CAMEL_ROOT}/task-script/"* "$TARGET_DIR/" 2>/dev/null || true
        rm -rf "${CAMEL_ROOT}/task-script"
    else
        mv "${CAMEL_ROOT}/task-script" "$TARGET_DIR"
    fi
    echo "[INFO] Scripts saved to: $TARGET_DIR"
else
    echo "[WARN] task-script directory not found after run"
fi

echo ""
echo "=============================================="
echo "Evolve Pipeline Complete!"
echo "=============================================="
echo "Evolve task list: $EVOLVE_TASK_LIST"
echo "Experiment ID:    $EXP_ID"
echo ""
