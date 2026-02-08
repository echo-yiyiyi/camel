#!/bin/bash
# 确保从 camel 目录运行
cd "$(dirname "$0")/../.." || exit 1

# python code-agent/generic/generic_code_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-refactored -P 2
# mv task-script task-script-w-context-refactored

    python code-agent/generic/generic_code_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-evolve -P 2 --model openai/gpt-4.1-mini
    if [ -d "task-script" ]; then
    mv task-script task-script-evolve
else
    echo "Warning: task-script directory not found, skipping mv command"
fi

# python code-agent/generic/generic_code_single_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-single
# mv task-script task-script-w-context-single