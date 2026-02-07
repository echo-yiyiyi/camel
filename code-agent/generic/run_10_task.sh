# python code-agent/generic/generic_code_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-refactored -P 2
# mv task-script task-script-w-context-refactored

python code-agent/generic/generic_code_agent.py --no-context --tasks code-agent/task_list.json --exp-id no-context-refactored-5.1 -P 2 --model openai/gpt-5.1
mv task-script task-script-no-context-refactored-5.1

# python code-agent/generic/generic_code_single_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-single
# mv task-script task-script-w-context-single