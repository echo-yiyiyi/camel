# python code-agent/generic/generic_code_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context
# mv task-script task-script-w-context

# python code-agent/generic/generic_code_agent.py --no-context --tasks code-agent/task_list.json --exp-id no-context
# mv task-script task-script-no-context

python code-agent/generic/generic_code_single_agent.py --context code-agent/generic/CAMEL.md --tasks code-agent/task_list.json --exp-id w-context-single
mv task-script task-script-w-context-single