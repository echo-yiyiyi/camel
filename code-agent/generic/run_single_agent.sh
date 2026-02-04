#!/bin/bash
# Run the single agent for a quick test

cd "$(dirname "$0")"

python generic_code_single_agent.py \
    --task "create an agent with the weather tool using Qwen2.5-14B-Instruct, and then it can be used to answer question about weather. Save the script to task-script/single_agent/test_weather_agent.py" \
    --model openai/gpt-4-1-mini \
    --timeout 300
