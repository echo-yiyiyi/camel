# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Optimized Explore Agent for fast and precise code exploration.

Key improvements over the original:
1. Uses specialized CodeSearchToolkit instead of generic TerminalToolkit
2. Native Python glob instead of shell `find` - 5-10x faster
3. Ripgrep for content search - 10-100x faster than grep
4. Structured output format for better LLM understanding
5. Optimized system prompt for efficient multi-tool usage
"""

import os
import json
import time
from datetime import datetime
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Import the optimized code search toolkit
from code_search_toolkit import CodeSearchToolkit

exp_id = "_optimized"

set_log_level('INFO')

# Directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))
camel_dir = os.path.dirname(base_dir)
project_root = os.path.dirname(camel_dir)
logs_dir = os.path.join(base_dir, f"logs{exp_id}")
os.makedirs(logs_dir, exist_ok=True)

# Initialize optimized code search toolkit
code_toolkit = CodeSearchToolkit(
    working_directory=camel_dir,
    exclude_dirs={
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build',
        '.initial_env', 'code-agent', 'task-script-v0', 'task-script_v0'
    },
    max_results=50,
)

# Get tools from the optimized toolkit
tools = code_toolkit.get_tools()

# Model configuration
model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4_1_MINI,
    model_config_dict=model_config_dict,
)

# Optimized system prompt with search techniques
EXPLORE_SYSTEM_PROMPT = """You are a code exploration specialist. Find relevant files for a task.

## Tools

1. **glob_search(pattern, path?, max_results?)** - Find files by name pattern
2. **grep_search(pattern, path?, glob_filter?, ignore_case?, output_mode?)** - Search file contents
3. **read_file(file_path, start_line?, end_line?)** - Read file contents
4. **list_directory(path?)** - List directory contents
5. **find_definition(name, definition_type?)** - Find class/function definitions
6. **find_imports(module_name, ignore_case?)** - Find files that import a module (finds real usage!)

## Search Techniques (CRITICAL)

### Technique 1: Search whole repo first, NEVER restrict path initially
```python
# WRONG - misses examples/models/qwen_model_example.py
glob_search("**/*qwen*.py", path="camel/models")

# CORRECT - searches entire repo, finds ALL qwen files
glob_search("**/*qwen*.py")
```

### Technique 2: Search multiple naming variants
```python
glob_search("**/*qwen*.py")   # lowercase
glob_search("**/*Qwen*.py")   # capitalized
glob_search("**/*qwen2*.py")  # with version
```

### Technique 3: Use find_imports to find REAL usage (most important!)
```python
# Find files that actually USE these modules
find_imports("WeatherToolkit")
find_imports("QwenModel")
find_imports("qwen")
```
This is better than grep because it specifically finds import statements.

### Technique 4: Read __init__.py to understand module exports
```python
# Quickly see what's available in a module
read_file("camel/toolkits/__init__.py")
read_file("camel/models/__init__.py")
```

### Technique 5: Search class definition vs instantiation
```python
# Find where class is DEFINED
grep_search("^class.*Qwen", output_mode="content")

# Find where class is USED (instantiated)
grep_search("QwenModel\\(|Qwen\\(", output_mode="files")
```

### Technique 6: Parallel search for ALL keywords at once
For task "weather agent with Qwen", call ALL of these in parallel:
```python
glob_search("**/*qwen*.py")
glob_search("**/*weather*.py")
find_imports("qwen")
find_imports("WeatherToolkit")
```

### Technique 7: Tests contain best usage examples
```python
# Test files show complete working examples
glob_search("**/test_*qwen*.py")
glob_search("**/test_*weather*.py")
```

## Project Structure Knowledge

- `examples/models/` - Model usage examples (qwen_model_example.py, etc.)
- `examples/toolkits/` - Toolkit examples
- `camel/models/` - Model implementations
- `camel/toolkits/` - Toolkit implementations
- `camel/configs/` - Configuration classes
- `test/` - Test files (great for usage examples)

## Output Format

```
## Examples (check these first for writing scripts)
- examples/models/qwen_model_example.py - shows QwenModel usage

## Implementation
- camel/models/qwen_model.py - QwenModel class definition

## Tests (also good usage references)
- test/models/test_qwen_model.py - test cases showing usage
```

## Rules
- READ-ONLY mode
- Search WHOLE repo first (no path restriction)
- Use parallel searches for multiple keywords
- No emojis
"""

# Architecture description for context
CAMEL_ARCHITECTURE = """
## CAMEL Repository Structure

### Core Modules
- `camel/agents/` - Agent implementations (ChatAgent is the base)
- `camel/models/` - LLM provider integrations (50+)
- `camel/configs/` - Model configuration classes
- `camel/toolkits/` - Tool integrations (50+)
- `camel/messages/` - Message types
- `camel/types/` - Enums (ModelPlatformType, ModelType, RoleType)
- `camel/memories/` - Memory systems
- `camel/runtimes/` - Execution environments
- `camel/societies/` - Multi-agent coordination
- `camel/datagen/` - Synthetic data generation

### Other Directories
- `examples/` - Usage examples
- `test/` - Test files
- `docs/` - Documentation
"""

# Create the optimized explore agent
explore_agent = ChatAgent(
    system_message=EXPLORE_SYSTEM_PROMPT,
    model=model,
    tools=tools,
)


def run_exploration(task_name: str, task_description: str) -> dict:
    """Run exploration for a single task and return results."""
    explore_agent.reset()

    start_time = time.time()
    start_datetime = datetime.now()

    # Construct the exploration prompt
    explore_prompt = f"""Find relevant files for this task:

**Task**: {task_description}

**Repository Context**:
{CAMEL_ARCHITECTURE}

Search for implementation files, examples, tests, and documentation related to this task.
Return the most relevant file paths with brief explanations.
"""

    # Run exploration
    explore_response = explore_agent.step(explore_prompt)

    end_time = time.time()
    end_datetime = datetime.now()
    duration_seconds = end_time - start_time

    # Extract results
    task_output = (
        explore_response.msgs[0].content
        if explore_response and explore_response.msgs
        else ""
    )

    tool_calls = explore_response.info.get("tool_calls", []) if hasattr(explore_response, "info") else []

    return {
        "task_name": task_name,
        "task_description": task_description,
        "output": task_output,
        "tool_calls": tool_calls,
        "duration_seconds": duration_seconds,
        "start_time": start_datetime,
        "end_time": end_datetime,
        "chat_history": explore_agent.chat_history,
    }


def save_log(result: dict, logs_dir: str):
    """Save exploration results to a log file."""
    timestamp = result["start_time"].strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"explore_{result['task_name']}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    with open(log_filepath, "w", encoding="utf-8") as log_file:
        log_file.write(f"Task: {result['task_name']}\n")
        log_file.write(f"Start Time: {result['start_time'].isoformat()}\n")
        log_file.write(f"End Time: {result['end_time'].isoformat()}\n")
        log_file.write(f"Duration: {result['duration_seconds']:.2f} seconds\n")
        log_file.write(f"Description: {result['task_description']}\n")
        log_file.write("=" * 80 + "\n\n")

        # Tool calls summary
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            log_file.write(f"Tool Calls ({len(tool_calls)} total):\n")
            log_file.write("-" * 80 + "\n")
            for i, tool_call in enumerate(tool_calls, 1):
                log_file.write(f"\n[{i}] {tool_call.tool_name}\n")
                log_file.write(f"Args: {json.dumps(tool_call.args, indent=2, ensure_ascii=False)}\n")
                result_preview = str(tool_call.result)[:500]
                log_file.write(f"Result: {result_preview}...\n" if len(str(tool_call.result)) > 500 else f"Result: {result_preview}\n")
            log_file.write("=" * 80 + "\n\n")

        # Final output
        log_file.write("Final Output:\n")
        log_file.write("-" * 80 + "\n")
        log_file.write(result["output"])
        log_file.write("\n")

    return log_filepath


# Main execution
if __name__ == "__main__":
    with open("code-agent/task_list.json", "r") as f:
        task_list = json.load(f)

    cnt = 0
    for task_name, task_description in task_list.items():
        print(f"\n{'='*60}")
        print(f"Processing: {task_name}")
        print(f"{'='*60}")

        result = run_exploration(task_name, task_description)

        print(f"Completed in {result['duration_seconds']:.2f} seconds")
        print(f"Output preview: {result['output'][:300]}...")

        log_path = save_log(result, logs_dir)
        print(f"Log saved to: {log_path}")

        cnt += 1
        if cnt >= 20:  # Limit for testing
            break

    print(f"\nCompleted {cnt} tasks. Logs saved to {logs_dir}")
