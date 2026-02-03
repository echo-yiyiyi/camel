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
Optimized Code Agent with two-phase approach:
1. Explore Phase: Fast code exploration using optimized CodeSearchToolkit
2. Code Phase: Write and execute scripts using TerminalToolkit

Key improvements:
- Optimized file search (glob/grep/find_imports)
- Separate agents for exploration and code generation
- Better error handling and debugging loop
"""

import os
import json
import time
from datetime import datetime
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool, Crawl4AIToolkit
from camel.types import ModelPlatformType, ModelType

# Import the optimized code search toolkit
from code_search_toolkit import CodeSearchToolkit

exp_id = "_code_optimized_read_file_4"

set_log_level('INFO')

# =============================================================================
# Directory Setup
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
camel_dir = os.path.dirname(base_dir)
project_root = os.path.dirname(camel_dir)
workspace_dir = os.path.join(project_root, "workspace")
logs_dir = os.path.join(base_dir, f"logs{exp_id}")
os.makedirs(logs_dir, exist_ok=True)

# =============================================================================
# Model Configuration
# =============================================================================
model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4_1_MINI,
    model_config_dict=model_config_dict,
)

# =============================================================================
# Explore Agent Setup (Optimized)
# =============================================================================
explore_toolkit = CodeSearchToolkit(
    working_directory=camel_dir,
    exclude_dirs={
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build',
        '.initial_env', 'code-agent', 'task-script*'
    },
    max_results=50,
)

explore_tools = explore_toolkit.get_tools()

EXPLORE_SYSTEM_PROMPT = """You are a code exploration specialist. Find relevant files for a task.

## Tools

1. **glob_search(pattern, path?, max_results?)** - Find files by name pattern
2. **grep_search(pattern, path?, glob_filter?, ignore_case?, output_mode?)** - Search file contents
3. **read_file(file_path)** - Read file contents (auto-limits to 2000 lines for large files)
4. **list_directory(path?)** - List directory contents
5. **find_definition(name, definition_type?)** - Find class/function definitions
6. **find_imports(module_name, ignore_case?)** - Find files that import a module

## Search Techniques (CRITICAL)

### Technique 1: Search whole repo first, NEVER restrict path initially
```python
# WRONG - misses examples/models/llama_model_example.py
glob_search("**/*llama*.py", path="camel/models")

# CORRECT - searches entire repo
glob_search("**/*llama*.py")
```

### Technique 2: Use find_imports to find REAL usage
```python
find_imports("WeatherToolkit")
find_imports("LlamaModel")
```

### Technique 3: Read __init__.py to understand module exports
```python
read_file("camel/toolkits/__init__.py")
read_file("camel/models/__init__.py")
```

### Technique 4: Parallel search for ALL keywords
For task "weather agent with Llama", call ALL in parallel:
```python
glob_search("**/*llama*.py")
glob_search("**/*weather*.py")
find_imports("llama")
find_imports("WeatherToolkit")
```

### Technique 5: Tests and examples contain best usage patterns
```python
glob_search("**/test_*llama*.py")
glob_search("**/*example*.py")
```

### Technique 6: Find EXACT enum values for models/tools
When task specifies a model like "Llama-3.1-8B-Instruct", search for the exact ModelType enum:
```python
# Search with partial match (enum names use underscores, not hyphens)
grep_search("LLAMA_3_1.*8B", path="camel/types/enums.py")
grep_search("LLAMA_3_1", path="camel/types/enums.py")
```
CAMEL ModelType naming convention:
- No "_INSTRUCT" suffix usually (e.g., LLAMA_3_1_8B not LLAMA_3_1_8B_INSTRUCT)
- Use underscores not hyphens (e.g., LLAMA_3_1 not LLAMA-3.1)

### Technique 7: Find SPECIFIC tool methods
When task specifies a tool like "brave search", find the exact method:
```python
grep_search("search_brave", path="camel/toolkits")
grep_search("def search_brave")
```
Don't use get_tools() if task asks for a specific tool - use the specific method directly.

### Technique 8: When file name search fails, read __init__.py
Class names often don't match file names. When glob_search by name fails:
```python
# Task asks for "longterm memory" but glob_search("**/*longterm*.py") finds nothing
# Solution: Read __init__.py to see all exported classes
read_file("camel/memories/__init__.py")
# This reveals: LongtermAgentMemory is in agent_memories.py!

# Or search for the class name directly
grep_search("class.*Longterm", ignore_case=True)
```
Always read the module's `__init__.py` to discover all available classes.

## Project Structure

- `examples/models/` - Model usage examples
- `examples/toolkits/` - Toolkit examples
- `examples/memories/` - Memory usage examples
- `camel/models/` - Model implementations
- `camel/toolkits/` - Toolkit implementations
- `camel/memories/` - Memory implementations (ChatHistoryMemory, LongtermAgentMemory, VectorDBMemory)
- `camel/configs/` - Configuration classes
- `camel/types/enums.py` - ModelType, ModelPlatformType enums
- `test/` - Test files

## Output Format

For each file, **list the core classes/functions inside**:

```
## Examples (MOST IMPORTANT)
- examples/models/llama_model_example.py
  - Shows: ModelFactory.create() with LLAMA model
  - Key usage: model_platform=ModelPlatformType.TOGETHER, model_type=ModelType.LLAMA_3_1_8B

## Implementation
- camel/memories/agent_memories.py
  - Classes: AgentMemory, ChatHistoryMemory, VectorDBMemory, **LongtermAgentMemory**
  - Note: LongtermAgentMemory is for persistent memory across sessions

- camel/models/togetherai_model.py
  - Classes: TogetherAIModel
  - Config: TogetherAIConfig

## Enums (for exact values)
- camel/types/enums.py:317 - LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

## Tests
- test/models/test_qwen_model.py - test cases showing parameter usage
```

**CRITICAL**: Always read `__init__.py` to discover all exported classes. Class names often don't match file names (e.g., `LongtermAgentMemory` is in `agent_memories.py`, not `longterm_memory.py`).

## Rules
- READ-ONLY mode
- Search WHOLE repo first (no path restriction)
- Use parallel searches
- Find EXACT enum values for specified models
- Find SPECIFIC tool methods for specified tools
- **Read __init__.py to discover all available classes**
- **List core classes/functions for each file you find**
- No emojis
"""

explore_agent = ChatAgent(
    system_message=EXPLORE_SYSTEM_PROMPT,
    model=model,
    tools=explore_tools,
)

# =============================================================================
# Code Agent Setup (with TerminalToolkit for writing/executing)
# =============================================================================
terminal_toolkit = TerminalToolkit(
    working_directory=camel_dir,
    clone_current_env=True,
    timeout=300.0,  # 5 minutes timeout for code execution
)

terminal_tools = terminal_toolkit.get_tools()

# Create read_file tool using CodeSearchToolkit (shell-based for consistency)
def read_file_tool(file_path: str) -> str:
    r"""Read the content of a file.

    If the file is less than 200 lines, reads the whole file.
    Otherwise, reads the first 2000 lines.

    Args:
        file_path (str): The path of the file to read.

    Returns:
        str: The content of the file.
    """
    return explore_toolkit.read_file(file_path)


CODE_SYSTEM_PROMPT = """You are a Code Agent that writes and executes Python scripts.

## Tools

You have access to terminal tools for:
- `shell_exec`: Execute shell commands
- `shell_write_content_to_file`: Write content to a file
- `read_file_tool`: Read file contents

## Workflow (MUST FOLLOW IN ORDER)

### Step 1: READ BEFORE WRITE (MANDATORY)
**You MUST read files that are relevant to what you're going to write before writing any code.**
- Read example files or implementation files that relate to your task
- You don't need to read all files - only the ones relevant to your task
- **DO NOT write code based only on the summary** - the summary may be incomplete

### Step 2: Write the script
- Copy import statements and API patterns EXACTLY from the files you read
- Use `shell_write_content_to_file` to save the script

### Step 3: Execute the script
- Use `shell_exec` with `python <script_path>`

### Step 4: Debug if needed
- If errors occur, read more files to understand the correct API, then fix and re-run

## CRITICAL: Strict Instruction Following

**Use EXACTLY what the task specifies. Do NOT substitute "similar" or "better" alternatives.**

### Model Selection
- If task does NOT specify a model -> Use `ModelPlatformType.DEFAULT` and `ModelType.DEFAULT`
- Task says "Llama-3.1-8B-Instruct" -> Use `ModelType.LLAMA_3_1_8B` EXACTLY
- Do NOT use LLAMA_3_2_3B as "closest alternative"
- Do NOT guess enum names - verify from enums.py or examples
- ModelPlatformType and ModelType MUST match according to enums.py (e.g., MODELSCOPE models use MODELSCOPE platform, QWEN models use QWEN platform)

### Tool Selection
- Task says "brave search" -> Use `search_toolkit.search_brave` EXACTLY
- Do NOT use `get_tools()` which returns ALL tools
- Do NOT substitute with other search engines

### URL Handling
- If task provides a URL (e.g., "https://mcpservers.org/servers/devin/deepwiki"):
  - Prioritize DIRECT HTTP access using `requests` library
  - Create a simple FunctionTool that calls the URL
  - Do NOT assume it requires complex registry (ACI, etc.)
  - Example:
    ```python
    def query_server(query: str) -> str:
        response = requests.post(url, json={"query": query})
        return response.text
    tool = FunctionTool(func=query_server)
    ```

## Important Rules

1. **Use the provided file paths** - The explore agent has found relevant examples and implementations
2. **Follow existing patterns** - Look at examples to understand how to use the APIs
3. **Handle errors gracefully** - If execution fails, analyze the error and fix it
4. **Environment is ready** - All API keys and dependencies are configured
5. **Save to correct location** - Follow the task's specified output path
6. **Use EXACT values** - Copy enum values, method names exactly from explored files

## Script Writing Tips

- Import from camel package: `from camel.agents import ChatAgent`
- Use ModelFactory for models: `ModelFactory.create(model_platform=..., model_type=...)`
- If task does NOT specify a model, use DEFAULT: `ModelPlatformType.DEFAULT, ModelType.DEFAULT`
- Copy exact enum values from enums.py (e.g., `ModelType.LLAMA_3_1_8B`)
- Use specific tool methods, not get_tools() (e.g., `toolkit.search_brave`)

## Debugging Tips

- If you see import errors, check the correct module path in __init__.py
- If you see AttributeError for enum, verify the exact enum name in enums.py
- If you see API errors, it's likely a script issue, not environment
- Read more files if you need to understand the API better
- **If a file does not exist, do NOT retry the same path** - search for the correct path instead (e.g., use glob_search or grep)
"""

crawl4ai_toolkit = Crawl4AIToolkit()

code_agent = ChatAgent(
    system_message=CODE_SYSTEM_PROMPT,
    model=model,
    tools=[FunctionTool(read_file_tool)] + terminal_tools + crawl4ai_toolkit.get_tools(),
)

# =============================================================================
# Architecture Description
# =============================================================================
CAMEL_ARCHITECTURE = """
## CAMEL Repository Structure

### Core Modules
- `camel/agents/` - Agent implementations (ChatAgent is the base)
- `camel/models/` - LLM provider integrations (50+)
- `camel/memories/` - Memory systems (ChatHistoryMemory, LongtermAgentMemory, VectorDBMemory)
- `camel/configs/` - Model configuration classes
- `camel/toolkits/` - Tool integrations (50+)
- `camel/types/` - Enums (ModelPlatformType, ModelType, RoleType)

### Other Directories
- `examples/` - Usage examples
- `test/` - Test files
- `docs/` - Documentation
"""


# =============================================================================
# Main Execution Functions
# =============================================================================
def run_explore_phase(task_description: str) -> tuple:
    """Run the exploration phase to find relevant files."""
    explore_agent.reset()

    explore_prompt = f"""Find relevant files for this task:

**Task**: {task_description}

**Repository Context**:
{CAMEL_ARCHITECTURE}

Search for implementation files, examples, tests, and documentation.
Return the most relevant file paths with brief explanations.
Focus on finding:
1. Example scripts that show similar usage patterns
2. Implementation files for the required components
3. Configuration/type definitions
"""

    error_msg = None
    try:
        response = explore_agent.step(explore_prompt)
        output = response.msgs[0].content if response and response.msgs else ""
        tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []
    except KeyboardInterrupt:
        error_msg = "Task interrupted by user (Ctrl+C)"
        print(f"\n[!] {error_msg}")
        output = f"[ERROR] {error_msg}"
        tool_calls = []
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[!] Error in explore phase: {error_msg}")
        output = f"[ERROR] {error_msg}"
        tool_calls = []

    return output, tool_calls, explore_agent.chat_history, error_msg


def run_code_phase(task_description: str, explore_output: str) -> tuple:
    """Run the code generation phase to write and execute the script."""
    code_agent.reset()

    # Extract all .py file paths from explore output for explicit instruction
    relevant_files = []
    for line in explore_output.split('\n'):
        line = line.strip()
        # Match lines like "- examples/xxx.py" or "- camel/xxx.py" or just "examples/xxx.py"
        if '.py' in line:
            # Extract path that ends with .py
            parts = line.lstrip('- ').split()
            for part in parts:
                if part.endswith('.py') and ('examples/' in part or 'camel/' in part):
                    relevant_files.append(part)
                    break

    # Remove duplicates while preserving order
    seen = set()
    relevant_files = [f for f in relevant_files if not (f in seen or seen.add(f))]

    file_instruction = ""
    if relevant_files:
        file_instruction = f"""
**Available files** (use `read_file_tool` to read):
{chr(10).join(f'- {f}' for f in relevant_files[:5])}

**IMPORTANT**: Before writing code, you MUST read the files that are relevant to your task.
You don't need to read all files - only the ones related to what you're going to write.
The summary may be incomplete - verify API usage by reading actual files.
"""

    code_prompt = f"""**Task**: {task_description}

**Relevant files found by exploration**:
{explore_output}
{file_instruction}
**Instructions**:
1. **FIRST**: Read the files that are relevant to what you're going to write
2. Write the script using `shell_write_content_to_file`
3. Execute with `shell_exec python <script_path>`
4. If errors occur, read more files to understand the correct API, then fix and re-run

**WARNINGS**:
- DO NOT write code based only on the summary - read actual files first
- Copy code patterns EXACTLY from the files you read
- If task does NOT specify a model, use DEFAULT: ModelPlatformType.DEFAULT, ModelType.DEFAULT

Note: The environment and APIs are correctly configured. If you encounter errors,
it's likely a script issue that needs fixing.
"""
    error_msg = None
    try:
        response = code_agent.step(code_prompt)
        output = response.msgs[0].content if response and response.msgs else ""
        tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []
    except KeyboardInterrupt:
        error_msg = "Task interrupted by user (Ctrl+C)"
        print(f"\n[!] {error_msg}")
        output = f"[ERROR] {error_msg}"
        tool_calls = []
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[!] Error in code phase: {error_msg}")
        output = f"[ERROR] {error_msg}"
        tool_calls = []

    return output, tool_calls, code_agent.chat_history, error_msg


def save_log(
    task_name: str,
    task_description: str,
    explore_output: str,
    explore_tool_calls: list,
    explore_history: list,
    explore_error: str,
    code_output: str,
    code_tool_calls: list,
    code_history: list,
    code_error: str,
    start_time: datetime,
    end_time: datetime,
    explore_duration: float,
    code_duration: float,
):
    """Save complete execution log."""
    total_duration = (end_time - start_time).total_seconds()
    timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Mark filename if there was an error
    status = "error" if (explore_error or code_error) else "ok"
    log_filename = f"code_{task_name}_{status}_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    with open(log_filepath, "w", encoding="utf-8") as f:
        # Header
        f.write(f"Task: {task_name}\n")
        f.write(f"Status: {'ERROR' if (explore_error or code_error) else 'SUCCESS'}\n")
        f.write(f"Start Time: {start_time.isoformat()}\n")
        f.write(f"End Time: {end_time.isoformat()}\n")
        f.write(f"Total Duration: {total_duration:.2f} seconds\n")
        f.write(f"  - Explore Phase: {explore_duration:.2f} seconds\n")
        f.write(f"  - Code Phase: {code_duration:.2f} seconds\n")
        f.write(f"Description: {task_description}\n")
        if explore_error:
            f.write(f"Explore Error: {explore_error}\n")
        if code_error:
            f.write(f"Code Error: {code_error}\n")
        f.write("=" * 80 + "\n\n")

        # Explore Phase
        f.write("EXPLORE PHASE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Tool Calls: {len(explore_tool_calls)}\n")
        f.write("-" * 40 + "\n")
        for i, tc in enumerate(explore_tool_calls, 1):
            f.write(f"\n[{i}] {tc.tool_name}\n")
            f.write(f"Args: {json.dumps(tc.args, indent=2, ensure_ascii=False)}\n")
            result_str = str(tc.result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            f.write(f"Result: {result_str}\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write("Output:\n")
        f.write(explore_output)
        f.write("\n" + "=" * 80 + "\n\n")

        # Code Phase
        f.write("CODE PHASE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Tool Calls: {len(code_tool_calls)}\n")
        f.write("-" * 40 + "\n")
        for i, tc in enumerate(code_tool_calls, 1):
            f.write(f"\n[{i}] {tc.tool_name}\n")
            f.write(f"Args: {json.dumps(tc.args, indent=2, ensure_ascii=False)}\n")
            result_str = str(tc.result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "..."
            f.write(f"Result: {result_str}\n")
        f.write("\n" + "-" * 40 + "\n")
        f.write("Output:\n")
        f.write(code_output)
        f.write("\n" + "=" * 80 + "\n\n")

        # Explore Agent Chat History
        f.write("EXPLORE AGENT CHAT HISTORY\n")
        f.write("=" * 80 + "\n")
        for i, msg in enumerate(explore_history, 1):
            f.write(f"\n[Message {i}]\n")
            f.write(f"Role: {msg.get('role', 'unknown')}\n")
            if 'content' in msg and msg['content']:
                content = msg['content']
                if isinstance(content, str) and len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)\n"
                f.write(f"Content: {content}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Code Agent Chat History
        f.write("CODE AGENT CHAT HISTORY\n")
        f.write("=" * 80 + "\n")
        for i, msg in enumerate(code_history, 1):
            f.write(f"\n[Message {i}]\n")
            f.write(f"Role: {msg.get('role', 'unknown')}\n")
            if 'content' in msg and msg['content']:
                content = msg['content']
                if isinstance(content, str) and len(content) > 5000:
                    content = content[:5000] + "\n... (truncated)\n"
                f.write(f"Content: {content}\n")
            if 'tool_calls' in msg and msg['tool_calls']:
                f.write(f"Tool Calls: {json.dumps(msg['tool_calls'], indent=2, ensure_ascii=False)}\n")
        f.write("\n")

    return log_filepath


def run_task(task_name: str, task_description: str) -> dict:
    """Run complete task: explore + code generation."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    start_time = datetime.now()
    explore_error = None
    code_error = None

    # Phase 1: Explore
    print("\n[Phase 1: Exploring...]")
    explore_start = time.time()
    explore_output, explore_tool_calls, explore_history, explore_error = run_explore_phase(task_description)
    explore_duration = time.time() - explore_start
    if explore_error:
        print(f"Explore phase error: {explore_error}")
    else:
        print(f"Explore completed in {explore_duration:.2f}s, found {len(explore_tool_calls)} tool calls")

    # Phase 2: Code (skip if explore failed completely)
    print("\n[Phase 2: Writing and executing code...]")
    code_start = time.time()
    if explore_error and "[ERROR]" in explore_output:
        code_output = "[SKIPPED] Explore phase failed"
        code_tool_calls = []
        code_history = []
    else:
        code_output, code_tool_calls, code_history, code_error = run_code_phase(task_description, explore_output)
    code_duration = time.time() - code_start
    if code_error:
        print(f"Code phase error: {code_error}")
    else:
        print(f"Code completed in {code_duration:.2f}s, made {len(code_tool_calls)} tool calls")

    end_time = datetime.now()

    # Save log (always save, even on error)
    log_path = save_log(
        task_name=task_name,
        task_description=task_description,
        explore_output=explore_output,
        explore_tool_calls=explore_tool_calls,
        explore_history=explore_history,
        explore_error=explore_error,
        code_output=code_output,
        code_tool_calls=code_tool_calls,
        code_history=code_history,
        code_error=code_error,
        start_time=start_time,
        end_time=end_time,
        explore_duration=explore_duration,
        code_duration=code_duration,
    )

    total_duration = (end_time - start_time).total_seconds()
    status = "ERROR" if (explore_error or code_error) else "SUCCESS"
    print(f"\n[{status}] Total time: {total_duration:.2f}s")
    print(f"Log saved to: {log_path}")

    return {
        "task_name": task_name,
        "explore_output": explore_output,
        "code_output": code_output,
        "total_duration": total_duration,
        "log_path": log_path,
        "has_error": bool(explore_error or code_error),
    }


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    with open("code-agent/task_list.json", "r") as f:
        task_list = json.load(f)

    cnt = 0
    for task_name, task_description in task_list.items():
        cnt += 1
        if cnt <= 0:
            continue
        result = run_task(task_name, task_description)

        print(f"\n--- Result Preview ---")
        print(f"Code output: {result['code_output'][:500]}...")

        if cnt >= 10:  # Limit for testing
            break

    print(f"\n{'='*60}")
    print(f"Completed {cnt} tasks. Logs saved to {logs_dir}")
