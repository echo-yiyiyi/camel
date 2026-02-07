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
Generic Code Agent with two-phase approach:
1. Explore Phase: Fast code exploration using GenericCodeSearchToolkit
2. Code Phase: Write and execute scripts using TerminalToolkit

This is a framework-agnostic version that can be used with any codebase.
Load project-specific context from a markdown file before running tasks.
"""

import asyncio
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool, Crawl4AIToolkit
from camel.types import ModelPlatformType, ModelType

# Import the generic code search toolkit
from generic_code_search_toolkit import GenericCodeSearchToolkit

set_log_level('INFO')


# =============================================================================
# Generic System Prompts (Framework-Agnostic)
# =============================================================================

GENERIC_EXPLORE_SYSTEM_PROMPT = """You are a code exploration specialist. Find relevant files for a task.

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
# WRONG - may miss files in other directories
glob_search("**/*handler*.py", path="src/handlers")

# CORRECT - searches entire repo
glob_search("**/*handler*.py")
```

### Technique 2: Use find_imports to find REAL usage
```python
find_imports("requests")
find_imports("UserService")
```

### Technique 3: Read __init__.py when class names don't match file names
```python
# If glob_search("**/*usermanager*.py") finds nothing, read the module index
read_file("src/users/__init__.py")
# Or search for the class definition directly
grep_search("class.*UserManager", ignore_case=True)
```

### Technique 4: Parallel search for ALL keywords
For any multi-keyword task, search ALL keywords in parallel:
```python
glob_search("**/*keyword1*.py")
glob_search("**/*keyword2*.py")
find_imports("relevant_module")
```

### Technique 5: Tests and examples contain best usage patterns
```python
glob_search("**/test_*.py")
glob_search("**/*example*.py")
```

### Technique 6: Find SPECIFIC function/method when task specifies it
When task mentions a specific feature (e.g., "send email"), find the exact function:
```python
grep_search("def.*send.*email", ignore_case=True)
grep_search("def.*mail", ignore_case=True)
```
Report the exact function name, not just the file path.

## Output Format

For each file, **list the core classes/functions inside**:

```
## Documentation
- path/to/doc.md - Brief description of content

## Examples (MOST IMPORTANT)
- path/to/example - What it demonstrates
  - Key functions: func1(), func2()

## Implementation
- path/to/module
  - Classes: ClassName1, ClassName2
  - Functions: func1, func2, func3

## Configuration
- path/to/config - What settings it contains

## Tests
- path/to/tests - Test cases showing usage
```

## Rules
- READ-ONLY mode
- Search WHOLE repo first (no path restriction)
- Use parallel searches
- **Read __init__.py to discover all available classes**
- **Check docs/ for documentation and tutorials**
- **List core classes/functions for each file you find**
- No emojis
"""


GENERIC_CODE_SYSTEM_PROMPT = """You are a Code Agent that writes and executes scripts.

## Tools

You have access to terminal tools for:
- `shell_exec`: Execute shell commands
- `shell_write_content_to_file`: Write content to a file
- `read_file_tool`: Read file contents

## Workflow (MUST FOLLOW IN ORDER)

### Step 0: FOLLOW PROJECT CONTEXT RULES
**If Project-Specific Context is provided below, follow its rules EXACTLY.**
- These rules override any patterns you see in example files
- Pay special attention to "CRITICAL" or "ALWAYS/NEVER" instructions

### Step 1: READ BEFORE WRITE (MANDATORY)
**You MUST read files that are relevant to what you're going to write before writing any code.**
- Read example files or implementation files that relate to your task
- You don't need to read all files - only the ones relevant to your task
- **DO NOT write code based only on the summary** - the summary may be incomplete

### Step 2: Write the script
- Copy import statements and API patterns EXACTLY from the files you read
- But if Project Context has different rules, follow Project Context
- Use `shell_write_content_to_file` to save the script

### Step 3: Execute the script
- Use `shell_exec` with the appropriate interpreter (python, node, etc.)

### Step 4: Debug if needed
- If errors occur, read more files to understand the correct API, then fix and re-run

## CRITICAL: Strict Instruction Following

**Use EXACTLY what the task specifies. Do NOT substitute "similar" or "better" alternatives.**

### General Rules
- If task specifies an exact name/value -> Use it EXACTLY, not a "similar" alternative
- If task specifies a function/method -> Find and use that EXACT function
- If task specifies a library or version -> Use that EXACT library/version
- If task specifies a file path -> Save to that EXACT path
- Do NOT add features, parameters, or "improvements" not requested in the task

### Finding Specific Functions
When task asks for a specific feature or function:
1. Search for the exact function name in the codebase
2. Use that specific function directly, not a wrapper or helper class
3. Example: Task says "send email" -> Find and use `send_email()`, not a generic `EmailService` class

### URL Handling
If task provides a URL, prioritize direct HTTP access (requests, fetch, etc.) over complex setup

## Important Rules

1. **Use the provided file paths** - The explore agent has found relevant examples and implementations
2. **Follow existing patterns** - Look at examples to understand how to use the APIs
3. **Handle errors gracefully** - If execution fails, analyze the error and fix it
4. **Save to correct location** - Follow the task's specified output path
5. **Use EXACT values** - Copy constants, method names exactly from explored files

## Debugging Tips

- If you see import errors, check the correct module path in __init__.py
- If you see AttributeError, verify the exact attribute name in source files
- Read more files if you need to understand the API better
- **If a file does not exist, do NOT retry the same path** - search for the correct path instead
"""


# =============================================================================
# Generic Code Agent Class
# =============================================================================

class GenericCodeAgent:
    """A generic code agent that can work with any codebase.

    This agent uses a two-phase approach:
    1. Explore Phase: Find relevant files using GenericCodeSearchToolkit
    2. Code Phase: Write and execute scripts using TerminalToolkit

    Project-specific context can be loaded from a markdown file.
    """

    def __init__(
        self,
        working_directory: str,
        context_file: Optional[str] = None,
        logs_dir: Optional[str] = None,
        exclude_dirs: Optional[set] = None,
        model_platform: ModelPlatformType = ModelPlatformType.OPENAI,
        model_type: ModelType = ModelType.GPT_4_1_MINI,
        exp_id: str = "",
    ):
        """Initialize the GenericCodeAgent.

        Args:
            working_directory: The root directory for code exploration.
            context_file: Path to a markdown file with project-specific context.
            logs_dir: Directory to save logs. Defaults to working_directory/logs.
            exclude_dirs: Additional directories to exclude from search.
            model_platform: The model platform to use.
            model_type: The model type to use.
            exp_id: Experiment ID for logging.
        """
        self.working_dir = Path(working_directory).resolve()
        self.exp_id = exp_id

        # Setup logs directory
        if logs_dir:
            self.logs_dir = Path(logs_dir)
        else:
            self.logs_dir = self.working_dir / f"logs{exp_id}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load project-specific context
        self.project_context = ""
        if context_file:
            context_path = Path(context_file)
            if context_path.exists():
                self.project_context = context_path.read_text(encoding='utf-8')

        # Setup model
        model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()
        self.model = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_type,
            model_config_dict=model_config_dict,
        )

        # Setup explore toolkit
        self.explore_toolkit = GenericCodeSearchToolkit(
            working_directory=str(self.working_dir),
            exclude_dirs=exclude_dirs,
            max_results=50,
        )
        explore_tools = self.explore_toolkit.get_tools()

        # Build explore system prompt with project context
        explore_prompt = GENERIC_EXPLORE_SYSTEM_PROMPT
        if self.project_context:
            explore_prompt += f"\n\n## Project-Specific Context\n\n{self.project_context}"
        
        # print(f"explore_prompt: {explore_prompt}")
        # exit()

        self.explore_agent = ChatAgent(
            system_message=explore_prompt,
            model=self.model,
            tools=explore_tools,
        )

        # Setup terminal toolkit for code phase
        self.terminal_toolkit = TerminalToolkit(
            working_directory=str(self.working_dir),
            clone_current_env=True,
            timeout=300.0,
        )
        terminal_tools = self.terminal_toolkit.get_tools()

        # Create read_file tool using explore toolkit
        def read_file_tool(file_path: str) -> str:
            r"""Read the content of a file.

            If the file is less than 200 lines, reads the whole file.
            Otherwise, reads the first 2000 lines.

            Args:
                file_path (str): The path of the file to read.

            Returns:
                str: The content of the file.
            """
            return self.explore_toolkit.read_file(file_path)

        # Build code system prompt with project context
        code_prompt = GENERIC_CODE_SYSTEM_PROMPT
        if self.project_context:
            code_prompt += f"\n\n## Project-Specific Context\n\n{self.project_context}"

        crawl4ai_toolkit = Crawl4AIToolkit()

        self.code_agent = ChatAgent(
            system_message=code_prompt,
            model=self.model,
            tools=[FunctionTool(read_file_tool)] + terminal_tools + crawl4ai_toolkit.get_tools(),
        )

    async def run_explore_phase(self, task_description: str) -> tuple:
        """Run the exploration phase to find relevant files."""
        self.explore_agent.reset()

        explore_prompt = f"""Find relevant files for this task:

**Task**: {task_description}

Search for implementation files, examples, tests, and documentation.
Return the most relevant file paths with brief explanations.
Focus on finding:
1. Example scripts that show similar usage patterns
2. Implementation files for the required components
3. Configuration/type definitions
"""

        error_msg = None
        try:
            response = await self.explore_agent.astep(explore_prompt)
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

        return output, tool_calls, self.explore_agent.chat_history, error_msg

    async def run_code_phase(self, task_description: str, explore_output: str) -> tuple:
        """Run the code generation phase to write and execute the script."""
        self.code_agent.reset()

        # Extract all file paths from explore output
        relevant_files = []
        for line in explore_output.split('\n'):
            line = line.strip()
            # Match lines with file extensions
            if any(ext in line for ext in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.md']):
                parts = line.lstrip('- ').split()
                for part in parts:
                    if any(part.endswith(ext) for ext in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.md']):
                        relevant_files.append(part)
                        break

        # Remove duplicates while preserving order
        seen = set()
        relevant_files = [f for f in relevant_files if not (f in seen or seen.add(f))]

        file_instruction = ""
        if relevant_files:
            file_instruction = f"""
**Available files** (use `read_file_tool` to read):
{chr(10).join(f'- {f}' for f in relevant_files[:10])}

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
3. Execute with appropriate command
4. If errors occur, read more files to understand the correct API, then fix and re-run

**WARNINGS**:
- DO NOT write code based only on the summary - read actual files first
- Copy code patterns EXACTLY from the files you read

Note: The environment and APIs are correctly configured. If you encounter errors,
it's likely a script issue that needs fixing.
"""
        error_msg = None
        try:
            response = await self.code_agent.astep(code_prompt)
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

        return output, tool_calls, self.code_agent.chat_history, error_msg

    def save_log(
        self,
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
        log_filepath = self.logs_dir / log_filename

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

        return str(log_filepath)

    async def run_task(self, task_name: str, task_description: str) -> dict:
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
        explore_output, explore_tool_calls, explore_history, explore_error = await self.run_explore_phase(task_description)
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
            code_output, code_tool_calls, code_history, code_error = await self.run_code_phase(task_description, explore_output)
        code_duration = time.time() - code_start
        if code_error:
            print(f"Code phase error: {code_error}")
        else:
            print(f"Code completed in {code_duration:.2f}s, made {len(code_tool_calls)} tool calls")

        end_time = datetime.now()

        # Save log (always save, even on error)
        log_path = self.save_log(
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
# Helper function for model parsing
# =============================================================================
def parse_model(model_str: str) -> tuple:
    """Parse model string like 'openai/gpt-4o-mini' into platform and type."""
    model_mapping = {
        "openai/gpt-4o-mini": (ModelPlatformType.OPENAI, ModelType.GPT_4O_MINI),
        "openai/gpt-4o": (ModelPlatformType.OPENAI, ModelType.GPT_4O),
        "openai/gpt-4-1-mini": (ModelPlatformType.OPENAI, ModelType.GPT_4_1_MINI),
        "openai/gpt-4-1": (ModelPlatformType.OPENAI, ModelType.GPT_4_1),
        "anthropic/claude-3-5-sonnet": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_5_SONNET),
        "anthropic/claude-3-opus": (ModelPlatformType.ANTHROPIC, ModelType.CLAUDE_3_OPUS),
        "default": (ModelPlatformType.OPENAI, ModelType.GPT_4_1_MINI),
    }
    return model_mapping.get(model_str.lower(), model_mapping["default"])


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse
    import sys

    # Setup default paths
    # base_dir is generic/, go up twice to reach camel root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    code_agent_dir = os.path.dirname(base_dir)  # code-agent/
    default_camel_dir = os.path.dirname(code_agent_dir)  # camel root

    parser = argparse.ArgumentParser(
        description="GenericCodeAgent - A two-phase code agent for any codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on CAMEL project with CAMEL.md context (default)
    python generic_code_agent.py --task "Create an agent that uses GPT-4"

    # Run on CAMEL project WITHOUT context (pure generic mode)
    python generic_code_agent.py --no-context --task "Create an agent"

    # Run on a custom project with custom context
    python generic_code_agent.py \\
        --project /path/to/myproject \\
        --context /path/to/context.md \\
        --task "Implement user authentication"

    # Run on a custom project without context
    python generic_code_agent.py \\
        --project /path/to/myproject \\
        --no-context \\
        --task "Find the main entry point"

    # Run multiple tasks from JSON file
    python generic_code_agent.py --tasks tasks.json --max-tasks 5

    # Use a different model
    python generic_code_agent.py --model openai/gpt-4o --task "Refactor the module"
        """
    )

    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help=f"Project directory to explore. Default: {default_camel_dir}"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Path to project context markdown file. Default: CAMEL.md for CAMEL project."
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        default=False,
        help="Run without loading any project context (pure generic mode)."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="Single task description to run."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Path to JSON file with task list (dict of task_name: description)."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openai/gpt-4-1-mini",
        help="Model to use (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet)"
    )
    parser.add_argument(
        "--logs-dir", "-l",
        type=str,
        default=None,
        help="Directory to save logs. Default: <project>/logs<exp_id>"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Additional directories to exclude from search."
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to run from tasks file. Default: 10"
    )
    parser.add_argument(
        "--parallel", "-P",
        type=int,
        default=1,
        help="Number of parallel agent instances. Default: 1 (sequential)"
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default="",
        help="Experiment ID for logging (appended to logs dir name)."
    )

    args = parser.parse_args()

    # Validate: need either --task or --tasks
    if not args.task and not args.tasks:
        parser.error("Either --task or --tasks must be specified")

    # Setup project directory
    if args.project:
        project_dir = os.path.abspath(args.project)
    else:
        project_dir = default_camel_dir

    # Setup context file
    if args.no_context:
        context_file = None
    elif args.context:
        context_file = os.path.abspath(args.context)
    elif not args.project:
        # Default to CAMEL.md for CAMEL project
        context_file = os.path.join(base_dir, "CAMEL.md")
    else:
        context_file = None

    # Setup exclude directories
    exclude_dirs = {
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build',
        '.initial_env', 'task-script*', 'CAMEL_old.md'
    }
    if args.exclude:
        exclude_dirs.update(args.exclude)

    # Parse model
    model_platform, model_type = parse_model(args.model)

    # Validate paths
    if not os.path.isdir(project_dir):
        print(f"Error: Project directory does not exist: {project_dir}")
        sys.exit(1)

    if context_file and not os.path.isfile(context_file):
        print(f"Warning: Context file does not exist: {context_file}")
        context_file = None

    # Print configuration
    print("=" * 60)
    print("GenericCodeAgent")
    print("=" * 60)
    print(f"Project:  {project_dir}")
    print(f"Context:  {context_file or 'None (pure generic mode)'}")
    print(f"Model:    {args.model}")
    print(f"Logs:     {args.logs_dir or 'default'}")
    print("=" * 60)

    # Initialize agent
    agent = GenericCodeAgent(
        working_directory=project_dir,
        context_file=context_file,
        logs_dir=args.logs_dir,
        exclude_dirs=exclude_dirs,
        model_platform=model_platform,
        model_type=model_type,
        exp_id=args.exp_id,
    )

    # Helper function to create agent instance
    def create_agent():
        return GenericCodeAgent(
            working_directory=project_dir,
            context_file=context_file,
            logs_dir=args.logs_dir,
            exclude_dirs=exclude_dirs,
            model_platform=model_platform,
            model_type=model_type,
            exp_id=args.exp_id,
        )

    # Run tasks
    async def main():
        if args.task:
            # Single task mode
            result = await agent.run_task("single_task", args.task)
            print(f"\n--- Result ---")
            print(f"Status: {'ERROR' if result['has_error'] else 'SUCCESS'}")
            print(f"Duration: {result['total_duration']:.2f}s")
            print(f"Log: {result['log_path']}")

        elif args.tasks:
            # Multiple tasks from JSON file
            if not os.path.isfile(args.tasks):
                print(f"Error: Tasks file does not exist: {args.tasks}")
                sys.exit(1)

            with open(args.tasks, "r") as f:
                task_list = json.load(f)

            # Get tasks to run
            tasks_to_run = list(task_list.items())[:args.max_tasks]
            total_tasks = len(tasks_to_run)

            if args.parallel > 1:
                # Parallel execution with multiple agent instances
                print(f"\n[Parallel mode: {args.parallel} agents]")

                async def run_single_task(task_info):
                    task_name, task_desc = task_info
                    task_agent = create_agent()
                    return await task_agent.run_task(task_name, task_desc)

                # Process in batches
                results = []
                for i in range(0, total_tasks, args.parallel):
                    batch = tasks_to_run[i:i + args.parallel]
                    print(f"\n[Batch {i//args.parallel + 1}: "
                          f"tasks {i+1}-{min(i+len(batch), total_tasks)}]")
                    batch_results = await asyncio.gather(
                        *[run_single_task(t) for t in batch]
                    )
                    results.extend(batch_results)
            else:
                # Sequential execution
                results = []
                for task_name, task_description in tasks_to_run:
                    result = await agent.run_task(task_name, task_description)
                    results.append(result)

                    print(f"\n--- Result Preview ---")
                    output_preview = (result['code_output'][:500]
                                      if result['code_output'] else "N/A")
                    print(f"Code output: {output_preview}...")

            # Summary
            print(f"\n{'='*60}")
            print(f"Summary: {total_tasks} tasks completed")
            print(f"Success: {sum(1 for r in results if not r['has_error'])}")
            print(f"Errors:  {sum(1 for r in results if r['has_error'])}")
            print(f"Logs:    {agent.logs_dir}")

    asyncio.run(main())
