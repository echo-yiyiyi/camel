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
Generic Code Single Agent - Unified agent with structured steps.

Key design:
1. Single Agent with shared context (no duplicate file reads)
2. Structured steps: Search → Read → Write → Execute
3. Each step has clear goal, reducing context confusion

This is a framework-agnostic version that can be used with any codebase.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool, Crawl4AIToolkit
from camel.types import ModelPlatformType, ModelType

from generic_code_search_toolkit import GenericCodeSearchToolkit

set_log_level('INFO')


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a Code Agent that explores codebases and writes scripts.

## Tools Available

### Search Tools
- `glob_search(pattern, path?, max_results?)` - Find files by name pattern
- `grep_search(pattern, path?, glob_filter?, ignore_case?, output_mode?)` - Search file contents
- `find_definition(name, definition_type?)` - Find class/function definitions
- `find_imports(module_name, ignore_case?)` - Find files that import a module
- `list_directory(path?)` - List directory contents

### File Tools
- `read_file(file_path)` - Read file contents

### Code Execution Tools
- `shell_exec(id, command)` - Execute shell commands
- `shell_write_content_to_file(content, file_path)` - Write content to a file

## Important Rules

1. **Search whole repo first** - Don't restrict path initially
2. **Copy patterns exactly** - Use exact imports and API calls from files you read
3. **Don't re-read files** - Files stay in your memory after reading
4. **Use exact values from task** - Don't substitute similar alternatives

## Search Techniques

- `glob_search("**/*weather*.py")` - Find files by name
- `find_imports("WeatherToolkit")` - Find real usage examples
- `grep_search("class.*Manager", glob_filter="*.py")` - Search content
- Check `examples/` and `test/` for best usage patterns
"""


# =============================================================================
# Step Definitions
# =============================================================================

class StepResult:
    """Result of a single step execution."""
    def __init__(
        self,
        step_name: str,
        output: str,
        tool_calls: List[Any],
        duration: float,
        error: Optional[str] = None,
    ):
        self.step_name = step_name
        self.output = output
        self.tool_calls = tool_calls
        self.duration = duration
        self.error = error


# =============================================================================
# Generic Code Single Agent Class
# =============================================================================

class GenericCodeSingleAgent:
    """A unified code agent with structured step execution.

    Uses a single ChatAgent with shared context, but executes in clear steps:
    1. Search - Find relevant files
    2. Read - Read and understand key files
    3. Write - Write the script
    4. Execute - Run and debug

    Files read in earlier steps remain in context for later steps.
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
        step_timeout: float = 120.0,
    ):
        """Initialize the GenericCodeSingleAgent."""
        self.working_dir = Path(working_directory).resolve()
        self.exp_id = exp_id
        self.step_timeout = step_timeout

        # Setup logs directory
        if logs_dir:
            self.logs_dir = Path(logs_dir)
        else:
            self.logs_dir = self.working_dir / f"logs_single{exp_id}"
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

        # Setup toolkits
        self.search_toolkit = GenericCodeSearchToolkit(
            working_directory=str(self.working_dir),
            exclude_dirs=exclude_dirs,
            max_results=50,
        )

        self.terminal_toolkit = TerminalToolkit(
            working_directory=str(self.working_dir),
            clone_current_env=True,
            timeout=300.0,
        )

        crawl4ai_toolkit = Crawl4AIToolkit()

        # Combine all tools
        all_tools = (
            self.search_toolkit.get_tools() +
            self.terminal_toolkit.get_tools() +
            crawl4ai_toolkit.get_tools()
        )

        # Build system prompt with project context
        system_prompt = SYSTEM_PROMPT
        if self.project_context:
            system_prompt += f"\n\n## Project-Specific Context\n\n{self.project_context}"

        # Single agent with all tools
        self.agent = ChatAgent(
            system_message=system_prompt,
            model=self.model,
            tools=all_tools,
            step_timeout=self.step_timeout,
        )

    def _run_step(
        self,
        step_name: str,
        prompt: str,
        max_iterations: int = 3,
    ) -> StepResult:
        """Run a single step with the agent.

        Args:
            step_name: Name of the step for logging.
            prompt: The prompt for this step.
            max_iterations: Max iterations within this step.

        Returns:
            StepResult with output and tool calls.
        """
        print(f"\n[Step: {step_name}]")
        start_time = time.time()
        all_tool_calls = []
        final_output = ""
        error = None

        try:
            response = self.agent.step(prompt)
            final_output = response.msgs[0].content if response and response.msgs else ""
            tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []
            all_tool_calls.extend(tool_calls)

            # Allow continuation within step if agent is still working
            iteration = 1
            while iteration < max_iterations and tool_calls:
                # Check if agent indicates step is complete
                if any(phrase in final_output.lower() for phrase in [
                    "found the following",
                    "here are the",
                    "files identified",
                    "completed",
                    "ready to",
                    "now i will",
                    "next step",
                ]):
                    break

                response = self.agent.step("Continue.")
                final_output = response.msgs[0].content if response and response.msgs else ""
                tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []
                all_tool_calls.extend(tool_calls)
                iteration += 1

        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            print(f"  Error: {error}")

        duration = time.time() - start_time
        print(f"  Completed in {duration:.2f}s, {len(all_tool_calls)} tool calls")

        return StepResult(
            step_name=step_name,
            output=final_output,
            tool_calls=all_tool_calls,
            duration=duration,
            error=error,
        )

    def run_task(self, task_name: str, task_description: str) -> dict:
        """Run a complete task with structured steps.

        Steps:
        1. Search - Find relevant files
        2. Read - Read key files to understand patterns
        3. Write - Write the script
        4. Execute - Run and debug if needed
        """
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        self.agent.reset()
        start_time = datetime.now()
        step_results: List[StepResult] = []

        # =====================================================================
        # Step 1: Search
        # =====================================================================
        search_prompt = f"""**Task**: {task_description}

**Current Step**: SEARCH - Find relevant files in the codebase.

**Instructions**:
1. Use glob_search to find files by name patterns related to the task
2. Use find_imports to find real usage examples
3. Search the WHOLE repo first (don't restrict path)
4. Look in examples/, test/, docs/ for usage patterns

**Output**: List the files you found, organized by category:
- Examples (most important)
- Implementation files
- Tests
- Documentation

Do NOT read any files yet - just search and list what you found.
"""
        step1 = self._run_step("Search", search_prompt, max_iterations=3)
        step_results.append(step1)

        if step1.error:
            return self._finalize_task(task_name, task_description, step_results, start_time)

        # =====================================================================
        # Step 2: Read
        # =====================================================================
        read_prompt = f"""**Current Step**: READ - Read the most relevant files.

Based on your search results, now read the key files to understand:
1. How to import and use the required modules
2. The correct API patterns and method signatures
3. Configuration and initialization patterns

**Instructions**:
1. Read 2-4 most relevant example files first
2. Read implementation files if needed for API details
3. Focus on files that show similar usage patterns

**Important**: Files you read now will stay in your memory. You won't need to re-read them.

After reading, summarize:
- Key imports needed
- Main API patterns
- Any configuration required
"""
        step2 = self._run_step("Read", read_prompt, max_iterations=3)
        step_results.append(step2)

        if step2.error:
            return self._finalize_task(task_name, task_description, step_results, start_time)

        # =====================================================================
        # Step 3: Write
        # =====================================================================
        write_prompt = f"""**Current Step**: WRITE - Write the script.

Now write the script based on the files you read.

**Task reminder**: {task_description}

**Instructions**:
1. Copy import statements EXACTLY from the files you read
2. Follow the API patterns you observed
3. Use shell_write_content_to_file to save the script
4. Save to the path specified in the task

**Important**:
- Use EXACT model names, class names, method names from source files
- Don't re-read files - use what you already learned
- Include proper error handling if shown in examples
"""
        step3 = self._run_step("Write", write_prompt, max_iterations=2)
        step_results.append(step3)

        if step3.error:
            return self._finalize_task(task_name, task_description, step_results, start_time)

        # =====================================================================
        # Step 4: Execute
        # =====================================================================
        execute_prompt = f"""**Current Step**: EXECUTE - Run the script and verify.

**Instructions**:
1. Execute the script using shell_exec with the appropriate command (e.g., python script.py)
2. Check the output for errors
3. If there are errors:
   - Analyze the error message
   - Read additional files if needed to understand the correct API
   - Fix the script and re-run (use shell_write_content_to_file to update)

**Success criteria**: The script runs without errors and produces expected output.
"""
        step4 = self._run_step("Execute", execute_prompt, max_iterations=5)
        step_results.append(step4)

        return self._finalize_task(task_name, task_description, step_results, start_time)

    def _finalize_task(
        self,
        task_name: str,
        task_description: str,
        step_results: List[StepResult],
        start_time: datetime,
    ) -> dict:
        """Finalize task and save logs."""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Aggregate statistics
        all_tool_calls = []
        for sr in step_results:
            all_tool_calls.extend(sr.tool_calls)

        has_error = any(sr.error for sr in step_results)

        # Categorize tool calls
        search_calls = [tc for tc in all_tool_calls if tc.tool_name in [
            'glob_search', 'grep_search', 'find_definition', 'find_imports', 'list_directory'
        ]]
        read_calls = [tc for tc in all_tool_calls if tc.tool_name == 'read_file']
        exec_calls = [tc for tc in all_tool_calls if tc.tool_name in [
            'shell_exec', 'shell_write_content_to_file'
        ]]

        # Check for duplicate reads
        read_paths = [tc.args.get('file_path', '') for tc in read_calls]
        unique_reads = len(set(read_paths))
        duplicate_reads = len(read_paths) - unique_reads

        # Print summary
        print(f"\n{'='*60}")
        print(f"Task Complete: {'ERROR' if has_error else 'SUCCESS'}")
        print(f"{'='*60}")
        print(f"Total time: {total_duration:.2f}s")
        for sr in step_results:
            status = "ERROR" if sr.error else "OK"
            print(f"  - {sr.step_name}: {sr.duration:.2f}s ({len(sr.tool_calls)} calls) [{status}]")
        print(f"\nTool calls: {len(all_tool_calls)} total")
        print(f"  - Search: {len(search_calls)}")
        print(f"  - Read: {len(read_calls)} ({unique_reads} unique)")
        print(f"  - Exec: {len(exec_calls)}")
        if duplicate_reads > 0:
            print(f"  - WARNING: {duplicate_reads} duplicate reads")

        # Save log
        log_path = self._save_log(
            task_name=task_name,
            task_description=task_description,
            step_results=step_results,
            start_time=start_time,
            end_time=end_time,
        )
        print(f"\nLog: {log_path}")

        return {
            "task_name": task_name,
            "output": step_results[-1].output if step_results else "",
            "total_duration": total_duration,
            "log_path": log_path,
            "has_error": has_error,
            "stats": {
                "search_calls": len(search_calls),
                "read_calls": len(read_calls),
                "unique_reads": unique_reads,
                "duplicate_reads": duplicate_reads,
                "exec_calls": len(exec_calls),
                "total_tool_calls": len(all_tool_calls),
            },
            "step_durations": {sr.step_name: sr.duration for sr in step_results},
        }

    def _save_log(
        self,
        task_name: str,
        task_description: str,
        step_results: List[StepResult],
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """Save execution log to file."""
        total_duration = (end_time - start_time).total_seconds()
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")

        has_error = any(sr.error for sr in step_results)
        status = "error" if has_error else "ok"
        log_filename = f"single_{task_name}_{status}_{timestamp}.log"
        log_filepath = self.logs_dir / log_filename

        with open(log_filepath, "w", encoding="utf-8") as f:
            # Header
            f.write(f"Task: {task_name}\n")
            f.write(f"Status: {'ERROR' if has_error else 'SUCCESS'}\n")
            f.write(f"Start Time: {start_time.isoformat()}\n")
            f.write(f"End Time: {end_time.isoformat()}\n")
            f.write(f"Total Duration: {total_duration:.2f} seconds\n")
            f.write(f"Description: {task_description}\n")
            f.write("=" * 80 + "\n\n")

            # Step Summary
            f.write("STEP SUMMARY\n")
            f.write("=" * 80 + "\n")
            for sr in step_results:
                status_str = "ERROR" if sr.error else "OK"
                f.write(f"{sr.step_name}: {sr.duration:.2f}s, {len(sr.tool_calls)} tool calls [{status_str}]\n")
                if sr.error:
                    f.write(f"  Error: {sr.error}\n")
            f.write("\n")

            # Aggregate stats
            all_tool_calls = []
            for sr in step_results:
                all_tool_calls.extend(sr.tool_calls)

            read_calls = [tc for tc in all_tool_calls if tc.tool_name == 'read_file']
            read_paths = [tc.args.get('file_path', '') for tc in read_calls]

            if len(read_paths) > len(set(read_paths)):
                f.write("WARNING: Duplicate file reads detected!\n")
                from collections import Counter
                path_counts = Counter(read_paths)
                for path, count in path_counts.items():
                    if count > 1:
                        f.write(f"  - {path}: read {count} times\n")
                f.write("\n")

            f.write("=" * 80 + "\n\n")

            # Each step details
            for sr in step_results:
                f.write(f"STEP: {sr.step_name.upper()}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Duration: {sr.duration:.2f}s\n")
                f.write(f"Tool Calls: {len(sr.tool_calls)}\n")
                if sr.error:
                    f.write(f"Error: {sr.error}\n")
                f.write("-" * 40 + "\n")

                for i, tc in enumerate(sr.tool_calls, 1):
                    f.write(f"\n[{i}] {tc.tool_name}\n")
                    f.write(f"Args: {json.dumps(tc.args, indent=2, ensure_ascii=False)}\n")
                    result_str = str(tc.result)
                    max_len = 1000 if tc.tool_name == 'read_file' else 500
                    if len(result_str) > max_len:
                        result_str = result_str[:max_len] + "..."
                    f.write(f"Result: {result_str}\n")

                f.write("\n" + "-" * 40 + "\n")
                f.write("Output:\n")
                f.write(sr.output[:3000] if len(sr.output) > 3000 else sr.output)
                f.write("\n" + "=" * 80 + "\n\n")

            # Chat History
            f.write("CHAT HISTORY\n")
            f.write("=" * 80 + "\n")
            for i, msg in enumerate(self.agent.chat_history, 1):
                f.write(f"\n[Message {i}]\n")
                f.write(f"Role: {msg.get('role', 'unknown')}\n")
                if 'content' in msg and msg['content']:
                    content = msg['content']
                    if isinstance(content, str) and len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)\n"
                    f.write(f"Content: {content}\n")
            f.write("\n")

        return str(log_filepath)


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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    code_agent_dir = os.path.dirname(base_dir)
    default_camel_dir = os.path.dirname(code_agent_dir)

    parser = argparse.ArgumentParser(
        description="GenericCodeSingleAgent - Unified agent with structured steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run single task
    python generic_code_single_agent.py --task "Create an agent with weather tools"

    # Run multiple tasks
    python generic_code_single_agent.py --tasks tasks.json --max-tasks 5

    # Use different model
    python generic_code_single_agent.py --model openai/gpt-4o --task "Create agent"
        """
    )

    parser.add_argument("--project", "-p", type=str, default=None)
    parser.add_argument("--context", "-c", type=str, default=None)
    parser.add_argument("--no-context", action="store_true", default=False)
    parser.add_argument("--task", "-t", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default="openai/gpt-4-1-mini")
    parser.add_argument("--logs-dir", "-l", type=str, default=None)
    parser.add_argument("--exclude", type=str, nargs="+", default=None)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--exp-id", type=str, default="")
    parser.add_argument("--timeout", type=float, default=120.0)

    args = parser.parse_args()

    if not args.task and not args.tasks:
        parser.error("Either --task or --tasks must be specified")

    # Setup paths
    project_dir = os.path.abspath(args.project) if args.project else default_camel_dir

    if args.no_context:
        context_file = None
    elif args.context:
        context_file = os.path.abspath(args.context)
    elif not args.project:
        context_file = os.path.join(base_dir, "CAMEL.md")
    else:
        context_file = None

    exclude_dirs = {
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build',
        '.initial_env', 'task-script*'
    }
    if args.exclude:
        exclude_dirs.update(args.exclude)

    model_platform, model_type = parse_model(args.model)

    if not os.path.isdir(project_dir):
        print(f"Error: Project directory does not exist: {project_dir}")
        sys.exit(1)

    if context_file and not os.path.isfile(context_file):
        print(f"Warning: Context file does not exist: {context_file}")
        context_file = None

    # Print config
    print("=" * 60)
    print("GenericCodeSingleAgent (Structured Steps)")
    print("=" * 60)
    print(f"Project:  {project_dir}")
    print(f"Context:  {context_file or 'None'}")
    print(f"Model:    {args.model}")
    print(f"Timeout:  {args.timeout}s per step")
    print("=" * 60)

    # Initialize
    agent = GenericCodeSingleAgent(
        working_directory=project_dir,
        context_file=context_file,
        logs_dir=args.logs_dir,
        exclude_dirs=exclude_dirs,
        model_platform=model_platform,
        model_type=model_type,
        exp_id=args.exp_id,
        step_timeout=args.timeout,
    )

    # Run
    if args.task:
        result = agent.run_task("single_task", args.task)
        print(f"\n--- Final Result ---")
        print(f"Status: {'ERROR' if result['has_error'] else 'SUCCESS'}")
        print(f"Duration: {result['total_duration']:.2f}s")
        print(f"Steps: {result['step_durations']}")
        print(f"Stats: {result['stats']}")

    elif args.tasks:
        if not os.path.isfile(args.tasks):
            print(f"Error: Tasks file does not exist: {args.tasks}")
            sys.exit(1)

        with open(args.tasks, "r") as f:
            task_list = json.load(f)

        results = []
        for i, (task_name, task_desc) in enumerate(task_list.items()):
            if i >= args.max_tasks:
                break
            result = agent.run_task(task_name, task_desc)
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print(f"Summary: {len(results)} tasks")
        print(f"Success: {sum(1 for r in results if not r['has_error'])}")
        print(f"Errors:  {sum(1 for r in results if r['has_error'])}")

        total_reads = sum(r['stats']['read_calls'] for r in results)
        total_unique = sum(r['stats']['unique_reads'] for r in results)
        total_dups = sum(r['stats']['duplicate_reads'] for r in results)
        print(f"\nRead Stats:")
        print(f"  - Total reads: {total_reads}")
        print(f"  - Unique files: {total_unique}")
        print(f"  - Duplicate reads: {total_dups}")

        print(f"\nLogs: {agent.logs_dir}")
