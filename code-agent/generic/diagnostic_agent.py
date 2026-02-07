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
Diagnostic Agent: LLM-as-Judge for Code Agent Analysis

This agent analyzes task logs and ground truth to identify problems in:
1. Code Agent behavior (wrong patterns, missed files, incorrect API usage)
2. CAMEL codebase issues (missing examples, confusing APIs, documentation gaps)

It uses the code search toolkit to verify issues against the actual codebase.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelType

from generic_code_search_toolkit import GenericCodeSearchToolkit


DIAGNOSTIC_SYSTEM_PROMPT = """You are a Diagnostic Agent (LLM-as-Judge) that evaluates code generation results.

## Your Role
You evaluate whether a generated script is correct by comparing it against:
1. The ground truth (expected correct implementation)
2. The task requirements
3. Whether the script successfully executed

## Tools Available
You have code search tools to verify issues against the actual codebase:
- `glob_search(pattern)` - Find files by name pattern
- `grep_search(pattern)` - Search file contents
- `read_file(file_path)` - Read file contents
- `find_definition(name)` - Find class/function definitions
- `find_imports(module_name)` - Find files that import a module

## Evaluation Criteria

### PASS Conditions (script is CORRECT):
1. **Exact Match**: Generated script matches ground truth
2. **Functional Match**: Script differs but:
   - Successfully executed (check log for errors)
   - Differences are non-essential (extra comments, different variable names)
   - Extra parameters are NOT required by the task
3. **Better Implementation**: Script is actually better than ground truth

### FAIL Conditions (script is INCORRECT):
1. **Wrong API**: Used non-existent or wrong API (e.g., `QwenModel` instead of `ModelFactory.create`)
2. **Wrong Values**: Used wrong enum values, model names, etc.
3. **Missing Required Features**: Task required something that's missing
4. **Execution Failed**: Script failed to run (check log)
5. **Wrong Tool/Method**: Task specified a specific tool but used different one

## Analysis Steps

### Step 1: Parse Task Requirements
- What does the task require?
- What model/tool/API is specified?
- What is the expected output file path?

### Step 2: Compare Generated vs Ground Truth
For each difference, determine:
- Is it a critical difference (wrong API, wrong model)?
- Is it acceptable (extra imports, different formatting)?
- Is the extra parameter required by the task?

### Step 3: Check Execution Status
From the log:
- Did the script execute successfully?
- Were there any errors?
- Did it produce expected output?

### Step 4: Final Verdict

## Output Format

```
## Task: [Task Name]

### Task Requirements
- Model: [required model if specified]
- Tool: [required tool if specified]
- Output: [expected output path]

### Comparison Results

| Aspect | Ground Truth | Generated | Match | Acceptable |
|--------|--------------|-----------|-------|------------|
| Model Creation | ModelFactory.create() | QwenModel() | ❌ | ❌ |
| Model Type | QWEN_2_5_14B | QWEN_2_5B_INSTRUCT | ❌ | ❌ |
| Tool Usage | WeatherToolkit | WeatherToolkit | ✅ | ✅ |

### Differences Detail
1. **[Difference 1]**
   - Ground Truth: `...`
   - Generated: `...`
   - Required by Task: Yes/No
   - Acceptable: Yes/No
   - Reason: [why acceptable or not]

### Execution Status
- **Status**: SUCCESS / FAILED
- **Error** (if any): [error message]

### Final Verdict
- **Result**: ✅ PASS / ❌ FAIL
- **Reason**: [brief explanation]
- **Issues Found**: [list of issues if FAIL]
```

## Important Rules
- A script that runs successfully with non-essential differences is PASS
- A script with wrong API/model even if it runs is FAIL (unless task didn't specify)
- Extra parameters not required by task are acceptable if script runs
- Use code search to verify if APIs/classes exist in the codebase
"""


class DiagnosticAgent:
    """Agent that diagnoses code generation issues using LLM-as-Judge approach."""

    def __init__(
        self,
        working_directory: str,
        model_type: ModelType = ModelType.GPT_4O,
        output_dir: Optional[str] = None,
    ):
        self.working_dir = Path(working_directory).resolve()
        self.output_dir = Path(output_dir) if output_dir else self.working_dir / "diagnostics"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup model
        model_config = ChatGPTConfig(temperature=0.0).as_dict()
        self.model = ModelFactory.create(
            model_type=model_type,
            model_config_dict=model_config,
        )

        # Setup code search toolkit
        exclude_dirs = {
            'node_modules', '.venv', '.git', '__pycache__', '.tox',
            '.mypy_cache', '.pytest_cache', 'dist', 'build',
        }
        self.search_toolkit = GenericCodeSearchToolkit(
            working_directory=str(self.working_dir),
            exclude_dirs=exclude_dirs,
            max_results=50,
        )

        # Create agent with search tools
        self.agent = ChatAgent(
            system_message=DIAGNOSTIC_SYSTEM_PROMPT,
            model=self.model,
            tools=self.search_toolkit.get_tools(),
        )

    def diagnose(
        self,
        log_path: str,
        script_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        task_description: Optional[str] = None,
    ) -> dict:
        """Evaluate a generated script against ground truth.

        Args:
            log_path: Path to the execution log file
            script_path: Path to the generated script file
            ground_truth_path: Path to the ground truth (expected) script
            task_description: The original task description
        """

        # Read log file
        log_content = Path(log_path).read_text(encoding='utf-8')

        # Check if execution was successful
        exec_success = "Status: SUCCESS" in log_content[:500]

        # Read generated script if provided
        generated_script = ""
        if script_path and Path(script_path).exists():
            generated_script = Path(script_path).read_text(encoding='utf-8')

        # Read ground truth if provided
        ground_truth = ""
        if ground_truth_path and Path(ground_truth_path).exists():
            ground_truth = Path(ground_truth_path).read_text(encoding='utf-8')

        # Build prompt
        prompt = f"""## Task Description

{task_description or "Not provided"}

## Generated Script

```python
{generated_script}
```

## Ground Truth (Expected Implementation)

```python
{ground_truth}
```

## Execution Log Summary

- **Execution Status**: {"SUCCESS" if exec_success else "FAILED"}

### Log Content (first 10000 chars):
```
{log_content[:10000]}
```

## Instructions

Compare the generated script against the ground truth and evaluate:

1. **Identify all differences** between generated and ground truth
2. **For each difference**, determine:
   - Is it required by the task? (check task description)
   - Does it affect functionality?
   - Is it acceptable?
3. **Check execution status** - did the script run successfully?
4. **Use code search tools** to verify if APIs/classes used actually exist
5. **Give final verdict**: PASS or FAIL

Remember:
- If script runs successfully and differences are non-essential → PASS
- If script uses wrong/non-existent API → FAIL (even if it somehow runs)
- Extra parameters not in task requirements are acceptable if script runs

Start your evaluation now.
"""

        # Run diagnosis
        self.agent.reset()
        response = self.agent.step(prompt)

        diagnosis = response.msgs[0].content if response and response.msgs else ""
        tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []

        # Save diagnosis report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = Path(log_path).stem
        report_path = self.output_dir / f"eval_{log_name}_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"- **Log File**: {log_path}\n")
            f.write(f"- **Generated Script**: {script_path or 'Not provided'}\n")
            f.write(f"- **Ground Truth**: {ground_truth_path or 'Not provided'}\n")
            f.write(f"- **Execution Status**: {'SUCCESS' if exec_success else 'FAILED'}\n")
            f.write(f"- **Report Generated**: {timestamp}\n")
            f.write(f"- **Tool Calls**: {len(tool_calls)}\n\n")
            f.write("---\n\n")
            f.write(diagnosis)

        print(f"\nDiagnosis saved to: {report_path}")

        return {
            "diagnosis": diagnosis,
            "tool_calls": tool_calls,
            "report_path": str(report_path),
        }

    def batch_evaluate(
        self,
        log_dir: str,
        script_dir: str,
        ground_truth_dir: str,
        task_list_path: Optional[str] = None,
        max_tasks: int = 10,
    ) -> list:
        """Evaluate multiple generated scripts against ground truth.

        Args:
            log_dir: Directory containing execution logs
            script_dir: Directory containing generated scripts
            ground_truth_dir: Directory containing ground truth scripts
            task_list_path: Path to task_list.json with task descriptions
            max_tasks: Maximum number of tasks to evaluate
        """
        import json as json_module

        # Load task list if provided
        task_list = {}
        if task_list_path and Path(task_list_path).exists():
            with open(task_list_path) as f:
                task_list = json_module.load(f)

        # Find all ground truth files (these define which tasks to evaluate)
        gt_dir = Path(ground_truth_dir)
        gt_files = sorted(gt_dir.glob("*.py"))[:max_tasks]

        results = []
        pass_count = 0
        fail_count = 0

        for gt_file in gt_files:
            print(f"\n{'='*60}")
            print(f"Evaluating: {gt_file.name}")
            print(f"{'='*60}")

            # Extract task number from filename (e.g., "1_weather_agent.py" -> "1")
            task_num = gt_file.stem.split('_')[0]
            task_key = f"task_{task_num}"

            # Find matching script
            script_dir_path = Path(script_dir)
            script_matches = list(script_dir_path.glob(f"{task_num}_*.py"))
            script_path = str(script_matches[0]) if script_matches else None

            # Find matching log
            log_dir_path = Path(log_dir)
            log_matches = list(log_dir_path.glob(f"*task_{task_num}_*.log"))
            log_path = str(log_matches[0]) if log_matches else None

            # Get task description
            task_desc = task_list.get(task_key, "Task description not found")

            if not script_path:
                print(f"  [SKIP] No generated script found for {gt_file.name}")
                continue
            if not log_path:
                print(f"  [SKIP] No log found for {gt_file.name}")
                continue

            result = self.diagnose(
                log_path=log_path,
                script_path=script_path,
                ground_truth_path=str(gt_file),
                task_description=task_desc,
            )

            # Check if PASS or FAIL from diagnosis
            is_pass = "PASS" in result["diagnosis"] and "FAIL" not in result["diagnosis"].split("PASS")[0]

            if is_pass:
                pass_count += 1
                print(f"  Result: ✅ PASS")
            else:
                fail_count += 1
                print(f"  Result: ❌ FAIL")

            results.append({
                "task": gt_file.name,
                "script": script_path,
                "log": log_path,
                "ground_truth": str(gt_file),
                "passed": is_pass,
                **result
            })

        # Summary
        total = pass_count + fail_count
        accuracy = (pass_count / total * 100) if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"Evaluation Complete")
        print(f"{'='*60}")
        print(f"Total: {total}")
        print(f"Passed: {pass_count} ({accuracy:.1f}%)")
        print(f"Failed: {fail_count}")
        print(f"Reports saved to: {self.output_dir}")

        # Save summary report
        summary_path = self.output_dir / "evaluation_summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Evaluation Summary\n\n")
            f.write(f"- **Log Directory**: {log_dir}\n")
            f.write(f"- **Script Directory**: {script_dir}\n")
            f.write(f"- **Ground Truth Directory**: {ground_truth_dir}\n\n")
            f.write(f"## Results\n\n")
            f.write(f"| Task | Result | Report |\n")
            f.write(f"|------|--------|--------|\n")
            for r in results:
                status = "✅ PASS" if r["passed"] else "❌ FAIL"
                f.write(f"| {r['task']} | {status} | [Report]({r['report_path']}) |\n")
            f.write(f"\n## Summary\n\n")
            f.write(f"- **Total**: {total}\n")
            f.write(f"- **Passed**: {pass_count} ({accuracy:.1f}%)\n")
            f.write(f"- **Failed**: {fail_count}\n")

        print(f"Summary saved to: {summary_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic Agent - LLM-as-Judge for Code Generation Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file evaluation
    python diagnostic_agent.py \\
        --log logs/code_task_1.log \\
        --script task-script/1_weather.py \\
        --ground-truth ground_truth/1_weather.py \\
        --task "Create weather agent with Qwen"

    # Batch evaluation
    python diagnostic_agent.py \\
        --log-dir logs/ \\
        --script-dir task-script/single_agent/ \\
        --ground-truth-dir ground_truth/single_agent/ \\
        --task-list task_list.json
        """
    )

    # Single file mode
    parser.add_argument(
        "--log", "-l",
        type=str,
        help="Path to a single task log file"
    )
    parser.add_argument(
        "--script", "-s",
        type=str,
        help="Path to generated script file"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        type=str,
        help="Path to ground truth file"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task description"
    )

    # Batch mode
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory containing log files"
    )
    parser.add_argument(
        "--script-dir",
        type=str,
        help="Directory containing generated scripts"
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        help="Directory containing ground truth scripts"
    )
    parser.add_argument(
        "--task-list",
        type=str,
        help="Path to task_list.json with task descriptions"
    )

    # Common options
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Project directory to search (default: CAMEL root)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for evaluation reports"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="Model to use for evaluation (default: gpt-4o)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to evaluate in batch mode"
    )

    args = parser.parse_args()

    # Validate arguments
    is_single_mode = args.log or args.script or args.ground_truth
    is_batch_mode = args.log_dir or args.script_dir or args.ground_truth_dir

    if not is_single_mode and not is_batch_mode:
        parser.error("Must specify either single file args or batch mode args")

    if is_batch_mode:
        if not (args.log_dir and args.script_dir and args.ground_truth_dir):
            parser.error("Batch mode requires --log-dir, --script-dir, and --ground-truth-dir")

    # Setup project directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    code_agent_dir = os.path.dirname(base_dir)
    default_project_dir = os.path.dirname(code_agent_dir)

    project_dir = args.project if args.project else default_project_dir

    # Parse model
    model_mapping = {
        "gpt-4o": ModelType.GPT_4O,
        "gpt-4o-mini": ModelType.GPT_4O_MINI,
        "gpt-4-1": ModelType.GPT_4_1,
        "gpt-4-1-mini": ModelType.GPT_4_1_MINI,
        "claude-3-5-sonnet": ModelType.CLAUDE_3_5_SONNET,
    }
    model_type = model_mapping.get(args.model.lower(), ModelType.GPT_4O)

    # Create diagnostic agent
    agent = DiagnosticAgent(
        working_directory=project_dir,
        model_type=model_type,
        output_dir=args.output,
    )

    # Run evaluation
    if is_single_mode:
        result = agent.diagnose(
            log_path=args.log,
            script_path=args.script,
            ground_truth_path=args.ground_truth,
            task_description=args.task,
        )
        print("\n" + "="*60)
        print("EVALUATION RESULT")
        print("="*60)
        print(result["diagnosis"][:3000])
        if len(result["diagnosis"]) > 3000:
            print("\n... (truncated, see full report)")
        print(f"\nFull report: {result['report_path']}")
    else:
        agent.batch_evaluate(
            log_dir=args.log_dir,
            script_dir=args.script_dir,
            ground_truth_dir=args.ground_truth_dir,
            task_list_path=args.task_list,
            max_tasks=args.max_tasks,
        )


if __name__ == "__main__":
    main()
