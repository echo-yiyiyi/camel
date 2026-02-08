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
Diagnose Agent: Analyzes failed code generation tasks and improves CAMEL.md

This agent:
1. Takes failed scripts from analysis agent output
2. Analyzes why they failed (explore phase or code phase)
3. Modifies CAMEL.md to improve future generations
4. Optionally re-runs the explore/code phase to verify improvements

Inherits capabilities from GenericCodeAgent.
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool, Crawl4AIToolkit
from camel.types import ModelType

from generic_code_search_toolkit import GenericCodeSearchToolkit
from generic_code_agent import GENERIC_EXPLORE_SYSTEM_PROMPT, GENERIC_CODE_SYSTEM_PROMPT


# =============================================================================
# System Prompts
# =============================================================================

DIAGNOSE_SYSTEM_PROMPT = """You are a Diagnose Agent that analyzes why code generation failed and improves the project context file (CAMEL.md).

## Your Task

1. Analyze the failed task based on:
   - Analysis report (from analysis agent)
   - Execution log (from code agent)
   - Task description
   - Ground truth code (what it should look like)

2. Determine the root cause:
   - **Explore Phase Issue**: The explore agent failed to find relevant files or provided wrong information
   - **Code Phase Issue**: The explore was fine but code agent wrote incorrect code

3. Suggest improvements to CAMEL.md that will help the code agent succeed next time

## How to Diagnose

### Step 1: Check Explore Phase Output

Look for the "EXPLORE PHASE" section in the log:
- Did it find the relevant files?
- Did it identify the correct classes/methods?
- Did it miss important documentation or examples?

If explore output is incomplete/wrong -> **Explore Phase Issue**

### Step 2: Check Code Phase Behavior

If explore phase looks good, check the "CODE PHASE" section:
- Did code agent read the files suggested by explore?
- Did code agent copy patterns correctly?
- Did code agent use wrong API/methods despite correct information?

If code agent ignored good information -> **Code Phase Issue**

### Step 3: Compare Generated vs Ground Truth

Look at what's different:
- Model type?
- Toolkit class or method?
- API usage pattern?
- Missing imports?

## Output Format

```markdown
# Diagnosis: [task_name]

## Root Cause Analysis

**Phase with Issue**: Explore Phase / Code Phase

**What went wrong**:
[Explain the specific failure]

**Evidence from log**:
```
[Quote relevant log section]
```

## Comparison: Generated vs Ground Truth

| Aspect | Generated | Ground Truth | Issue |
|--------|-----------|--------------|-------|
| Model | ... | ... | ... |
| Toolkit | ... | ... | ... |
| Method | ... | ... | ... |

## Suggested CAMEL.md Improvements

### New Rule to Add

[Specific rule that would have prevented this error]

### Example Pattern to Add

```python
# Example code pattern to add to CAMEL.md
```

### Search Technique to Add

[If explore phase missed something, add search technique]
```

## Rules

- Be specific about what went wrong
- Quote actual code/log content as evidence
- Suggest concrete, actionable improvements
- Focus on patterns that generalize to similar tasks
"""

ROOT_CAUSE_ANALYSIS_PROMPT = """You are analyzing the root cause of a code generation failure.

## Failed Item
{failed_item}

## Original Task
{task_description}

## Execution Log
{log_content}

## Generated Code
{generated_code}

## Ground Truth Code
{ground_truth}

## Your Task

Analyze the log to determine WHY this specific item failed. You MUST quote specific log sections as evidence.

### Step 1: Find EXPLORE PHASE Evidence
Search the log for "EXPLORE PHASE" section and quote:
- What files/classes did explore find?
- Did it find the correct component for this failed item?
- If not, what did it search for and what did it miss?

### Step 2: Find CODE PHASE Evidence
Search the log for "CODE PHASE" section and quote:
- Which files did code agent read (look for "read_file" calls)?
- What code patterns did it copy?
- Where did it make the wrong decision?

### Step 3: Find the Decision Point
Identify the EXACT moment things went wrong:
- Did explore return wrong info? Quote the explore output.
- Did code agent ignore correct info? Quote what it ignored.
- Did code agent read wrong file? Quote which file it read.

### Step 4: Check CAMEL Codebase
Use search tools to verify:
- Does a correct example exist? Search for it.
- Is the API documented? Search for documentation.
- Quote what you find (or note that nothing exists).

### Step 5: Determine Root Cause

**Code Agent Issue** - if log shows:
- Explore found correct info but code agent ignored it
- Code agent read correct file but copied wrong pattern
- Code agent made wrong decision despite having correct info

**CAMEL Codebase Issue** - if log shows:
- Explore searched but couldn't find relevant examples
- No documentation exists for this use case
- API is confusing with no clear usage pattern

## Output Format (MUST include all sections with log quotes)

```markdown
### Root Cause: {failed_item}

**1. Explore Phase Evidence**:
```
[EXACT quote from EXPLORE PHASE section of log]
[Show what explore found or failed to find]
```

**2. Code Phase Evidence**:
```
[EXACT quote from CODE PHASE section of log]
[Show what files code agent read and what it wrote]
```

**3. Decision Point**:
```
[EXACT quote showing where the wrong decision was made]
```

**4. CAMEL Codebase Check**:
- Searched for: [what you searched]
- Found: [quote from CAMEL source/examples, or "No relevant examples found"]

**5. Root Cause**: Code Agent Issue / CAMEL Codebase Issue

**6. Explanation**:
[Detailed explanation connecting the evidence to the cause]

**7. How to Fix**:
- If Code Agent Issue: [What the agent should have done at the decision point]
- If CAMEL Codebase Issue: [What documentation/examples should be added]
```
"""

CAMEL_MD_UPDATE_PROMPT = """Update CAMEL.md based on diagnosis reports.

## Current CAMEL.md
{camel_md_content}

## Diagnosis Reports
{diagnosis_reports}

## Rules

1. Add SHORT techniques (not task answers)
2. Use DIFFERENT example than the failed task
3. No duplicates - skip if similar rule exists
4. Max 1 new technique per diagnosis

## Format

✅ GOOD - technique with brief few-shot (different example):
```
### Technique: Find SPECIFIC tool methods
When task specifies a tool like "brave search", find the exact method:
grep_search("search_brave", path="camel/toolkits")
```

❌ BAD - full code example or task answer:
```python
from camel.toolkits import SearchToolkit
search_tool = SearchToolkit().search_duckduckgo
agent = ChatAgent(tools=[search_tool])
```

## Output

Output the complete updated CAMEL.md.
"""


class DiagnoseAgent:
    """Agent that diagnoses failed code generation and improves CAMEL.md."""

    def __init__(
        self,
        working_directory: str,
        camel_md_path: str,
        model_type: ModelType = ModelType.GPT_4_1_MINI,
        output_dir: Optional[str] = None,
    ):
        """Initialize the DiagnoseAgent.

        Args:
            working_directory: The root directory for code exploration (CAMEL repo).
            camel_md_path: Path to CAMEL.md file.
            model_type: The model type to use.
            output_dir: Directory to save diagnosis reports.
        """
        self.working_dir = Path(working_directory).resolve()
        self.camel_md_path = Path(camel_md_path)
        self.output_dir = Path(output_dir) if output_dir else self.working_dir / "diagnoses"
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
            '.initial_env', 'CAMEL_old.md', 'CAMEL_remove_model_platform.md'
        }
        self.search_toolkit = GenericCodeSearchToolkit(
            working_directory=str(self.working_dir),
            exclude_dirs=exclude_dirs,
            max_results=50,
        )

        # Setup terminal toolkit for potential re-execution
        self.terminal_toolkit = TerminalToolkit(
            working_directory=str(self.working_dir),
            clone_current_env=True,
            timeout=300.0,
        )

        # Create diagnose agent with search tools
        self.diagnose_agent = ChatAgent(
            system_message=DIAGNOSE_SYSTEM_PROMPT,
            model=self.model,
            tools=self.search_toolkit.get_tools(),
        )

        # Create CAMEL.md update agent
        self.update_agent = ChatAgent(
            system_message="You update CAMEL.md based on diagnosis reports.",
            model=self.model,
        )

        # Create root cause analysis agent (has search tools to read CAMEL source)
        self.root_cause_agent = ChatAgent(
            system_message="You analyze root causes of code generation failures.",
            model=self.model,
            tools=self.search_toolkit.get_tools(),
        )

    def _load_camel_md(self) -> str:
        """Load current CAMEL.md content."""
        if self.camel_md_path.exists():
            return self.camel_md_path.read_text(encoding='utf-8')
        return ""

    def _save_camel_md(self, content: str) -> None:
        """Save updated CAMEL.md content."""
        # Backup original
        backup_path = self.camel_md_path.with_suffix('.md.bak')
        if self.camel_md_path.exists():
            original = self.camel_md_path.read_text(encoding='utf-8')
            backup_path.write_text(original, encoding='utf-8')

        self.camel_md_path.write_text(content, encoding='utf-8')
        print(f"Updated CAMEL.md (backup at {backup_path})")

    def _extract_explore_output_from_log(self, log_content: str) -> str:
        """Extract explore phase output from log file."""
        # Find EXPLORE PHASE section
        explore_start = log_content.find("EXPLORE PHASE")
        if explore_start == -1:
            return ""

        # Find CODE PHASE section (end of explore)
        code_start = log_content.find("CODE PHASE", explore_start)
        if code_start == -1:
            return log_content[explore_start:]

        return log_content[explore_start:code_start]

    def _extract_code_output_from_log(self, log_content: str) -> str:
        """Extract code phase output from log file."""
        code_start = log_content.find("CODE PHASE")
        if code_start == -1:
            return ""

        # Find next major section or end
        explore_history_start = log_content.find("EXPLORE AGENT CHAT HISTORY", code_start)
        if explore_history_start == -1:
            return log_content[code_start:]

        return log_content[code_start:explore_history_start]

    def _extract_failed_items_from_analysis(self, analysis_report: str) -> List[str]:
        """Extract NOT Acceptable items from analysis report.

        Looks for table rows with ❌ NOT Acceptable or ❌ in the acceptable column.

        Args:
            analysis_report: The analysis report content

        Returns:
            List of failed item descriptions
        """
        failed_items = []

        # Find table rows with ❌
        for line in analysis_report.split('\n'):
            if '❌' in line and '|' in line:
                # Parse table row: | item | ground truth | generated | match | acceptable |
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    item_name = parts[1] if len(parts) > 1 else ""
                    if item_name and item_name not in ['Aspect', '---', '']:
                        # Include the full row for context
                        failed_items.append(line.strip())

        return failed_items

    def _analyze_root_causes(
        self,
        failed_items: List[str],
        task_description: str,
        log_content: str,
        generated_script: str,
        ground_truth: str,
    ) -> str:
        """Analyze root cause for each failed item.

        Args:
            failed_items: List of failed item descriptions from analysis
            task_description: The original task description
            log_content: Full execution log
            generated_script: The generated code
            ground_truth: The ground truth code

        Returns:
            Root cause analysis markdown string
        """
        if not failed_items:
            return ""

        print(f"\n[Analyzing root causes for {len(failed_items)} failed items...]")

        analyses = []
        for i, item in enumerate(failed_items, 1):
            print(f"  [{i}/{len(failed_items)}] Analyzing: {item[:60]}...")

            prompt = ROOT_CAUSE_ANALYSIS_PROMPT.format(
                failed_item=item,
                task_description=task_description,
                log_content=log_content[:15000],  # Truncate for context
                generated_code=generated_script,
                ground_truth=ground_truth,
            )

            self.root_cause_agent.reset()
            response = self.root_cause_agent.step(prompt)

            if response and response.msgs:
                analysis = response.msgs[0].content
                analyses.append(analysis)

        if not analyses:
            return ""

        # Combine all analyses
        result = "\n\n## Deep Root Cause Analysis\n\n"
        result += "_This section analyzes each NOT Acceptable item by reading the execution log and CAMEL source code._\n\n"
        result += "\n\n---\n\n".join(analyses)

        return result

    def _determine_failure_phase(self, log_content: str, analysis_report: str) -> str:
        """Determine which phase caused the failure."""
        explore_output = self._extract_explore_output_from_log(log_content)

        # Check for explore phase errors
        if "[ERROR]" in explore_output or "Error" in explore_output[:500]:
            return "explore"

        # Check if analysis report indicates wrong components were found
        # This is a heuristic - the diagnose agent will refine this
        if "explore" in analysis_report.lower() and "not found" in analysis_report.lower():
            return "explore"

        return "code"

    def _create_explore_agent(self) -> ChatAgent:
        """Create explore agent with current CAMEL.md context."""
        project_context = self._load_camel_md()
        explore_prompt = GENERIC_EXPLORE_SYSTEM_PROMPT
        if project_context:
            explore_prompt += f"\n\n## Project-Specific Context\n\n{project_context}"

        return ChatAgent(
            system_message=explore_prompt,
            model=self.model,
            tools=self.search_toolkit.get_tools(),
        )

    def _create_code_agent(self) -> ChatAgent:
        """Create code agent with current CAMEL.md context."""
        project_context = self._load_camel_md()
        code_prompt = GENERIC_CODE_SYSTEM_PROMPT
        if project_context:
            code_prompt += f"\n\n## Project-Specific Context\n\n{project_context}"

        # Create read_file tool
        def read_file_tool(file_path: str) -> str:
            r"""Read the content of a file.

            Args:
                file_path (str): The path of the file to read.

            Returns:
                str: The content of the file.
            """
            return self.search_toolkit.read_file(file_path)

        terminal_tools = self.terminal_toolkit.get_tools()
        crawl4ai_toolkit = Crawl4AIToolkit()

        return ChatAgent(
            system_message=code_prompt,
            model=self.model,
            tools=[FunctionTool(read_file_tool)] + terminal_tools + crawl4ai_toolkit.get_tools(),
        )

    def run_explore_phase(self, task_description: str) -> str:
        """Run explore phase with current CAMEL.md context.

        Args:
            task_description: The task description

        Returns:
            Explore output string
        """
        explore_agent = self._create_explore_agent()
        explore_agent.reset()

        prompt = f"""Find relevant files for this task:

**Task**: {task_description}

Search for implementation files, examples, tests, and documentation.
Return the most relevant file paths with brief explanations.
Focus on finding:
1. Example scripts that show similar usage patterns
2. Implementation files for the required components
3. Configuration/type definitions
"""

        response = explore_agent.step(prompt)
        return response.msgs[0].content if response and response.msgs else ""

    def run_code_phase(self, task_description: str, explore_output: str) -> str:
        """Run code phase with current CAMEL.md context.

        Args:
            task_description: The task description
            explore_output: Output from explore phase (new or from log)

        Returns:
            Code output string
        """
        code_agent = self._create_code_agent()
        code_agent.reset()

        # Extract relevant files from explore output
        relevant_files = []
        for line in explore_output.split('\n'):
            line = line.strip()
            if any(ext in line for ext in ['.py', '.md']):
                parts = line.lstrip('- ').split()
                for part in parts:
                    if any(part.endswith(ext) for ext in ['.py', '.md']):
                        relevant_files.append(part)
                        break

        # Remove duplicates
        seen = set()
        relevant_files = [f for f in relevant_files if not (f in seen or seen.add(f))]

        file_instruction = ""
        if relevant_files:
            file_instruction = f"""
**Available files** (use `read_file_tool` to read):
{chr(10).join(f'- {f}' for f in relevant_files[:10])}

**IMPORTANT**: Before writing code, you MUST read the files that are relevant to your task.
"""

        prompt = f"""**Task**: {task_description}

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
"""

        response = code_agent.step(prompt)
        return response.msgs[0].content if response and response.msgs else ""

    def rerun_task(
        self,
        task_description: str,
        failure_phase: str,
        explore_output_from_log: Optional[str] = None,
    ) -> Dict:
        """Re-run a task after CAMEL.md has been updated.

        Args:
            task_description: The task description
            failure_phase: "explore" or "code"
            explore_output_from_log: Explore output from original log (used if code phase failed)

        Returns:
            Dictionary with re-run results
        """
        print(f"\n[Re-running task - failure was in {failure_phase} phase]")

        if failure_phase == "explore":
            # Re-run both explore and code phases
            print("[Step 1] Running explore phase with updated CAMEL.md...")
            explore_output = self.run_explore_phase(task_description)
            print(f"Explore output: {explore_output[:500]}...")

            print("[Step 2] Running code phase with new explore output...")
            code_output = self.run_code_phase(task_description, explore_output)
        else:
            # Skip explore, use output from log
            print("[Step 1] Skipping explore phase (using output from log)...")
            explore_output = explore_output_from_log or ""

            print("[Step 2] Running code phase with updated CAMEL.md...")
            code_output = self.run_code_phase(task_description, explore_output)

        return {
            "explore_output": explore_output,
            "code_output": code_output,
        }

    def diagnose_single(
        self,
        task_name: str,
        task_description: str,
        log_path: str,
        script_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        analysis_report_path: Optional[str] = None,
    ) -> Dict:
        """Diagnose a single failed task.

        Args:
            task_name: Name of the task (e.g., "task_1")
            task_description: The original task description
            log_path: Path to the execution log file
            script_path: Path to the generated script
            ground_truth_path: Path to the ground truth script
            analysis_report_path: Path to the analysis agent's report

        Returns:
            Dictionary with diagnosis results
        """
        print(f"\n{'='*60}")
        print(f"Diagnosing: {task_name}")
        print(f"{'='*60}")

        # Read all inputs
        log_content = Path(log_path).read_text(encoding='utf-8') if Path(log_path).exists() else ""

        generated_script = ""
        if script_path and Path(script_path).exists():
            generated_script = Path(script_path).read_text(encoding='utf-8')

        ground_truth = ""
        if ground_truth_path and Path(ground_truth_path).exists():
            ground_truth = Path(ground_truth_path).read_text(encoding='utf-8')

        analysis_report = ""
        if analysis_report_path and Path(analysis_report_path).exists():
            analysis_report = Path(analysis_report_path).read_text(encoding='utf-8')

        # Extract key log sections
        explore_output = self._extract_explore_output_from_log(log_content)
        code_output = self._extract_code_output_from_log(log_content)

        # Determine which phase likely failed
        failure_phase = self._determine_failure_phase(log_content, analysis_report)
        print(f"Initial diagnosis: {failure_phase} phase issue")

        # Build diagnosis prompt
        prompt = f"""Diagnose why this code generation task failed.

## Task Description
{task_description}

## Analysis Report (from Analysis Agent)
{analysis_report[:5000] if analysis_report else "Not provided"}

## Explore Phase Output (from Log)
```
{explore_output[:8000]}
```

## Code Phase Output (from Log)
```
{code_output[:8000]}
```

## Generated Script
```python
{generated_script}
```

## Ground Truth Script
```python
{ground_truth}
```

## Your Task
1. Identify the root cause (explore or code phase issue)
2. Explain what went wrong specifically
3. Suggest concrete improvements for CAMEL.md
"""

        # Run diagnosis
        self.diagnose_agent.reset()
        response = self.diagnose_agent.step(prompt)

        diagnosis = response.msgs[0].content if response and response.msgs else ""

        # Extract failed items from analysis report for root cause analysis
        failed_items = self._extract_failed_items_from_analysis(analysis_report)

        # Run deep root cause analysis for each failed item
        root_cause_analysis = ""
        if failed_items:
            root_cause_analysis = self._analyze_root_causes(
                failed_items=failed_items,
                task_description=task_description,
                log_content=log_content,
                generated_script=generated_script,
                ground_truth=ground_truth,
            )

        # Combine diagnosis with root cause analysis
        full_diagnosis = diagnosis
        if root_cause_analysis:
            full_diagnosis += "\n\n" + root_cause_analysis

        # Save diagnosis report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.output_dir / f"diagnose_{task_name}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_diagnosis)

        print(f"Diagnosis saved to: {report_path}")

        return {
            "task_name": task_name,
            "failure_phase": failure_phase,
            "diagnosis": full_diagnosis,
            "report_path": str(report_path),
            "explore_output": explore_output[:2000],  # For potential re-use
            "failed_items_count": len(failed_items),
        }

    def update_camel_md(self, diagnoses: List[Dict]) -> str:
        """Update CAMEL.md based on multiple diagnoses.

        Args:
            diagnoses: List of diagnosis results

        Returns:
            Updated CAMEL.md content
        """
        print(f"\n{'='*60}")
        print("Updating CAMEL.md based on diagnoses")
        print(f"{'='*60}")

        current_camel_md = self._load_camel_md()

        # Combine diagnosis reports
        diagnosis_reports = "\n\n---\n\n".join([
            f"## {d['task_name']}\n\n{d['diagnosis']}"
            for d in diagnoses
        ])

        prompt = CAMEL_MD_UPDATE_PROMPT.format(
            camel_md_content=current_camel_md,
            diagnosis_reports=diagnosis_reports,
        )

        self.update_agent.reset()
        response = self.update_agent.step(prompt)

        updated_content = response.msgs[0].content if response and response.msgs else current_camel_md

        # Clean up markdown code block if present
        if updated_content.startswith("```markdown"):
            updated_content = updated_content[len("```markdown"):].strip()
        if updated_content.startswith("```"):
            updated_content = updated_content[3:].strip()
        if updated_content.endswith("```"):
            updated_content = updated_content[:-3].strip()

        # Save updated CAMEL.md
        self._save_camel_md(updated_content)

        return updated_content

    def batch_diagnose(
        self,
        log_dir: str,
        script_dir: str,
        ground_truth_dir: str,
        analysis_output_dir: str,
        task_list_path: str,
        max_tasks: int = 10,
        update_batch_size: int = 1,
    ) -> List[Dict]:
        """Diagnose multiple failed tasks.

        Args:
            log_dir: Directory containing execution logs
            script_dir: Directory containing generated scripts
            ground_truth_dir: Directory containing ground truth scripts
            analysis_output_dir: Directory containing analysis agent reports
            task_list_path: Path to task_list.json
            max_tasks: Maximum number of tasks to diagnose
            update_batch_size: Update CAMEL.md every N tasks (default: 1)

        Returns:
            List of diagnosis results
        """
        # Load task list
        with open(task_list_path) as f:
            task_list = json.load(f)

        # Find analysis reports to identify failed tasks
        analysis_dir = Path(analysis_output_dir)

        # Read evaluation summary to find failed tasks
        summary_path = analysis_dir / "evaluation_summary.md"
        failed_tasks = []

        if summary_path.exists():
            summary_content = summary_path.read_text(encoding='utf-8')
            # Parse summary to find failed tasks
            for line in summary_content.split('\n'):
                if '❌ FAIL' in line:
                    # Extract task number from line like "| 2_duckduckgo_agent.py | ❌ FAIL |"
                    match = re.search(r'\| (\d+)_', line)
                    if match:
                        failed_tasks.append(int(match.group(1)))
        else:
            # If no summary, diagnose all tasks
            failed_tasks = list(range(1, max_tasks + 1))

        print(f"Failed tasks to diagnose: {failed_tasks[:max_tasks]}")

        diagnoses = []
        gt_dir = Path(ground_truth_dir)
        log_dir_path = Path(log_dir)
        script_dir_path = Path(script_dir)

        for task_num in failed_tasks[:max_tasks]:
            task_key = f"task_{task_num}"
            task_desc = task_list.get(task_key, "Task description not found")

            # Find matching files
            gt_matches = list(gt_dir.glob(f"{task_num}_*.py"))
            log_matches = list(log_dir_path.glob(f"*task_{task_num}_*.log"))
            script_matches = list(script_dir_path.glob(f"{task_num}_*.py"))

            # Find analysis report
            analysis_matches = list(analysis_dir.glob(f"eval_*task_{task_num}_*.md"))

            if not log_matches:
                print(f"[SKIP] No log found for task_{task_num}")
                continue

            diagnosis = self.diagnose_single(
                task_name=task_key,
                task_description=task_desc,
                log_path=str(log_matches[0]),
                script_path=str(script_matches[0]) if script_matches else None,
                ground_truth_path=str(gt_matches[0]) if gt_matches else None,
                analysis_report_path=str(analysis_matches[0]) if analysis_matches else None,
            )
            diagnoses.append(diagnosis)

            # Update CAMEL.md every update_batch_size tasks
            if len(diagnoses) % update_batch_size == 0:
                batch_start = len(diagnoses) - update_batch_size
                batch = diagnoses[batch_start:]
                print(f"\n[Updating CAMEL.md with {len(batch)} diagnoses...]")
                self.update_camel_md(batch)

        # Final update for remaining diagnoses
        remaining = len(diagnoses) % update_batch_size
        if remaining > 0:
            batch = diagnoses[-remaining:]
            print(f"\n[Updating CAMEL.md with remaining {len(batch)} diagnoses...]")
            self.update_camel_md(batch)

        # Save diagnosis summary
        summary_path = self.output_dir / "diagnosis_summary.md"
        with open(summary_path, 'w') as f:
            f.write("# Diagnosis Summary\n\n")
            f.write(f"- **Total diagnosed**: {len(diagnoses)}\n")
            f.write(f"- **Explore phase issues**: {sum(1 for d in diagnoses if d['failure_phase'] == 'explore')}\n")
            f.write(f"- **Code phase issues**: {sum(1 for d in diagnoses if d['failure_phase'] == 'code')}\n\n")
            f.write("## Tasks Diagnosed\n\n")
            for d in diagnoses:
                f.write(f"- [{d['task_name']}]({d['report_path']}) - {d['failure_phase']} phase issue\n")

        print(f"\nDiagnosis summary saved to: {summary_path}")

        return diagnoses


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Agent - Analyzes failed code generation and improves CAMEL.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single task diagnosis
    python diagnose_agent.py \\
        --task-name task_2 \\
        --task-desc "create agent using Gemini with DuckDuckGo" \\
        --log logs/code_task_2.log \\
        --script task-script/single_agent/2_duckduckgo.py \\
        --ground-truth ground_truth/single_agent/2_duckduckgo.py \\
        --analysis-report evaluation_reports/eval_task_2.md

    # Batch diagnosis (diagnose all failed tasks)
    python diagnose_agent.py \\
        --log-dir logs/ \\
        --script-dir task-script/single_agent/ \\
        --ground-truth-dir ../ground_truth/single_agent/ \\
        --analysis-dir evaluation_reports/w_context_refactored/ \\
        --task-list ../task_list.json
        """
    )

    # Single task mode
    parser.add_argument("--task-name", type=str, help="Task name (e.g., task_1)")
    parser.add_argument("--task-desc", type=str, help="Task description")
    parser.add_argument("--log", type=str, help="Path to execution log file")
    parser.add_argument("--script", type=str, help="Path to generated script")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth script")
    parser.add_argument("--analysis-report", type=str, help="Path to analysis agent report")

    # Batch mode
    parser.add_argument("--log-dir", type=str, help="Directory containing log files")
    parser.add_argument("--script-dir", type=str, help="Directory containing generated scripts")
    parser.add_argument("--ground-truth-dir", type=str, help="Directory containing ground truth")
    parser.add_argument("--analysis-dir", type=str, help="Directory containing analysis reports")
    parser.add_argument("--task-list", type=str, help="Path to task_list.json")

    # Common options
    parser.add_argument(
        "--project", "-p",
        type=str,
        default=None,
        help="Project directory (CAMEL repo root)"
    )
    parser.add_argument(
        "--camel-md",
        type=str,
        default=None,
        help="Path to CAMEL.md file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for diagnosis reports"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4-1",
        help="Model to use (default: gpt-4-1)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Maximum number of tasks to diagnose"
    )
    parser.add_argument(
        "--update-batch-size",
        type=int,
        default=2,
        help="Update CAMEL.md every N tasks (default: 2)"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run the task after updating CAMEL.md"
    )

    args = parser.parse_args()

    # Validate mode
    is_single_mode = args.task_name or args.log
    is_batch_mode = args.log_dir or args.analysis_dir

    if not is_single_mode and not is_batch_mode:
        parser.error("Must specify either single task args or batch mode args")

    if is_batch_mode:
        required = ["log_dir", "script_dir", "ground_truth_dir", "analysis_dir", "task_list"]
        missing = [r for r in required if not getattr(args, r.replace("-", "_"), None)]
        if missing:
            parser.error(f"Batch mode requires: {', '.join('--' + r for r in missing)}")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    code_agent_dir = os.path.dirname(base_dir)
    default_project_dir = os.path.dirname(code_agent_dir)

    project_dir = args.project if args.project else default_project_dir
    camel_md_path = args.camel_md if args.camel_md else os.path.join(base_dir, "CAMEL.md")

    # Parse model
    model_mapping = {
        "gpt-4o": ModelType.GPT_4O,
        "gpt-4o-mini": ModelType.GPT_4O_MINI,
        "gpt-4-1": ModelType.GPT_4_1,
        "gpt-4-1-mini": ModelType.GPT_4_1_MINI,
    }
    model_type = model_mapping.get(args.model.lower(), ModelType.GPT_4_1)

    # Create agent
    agent = DiagnoseAgent(
        working_directory=project_dir,
        camel_md_path=camel_md_path,
        model_type=model_type,
        output_dir=args.output,
    )

    # Run
    if is_single_mode:
        result = agent.diagnose_single(
            task_name=args.task_name or "unknown",
            task_description=args.task_desc or "",
            log_path=args.log,
            script_path=args.script,
            ground_truth_path=args.ground_truth,
            analysis_report_path=args.analysis_report,
        )
        print("\n" + "="*60)
        print("DIAGNOSIS")
        print("="*60)
        print(result["diagnosis"][:3000])
        if len(result["diagnosis"]) > 3000:
            print("\n... (truncated, see full report)")
        print(f"\nFull report: {result['report_path']}")

        # Update CAMEL.md automatically
        agent.update_camel_md([result])
    else:
        agent.batch_diagnose(
            log_dir=args.log_dir,
            script_dir=args.script_dir,
            ground_truth_dir=args.ground_truth_dir,
            analysis_output_dir=args.analysis_dir,
            task_list_path=args.task_list,
            max_tasks=args.max_tasks,
            update_batch_size=args.update_batch_size,
        )


if __name__ == "__main__":
    main()
