#!/usr/bin/env python3
"""Parse evaluation reports to extract failed tasks and generate evolve_task_list.json."""

import json
import re
import sys
import os


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <eval_dir> <task_list.json> <output.json>")
        sys.exit(1)

    eval_dir = sys.argv[1]
    task_list_file = sys.argv[2]
    output_file = sys.argv[3]

    summary_file = os.path.join(eval_dir, "evaluation_summary.md")

    # Read original task list
    with open(task_list_file, 'r') as f:
        original_tasks = json.load(f)

    # Read evaluation summary to find failed tasks
    with open(summary_file, 'r') as f:
        summary = f.read()

    # Parse failed tasks from summary table
    # Format: | script_name | ❌ FAIL | [Report](path) |
    failed_tasks = []
    for line in summary.split('\n'):
        if '\u274c FAIL' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 3:
                script_name = parts[0]
                match = re.match(r'(\d+)_', script_name)
                if match:
                    task_num = int(match.group(1))
                    report_match = re.search(r'\[Report\]\(([^)]+)\)', parts[2])
                    report_path = report_match.group(1) if report_match else None
                    failed_tasks.append({
                        'task_num': task_num,
                        'script_name': script_name,
                        'report_path': report_path,
                    })

    if not failed_tasks:
        print('[INFO] No failed tasks found. Nothing to evolve.')
        # Write empty dict so caller knows
        with open(output_file, 'w') as f:
            json.dump({}, f)
        sys.exit(0)

    print(f'[INFO] Found {len(failed_tasks)} failed task(s): '
          f'{[t["task_num"] for t in failed_tasks]}')

    # For each failed task, read the report and extract fix information
    evolve_tasks = {}
    for task_info in failed_tasks:
        task_num = task_info['task_num']
        task_key = f'task_{task_num}'
        report_path = task_info['report_path']

        # Get original task description
        original_desc = original_tasks.get(task_key, '')
        if not original_desc:
            print(f'[WARN] No original task found for {task_key}, skipping')
            continue

        # Read the evaluation report and extract how-to-fix content
        fix_hints = ''
        if report_path and os.path.isfile(report_path):
            with open(report_path, 'r') as f:
                report = f.read()

            # Extract all "6. Systematic Improvement" sections
            improvements = re.findall(
                r'(\*\*6\. Systematic Improvement\*\*:.*?)(?=\n---|\n### Root Cause:|\Z)',
                report, re.DOTALL
            )
            if improvements:
                fix_hints = '\n\n'.join(s.strip() for s in improvements)

        # Build evolved task description
        if fix_hints:
            evolved_desc = (
                fix_hints
                + '\n\nNOTE: Read the relevant source code and '
                'decide what to change based on the issue type:\n'
                '- For **Code Agent Issue** (general skills like '
                'search strategies, prompt patterns): modify the '
                'code agent code (e.g. system prompts, search '
                'techniques in generic_code_agent.py)\n'
                '- For **CAMEL Codebase Issue** (framework-specific '
                'patterns, API usage): APPEND a new Technique '
                'section to code-agent/generic/CAMEL.md, or fix '
                'the CAMEL source code directly\n'
                'When updating any file, ALWAYS read its full '
                'content first and preserve all existing content.'
            )
        else:
            # No Systematic Improvement found, skip this task
            print(f'[INFO] {task_key}: no Systematic Improvement, skipping')
            continue

        evolve_tasks[task_key] = evolved_desc
        print(f'[INFO] {task_key}: added with fix hints')

    # Write evolve_task_list.json
    with open(output_file, 'w') as f:
        json.dump(evolve_tasks, f, indent=4, ensure_ascii=False)

    print(f'[INFO] Wrote {len(evolve_tasks)} task(s) to {output_file}')

    # Preview
    print('\n--- evolve_task_list.json ---')
    for name, desc in evolve_tasks.items():
        preview = desc[:200] + '...' if len(desc) > 200 else desc
        print(f'  {name}: {preview}')
    print('---')


if __name__ == '__main__':
    main()
