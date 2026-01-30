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


import os
import json
import time
from datetime import datetime, timedelta
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit, FunctionTool
from camel.types import ModelPlatformType, ModelType

exp_id = "_v0"

# Enable verbose logging to see tool execution details
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
set_log_level('INFO')  # Use 'DEBUG' for even more detailed output

# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# Get camel directory (parent of terminal_agent)
camel_dir = os.path.dirname(base_dir)
# Get project root directory (parent of camel directory)
project_root = os.path.dirname(camel_dir)
# Define workspace directory for the toolkit
workspace_dir = os.path.join(project_root, "workspace")
# Define logs directory for task outputs
logs_dir = os.path.join(base_dir, f"logs{exp_id}")
os.makedirs(logs_dir, exist_ok=True)

# Define system message
sys_msg = (
    "You are a Code Agent helping with code generation tasks. "
    "You have access to terminal tools that can help you execute "
    "shell commands, search files and write scripts. "
)

# Set model config - use camel directory as working directory
# Agent will start in camel folder, can directly use 'ls' to see camel contents
tools = TerminalToolkit(
    working_directory=camel_dir,
    clone_current_env=True,
    timeout=300.0, # 10 minutes timeout for complex code generation tasks
).get_tools()

model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4_1_MINI,
    model_config_dict=model_config_dict,
)


def read_file_tool(file_path: str) -> str:
    r"""Read the content of a file.

    Args:
        file_path (str): The path of the file to read.

    Returns:
        str: The content of the file.
    """
    print(f"tools: {tools}")
    content = tools[0](
        id=f"read_file_{file_path}", 
        # If the file is less than 200 lines, read the whole file. Otherwise, read the first 2000 lines.
        command=f"lines=$(wc -l < \"{file_path}\") && if [ $lines -lt 200 ]; then cat \"{file_path}\"; else head -2000 \"{file_path}\"; fi")
    return content

# Set agent
# Increase step_timeout to 600 seconds (10 minutes) to handle complex tasks
code_agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=[FunctionTool(read_file_tool)] + tools,
    step_timeout=600.0,  # 10 minutes timeout for complex code generation tasks
)


GLOB_TOOL_NAME = "TerminalToolkit and the `glob` command"
GREP_TOOL_NAME = "TerminalToolkit and the `grep` command"
READ_TOOL_NAME = "`read_file_tool`"
BASH_TOOL_NAME = "TerminalToolkit's `shell_exec` tool"

explore_sys_prompt = """
<!--
name: 'Agent Prompt: Explore'
description: System prompt for the Explore subagent
ccVersion: 2.0.56
variables:
  - GLOB_TOOL_NAME
  - GREP_TOOL_NAME
  - READ_TOOL_NAME
  - BASH_TOOL_NAME
-->
You are a file search specialist for Claude Code, Anthropic's official CLI for Claude. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use ${GLOB_TOOL_NAME} for broad file pattern matching
- Use ${GREP_TOOL_NAME} for searching file contents with regex
- Use ${READ_TOOL_NAME} when you know the specific file path you need to read
- Use ${BASH_TOOL_NAME} ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use ${BASH_TOOL_NAME} for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly.

"""

explore_agent = ChatAgent(
    system_message=explore_sys_prompt,
    model=model,
    tools=[FunctionTool(read_file_tool)] + tools,
)

with open("code-agent/task_list.json", "r") as f:
    task_list = json.load(f)
    cnt = 0
    for task_name, task_description in task_list.items():
        print(f"Processing code task: {task_name}")
        code_agent.reset()
        # Record start time
        start_time = time.time()
        start_datetime = datetime.now()

        explore_prompt = f"""
        Your task is  {task_description}.
        Return the possible relevent path for the task.
        """

        explore_response = explore_agent.step(explore_prompt)

        code_prompt = f"""
        Your task is  {task_description}. The possible relevent path for the task are: {explore_response.msgs[0].content}.
        Read the files and write the script according to the task using the `shell_write_content_to_file` tool.
        Execute the script and debug the script until it has no error. If the script need to be modified, modify the script using the `shell_write_content_to_file` tool.
        If an error occurs, automatically fix it and re-run until successful.
        If you need to read more files for debugging, use the `read_file_tool` to read the files.
        Note that if you encouter an environment variable error or api error, it is more likely to be a problem in the script, since the environment and api are correctly deployed.
        """
        code_response = code_agent.step(code_prompt)

        # Collect all responses, tool calls, and reasoning from both steps
        all_responses = [
            ("explore", explore_response),
            ("code", code_response),
        ]
        all_tool_calls = []
        all_reasoning_contents = []

        # 从每一步里收集 tool_calls 和 reasoning_content
        for step_name, response in all_responses:
            if response is None:
                continue

            # Collect tool calls from this step
            step_tool_calls = response.info.get("tool_calls", []) if hasattr(response, "info") else []
            if step_tool_calls:
                all_tool_calls.extend(step_tool_calls)

            # Collect reasoning content from this step
            for msg in response.msgs:
                if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                    all_reasoning_contents.append((step_name, msg.reasoning_content))
        
        # Record end time and calculate duration
        end_time = time.time()
        end_datetime = datetime.now()
        duration_seconds = end_time - start_time
        duration = timedelta(seconds=duration_seconds)
        
        # Get the final response (last step)
        final_response = all_responses[-1][1] if all_responses else None
        task_output = (
            final_response.msgs[0].content
            if final_response and final_response.msgs
            else ""
        )

        # Get full conversation history（从 code_agent 视角）
        chat_history = code_agent.chat_history

        # Use collected tool calls and reasoning from all steps
        tool_calls = all_tool_calls
        reasoning_contents = [reasoning for _, reasoning in all_reasoning_contents]
        
        print(f"Task {task_name} completed in {duration} ({duration_seconds:.2f} seconds)")
        print(f"Task {task_name} completed: {task_output}")
        
        # Save task output to log file
        timestamp = start_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{task_name}_{timestamp}.log"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        with open(log_filepath, "w", encoding="utf-8") as log_file:
            log_file.write(f"Task: {task_name}\n")
            log_file.write(f"Start Time: {start_datetime.isoformat()}\n")
            log_file.write(f"End Time: {end_datetime.isoformat()}\n")
            log_file.write(f"Duration: {duration} ({duration_seconds:.2f} seconds)\n")
            log_file.write(f"Description: {task_description}\n")
            log_file.write("=" * 80 + "\n\n")
            
            # Write step-by-step responses
            log_file.write("Step-by-Step Responses:\n")
            log_file.write("-" * 80 + "\n")
            for step_name, step_response in all_responses:
                log_file.write(f"\n[{step_name}]\n")
                step_output = step_response.msgs[0].content if step_response.msgs else ""
                log_file.write(f"Output: {step_output[:500]}...\n" if len(step_output) > 500 else f"Output: {step_output}\n")
            log_file.write("\n" + "=" * 80 + "\n\n")
            
            # Write reasoning content if available (grouped by step)
            if all_reasoning_contents:
                log_file.write("Reasoning Content (by Step):\n")
                log_file.write("-" * 80 + "\n")
                for step_name, reasoning in all_reasoning_contents:
                    log_file.write(f"\n[{step_name}]\n")
                    log_file.write(str(reasoning))
                    log_file.write("\n")
                log_file.write("\n" + "=" * 80 + "\n\n")
            
            # Write tool calls (all steps combined)
            if tool_calls:
                log_file.write(f"Tool Calls (Total: {len(tool_calls)}):\n")
                log_file.write("-" * 80 + "\n")
                for i, tool_call in enumerate(tool_calls, 1):
                    log_file.write(f"\n[Tool Call {i}]\n")
                    log_file.write(f"Tool Name: {tool_call.tool_name}\n")
                    log_file.write(f"Tool Call ID: {tool_call.tool_call_id}\n")
                    log_file.write(f"Arguments: {json.dumps(tool_call.args, indent=2, ensure_ascii=False)}\n")
                    log_file.write(f"Result:\n{str(tool_call.result)}\n")
                    log_file.write("\n")
                log_file.write("=" * 80 + "\n\n")
            
            # Write full conversation history
            log_file.write("Full Conversation History:\n")
            log_file.write("-" * 80 + "\n")
            for i, msg in enumerate(chat_history, 1):
                log_file.write(f"\n[Message {i}]\n")
                log_file.write(f"Role: {msg.get('role', 'unknown')}\n")
                if 'content' in msg and msg['content']:
                    content = msg['content']
                    # Truncate very long content
                    if isinstance(content, str) and len(content) > 5000:
                        content = content[:5000] + f"\n... (truncated, total length: {len(msg['content'])})\n"
                    log_file.write(f"Content: {content}\n")
                if 'tool_calls' in msg and msg['tool_calls']:
                    log_file.write(f"Tool Calls: {json.dumps(msg['tool_calls'], indent=2, ensure_ascii=False)}\n")
                log_file.write("\n")
            log_file.write("=" * 80 + "\n\n")
            
            # Write final task output
            log_file.write("Final Task Output:\n")
            log_file.write("-" * 80 + "\n")
            log_file.write(task_output)
            log_file.write("\n")
        
        print(f"Task output saved to: {log_filepath}")
        cnt += 1
        if cnt >= 1:
            break
