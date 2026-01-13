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
logs_dir = os.path.join(base_dir, "logs")
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


def get_file_description(keywords: str = "") -> str:
    r"""Get the description of files in the current directory, optionally filtered by keywords.

    Args:
        keywords (str, optional): Keywords to filter files. If provided, only files matching 
            the keywords will be returned. If empty, returns a summary with file count and 
            sample paths. (default: "")

    Returns:
        str: The description of files. If keywords provided, returns matching file descriptions.
            If empty, returns a summary to avoid overwhelming the context.
    """
    with open("code-agent/file_descriptions_2026-01-13_12-57-52.md", "r") as f:
        content = f.read()
    
    # If no keywords, return a summary instead of the full file
    if not keywords or keywords.strip() == "":
        # Count files
        file_count = content.count("## ./")
        # Extract first 20 file paths as samples
        import re
        file_paths = re.findall(r"## (\./camel/[^\n]+)", content)[:20]
        sample_paths = "\n".join(f"  - {path}" for path in file_paths)
        
        return f"""File Descriptions Summary:
Total files: {file_count}

Sample file paths:
{sample_paths}
...

To get specific file descriptions, use this tool with keywords related to your task.
For example: get_file_description(keywords="weather agent toolkit")
This will return only files matching those keywords.

Available categories include:
- agents (ChatAgent, BaseAgent, etc.)
- toolkits (WeatherToolkit, BrowserToolkit, etc.)
- models (QwenModel, OpenAIModel, etc.)
- examples (example scripts and tutorials)
- docs (documentation files)
"""
    
    # Filter by keywords
    keywords_lower = keywords.lower()
    lines = content.split('\n')
    result = []
    current_section = []
    in_matching_section = False
    
    for i, line in enumerate(lines):
        if line.startswith("## ./"):
            # Save previous section if it matched
            if in_matching_section and current_section:
                result.extend(current_section)
                result.append("")  # Add separator
            
            # Check if this file path or description matches keywords
            current_section = [line]
            in_matching_section = keywords_lower in line.lower()
        elif in_matching_section:
            current_section.append(line)
            # Stop at separator
            if line.strip() == "---":
                in_matching_section = False
    
    # Add last section if matching
    if in_matching_section and current_section:
        result.extend(current_section)
    
    if result:
        return "\n".join(result)
    else:
        return f"No files found matching keywords: {keywords}\n\nUse get_file_description() without arguments to see available file categories."


# Set agent
# Increase step_timeout to 600 seconds (10 minutes) to handle complex tasks
camel_agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=[FunctionTool(read_file_tool), FunctionTool(get_file_description)] + tools,
    step_timeout=600.0,  # 10 minutes timeout for complex code generation tasks
)


with open("code-agent/task_list.json", "r") as f:
    task_list = json.load(f)
    for task_name, task_description in task_list.items():
        print(f"Processing task: {task_name}")
        camel_agent.reset()
        # Record start time
        start_time = time.time()
        start_datetime = datetime.now()

        cot_1_prompt = f"""
        Your task is: {task_description}
        
        IMPORTANT: In this step, you should ONLY get the file descriptions. Do NOT start implementing the task yet.
        
        First, call `get_file_description()` without arguments to see a summary of available files.
        Then, based on the task keywords, call `get_file_description(keywords="your_keywords")` to get 
        relevant file descriptions. Extract keywords from the task like: agent, toolkit, model, etc.
        
        This will help you understand what files are available for reference in the next steps.
        """

        cot_2_prompt = f"""
        Your task is: {task_description}
        
        Based on the file descriptions you obtained in Step 1:
        1. Extract the keywords from the task description.
        2. Identify relevant files from the file descriptions list that match these keywords.
        3. Read the relevant example files, documentation, and source code using the `read_file_tool`.
        
        IMPORTANT: 
        - Only use file paths that exist in the file descriptions list from Step 1.
        - Do not generate or guess file paths that are not in the list.
        - Focus on reading files related to: examples, documentation, and core source code that are relevant to the task keywords.
        """

        cot_3_prompt = f"""
        Your task is: {task_description}
        
        Based on the files you read in Step 2, write the complete script according to the task requirements.
        
        Use the `shell_write_content_to_file` tool to create the script.
        Make sure to:
        - Use the correct imports and classes based on the files you read
        - Follow the patterns and examples you found in the codebase
        - Save the script to the correct path as specified in the task
        """

        cot_4_prompt = f"""
        Your task is: {task_description}
        
        IMPORTANT: All packages and environment variables have been deployed correctly.
        
        Execute the script you created in Step 3 and debug it until it runs without errors.
        
        Process:
        1. Execute the script using the terminal tools
        2. If errors occur, analyze the error messages carefully
        3. Fix the script using the `shell_write_content_to_file` tool
        4. Re-run the script and repeat until it succeeds
        5. If you need to read more files for debugging, use the `read_file_tool`
        
        Note: If you encounter environment variable errors or API errors, it is more likely to be a problem in the script (wrong API usage, incorrect model names, etc.), since the environment and API are correctly deployed.
        """

        # Collect all responses, tool calls, and reasoning from all steps
        all_responses = []
        all_tool_calls = []
        all_reasoning_contents = []
        
        cot_steps = [
            ("Step 1: Get file descriptions", cot_1_prompt),
            ("Step 2: Extract keywords and read relevant files", cot_2_prompt),
            ("Step 3: Write the script", cot_3_prompt),
            ("Step 4: Execute and debug", cot_4_prompt),
        ]
        
        for step_name, cot_prompt in cot_steps:
            print(f"\n{step_name}")
            response = camel_agent.step(cot_prompt)
            all_responses.append((step_name, response))
            print(response.msgs[0].content if response.msgs else "")
            
            # Collect tool calls from this step
            step_tool_calls = response.info.get("tool_calls", [])
            if step_tool_calls:
                all_tool_calls.extend(step_tool_calls)
            
            # Collect reasoning content from this step
            for msg in response.msgs:
                if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                    all_reasoning_contents.append((step_name, msg.reasoning_content))
        
        # Record end time and calculate duration
        end_time = time.time()
        end_datetime = datetime.now()
        duration_seconds = end_time - start_time
        duration = timedelta(seconds=duration_seconds)
        
        # Get the final response (last step)
        final_response = all_responses[-1][1] if all_responses else None
        task_output = final_response.msgs[0].content if final_response and final_response.msgs else ""
        
        # Get full conversation history
        chat_history = camel_agent.chat_history
        
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
        break
