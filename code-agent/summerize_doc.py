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
import subprocess
import asyncio
from datetime import datetime
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Enable verbose logging
set_log_level('INFO')

# Get current script directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# Get camel directory (parent of code-agent)
camel_dir = os.path.dirname(base_dir)
# Get project root directory (parent of camel directory)
project_root = os.path.dirname(camel_dir)

# Change to project root directory
os.chdir(project_root)

# Define system message for generating descriptions
sys_msg = (
    "You are a helpful assistant that generates concise one-sentence descriptions "
    "for code files. Given the file path and content, provide a clear, informative "
    "one-sentence description in English that explains what the file does or contains."
)

# Set model config
model_config_dict = ChatGPTConfig(
    temperature=0.0,
).as_dict()

model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config_dict,
)

# Create agent for generating descriptions
agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    step_timeout=60.0,
)

def get_file_list():
    """Execute find commands to get list of files."""
    files = []
    
    # First find command: Python files
    cmd1 = [
        'find', '.', '-name', '*.py',
        '-not', '-path', '*/node_modules/*',
        '-not', '-path', '*/.venv/*',
        '-not', '-path', '*/.initial_env/*',
        '-not', '-path', '*/__pycache__/*',
        '-not', '-path', '*/code-agent/*',
        '-not', '-path', '*/task-script-v0/*'
    ]
    
    # Second find command: docs files
    cmd2 = [
        'find', './docs', '-type', 'f',
        '(', '-name', '*.md', '-o',
        '-name', '*.rst', '-o',
        '-name', '*.mdx', '-o',
        '-name', '*.py', '-o',
        '-name', '*.ipynb', ')'
    ]
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
        files.extend([f.strip() for f in result1.stdout.split('\n') if f.strip()])
    except subprocess.CalledProcessError as e:
        print(f"Error executing first find command: {e}")
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
        files.extend([f.strip() for f in result2.stdout.split('\n') if f.strip()])
    except subprocess.CalledProcessError as e:
        print(f"Error executing second find command: {e}")
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files

async def read_file_content(file_path):
    """Read file content completely asynchronously."""
    def _read_file():
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _read_file)
        return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

async def generate_description(file_path, content, agent):
    """Generate a one-sentence description for a file using LLM asynchronously."""
    # Limit content to first 5000 characters to avoid token limits
    content_preview = content[:5000] if len(content) > 5000 else content
    prompt = f"""Please provide a concise one-sentence description in English for this file:

File path: {file_path}

File content:
{content_preview}

Provide only the description, no additional text."""
    
    try:
        agent.reset()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, agent.step, prompt)
        description = response.msgs[0].content if response.msgs else "No description generated"
        # Clean up the description (remove quotes, extra whitespace)
        description = description.strip().strip('"').strip("'")
        return description
    except Exception as e:
        print(f"Error generating description for {file_path}: {e}")
        return f"Error generating description: {str(e)}"

async def process_file(file_path, agent, semaphore):
    """Process a single file: read content and generate description."""
    async with semaphore:  # Limit concurrent LLM calls
        print(f"Processing: {file_path}")
        
        # Read file content
        content = await read_file_content(file_path)
        
        if not content:
            description = "Empty file or could not read content"
        else:
            # Generate description using LLM
            description = await generate_description(file_path, content, agent)
        
        print(f"  Completed: {file_path} - {description[:80]}...")
        return file_path, description

async def main_async():
    """Main async function to process all files and generate markdown."""
    print("Getting file list...")
    files = get_file_list()
    print(f"Found {len(files)} files to process")
    
    # Create multiple agents for concurrent processing
    # Limit concurrent LLM calls to avoid rate limits (adjust based on your API limits)
    max_concurrent = 5
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create agents for concurrent processing
    agents = []
    for _ in range(max_concurrent):
        agent = ChatAgent(
            system_message=sys_msg,
            model=model,
            step_timeout=60.0,
        )
        agents.append(agent)
    
    # Process files concurrently
    tasks = []
    for i, file_path in enumerate(files):
        agent = agents[i % len(agents)]  # Round-robin agent assignment
        task = process_file(file_path, agent, semaphore)
        tasks.append(task)
    
    print(f"\nProcessing {len(files)} files with {max_concurrent} concurrent workers...")
    results = await asyncio.gather(*tasks)
    
    # Sort results by file path to maintain consistent output order
    results.sort(key=lambda x: x[0])
    
    # Output markdown file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(base_dir, f"file_descriptions_{timestamp}.md")
    
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# File Descriptions\n\n")
        md_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md_file.write("---\n\n")
        
        for file_path, description in results:
            md_file.write(f"## {file_path}\n\n")
            md_file.write(f"{description}\n\n")
            md_file.write("---\n\n")
    
    print(f"\nCompleted! Output saved to: {output_file}")

def main():
    """Main function entry point."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
