# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import asyncio
import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import HybridBrowserToolkit, PPTXToolkit


USER_DATA_DIR = "User_Data"

# Create the model
model = ModelFactory.create(
    model_type="gpt-4o",
    model_config_dict={"temperature": 0.0, "top_p": 1},
)

# Create the browser toolkit with custom tools
custom_tools = [
    "browser_open",
    "browser_close",
    "browser_visit_page",
    "browser_back",
    "browser_forward",
    "browser_click",
    "browser_type",
    "browser_switch_tab",
    "browser_enter",
]

browser_toolkit = HybridBrowserToolkit(
    headless=False,
    user_data_dir=USER_DATA_DIR,
    enabled_tools=custom_tools,
    browser_log_to_file=True,
    stealth=True,
    viewport_limit=True,
)

# Create the PPTX toolkit
pptx_toolkit = PPTXToolkit(
    working_directory="./pptx_outputs",
)

# Create the agent with both toolkits
agent = ChatAgent(
    model=model,
    tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
    max_iteration=10,
)

TASK_PROMPT = r"""
Search for information about CAMEL-AI using the browser tools.
Gather relevant information and generate a PowerPoint presentation about CAMEL-AI.
Use the PPTX tools to create the slides.
"""


async def main():
    try:
        response = await agent.astep(TASK_PROMPT)
        print("Task:", TASK_PROMPT)
        print(f"Using user data directory: {USER_DATA_DIR}")
        print(f"Enabled tools: {browser_toolkit.enabled_tools}")
        print("\nResponse from agent:")
        print(response.msgs[0].content if response.msgs else "<no response>")
    finally:
        print("\nClosing browser...")
        await browser_toolkit.browser_close()
        print("Browser closed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
