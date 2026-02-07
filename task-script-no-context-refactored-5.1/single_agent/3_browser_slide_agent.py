import asyncio
import logging

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import HybridBrowserToolkit, PPTXToolkit
from camel.types import ModelType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

USER_DATA_DIR = "User_Data"

# Create model backend
model_backend = ModelFactory.create(
    model_type=ModelType.GPT_4O,
    model_config_dict={"temperature": 0.0, "top_p": 1},
)

# Create browser toolkit with custom tools
custom_browser_tools = [
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
    enabled_tools=custom_browser_tools,
    browser_log_to_file=True,
    stealth=True,
    viewport_limit=True,
)

# Create PPTX toolkit
pptx_toolkit = PPTXToolkit(working_directory="./pptx_outputs")

# Combine tools from both toolkits
combined_tools = [*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()]

# Create ChatAgent with combined tools
agent = ChatAgent(
    model=model_backend,
    tools=combined_tools,
    max_iteration=10,
)

TASK_PROMPT = r"""
Search for information about CAMEL-AI, including its features, applications, community, and impact.
Gather relevant information from multiple web sources.
Then generate a PowerPoint presentation summarizing the key points about CAMEL-AI.
Use the PPTX tools to create slides with appropriate titles, bullet points, and images if possible.
"""

async def main() -> None:
    try:
        response = await agent.astep(TASK_PROMPT)
        print("Task:", TASK_PROMPT)
        print(f"Using user data directory: {USER_DATA_DIR}")
        print(f"Enabled tools: {[tool.get_function_name() for tool in combined_tools]}")
        print("\nResponse from agent:")
        print(response.msgs[0].content if response.msgs else "<no response>")
    finally:
        print("\nClosing browser...")
        await browser_toolkit.browser_close()
        print("Browser closed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
