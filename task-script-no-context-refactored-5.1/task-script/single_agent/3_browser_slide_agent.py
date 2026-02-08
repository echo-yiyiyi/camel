import asyncio
import json

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import HybridBrowserToolkit, PPTXToolkit

USER_DATA_DIR = "User_Data"

# Initialize the model
model = ModelFactory.create(
    model_type="gpt-4o",
    model_config_dict={"temperature": 0.0, "top_p": 1},
)

# Initialize the browser toolkit with custom tools
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
    headless=True,
    user_data_dir=USER_DATA_DIR,
    enabled_tools=custom_tools,
    browser_log_to_file=True,
    stealth=True,
    viewport_limit=True,
)

# Initialize the PPTX toolkit
pptx_toolkit = PPTXToolkit(
    working_directory="./pptx_outputs",
)

# Create the agent with both browser and PPTX tools
agent = ChatAgent(
    model=model,
    tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
    max_iteration=10,
)

TASK_PROMPT = """
Search for information about CAMEL-AI using the browser tools.
Gather relevant information and generate a PowerPoint presentation
about CAMEL-AI using the PPTX tools.
"""

async def main():
    try:
        response = await agent.astep(TASK_PROMPT)
        print("Task:", TASK_PROMPT)
        print("Response from agent:")
        print(response.msgs[0].content if response.msgs else "<no response>")
    finally:
        print("\nClosing browser...")
        await browser_toolkit.browser_close()
        print("Browser closed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
