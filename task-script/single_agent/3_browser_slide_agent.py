import asyncio
import os

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import HybridBrowserToolkit, PPTXToolkit
from camel.types import ModelPlatformType, ModelType


USER_DATA_DIR = "User_Data"


def run_browser_pptx_agent():
    # Initialize the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict={"temperature": 0.0, "top_p": 1},
    )

    # Initialize browser toolkit with a custom set of tools
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
        headless=True,
        user_data_dir=USER_DATA_DIR,
        enabled_tools=custom_browser_tools,
        browser_log_to_file=True,
        stealth=True,
        viewport_limit=True,
    )

    # Initialize PPTX toolkit
    pptx_toolkit = PPTXToolkit(
        working_directory="./pptx_outputs",
    )

    # Create the agent with both browser and pptx tools
    agent = ChatAgent(
        model=model,
        tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
        max_iteration=10,
    )

    # Task prompt to search for CAMEL-AI information and generate slides
    TASK_PROMPT = r"""
    Use the browser tools to search for information about CAMEL-AI.
    Gather relevant details about CAMEL-AI's framework, features, community, and applications.
    Then use the PPTX tools to create a PowerPoint presentation summarizing the information you found.
    Follow the PPTXToolkit guidelines for slide content formatting and template selection.
    """

    async def main():
        try:
            response = await agent.astep(TASK_PROMPT)
            print("Task prompt:", TASK_PROMPT)
            print("\nAgent response:")
            print(response.msgs[0].content if response.msgs else "<no response>")
            print(f"\nTool calls: {response.info.get('tool_calls', [])}")
        finally:
            print("\nClosing browser...")
            await browser_toolkit.browser_close()
            print("Browser closed successfully.")

    asyncio.run(main())


if __name__ == "__main__":
    run_browser_pptx_agent()
