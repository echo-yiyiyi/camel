from camel.agents import ChatAgent
from camel.societies.workforce.workforce import Workforce
from camel.societies.workforce.role_playing_worker import RolePlayingWorker
from camel.toolkits.search_toolkit import SearchToolkit
from camel.toolkits.browser_toolkit import BrowserToolkit
from camel.models import ModelFactory
from camel.types.enums import ModelType


def main():
    # Define the question to answer
    question = (
        "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
        "from Jan-May 2018 as listed on the NIH website?"
    )

    # Create a search toolkit instance and get the Brave search tool
    search_toolkit = SearchToolkit()
    brave_search_tool = None
    for tool in search_toolkit.get_tools():
        if tool.get_function_name() == "search_brave":
            brave_search_tool = tool
            break

    if brave_search_tool is None:
        raise RuntimeError("Brave search tool not found in SearchToolkit")

    # Assistant agent kwargs for search worker
    search_assistant_agent_kwargs = {
        "system_message": "You are a helpful search assistant.",
        "model": ModelFactory.create(
            model_type=ModelType.DEFAULT,
        ),
        "tools": [brave_search_tool],
    }

    # Create a search worker
    search_worker = RolePlayingWorker(
        description="Search Worker",
        assistant_role_name="Search Assistant",
        user_role_name="Search User",
        assistant_agent_kwargs=search_assistant_agent_kwargs,
    )

    # Create a browser toolkit instance and get the first browser tool
    browser_toolkit = BrowserToolkit()
    browser_tools = browser_toolkit.get_tools()
    if not browser_tools:
        raise RuntimeError("No browser tools found in BrowserToolkit")
    browser_tool = browser_tools[0]

    # Assistant agent kwargs for browser worker
    browser_assistant_agent_kwargs = {
        "system_message": "You are a helpful browser assistant.",
        "model": ModelFactory.create(
            model_type=ModelType.DEFAULT,
        ),
        "tools": [browser_tool],
    }

    # Create a browser worker
    browser_worker = RolePlayingWorker(
        description="Browser Worker",
        assistant_role_name="Browser Assistant",
        user_role_name="Browser User",
        assistant_agent_kwargs=browser_assistant_agent_kwargs,
    )

    # Create a workforce with the workers as children
    workforce = Workforce(
        description="Workforce to answer clinical trial enrollment question",
        children=[search_worker, browser_worker],
    )

    # Add a main task
    workforce.add_main_task(question)

    # Start the workforce and wait for completion
    import asyncio
    asyncio.run(workforce.start())

    # Get the result of the main task
    main_task = workforce.get_main_task()
    result = main_task.result if main_task else None

    # Print the result
    print("Answer:", result)


if __name__ == "__main__":
    main()
