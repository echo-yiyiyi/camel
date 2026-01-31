import asyncio
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import ChatGPTConfig
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import BrowserToolkit, FunctionTool, SearchToolkit
from camel.types import ModelPlatformType, ModelType

async def main():
    # Create coordinator agent
    coordinator_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )
    coordinator_agent = ChatAgent(
        system_message="You are a coordinator agent that manages task decomposition and assignment.",
        model=coordinator_model,
    )

    # Create task agent
    task_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )
    task_agent = ChatAgent(
        system_message="You are a task agent that specifies and decomposes tasks.",
        model=task_model,
    )

    # Create browser agent with BrowserToolkit
    browser_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )
    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )
    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    browser_toolkit = BrowserToolkit(
        headless=True,
        web_agent_model=web_agent_model,
        planning_agent_model=planning_agent_model,
        channel="chromium",
    )

    browser_agent = ChatAgent(
        system_message="You are a helpful assistant with web browsing capabilities.",
        model=browser_model,
        tools=browser_toolkit.get_tools(),
    )

    # Create search agent with SearchToolkit
    search_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    search_tool = SearchToolkit()
    search_agent = ChatAgent(
        system_message="You are a helpful assistant with web search capabilities.",
        model=search_model,
        tools=[FunctionTool(search_tool.search_brave)],
    )

    # Create workforce with coordinator and task agents
    workforce = Workforce(
        "Clinical Trial Enrollment Info Team",
        coordinator_agent=coordinator_agent,
        task_agent=task_agent,
    )

    # Add browser and search workers
    workforce.add_single_agent_worker(description="browser_worker", worker=browser_agent)
    workforce.add_single_agent_worker(description="search_worker", worker=search_agent)

    # Define the task
    question = (
        "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
        "from Jan-May 2018 as listed on the NIH website?"
    )

    # Add main task
    workforce.add_main_task(question)

    # Start workforce and wait for completion
    await workforce.start()

    # Print results from all workers
    for worker in workforce._children:
        print(f"Results from {worker.description}:")
        for task_id, result in worker.task_results.items():
            print(f"Task {task_id}: {result}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
