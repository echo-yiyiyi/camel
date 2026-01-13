# Agent script to search for CAMEL-AI information using browser tools and generate slides using PPT tools

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    import concurrent.futures
    import time
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("Starting agent script...")
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import signal
    def handler(signum, frame):
        raise TimeoutError("Execution timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60)  # 60 seconds timeout
    # Initialize model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    )

    # Initialize browser toolkit
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

    # Initialize PPTX toolkit
    pptx_toolkit = PPTXToolkit(
        working_directory="./pptx_outputs",
    )

    # Create agent with both toolkits
    agent = ChatAgent(
        system_message="You are a helpful assistant that can search the web and create PowerPoint presentations.",
        model=model,
        tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
    )

    # Step 1: Search for information about CAMEL-AI
    search_query = "Search for information about CAMEL-AI, its features, goals, and applications."
    print("Sending search query to agent...")
    start_time = time.time()
    search_response = agent.step(search_query)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(agent.step, search_query)
        search_response = future.result(timeout=10)
    print(f"Search query took {time.time() - start_time:.2f} seconds")
    print("Received search response from agent")
    print("Search Response:", search_response.msgs[0].content)

    # Step 2: Generate slides based on the search results
    slide_content = f"Create a PowerPoint presentation about CAMEL-AI based on the following content:\n{search_response.msgs[0].content}"
    print("Sending slide generation request to agent...")
    start_time = time.time()
    slide_response = agent.step(slide_content)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(agent.step, slide_content)
        slide_response = future.result(timeout=10)
    print(f"Slide generation took {time.time() - start_time:.2f} seconds")
    print("Received slide generation response from agent")
    print("Slide Generation Response:", slide_response.msgs[0].content)


if __name__ == "__main__":
    main()
