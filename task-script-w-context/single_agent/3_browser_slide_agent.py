from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types import ModelPlatformType, ModelType


def create_agent_with_browser_and_ppt_tools():
    # Create the model for the agent
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict={"temperature": 0.0},
    )

    # Initialize BrowserToolkit with the model
    browser_toolkit = BrowserToolkit(
        headless=True,
        web_agent_model=model,
        planning_agent_model=model,
    )

    # Initialize PPTXToolkit
    pptx_toolkit = PPTXToolkit()

    # Create the ChatAgent with system message and both toolkits
    system_message = "You are an assistant that can search the web using browser tools and generate PowerPoint slides using PPTX tools."

    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=browser_toolkit.get_tools() + pptx_toolkit.get_tools(),
    )

    return agent


if __name__ == "__main__":
    agent = create_agent_with_browser_and_ppt_tools()

    # Task: Search for information about CAMEL-AI and generate slides
    search_query = "Search for information about CAMEL-AI and generate a presentation with slides."

    response = agent.step(search_query)
    print("Agent response:")
    print(response.msgs[0].content)
