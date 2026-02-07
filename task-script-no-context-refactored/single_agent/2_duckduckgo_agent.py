import os

from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelType


def main():
    # Define system message
    sys_msg = "You are a helpful assistant."

    # Create Gemini model with GeminiConfig and GEMINI_3_PRO
    model = ModelFactory.create(
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create SearchToolkit instance
    search_toolkit = SearchToolkit()

    # Get duckduckgo search tool from toolkit
    duckduckgo_tools = [
        tool for tool in search_toolkit.get_tools()
        if tool.get_function_name() == "search_duckduckgo"
    ]

    # Create ChatAgent with system message, model, and duckduckgo tool
    agent = ChatAgent(system_message=sys_msg, model=model, tools=duckduckgo_tools)

    # Example question to ask
    question = "What is the latest news about AI?"

    # Use the agent to answer the question
    response = agent.step(question)

    # Print the agent's response content
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
