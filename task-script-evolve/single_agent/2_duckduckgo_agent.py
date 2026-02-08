import os

from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits import FunctionTool, SearchToolkit
from camel.types import ModelType


def main():
    # Define system message
    system_message = "You are a helpful assistant that can use DuckDuckGo search to answer questions."

    # Create Gemini model with config
    model = ModelFactory.create(
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create DuckDuckGo search tool
    duckduckgo_tool = FunctionTool(SearchToolkit().search_duckduckgo)

    # Create ChatAgent with Gemini model and DuckDuckGo tool
    agent = ChatAgent(system_message=system_message, model=model, tools=[duckduckgo_tool])

    # Example question
    question = "What is the latest news about artificial intelligence?"

    # Get response from agent
    response = agent.step(question)

    # Print the answer
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
