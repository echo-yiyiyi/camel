import sys
from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelType


def main():
    # Define system message
    system_message = "You are a helpful assistant."

    # Create Gemini model
    model = ModelFactory.create(
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create SearchToolkit and get DuckDuckGo search tool
    search_toolkit = SearchToolkit()
    # Filter to get only the duckduckgo search tool
    duckduckgo_tools = [tool for tool in search_toolkit.get_tools() if tool.func.__name__ == "search_duckduckgo"]

    # Create ChatAgent with Gemini model and DuckDuckGo search tool
    agent = ChatAgent(system_message=system_message, model=model, tools=duckduckgo_tools)

    # Example question
    question = "What is the capital of France?"
    if len(sys.argv) > 1:
        question = sys.argv[1]

    # Get response from agent
    response = agent.step(question)

    # Print the answer
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
