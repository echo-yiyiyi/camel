from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs.gemini_config import GeminiConfig
from camel.types.enums import ModelType
from camel.toolkits.search_toolkit import SearchToolkit


def main():
    # Create Gemini model config
    gemini_config = GeminiConfig()

    # Create Gemini model without passing tools in constructor
    model = ModelFactory.create(
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=gemini_config.__dict__,
    )

    # Get DuckDuckGo search tool from SearchToolkit
    search_toolkit = SearchToolkit()
    duckduckgo_tool = None
    for tool in search_toolkit.get_tools():
        if tool.func.__name__ == "search_duckduckgo":
            duckduckgo_tool = tool
            break

    if duckduckgo_tool is None:
        raise RuntimeError("DuckDuckGo search tool not found")

    # System message for the agent
    system_message = "You are a helpful assistant with access to DuckDuckGo search."

    # Create ChatAgent with Gemini model and DuckDuckGo search tool
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=[duckduckgo_tool]
    )

    # Example question to ask
    question = "What is the capital of France?"

    # Get agent response with question string
    response = agent.step(question)

    print(f"Question: {question}")
    print(f"Answer: {response}")


if __name__ == "__main__":
    main()
