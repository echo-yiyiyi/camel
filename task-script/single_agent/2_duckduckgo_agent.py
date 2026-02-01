from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.search_toolkit import SearchToolkit


def main():
    # Create Gemini model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO
    )

    # Get DuckDuckGo search tool from SearchToolkit
    search_toolkit = SearchToolkit()
    tools = search_toolkit.get_tools()
    duckduckgo_tool = None
    for tool in tools:
        # FunctionTool has attribute func.__name__ for function name
        if tool.func.__name__ == "search_duckduckgo":
            duckduckgo_tool = tool
            break

    if duckduckgo_tool is None:
        raise RuntimeError("DuckDuckGo search tool not found")

    # Create agent with system message and tools
    system_message = "You are a helpful assistant with access to DuckDuckGo search."
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=[duckduckgo_tool]
    )

    # Example question
    question = "What is the capital of France?"

    # Agent answers the question
    response = agent.step(question)
    print("Question:", question)
    print("Answer:", response)


if __name__ == "__main__":
    main()
