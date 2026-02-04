from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # Define system message
    system_message = "You are a helpful assistant. You can use DuckDuckGo search to answer questions."

    # Create Gemini model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create SearchToolkit and get duckduckgo search tool
    search_toolkit = SearchToolkit()
    tools = search_toolkit.get_tools()

    # Print tool info to find duckduckgo tool
    for tool in tools:
        print(f"Tool: {tool}")

    # Filter duckduckgo tool by function name
    duckduckgo_tools = [tool for tool in tools if tool.func.__name__ == "search_duckduckgo"]

    # Create ChatAgent with Gemini model and duckduckgo search tool
    agent = ChatAgent(system_message=system_message, model=model, tools=duckduckgo_tools)

    # Example question
    question = "What is the capital of France?"

    # Get response from agent
    response = agent.step(question)

    # Print the answer
    print("Question:", question)
    print("Answer:", response.msgs[0].content)


if __name__ == "__main__":
    main()
