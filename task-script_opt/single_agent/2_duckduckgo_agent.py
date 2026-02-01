from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import GeminiConfig
from camel.toolkits import SearchToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # System message for the agent
    sys_msg = "You are a helpful assistant. Use DuckDuckGo search to answer questions."

    # Create the Gemini flash model
    gemini_model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_FLASH,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create the SearchToolkit and get DuckDuckGo search tools
    search_toolkit = SearchToolkit()
    duckduckgo_tools = search_toolkit.get_tools()

    # Create the ChatAgent with Gemini model and DuckDuckGo tools
    agent = ChatAgent(system_message=sys_msg, model=gemini_model, tools=duckduckgo_tools)

    # Example question
    question = "What are the latest advancements in AI?"

    # Get the agent's response
    response = agent.step(question)

    # Print the response content
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
