from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.search_toolkit import SearchToolkit
from camel.types import ModelPlatformType, ModelType
from camel.configs import GeminiConfig


def main():
    # Initialize the Gemini model using ModelFactory with required parameters
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_FLASH,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Initialize the SearchToolkit and get the duckduckgo search tool
    search_toolkit = SearchToolkit()
    duckduckgo_tool = search_toolkit.search_duckduckgo

    # Create the ChatAgent with the Gemini model and the duckduckgo search tool
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=[duckduckgo_tool],
    )

    # Example question to ask the agent
    question = "What is the capital of France?"

    # Get the agent's answer
    response = agent.step(question)

    print(f"Question: {question}")
    print(f"Answer: {response.msgs[0].content}")


if __name__ == "__main__":
    main()
