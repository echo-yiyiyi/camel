from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.search_toolkit import SearchToolkit


def main():
    # Create Gemini model instance
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_3_PRO
    )

    # Create DuckDuckGo search tool
    search_toolkit = SearchToolkit()
    duckduckgo_tool = search_toolkit.search_duckduckgo

    # Create ChatAgent with Gemini model and DuckDuckGo tool
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=[duckduckgo_tool]
    )

    # Example usage: ask a question
    question = "What is the capital of France?"
    # Use agent.step() to get response
    response = agent.step(question)
    print(f"Q: {question}\nA: {response}")


if __name__ == "__main__":
    main()
