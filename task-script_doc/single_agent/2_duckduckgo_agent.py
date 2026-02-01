# task-script/single_agent/2_duckduckgo_agent.py
# Agent using Gemini model with DuckDuckGo search tool

from camel.agents import ChatAgent
from camel.models import GeminiModel
from camel.toolkits.search_toolkit import SearchToolkit


def main():
    # Initialize Gemini model
    gemini_model = GeminiModel(model_type="gemini-1")

    # Initialize search toolkit with DuckDuckGo search tool
    search_toolkit = SearchToolkit()
    duckduckgo_tool = search_toolkit.get_tools()[4]  # search_duckduckgo tool

    # Create agent with Gemini model and DuckDuckGo search tool
    agent = ChatAgent(
        system_message="You are a helpful assistant with access to DuckDuckGo search.",
        model=gemini_model,
        tools=[duckduckgo_tool],
        stream_accumulate=False,
    )

    agent.reset()

    # Example question
    question = "What is the latest news about AI?"

    # Agent answers the question using the Gemini model and DuckDuckGo search
    response = agent.step(question)

    print("Agent response:")
    print(response.msg.content)


if __name__ == "__main__":
    main()
