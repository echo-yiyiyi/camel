# task-script/single_agent/2_duckduckgo_agent.py
# Create an agent using Gemini model with DuckDuckGo search tool to answer questions

from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelType


def main():
    # Define system message
    system_message = "You are a helpful assistant that can search the web using DuckDuckGo to answer questions."

    # Create Gemini model instance
    model = ModelFactory.create(
        model_type=ModelType.GEMINI_3_PRO,
        model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
    )

    # Create SearchToolkit instance
    search_toolkit = SearchToolkit()

    # Get DuckDuckGo search tool from toolkit
    duckduckgo_tools = [
        tool for tool in search_toolkit.get_tools()
        if tool.get_function_name() == "search_duckduckgo"
    ]

    # Create ChatAgent with Gemini model and DuckDuckGo search tool
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=duckduckgo_tools,
    )

    # Example question
    question = "What is the latest news about artificial intelligence?"

    # Create user message
    from camel.messages import BaseMessage
    user_message = BaseMessage.make_user_message(role_name="User", content=question)

    # Get agent response
    response = agent.step(user_message)

    # Print the answer
    print("Agent answer:")
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
