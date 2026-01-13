import os
from camel.agents import ChatAgent
from camel.configs import GeminiConfig
from camel.models import ModelFactory
from camel.toolkits.search_toolkit import SearchToolkit
from camel.types import ModelPlatformType, ModelType

# Define system message for the agent
sys_msg = "You are a helpful assistant that can search the web using DuckDuckGo."

# Create the Gemini model instance
model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,
    model_type=ModelType.GEMINI_3_PRO,
    model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
)

# Create the DuckDuckGo search toolkit instance
search_toolkit = SearchToolkit()

# Create a list of tools for the agent, including the DuckDuckGo search tool
tools = [search_toolkit.search_duckduckgo]

# Create the chat agent with the Gemini model and the search tool
agent = ChatAgent(system_message=sys_msg, model=model, tools=tools)

def ask_agent(question: str) -> str:
    """Ask the agent a question and get the response."""
    response = agent.step(question)
    return response.msgs[0].content

if __name__ == "__main__":
    # Example usage
    question = "What is the capital of France?"
    answer = ask_agent(question)
    print(f"Q: {question}\nA: {answer}")
