from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.search_toolkit import SearchToolkit
from camel.types import ModelPlatformType, ModelType

# Create Gemini model
model = ModelFactory.create(
    model_platform=ModelPlatformType.GEMINI,  # Correct platform for Gemini
    model_type=ModelType.COMETAPI_GEMINI_2_5_PRO,  # Use a valid Gemini model type
)

# Create SearchToolkit instance
search_toolkit = SearchToolkit()

# Get DuckDuckGo search tool
duckduckgo_tool = search_toolkit.get_tools()
# Filter to get only the duckduckgo search tool
duckduckgo_tool = [tool for tool in duckduckgo_tool if tool.func.__name__ == "search_duckduckgo"]

# Create agent with system message and model, add duckduckgo tool
agent = ChatAgent(
    system_message="You are a helpful assistant with access to DuckDuckGo search.",
    model=model,
    tools=duckduckgo_tool,
)

# Example usage
if __name__ == "__main__":
    question = "What is the capital of France?"
    response = agent.step(question)
    print("Answer:", response.msgs[0].content)
