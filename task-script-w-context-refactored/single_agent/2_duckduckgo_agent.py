from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelType
from camel.configs import GeminiConfig

# Create Gemini model
model = ModelFactory.create(
    model_type=ModelType.GEMINI_3_PRO,
    model_config_dict=GeminiConfig(temperature=0.2).as_dict(),
)

# Create SearchToolkit and get DuckDuckGo search tools
search_toolkit = SearchToolkit()
tools = search_toolkit.get_tools()

# Create ChatAgent with system message, model, and tools
system_message = "You are a helpful assistant with access to DuckDuckGo search."
agent = ChatAgent(system_message=system_message, model=model, tools=tools)

# Example question to ask
question = "Who is the current president of the United States?"

# Get agent response
response = agent.step(question)

# Print the answer content
print(response.msgs[0].content)
