from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.memories.agent_memories import LongtermAgentMemory
from camel.memories import ScoreBasedContextCreator
from camel.toolkits import HumanToolkit
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

# Create the model
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Create the context creator for memory
context_creator = ScoreBasedContextCreator(
    token_counter=model.token_counter,
    token_limit=model.token_limit,
)

# Create the longterm memory with the context creator
longterm_memory = LongtermAgentMemory(context_creator=context_creator)
# Set a default current topic to avoid embedding errors
longterm_memory._current_topic = "Hello"

# Create the agent with default memory
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
    model=model,
    tools=[],
)

# Set the agent's memory to the longterm memory
agent.memory = longterm_memory

# Create the human interaction toolkit
human_toolkit = HumanToolkit()

# Add human interaction tools to the agent
for tool in human_toolkit.get_tools():
    agent.add_tool(tool)

# Example query to test the agent
query = "Remember that my favorite color is blue. Can you tell me what it is later?"
response = agent.step(query)
print("Agent response:", response.msgs[0].content)

# Follow-up query to test longterm memory
follow_up_query = "What is my favorite color?"
follow_up_response = agent.step(follow_up_query)
print("Follow-up response:", follow_up_response.msgs[0].content)
