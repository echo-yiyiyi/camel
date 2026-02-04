import os
from pathlib import Path

from camel.agents import ChatAgent
from camel.memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.models.model_factory import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

# Setup context creator for memory with token counter
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.DEFAULT),
    token_limit=1024,
)

# Create LongtermAgentMemory with the context creator
longterm_memory = LongtermAgentMemory(context_creator=context_creator, retrieve_limit=3, agent_id="agent_001")

# Setup human interaction tools using TerminalToolkit
workspace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
tools = TerminalToolkit(working_directory=workspace_dir).get_tools()

# Create model
model = ModelFactory.create(model_platform=ModelPlatformType.DEFAULT, model_type=ModelType.DEFAULT)

# Create ChatAgent with system message, model, memory and tools
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and terminal tools.",
    model=model,
    memory=longterm_memory,
    tools=tools,
    agent_id="agent_001"
)

# Example query to test the agent
user_input = "Hello, please remember that CAMEL is an AI framework. Also, create a directory named 'test_dir' in the workspace."
response = agent.step(user_input)
print("Agent response:", response.msgs[0].content if response.msgs else "No response")

# Save memory to file
memory_save_path = Path("./longterm_agent_memory.json")
agent.save_memory(memory_save_path)
print(f"Memory saved to {memory_save_path}")

# Load memory back to demonstrate persistence
new_agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and terminal tools.",
    model=model,
    tools=tools,
    agent_id="agent_001"
)
new_agent.load_memory_from_path(memory_save_path)

# Query new agent to verify memory loaded
query_after_load = "What did I ask you to remember earlier?"
response_after_load = new_agent.step(query_after_load)
print("New agent response after loading memory:", response_after_load.msgs[0].content if response_after_load.msgs else "No response")

# Clean up saved memory file
if memory_save_path.exists():
    memory_save_path.unlink()
