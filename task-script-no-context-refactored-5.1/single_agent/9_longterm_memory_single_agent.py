import os
from pathlib import Path

from camel.agents import ChatAgent
from camel.memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.models.model_factory import ModelFactory
from camel.types import ModelType
from camel.toolkits.memory_toolkit import MemoryToolkit

# Create a model instance
model = ModelFactory.create(model_type=ModelType.GPT_4O_MINI)

# Create a context creator for memory
context_creator = ScoreBasedContextCreator(
    token_counter=model.token_counter,
    token_limit=1024,
)

# Create LongtermAgentMemory instance
longterm_memory = LongtermAgentMemory(
    context_creator=context_creator,
    retrieve_limit=5,
)

# Create ChatAgent with longterm memory
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory.",
    agent_id="longterm_agent_001",
    model=model,
    memory=longterm_memory,
)

# Add MemoryToolkit tools for human interaction
memory_toolkit = MemoryToolkit(agent=agent)
for tool in memory_toolkit.get_tools():
    agent.add_tool(tool)

# Example query to test the agent
user_input_1 = "Hello, please remember that Python is a programming language."
response_1 = agent.step(user_input_1)
print("Agent response 1:", response_1.msgs[0].content if response_1.msgs else "No response")

user_input_2 = "Can you recall what I told you about Python?"
response_2 = agent.step(user_input_2)
print("Agent response 2:", response_2.msgs[0].content if response_2.msgs else "No response")

# Save memory to file
memory_file = Path("./longterm_memory.json")
agent.save_memory(memory_file)
print(f"Memory saved to {memory_file}")

# Clear memory via toolkit tool call
clear_response = agent.step("Please clear the memory.")
print("Clear memory response:", clear_response.msgs[0].content if clear_response.msgs else "No response")

# Load memory back from file via toolkit tool call
load_response = agent.step(f"Please load the memory from '{memory_file}'.")
print("Load memory response:", load_response.msgs[0].content if load_response.msgs else "No response")

# Query again after loading memory
user_input_3 = "What do you remember about Python now?"
response_3 = agent.step(user_input_3)
print("Agent response 3:", response_3.msgs[0].content if response_3.msgs else "No response")

# Clean up saved memory file
if memory_file.exists():
    os.remove(memory_file)
