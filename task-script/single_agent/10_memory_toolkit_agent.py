from pathlib import Path

from camel.agents import ChatAgent
from camel.memories import ChatHistoryMemory, ScoreBasedContextCreator
from camel.models.model_factory import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

# Create a token counter instance
token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)

# Create a context creator with token limit
context_creator = ScoreBasedContextCreator(
    token_counter=token_counter,  # Use proper token counter
    token_limit=1024,
)

# Create a model instance
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)

# Create a ChatHistoryMemory instance
memory = ChatHistoryMemory(
    context_creator=context_creator,
    agent_id="memory_toolkit_agent",
)

# Create a ChatAgent with memory
agent = ChatAgent(
    system_message="You are a helpful assistant with memory capabilities.",
    agent_id="memory_toolkit_agent",
    model=model,
    memory=memory,
)

# Example interaction to store memory
user_input_1 = "Remember this fact: The Earth revolves around the Sun."
response_1 = agent.step(user_input_1)
print("Agent response 1:", response_1.msgs[0].content if response_1.msgs else "No response")

# Example interaction to recall memory
user_input_2 = "What do you remember about the Earth?"
response_2 = agent.step(user_input_2)
print("Agent response 2:", response_2.msgs[0].content if response_2.msgs else "No response")

# Save memory to file
save_path = Path("./memory_toolkit_agent_memory.json")
agent.save_memory(save_path)
print(f"Memory saved to {save_path}.")

# Load memory from file into a new agent
new_agent = ChatAgent(
    system_message="You are a helpful assistant with memory capabilities.",
    agent_id="memory_toolkit_agent",
    model=model,
)
new_agent.load_memory_from_path(save_path)

# Query the new agent to verify memory load
user_input_3 = "Recall what you know about the Earth."
response_3 = new_agent.step(user_input_3)
print("New agent response:", response_3.msgs[0].content if response_3.msgs else "No response")

# Clean up saved memory file if desired
import os
if save_path.exists():
    os.remove(save_path)
    print(f"Memory file {save_path} removed.")
