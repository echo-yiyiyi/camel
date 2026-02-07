import os
from pathlib import Path

from camel.agents import ChatAgent
from camel.memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.models.model_factory import ModelFactory
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.storages.vectordb_storages import QdrantStorage
from camel.types import ModelType
from camel.utils import OpenAITokenCounter

# Setup vector storage for vector DB memory
vector_storage = QdrantStorage(
    vector_dim=1536,
    path=":memory:",  # in-memory storage for example
)

# Setup context creator with token limit
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
    token_limit=1024,
)

# Create the model
model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
)

# Create LongtermAgentMemory combining chat history and vector DB
longterm_memory = LongtermAgentMemory(
    context_creator=context_creator,
    chat_history_block=None,  # use default
    vector_db_block=None,    # use default with vector_storage
    retrieve_limit=3,
    agent_id="agent_longterm_001",
)

# Override vector_db_block storage to use our vector_storage
longterm_memory.vector_db_block.storage = vector_storage

# Create the ChatAgent with longterm memory
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory.",
    agent_id="agent_longterm_001",
    model=model,
    memory=longterm_memory,
)

# Add memory management tools for human interaction
memory_toolkit = MemoryToolkit(agent)
agent.add_tools(memory_toolkit.get_tools())

# Example query to test the agent
user_input = "Remember that Python is a programming language."
response = agent.step(user_input)
print("Agent response:", response.msgs[0].content if response.msgs else "No response")

# Another query to test recall
user_input2 = "What do you remember about Python?"
response2 = agent.step(user_input2)
print("Agent response:", response2.msgs[0].content if response2.msgs else "No response")

# Clean up vector storage if needed
# (No file to remove since in-memory)

