from camel.memories.records import MemoryRecord
from camel.types import ModelPlatformType
from camel.types import ModelType
from camel.models.model_factory import ModelFactory
"""
Agent with memory tools to manage memory and run query examples.
"""

from camel.agents.chat_agent import ChatAgent
from camel.memories import ChatHistoryMemory, ScoreBasedContextCreator
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.messages import BaseMessage
from camel.types import RoleType


def create_memory_toolkit_agent():
    # Setup context creator with token limit
    context_creator = ScoreBasedContextCreator(token_counter=None, token_limit=300)  # token_counter can be set as needed

    # Setup memory
    memory = ChatHistoryMemory(context_creator=context_creator)

    # Setup agent with memory
    agent = ChatAgent(model=ModelFactory.create(ModelPlatformType.OPENAI, ModelType.GPT_4), memory=memory)  # model_backend to be set as needed

    # Setup memory toolkit for the agent
    memory_toolkit = MemoryToolkit(agent=agent)

    # Example: Add a message to memory
    message = BaseMessage("Hello", RoleType.USER, None, "Hello, this is a test message.")
    memory.write_record(MemoryRecord(message=message, role_at_backend="assistant"))

    # Example: Save memory to file
    save_path = "memory.json"
    print(memory_toolkit.save(save_path))

    # Example: Load memory from JSON string
    memory_json = "[]"  # Empty memory example
    print(memory_toolkit.load(memory_json))

    # Example: Clear memory
    memory.clear()
    print("Memory cleared.")

    return agent, memory_toolkit


if __name__ == "__main__":
    agent, toolkit = create_memory_toolkit_agent()
    # Run example queries or interactions with the agent here
    print("Agent with memory toolkit created and examples run.")
