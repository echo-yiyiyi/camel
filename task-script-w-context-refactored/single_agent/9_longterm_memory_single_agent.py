from camel.agents import ChatAgent
from camel.memories import LongtermAgentMemory
from camel.memories.blocks import VectorDBBlock
from camel.models import ModelFactory
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.storages.vectordb_storages import QdrantStorage
from camel.types import ModelType
from camel.utils import OpenAITokenCounter


def run_longterm_memory_agent():
    # Create vector storage (in-memory for demo)
    vector_storage = QdrantStorage(
        vector_dim=1536,
        path=":memory:",
    )

    # Create a context creator for token limiting
    context_creator = ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=1024,
    )

    # Create the model
    model = ModelFactory.create(
        model_type=ModelType.GPT_4O_MINI,
    )

    # Create VectorDBBlock with storage
    vector_db_block = VectorDBBlock(storage=vector_storage)

    # Create the agent with longterm memory
    agent = ChatAgent(
        system_message="You are a helpful assistant with longterm memory and human interaction tools.",
        model=model,
        agent_id="longterm_agent_001",
    )

    # Create LongtermAgentMemory and assign to agent
    longterm_memory = LongtermAgentMemory(
        context_creator=context_creator,
        vector_db_block=vector_db_block,
        chat_history_block=None,
        retrieve_limit=3,
        agent_id=agent.agent_id,
    )
    # Set initial current topic to avoid empty embedding input error
    longterm_memory._current_topic = "general"

    agent.memory = longterm_memory

    # Add MemoryToolkit for human interaction tools
    memory_toolkit = MemoryToolkit(agent=agent)
    for tool in memory_toolkit.get_tools():
        agent.add_tool(tool)

    # Example query to test the agent
    print("--- Starting conversation with longterm memory agent ---")
    user_input_1 = "Tell me something interesting about space exploration."
    response_1 = agent.step(user_input_1)
    print(f"[User] {user_input_1}")
    print(f"[Agent] {response_1.msgs[0].content}")

    # Save memory via tool call
    save_command = "Please save the current memory to 'longterm_memory.json'."
    response_save = agent.step(save_command)
    print(f"[User] {save_command}")
    print(f"[Agent] {response_save.msgs[0].content}")

    # Clear memory via tool call
    clear_command = "Please clear the memory."
    response_clear = agent.step(clear_command)
    print(f"[User] {clear_command}")
    print(f"[Agent] {response_clear.msgs[0].content}")

    # Query after clearing memory
    user_input_2 = "What do you remember about space exploration?"
    response_2 = agent.step(user_input_2)
    print(f"[User] {user_input_2}")
    print(f"[Agent] {response_2.msgs[0].content}")

    # Load memory via tool call
    load_command = "Please load the memory from 'longterm_memory.json'."
    response_load = agent.step(load_command)
    print(f"[User] {load_command}")
    print(f"[Agent] {response_load.msgs[0].content}")

    # Query after loading memory
    user_input_3 = "What do you remember about space exploration?"
    response_3 = agent.step(user_input_3)
    print(f"[User] {user_input_3}")
    print(f"[Agent] {response_3.msgs[0].content}")


if __name__ == "__main__":
    run_longterm_memory_agent()
