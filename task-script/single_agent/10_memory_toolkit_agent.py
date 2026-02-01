from camel.agents import ChatAgent
from camel.memories.agent_memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter


def main():
    # Use a known available model for testing
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI
    )

    # Create a context creator with token counter and token limit
    context_creator = ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=1024,
    )

    # Create a chat history memory with context creator
    memory = ChatHistoryMemory(context_creator=context_creator)

    # Create a ChatAgent with memory
    agent = ChatAgent(
        model=model,
        system_message="You are a helpful assistant with memory.",
        memory=memory
    )

    # Create a MemoryToolkit for the agent
    memory_toolkit = MemoryToolkit(agent)

    # Add memory tools to the agent
    agent.add_tools(memory_toolkit.get_tools())

    # Example queries
    queries = [
        "Remember that my favorite color is blue.",
        "What is my favorite color?",
        "Clear the memory.",
        "What is my favorite color now?"
    ]

    for query in queries:
        print(f"User: {query}")
        response = agent.step(query)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
