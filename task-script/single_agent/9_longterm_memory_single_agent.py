from camel.agents import ChatAgent
from camel.memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.models.model_factory import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.toolkits.human_toolkit import HumanToolkit
from camel.utils import OpenAITokenCounter


def main():
    # Create a context creator for memory
    context_creator = ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=1024,
    )

    # Create a model instance
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Create a memory instance for longterm memory
    memory = ChatHistoryMemory(context_creator=context_creator)

    # Create the agent with memory
    agent = ChatAgent(
        model=model,
        memory=memory,
        system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
        agent_id="longterm_memory_agent",
    )

    # Add toolkits for memory management and human interaction
    memory_toolkit = MemoryToolkit(agent)
    human_toolkit = HumanToolkit()

    # Add individual function tools from the toolkits
    for tool in memory_toolkit.get_tools():
        agent.add_tool(tool)
    for tool in human_toolkit.get_tools():
        agent.add_tool(tool)

    # Example query to test the agent
    query = "Remember my favorite color is blue. What is my favorite color?"
    response = agent.step(query)
    print(f"User: {query}")
    if response.msgs:
        print(f"Agent: {response.msgs[0].content}")
    else:
        print("Agent: No response")


if __name__ == "__main__":
    main()
