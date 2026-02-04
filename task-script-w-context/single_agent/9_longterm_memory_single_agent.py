from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.memories.agent_memories import LongtermAgentMemory
from camel.memories import ScoreBasedContextCreator
from camel.toolkits.human_toolkit import HumanToolkit
from camel.types import ModelPlatformType, ModelType


def run_longterm_memory_agent_example():
    """
    Create a ChatAgent with LongtermAgentMemory and HumanToolkit tools.
    Test a simple query to demonstrate memory and human interaction.
    """

    # Create a model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create a context creator for memory
    context_creator = ScoreBasedContextCreator(
        token_counter=model.token_counter,
        token_limit=model.token_limit,
    )

    # Create longterm memory with context creator
    longterm_memory = LongtermAgentMemory(
        context_creator=context_creator
    )

    # Create the agent with system message
    system_message = (
        "You are an assistant with longterm memory and human interaction tools."
        " Use your memory to recall past conversations and ask the user for help when needed."
    )
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        memory=longterm_memory,
    )

    # Add human interaction tools
    human_toolkit = HumanToolkit()
    for tool in human_toolkit.get_tools():
        agent.add_tool(tool)

    # Test query
    user_query = "Hello, can you remember what we talked about before?"
    print(f"[User] {user_query}")
    response = agent.step(user_query)
    print(f"[Agent] {response.msgs[0].content}")


if __name__ == '__main__':
    run_longterm_memory_agent_example()
