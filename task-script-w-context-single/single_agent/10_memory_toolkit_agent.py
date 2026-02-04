from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.memory_toolkit import MemoryToolkit
from camel.types import ModelPlatformType, ModelType


def run_memory_toolkit_agent():
    # Create a model instance
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create a ChatAgent with a system message
    agent = ChatAgent(
        system_message="You are an assistant that manages memory using tools.",
        model=model,
    )

    # Create MemoryToolkit and add its tools to the agent
    memory_toolkit = MemoryToolkit(agent=agent)
    for tool in memory_toolkit.get_tools():
        agent.add_tool(tool)

    # Example conversation to populate memory
    print("\n--- Conversation Start ---")
    user_input_1 = "Remember that the sky is blue."
    print(f"[User] {user_input_1}")
    response_1 = agent.step(user_input_1)
    print(f"[Agent] {response_1.msgs[0].content}")

    user_input_2 = "Please save the memory to 'memory.json'."
    print(f"[User] {user_input_2}")
    response_2 = agent.step(user_input_2)
    print(f"[Agent] {response_2.msgs[0].content}")

    user_input_3 = "Clear the memory now."
    print(f"[User] {user_input_3}")
    response_3 = agent.step(user_input_3)
    print(f"[Agent] {response_3.msgs[0].content}")

    user_input_4 = "What do you remember about the sky?"
    print(f"[User] {user_input_4}")
    response_4 = agent.step(user_input_4)
    print(f"[Agent] {response_4.msgs[0].content}")

    user_input_5 = "Load the memory from 'memory.json'."
    print(f"[User] {user_input_5}")
    response_5 = agent.step(user_input_5)
    print(f"[Agent] {response_5.msgs[0].content}")

    user_input_6 = "What do you remember about the sky now?"
    print(f"[User] {user_input_6}")
    response_6 = agent.step(user_input_6)
    print(f"[Agent] {response_6.msgs[0].content}")


if __name__ == '__main__':
    run_memory_toolkit_agent()
