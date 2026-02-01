from camel.agents import ChatAgent
from camel.memories.agent_memories import LongtermAgentMemory
from camel.toolkits.human_toolkit import HumanToolkit
from camel.models import ModelFactory
from camel.types.enums import ModelType
from camel.memories.context_creators.score_based import ScoreBasedContextCreator


def main():
    # Create the model
    model = ModelFactory.create(model_platform="tongyi-qianwen", model_type=ModelType.QWEN_2_5_14B)

    # Create a ScoreBasedContextCreator with the model's token_counter
    context_creator = ScoreBasedContextCreator(
        token_counter=model.token_counter,
        token_limit=2048,
    )

    # Create longterm memory with context creator
    memory = LongtermAgentMemory(context_creator=context_creator, agent_id="longterm_agent")

    # Create human interaction tools
    human_toolkit = HumanToolkit()

    # Create the agent with longterm memory and human interaction tools
    agent = ChatAgent(
        system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
        model=model,
        memory=memory,
        tools=human_toolkit.get_tools(),
    )

    # Example query to test the agent
    query = "Hello! Can you remember this conversation for the long term?"
    response = agent.step(query)
    print(f"User: {query}")
    print(f"Agent: {response.msgs[0].content}")


if __name__ == "__main__":
    main()
