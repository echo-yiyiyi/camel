from camel.agents import ChatAgent
from camel.memories.agent_memories import LongtermAgentMemory
from camel.toolkits.human_toolkit import HumanToolkit
from camel.models.model_factory import ModelFactory
from camel.types import ModelType, ModelPlatformType
from camel.memories.context_creators.score_based import ScoreBasedContextCreator


def main():
    # Create the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=ModelType.GROQ_LLAMA_3_1_8B
    )

    # Create a context creator for the memory
    context_creator = ScoreBasedContextCreator(
        token_counter=model.token_counter,
        token_limit=model.token_limit
    )

    # Create longterm memory with context creator
    memory = LongtermAgentMemory(context_creator=context_creator)

    # Create human interaction toolkit
    human_toolkit = HumanToolkit()

    # Create the agent with memory and human interaction tools
    agent = ChatAgent(
        system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
        model=model,
        memory=memory,
        tools=human_toolkit.get_tools()  # Add human interaction tools here
    )

    # Example query to test the agent
    query = "Hello! Can you remember this conversation for the future?"
    agent.reset()
    response = agent.step(query)
    print(f"Agent response: {response.msg.content}")


if __name__ == "__main__":
    main()
