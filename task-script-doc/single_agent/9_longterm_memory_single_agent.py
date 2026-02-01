from camel.agents import ChatAgent
from camel.memories.blocks.chat_history_block import ChatHistoryBlock
from camel.memories import ChatHistoryMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.toolkits.human_toolkit import HumanToolkit
from camel.utils.token_counting import OpenAITokenCounter
from camel.types import ModelType


def main():
    # Initialize longterm memory with chat history block
    chat_history_block = ChatHistoryBlock()
    context_creator = ScoreBasedContextCreator(token_counter=OpenAITokenCounter(model=ModelType.GPT_4), token_limit=2048)
    memory = ChatHistoryMemory(context_creator=context_creator, storage=chat_history_block.storage)

    # Initialize human interaction toolkit
    human_toolkit = HumanToolkit()

    # Create the agent with longterm memory and human interaction tools
    agent = ChatAgent(
        system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
        memory=memory,
        tools=human_toolkit.get_tools(),
    )

    # Reset agent state
    agent.reset()

    # Example query to test the agent
    example_query = "Hello, can you remember our previous conversation and also ask me questions if you need clarification?"

    # Agent processes the example query
    response = agent.step(example_query)

    # Print the agent's response
    print("Agent response:", response.msg.content)


if __name__ == "__main__":
    main()
