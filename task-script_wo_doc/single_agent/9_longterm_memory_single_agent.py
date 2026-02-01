from camel.types import ModelType
from camel.types import UnifiedModelType
from camel.utils.token_counting import OpenAITokenCounter
from camel.utils import BaseTokenCounter
from camel.agents.chat_agent import ChatAgent
from camel.memories.agent_memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.toolkits.human_toolkit import HumanToolkit
from camel.messages import BaseMessage
from camel.types import RoleType


def main():
    # Initialize context creator for memory
    context_creator = ScoreBasedContextCreator(token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI), token_limit=1000)

    # Initialize longterm memory with context creator
    longterm_memory = LongtermAgentMemory(context_creator=context_creator, retrieve_limit=5)

    # Initialize human interaction toolkit
    human_toolkit = HumanToolkit()

    # Create system message
    system_message = BaseMessage(meta_dict=None, 
        role_name="LongtermMemoryAgent",
        role_type=RoleType.ASSISTANT,
        content="You are an assistant with longterm memory and human interaction capabilities."
    )

    # Create the chat agent with longterm memory and human interaction tools
    agent = ChatAgent(
        system_message=system_message,
        memory=longterm_memory,
        tools=[human_toolkit.ask_human_via_console, human_toolkit.send_message_to_user],
    )

    agent.reset()

    # Example query to test the agent
    example_query = "Remember my favorite color and ask me about it later."

    # Step the agent with the example query
    response = agent.step(example_query)

    print("Example Query:", example_query)
    print("Agent Response:", response.msg.content)


if __name__ == '__main__':
    main()
