from camel.agents import ChatAgent
from camel.memories import LongtermAgentMemory, ChatHistoryBlock, VectorDBBlock, MemoryRecord, ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import HumanToolkit
from camel.types import ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter

# Create the model
model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
)

# Create the longterm memory
memory = LongtermAgentMemory(
    context_creator=ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.GPT_4O_MINI),
        token_limit=1024,
    ),
    chat_history_block=ChatHistoryBlock(),
    vector_db_block=VectorDBBlock(),
)

# Create human interaction tools
human_toolkit = HumanToolkit()
tools = human_toolkit.get_tools()

# Create the agent with longterm memory and human interaction tools
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
    model=model,
    memory=memory,
    tools=tools,
)

# Example query to test the agent
query = "Hello, can you remember that CAMEL is an AI framework? Please confirm."

response = agent.step(query)

print("Agent response:")
print(response.msgs[0].content if response.msgs else "No response")
