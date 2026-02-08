from camel.agents import ChatAgent
from camel.memories.agent_memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.toolkits.browser_toolkit import BrowserToolkit
from camel.models.model_factory import ModelFactory
from camel.types import ModelType
from camel.utils import OpenAITokenCounter

# Create a token counter for the context creator
token_counter = OpenAITokenCounter(ModelType.DEFAULT)

# Create a context creator with token limit
context_creator = ScoreBasedContextCreator(
    token_counter=token_counter,
    token_limit=1024,
)

# Create longterm memory instance
longterm_memory = LongtermAgentMemory(
    context_creator=context_creator,
    retrieve_limit=5,
)

# Create browser toolkit instance for human interaction tool
browser_toolkit = BrowserToolkit()

# Create the agent with longterm memory and browser toolkit as tools
model = ModelFactory.create(model_type=ModelType.DEFAULT)
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and web browsing capabilities.",
    model=model,
    memory=longterm_memory,
    tools=browser_toolkit.get_tools(),
)

# Example query to test the agent
query = "Can you remember that CAMEL is an AI framework? Also, please browse the web and find the latest news about AI."
response = agent.step(query)
print("Agent response:", response.msgs[0].content if response.msgs else "No response")
