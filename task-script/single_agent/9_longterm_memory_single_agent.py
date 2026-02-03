from camel.agents import ChatAgent
from camel.memories.agent_memories import LongtermAgentMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.utils import OpenAITokenCounter
from camel.toolkits.human_toolkit import HumanToolkit
from camel.models.model_factory import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Create the context creator for memory
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.DEFAULT),
    token_limit=1024,
)

# Create the human toolkit instance
human_toolkit = HumanToolkit()

# Create the model instance
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Create the longterm memory instance
longterm_memory = LongtermAgentMemory(
    context_creator=context_creator,
    agent_id="longterm_agent",
)

# Create the agent with longterm memory and human interaction tools
agent = ChatAgent(
    system_message="You are a helpful assistant with longterm memory and human interaction capabilities.",
    model=model,
    memory=longterm_memory,
    tools=human_toolkit.get_tools(),
    agent_id="longterm_agent",
)

# Example query to test the agent
query = "Remember that the sky is blue. Then ask me what color the sky is."
response = agent.step(query)
print("Agent response:", response.msgs[0].content if response.msgs else "No response")
