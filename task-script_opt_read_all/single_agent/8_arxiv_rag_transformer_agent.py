from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.memories import VectorDBMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.storages.vectordb_storages import QdrantStorage
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter

# 1) Create ArxivToolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# 2) Create a vector storage for memory
vector_storage = QdrantStorage(
    vector_dim=1536,  # typical embedding dimension
    path=":memory:",  # in-memory storage
)

# 3) Create a ScoreBasedContextCreator for token limiting
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.DEFAULT),
    token_limit=1024,
)

# 4) Create the model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.QWEN_2_5_14B,
)

# 5) Create the ChatAgent with Arxiv tools and vector memory
agent = ChatAgent(
    system_message="You are a helpful assistant with access to Arxiv papers.",
    model=model,
    tools=tools,
    agent_id="arxiv_transformer_agent",
)

# 6) Create VectorDBMemory and assign to agent
vector_memory = VectorDBMemory(
    context_creator=context_creator,
    storage=vector_storage,
    retrieve_limit=3,
    agent_id=agent.agent_id,
)
agent.memory = vector_memory

# 7) Reset agent to initialize
agent.reset()

# 8) Search and download the paper "Attention Is All You Need"
search_response = agent.step("Search paper 'Attention Is All You Need'")

# Extract paper IDs from the search result tool call
paper_ids = []
for call in search_response.info.get('tool_calls', []):
    if call.func_name == 'search_papers':
        for paper in call.result:
            # Extract arxiv id from entry_id url
            entry_id = paper.get('entry_id', '')
            if entry_id.startswith('http://arxiv.org/abs/'):
                arxiv_id = entry_id.split('/')[-1]
                paper_ids.append(arxiv_id)

# Download the papers by IDs
if paper_ids:
    download_msg = f"Download paper 'Attention Is All You Need' for me with paper_ids {paper_ids}"
    download_response = agent.step(download_msg)
else:
    print("No paper IDs found to download.")

# 9) Ask the agent a question using vector retrieval
question = "What is a Transformer?"
answer_response = agent.step(question)

print("Question:", question)
print("Answer:", answer_response.msgs[0].content if answer_response.msgs else "No answer")
