import os
from pathlib import Path

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.memories import VectorDBMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.retrievers.vector_retriever import VectorRetriever
from camel.storages.vectordb_storages import QdrantStorage
from camel.types import ModelType
from camel.utils import OpenAITokenCounter

# NOTE: To process PDF files, please ensure you have installed the required dependencies:
# pip install "unstructured[pdf]"

# Define system message
sys_msg = "You are a helpful assistant."

# Create ArxivToolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(model_type=ModelType.DEFAULT)

# Create ChatAgent with Arxiv tools
agent = ChatAgent(system_message=sys_msg, model=model, tools=tools)
agent.reset()

# Define local directory to download papers
download_dir = Path("./downloaded_papers")
download_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Download the paper "Attention Is All You Need"
paper_title = "Attention Is All You Need"
download_response = agent.step(f'Download paper "{paper_title}" for me to my local path "{download_dir}"')
print("Download response:", download_response.msgs[0].content if download_response.msgs else "No response")

# Step 2: Setup VectorRetriever and embed the downloaded paper PDF
# We assume the PDF filename is exactly the paper title + .pdf
pdf_path = download_dir / (paper_title + ".pdf")

vector_storage = QdrantStorage(vector_dim=1536, path=":memory:")
vector_retriever = VectorRetriever(storage=vector_storage)

try:
    # Process the PDF to embed and store vectors
    vector_retriever.process(str(pdf_path))
except Exception as e:
    print(f"Error during vector processing: {e}")

# Step 3: Create a new ChatAgent with VectorDBMemory for retrieval
context_creator = ScoreBasedContextCreator(
    token_counter=OpenAITokenCounter(ModelType.DEFAULT),
    token_limit=1024,
)

retrieval_memory = VectorDBMemory(
    context_creator=context_creator,
    storage=vector_storage,
    retrieve_limit=3,
    agent_id="transformer_agent",
)

retrieval_agent = ChatAgent(
    system_message="You are an assistant with vector retrieval memory.",
    model=model,
    agent_id="transformer_agent",
)
retrieval_agent.memory = retrieval_memory

# Step 4: Ask the agent "What is a Transformer?"
query = "What is a Transformer?"
response = retrieval_agent.step(query)
print("Answer:", response.msgs[0].content if response.msgs else "No response")
