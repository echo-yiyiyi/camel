from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
from camel.types import ModelPlatformType, ModelType
import os

# System message for the agent
system_message = "You are a helpful assistant. Use the retrieved paper content to answer questions."

# Create ArxivToolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Create agent with Arxiv tools
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=tools,
)
agent.reset()

# Step 1: Search and download the paper "Attention Is All You Need"
search_query = "attention is all you need"
print(f"Searching for paper: {search_query}")
search_response = agent.step(f"Search paper '{search_query}' for me")
print("Search tool calls info:", search_response.info.get('tool_calls', []))

# Extract paper ids from search results
paper_ids = []
for call in search_response.info.get('tool_calls', []):
    if call.tool_name == 'search_papers':
        for paper in call.result:
            paper_id = paper.get('entry_id', '')
            if paper_id:
                # entry_id is like 'http://arxiv.org/abs/2407.15516v1'
                # extract arxiv id
                arxiv_id = paper_id.split('/')[-1]
                paper_ids.append(arxiv_id)

# Use the first paper id for download
if not paper_ids:
    raise ValueError("No paper ids found from search result.")

paper_id_to_download = paper_ids[0]

# Define output directory for downloaded papers
output_dir = os.path.join(os.path.dirname(__file__), "downloaded_papers")
os.makedirs(output_dir, exist_ok=True)

# Download the paper
download_msg = f"Download paper '{search_query}' for me to my local path '{output_dir}'"
download_response = agent.step(download_msg)
print("Download tool calls info:", download_response.info.get('tool_calls', []))

# The downloaded paper file path (PDF) - use this for vector retriever processing
pdf_file_path = os.path.join(output_dir, "Attention Is All You Need.pdf")

# Create VectorRetriever instance
vector_retriever = VectorRetriever()

# Process the downloaded paper content for vector retrieval
vector_retriever.process(content=pdf_file_path)

# Query the vector store
query = "What is a Transformer?"
retrieved_results = vector_retriever.query(query=query, top_k=5)

# Format retrieved context for agent input
retrieved_texts = "\n".join([res['text'] for res in retrieved_results])

# Create a new agent for answering with retrieved context
answer_agent = ChatAgent(system_message=system_message, model=model)

# Compose user message with retrieved context and question
user_message = f"Context:\n{retrieved_texts}\n\nQuestion: {query}\nAnswer based on the context above."

# Get answer from agent
answer_response = answer_agent.step(user_message)

print("Answer:")
print(answer_response.msg.content)
