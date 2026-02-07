from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
import os

# Define system message
sys_msg = "You are a helpful assistant."

# Create ArxivToolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(model_type="default")

# Create ChatAgent with Arxiv tools
agent = ChatAgent(system_message=sys_msg, model=model, tools=tools)
agent.reset()

# Define paper title and download directory
paper_title = "Attention Is All You Need"
download_dir = "./downloaded_papers"
os.makedirs(download_dir, exist_ok=True)

# Use agent to download the paper
download_msg = f'Download paper "{paper_title}" for me to my local path "{download_dir}"'
download_response = agent.step(download_msg)
print(f"Download response: {download_response.msg.content}")

# Find the downloaded PDF file path
# The paper title may have spaces and special characters, so we list files in download_dir
pdf_files = [f for f in os.listdir(download_dir) if f.lower().endswith('.pdf')]
if not pdf_files:
    raise FileNotFoundError("No PDF files found in the download directory.")

# For simplicity, pick the first PDF file
pdf_path = os.path.join(download_dir, pdf_files[0])
print(f"Using PDF file for vector retrieval: {pdf_path}")

# Create VectorRetriever and process the PDF file
vector_retriever = VectorRetriever()
vector_retriever.process(content=pdf_path)

# Query the vector retriever
query = "What is a Transformer?"
retrieved_results = vector_retriever.query(query=query, top_k=5)

# Format retrieved context for agent
retrieved_texts = "\n\n".join([res['text'] for res in retrieved_results])

# Create a new ChatAgent for answering with the same model but no tools
answer_agent = ChatAgent(system_message=sys_msg, model=model)
answer_agent.reset()

# Compose user message with retrieved context and query
user_message = f"Based on the following retrieved context, answer the question:\n{retrieved_texts}\n\nQuestion: {query}"

# Get answer from agent
answer_response = answer_agent.step(user_message)
print("Answer:")
print(answer_response.msg.content)
