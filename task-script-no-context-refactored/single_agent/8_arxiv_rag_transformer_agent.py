from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
from camel.types import ModelType
import os

# Define system message
sys_msg = "You are a helpful assistant"

# Create ArxivToolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(
    model_type=ModelType.DEFAULT,
)

# Create ChatAgent with Arxiv tools
agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=tools,
)
agent.reset()

# Step 1: Download the paper "Attention Is All You Need"
paper_query = "Attention Is All You Need"

# Use the agent to download the paper to a local directory
output_dir = "./downloaded_papers"
os.makedirs(output_dir, exist_ok=True)
download_msg = f'Download paper "{paper_query}" for me to my local path "{output_dir}"'
response = agent.step(download_msg)
print("Download response:", response.msg.content)

# Step 2: Use VectorRetriever to process the downloaded paper PDF
# We assume the paper PDF filename is "Attention Is All You Need.pdf" in output_dir
pdf_path = os.path.join(output_dir, "Attention Is All You Need.pdf")

# Check if file exists, if not, try to find the PDF file in the directory
if not os.path.exists(pdf_path):
    # Try to find a PDF file in the directory
    pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
    if pdf_files:
        pdf_path = os.path.join(output_dir, pdf_files[0])
    else:
        raise FileNotFoundError("No PDF file found in the download directory.")

# Create VectorRetriever
vector_retriever = VectorRetriever()

# Process the PDF file to build vector index
print(f"Processing PDF file for vector retrieval: {pdf_path}")
vector_retriever.process(pdf_path)

# Step 3: Query the vector retriever
query = "What is a Transformer?"
retrieved_results = vector_retriever.query(query=query, top_k=5)

# Format retrieved context for the agent
retrieved_texts = "\n\n".join([res['text'] for res in retrieved_results])

# Step 4: Use the agent to answer the question based on retrieved context
answer_prompt = f"Based on the following retrieved context, answer the question: {query}\n\nContext:\n{retrieved_texts}"
answer_response = agent.step(answer_prompt)

print("Answer:", answer_response.msg.content)
