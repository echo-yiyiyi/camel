import os
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
from camel.types import ModelPlatformType, ModelType

# Define system message
sys_msg = "You are a helpful assistant."

# Create Arxiv toolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Create agent with Arxiv tools
agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=tools,
)
agent.reset()

# Step 1: Download the paper "Attention Is All You Need"
paper_title = "Attention Is All You Need"

# Define output directory for downloads
output_dir = "./downloaded_papers"
os.makedirs(output_dir, exist_ok=True)

# Use the agent to call the download_papers tool with output_dir
print(f"Downloading paper '{paper_title}' to {output_dir} ...")
download_response = agent.step(f'Download paper "{paper_title}" for me to my local path "{output_dir}"')
print(f"Download response: {download_response.msg.content}")

# Step 2: Search papers to get paper info including pdf_url and text
papers = arxiv_toolkit.search_papers(query=paper_title, max_results=1)
if not papers:
    raise ValueError(f"No papers found for query: {paper_title}")

paper = papers[0]

# Extract paper text
paper_text = paper.get('paper_text', '')
if not paper_text:
    raise ValueError("Paper text is empty, cannot proceed with vector retrieval.")

# Save paper text to a temp file for vector processing
temp_text_path = os.path.join(output_dir, "attention_is_all_you_need.txt")
with open(temp_text_path, "w", encoding="utf-8") as f:
    f.write(paper_text)

print(f"Saved paper text to {temp_text_path}")

# Step 3: Use VectorRetriever to process the paper text
vector_retriever = VectorRetriever()
vector_retriever.process(content=temp_text_path)
print("Processed paper text into vector retriever.")

# Step 4: Query the vector retriever
query = "What is a Transformer?"
retrieved_info = vector_retriever.query(query=query, top_k=5)

# Format retrieved info as context string
context = "\n\n".join([item['text'] for item in retrieved_info])

# Step 5: Use agent to answer the question based on retrieved context
user_message = f"Context:\n{context}\n\nQuestion: {query}\nAnswer based on the context above."
answer = agent.step(user_message)

print("Answer:")
print(answer.msg.content)

# Cleanup temp file
os.remove(temp_text_path)
print(f"Removed temporary file {temp_text_path}")
