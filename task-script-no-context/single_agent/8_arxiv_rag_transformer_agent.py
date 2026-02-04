from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.types import ModelPlatformType, ModelType
from camel.retrievers.vector_retriever import VectorRetriever
import os

# Define system message
sys_msg = "You are a helpful assistant"

# Create ArxivToolkit tools
tools = ArxivToolkit().get_tools()

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Create ChatAgent with ArxivToolkit tools
agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=tools,
)
agent.reset()

# Step 1: Search and download the paper "Attention Is All You Need"
search_query = "Attention Is All You Need"
response = agent.step(f"Search paper '{search_query}' for me")

# Extract paper ids from tool call results
paper_ids = []
for call in response.info.get('tool_calls', []):
    if call.tool_name == 'search_papers':
        for paper in call.result:
            # We look for exact title match ignoring case
            if paper['title'].lower() == search_query.lower():
                # Extract arXiv id from entry_id url
                entry_id = paper['entry_id']
                # entry_id example: 'http://arxiv.org/abs/1706.03762v5'
                arxiv_id = entry_id.split('/')[-1]
                paper_ids.append(arxiv_id)

# If no exact match found, fallback to first paper id
if not paper_ids and response.info.get('tool_calls'):
    first_call = response.info['tool_calls'][0]
    if first_call.tool_name == 'search_papers' and first_call.result:
        paper_ids.append(first_call.result[0]['entry_id'].split('/')[-1])

# Define download directory
download_dir = "./downloaded_papers"
os.makedirs(download_dir, exist_ok=True)

# Download the paper
if paper_ids:
    download_response = agent.step(
        f"Download paper '{search_query}' for me to my local path '{download_dir}'"
    )
    print(f"Download response: {download_response.info.get('tool_calls')}")
else:
    print("No paper ids found to download.")

# Step 2: Use VectorRetriever to process the downloaded paper and query
retriever = VectorRetriever()

# We assume the paper is downloaded as a PDF file named by arxiv id in download_dir
# Find the first PDF file in the download_dir
import glob
pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))

if not pdf_files:
    print("No PDF files found in the download directory to process.")
else:
    paper_path = pdf_files[0]
    print(f"Processing paper file: {paper_path}")
    retriever.process(paper_path)

    # Query the retriever
    query = "What is a Transformer?"
    results = retriever.query(query)

    print(f"Query results for '{query}':")
    for res in results:
        similarity = res.get('similarity score', 'N/A')
        print(f"- Similarity: {similarity}")
        print(f"  Text: {res.get('text', '')}")
        print()

