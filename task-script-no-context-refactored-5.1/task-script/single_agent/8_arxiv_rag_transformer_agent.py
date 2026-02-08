from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers import VectorRetriever
from camel.types import ModelType

# Step 1: Create an agent with Arxiv tools
sys_msg = "You are a helpful assistant"

# Get Arxiv tools
arxiv_tools = ArxivToolkit().get_tools()

# Create a default model
model = ModelFactory.create(model_type=ModelType.DEFAULT)

# Create ChatAgent with Arxiv tools
agent = ChatAgent(system_message=sys_msg, model=model, tools=arxiv_tools)
agent.reset()

# Step 2: Download the paper "Attention Is All You Need"
paper_title = "Attention Is All You Need"

# Use the agent to search and download the paper
search_response = agent.step(f"Search paper '{paper_title}' for me")

# Extract paper ids from the search result tool call
paper_ids = []
for call in search_response.info.get('tool_calls', []):
    if call.func_name == 'search_papers':
        for paper in call.result:
            if paper_title.lower() in paper['title'].lower():
                # Extract arxiv id from entry_id url
                arxiv_id = paper['entry_id'].split('/')[-1]
                paper_ids.append(arxiv_id)

# Download the papers by ids
if paper_ids:
    download_response = agent.step(
        f"Download paper '{paper_title}' for me to my local path './downloaded_papers'"
    )
else:
    print(f"No papers found for title '{paper_title}'")

# Step 3: Use VectorRetriever to process the downloaded paper and answer question
retriever = VectorRetriever()

# Process the downloaded paper directory
retriever.process(content_input_path="./downloaded_papers")

# Query the retriever
query = "What is a Transformer?"
retrieved_info = retriever.query(query=query, top_k=5)

# Create an agent to answer based on retrieved info
answer_agent = ChatAgent(system_message=sys_msg, model=model)
user_msg = str(retrieved_info)
answer_response = answer_agent.step(user_msg)

print("Question:", query)
print("Answer:", answer_response.msg.content)
