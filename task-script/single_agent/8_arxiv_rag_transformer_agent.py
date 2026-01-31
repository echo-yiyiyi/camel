from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers import VectorRetriever
from camel.types import ModelPlatformType, ModelType

# Define system message
sys_msg = "You are a helpful assistant"

# Initialize ArxivToolkit and get tools
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

# Step 1: Search the paper "Attention Is All You Need"
search_msg = "Search paper 'Attention Is All You Need'"
search_response = agent.step(search_msg)

# Extract paper ids from the search response info
paper_ids = []
try:
    for record in search_response.info['tool_calls']:
        tool_name = getattr(record, 'tool_name', None)
        if tool_name == 'search_papers':
            results = getattr(record, 'result', [])
            if isinstance(results, list):
                for paper in results:
                    entry_id = paper.get('entry_id', '')
                    paper_id = entry_id.split('/')[-1]
                    paper_ids.append(paper_id)
except Exception as e:
    print("Error accessing tool call records:", e)

if not paper_ids:
    raise ValueError("No paper ids found from search response")

# Step 2: Download the paper using the download_papers tool
# We use the first paper id for download
download_msg = f"Download paper 'Attention Is All You Need' with paper_ids {paper_ids[:1]}"
download_response = agent.step(download_msg)

# The download tool returns a success message
print("Download response:", download_response.msg.content)

# Step 3: Use VectorRetriever to process the paper content
# We assume the downloaded paper content is accessible via the toolkit or agent
# For simplicity, we use the search results summary as content for vector retrieval
# In real case, we should load the actual paper content from downloaded files

paper_content = "".join([paper.get('summary', '') for paper in search_response.info['tool_calls'][0].result])

vector_retriever = VectorRetriever()
vector_retriever.process(content=paper_content)

# Step 4: Query the vector retriever for the question "What is a Transformer?"
query = "What is a Transformer?"
retrieved_info = vector_retriever.query(query=query, top_k=5)

# Step 5: Create a new agent to answer the question based on retrieved context
answer_sys_msg = "You are a helpful assistant. Answer the question based on the retrieved context."
answer_agent = ChatAgent(system_message=answer_sys_msg, model=model)

# Prepare user message with retrieved context and question
user_msg = f"Context: {retrieved_info}\nQuestion: {query}"

# Get answer
answer_response = answer_agent.step(user_msg)

print("Answer:", answer_response.msg.content)
