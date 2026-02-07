from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.memories.blocks.vectordb_block import VectorDBBlock
from camel.memories.agent_memories import VectorDBMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.types import ModelType
from camel.memories.records import MemoryRecord
import time

# Create Arxiv toolkit and get tools
arxiv_toolkit = ArxivToolkit()
tools = arxiv_toolkit.get_tools()

# Create model
model = ModelFactory.create(
    model_type=ModelType.DEFAULT,
)

# Create context creator with model's token counter
context_creator = ScoreBasedContextCreator(
    token_counter=model.token_counter,
    token_limit=4096,
)

# Create vector memory
vector_memory = VectorDBMemory(context_creator=context_creator)

# Create ChatAgent with Arxiv tools and vector memory
agent = ChatAgent(
    system_message="You are a helpful assistant.",
    model=model,
    tools=tools,
    memory=vector_memory,
)

agent.reset()

# Step 1: Download the paper "Attention Is All You Need"
download_response = agent.step('Download paper "Attention Is All You Need"')
print("Download response:", download_response.msg.content)

# Step 2: Search the paper to get paper text
search_results = arxiv_toolkit.search_papers(query="Attention Is All You Need", max_results=1)

# Extract paper text from search results
paper_text = ""
if search_results and 'paper_text' in search_results[0]:
    paper_text = search_results[0]['paper_text']

# Write paper text to vector memory
if paper_text:
    record = MemoryRecord(
        message=agent.memory._vectordb_block.embedding.get_output_dim() and agent.memory._vectordb_block.embedding and agent.memory._vectordb_block.embedding.__class__ and agent.memory._vectordb_block.embedding.__class__.__name__ and agent.memory._vectordb_block.embedding.embed and agent.memory._vectordb_block.embedding.embed.__class__ and agent.memory._vectordb_block.embedding.embed.__class__.__name__ and agent.memory._vectordb_block.embedding.embed.__code__ and agent.memory._vectordb_block.embedding.embed.__code__.co_varnames and agent.memory._vectordb_block.embedding.embed.__code__.co_varnames[0] and agent.memory._vectordb_block.embedding.embed.__code__.co_varnames[0] == 'text' and agent.memory._memory.message_class.make_assistant_message(
            content=paper_text
        ),
        role_at_backend=None,
        timestamp=time.time(),
        agent_id=agent.agent_id,
    )
    agent.memory.write_records([record])
    print("Paper text written to vector memory.")
else:
    print("No paper text found to write to vector memory.")

# Step 3: Ask the agent a question using vector retrieval
query = "What is a Transformer?"
retrieved_contexts = agent.memory.retrieve()

# Compose context string from retrieved contexts
context_str = "\n\n".join([ctx.memory_record.message.content for ctx in retrieved_contexts])

# Ask the agent with context
question_prompt = f"Based on the following context, answer the question:\n{context_str}\n\nQuestion: {query}"
response = agent.step(question_prompt)
print("Answer:", response.msg.content)
