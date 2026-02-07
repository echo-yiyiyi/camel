from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.memories import VectorDBMemory
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.storages.vectordb_storages import QdrantStorage
from camel.types import ModelType
from camel.utils import OpenAITokenCounter
from camel.memories.records import MemoryRecord


def main():
    # System message for the agent
    system_message = "You are a helpful assistant with access to Arxiv papers and vector retrieval."

    # Create Arxiv tools
    arxiv_tools = ArxivToolkit().get_tools()

    # Create model
    model = ModelFactory.create(
        model_type=ModelType.DEFAULT,
    )

    # Create vector storage (in-memory)
    vector_storage = QdrantStorage(
        vector_dim=1536,
        path=":memory:",
    )

    # Create context creator for token limiting
    context_creator = ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.DEFAULT),
        token_limit=1024,
    )

    # Create vector memory
    vector_memory = VectorDBMemory(
        context_creator=context_creator,
        storage=vector_storage,
        retrieve_limit=3,
        agent_id="arxiv_transformer_agent",
    )

    # Create agent with Arxiv tools but without memory initially
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=arxiv_tools,
        agent_id="arxiv_transformer_agent",
    )

    # Reset agent state
    agent.reset()

    # Step 1: Search the paper "Attention Is All You Need"
    search_query = "Attention Is All You Need"
    print(f"Searching for paper: {search_query}")
    search_response = agent.step(f"Search paper '{search_query}' for me")
    print("Search tool calls info:", search_response.info.get("tool_calls", []))

    # Extract paper_text from search results
    tool_calls = search_response.info.get("tool_calls", [])
    paper_text = ""
    if tool_calls:
        # The result of search_papers is in tool_calls[0].result
        results = tool_calls[0].result
        if results:
            # Use the first paper's text
            paper_text = results[0].get('paper_text', "")

    if not paper_text:
        print("No paper text found from search results.")
        return

    # Step 2: Add the paper text to vector memory as a MemoryRecord
    record = MemoryRecord(
        message=agent.model.message_class(content=paper_text),
        agent_id=agent.agent_id,
        role_at_backend="user",
    )
    vector_memory.write_records([record])

    # Assign the memory to the agent
    agent.memory = vector_memory

    # Step 3: Query the agent with vector retrieval
    question = "What is a Transformer?"
    print(f"Querying agent with: {question}")
    response = agent.step(question)
    print("Agent answer:", response.msgs[0].content if response.msgs else "No response")


if __name__ == "__main__":
    main()
