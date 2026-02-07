import time
from camel.agents import ChatAgent
from camel.toolkits.arxiv_toolkit import ArxivToolkit
from camel.memories import VectorDBMemory
from camel.memories.blocks.vectordb_block import VectorDBBlock
from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.memories.records import MemoryRecord
from camel.models import ModelFactory
from camel.embeddings import OpenAIEmbedding
from camel.storages.vectordb_storages import QdrantStorage
from camel.types.enums import ModelType
from camel.utils import OpenAITokenCounter


def wait_for_tool_call(agent, max_wait=10):
    # Wait for tool call info to appear in agent response
    for _ in range(max_wait):
        time.sleep(1)
        if agent._last_tool_call_record is not None:
            return True
    return False


def main():
    # Create the model
    model = ModelFactory.create(
        model_type=ModelType.DEFAULT
    )

    # Create embedding for vector DB
    embedding = OpenAIEmbedding()

    # Create vector storage
    vector_storage = QdrantStorage(
        vector_dim=embedding.get_output_dim(),
        path=":memory:"
    )

    # Create context creator
    context_creator = ScoreBasedContextCreator(
        token_counter=OpenAITokenCounter(ModelType.DEFAULT),
        token_limit=2048,
    )

    # Create ArxivToolkit
    arxiv_toolkit = ArxivToolkit()

    # Create vector DB memory
    vector_db_memory = VectorDBMemory(
        context_creator=context_creator,
        storage=vector_storage,
        retrieve_limit=3,
        agent_id="arxiv_rag_agent",
    )

    # Create the agent with Arxiv tools and vector memory
    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        memory=vector_db_memory,
        tools=arxiv_toolkit.get_tools(),
    )

    # Step 1: Search the paper "Attention Is All You Need"
    search_query = "Search paper 'Attention Is All You Need'"
    search_response = agent.step(search_query)

    if not wait_for_tool_call(agent):
        print("Timeout waiting for search tool call.")
        return

    # Extract paper id from tool call result
    tool_calls = search_response.info.get('tool_calls', [])
    if not tool_calls:
        print("No tool calls found for search.")
        return

    search_result = tool_calls[0].result
    if not search_result or len(search_result) == 0:
        print("No search results found.")
        return

    paper_id = search_result[0].get('id') or search_result[0].get('entry_id')
    if not paper_id:
        print("No paper id found in search result.")
        return

    # Step 2: Download the paper by id
    download_query = f"Download paper 'Attention Is All You Need' with id {paper_id}"
    download_response = agent.step(download_query)

    if not wait_for_tool_call(agent):
        print("Timeout waiting for download tool call.")
        return

    # Extract download result
    download_tool_calls = download_response.info.get('tool_calls', [])
    if not download_tool_calls:
        print("No tool calls found for download.")
        return

    download_result = download_tool_calls[0].result
    if not download_result:
        print("Download failed or no result.")
        return

    # Add the downloaded paper content to vector memory
    download_tool = [t for t in arxiv_toolkit.get_tools() if t.get_function_name() == "download_papers"]
    if not download_tool:
        print("Download tool not found.")
        return

    download_tool = download_tool[0]
    papers = download_tool.func(ids=[paper_id])

    for paper in papers:
        content = paper.get("content", "")
        if content:
            record = MemoryRecord(
                message=None,
                id=paper["id"],
                text=content,
                metadata={"title": paper["title"]}
            )
            vector_db_memory.write_records([record])

    # Now query the agent with vector retrieval
    question = "What is a Transformer?"

    # Retrieve relevant context from vector memory
    retrieved_docs = vector_db_memory.retrieve(question, limit=3)

    # Compose prompt with retrieved context
    context_texts = "\n\n".join([doc.memory_record.text for doc in retrieved_docs])
    prompt = f"Context:\n{context_texts}\n\nQuestion: {question}\nAnswer:" 

    # Use the model to answer the question with context
    answer = model.chat(prompt)

    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
