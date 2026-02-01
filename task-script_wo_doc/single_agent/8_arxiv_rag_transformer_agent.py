# Agent script to download paper from arXiv and answer question using vector retrieval
from camel.agents import ChatAgent
from camel.toolkits.arxiv_toolkit import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
from camel.embeddings.openai_compatible_embedding import OpenAICompatibleEmbedding
from camel.storages.vectordb_storages.faiss import FaissStorage


def arxiv_rag_transformer_agent():
    # Initialize Arxiv toolkit
    arxiv_toolkit = ArxivToolkit()

    # Download paper "Attention Is All You Need"
    paper_title = "Attention Is All You Need"
    search_results = list(arxiv_toolkit._get_search_results(query=paper_title, max_results=1))
    if not search_results:
        return "Paper not found."
    paper = search_results[0]
    paper_content = paper.summary  # Use summary as content

    # Initialize vector storage and embedding
    embedding_model = OpenAICompatibleEmbedding(model_type='text-embedding-3-large')
    storage = FaissStorage()

    # Initialize vector retriever
    vector_retriever = VectorRetriever(embedding_model=embedding_model, storage=storage)

    # Add paper content to vector storage
    vector_retriever.storage.add_texts([paper_content], metadatas=[{"title": paper.title}])

    # Query vector retriever
    query = "What is a Transformer?"
    retrieved_docs = vector_retriever.query(query=query, top_k=5)

    # Prepare context for agent
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    assistant_sys_msg = "You are a helpful assistant. Answer the question based on the context provided."
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    # Initialize chat agent
    agent = ChatAgent(assistant_sys_msg)

    # Get response
    response = agent.step(user_msg)
    return response.msg.content


if __name__ == "__main__":
    answer = arxiv_rag_transformer_agent()
    print("Answer:", answer)
