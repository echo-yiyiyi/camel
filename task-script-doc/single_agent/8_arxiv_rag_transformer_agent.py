from camel.agents.chat_agent import ChatAgent
from camel.toolkits.arxiv_toolkit import ArxivToolkit
from camel.retrievers.vector_retriever import VectorRetriever
from camel.embeddings.openai_embedding import OpenAIEmbedding
from camel.storages.vectordb_storages.qdrant import QdrantStorage
from camel.toolkits.function_tool import FunctionTool

def main():
    # Initialize Arxiv toolkit
    arxiv_toolkit = ArxivToolkit()

    # Search and download the paper "Attention Is All You Need"
    papers = arxiv_toolkit.search_papers(query="Attention Is All You Need", max_results=1)
    if not papers:
        print("Paper not found.")
        return
    paper = papers[0]
    print(f"Downloaded paper: {paper['title']}")

    # Initialize vector storage and retriever
    embedding_model = OpenAIEmbedding()
    vector_storage = QdrantStorage(vector_dim=embedding_model.get_output_dim())
    retriever = VectorRetriever(embedding_model=embedding_model, storage=vector_storage)

    # Define a wrapper function for retriever.process
    def vector_retrieval_tool_func(text: str):
        retriever.process(text)
        return "Processed text for vector retrieval."

    # Wrap the search_papers method and the wrapper function as tools
    arxiv_search_tool = FunctionTool(arxiv_toolkit.search_papers)
    vector_retrieval_tool = FunctionTool(vector_retrieval_tool_func)

    # Initialize chat agent with these tools
    agent = ChatAgent(
        system_message="You are an academic assistant with access to Arxiv papers and vector retrieval.",
        tools=[arxiv_search_tool, vector_retrieval_tool]
    )

    # Ask the agent to answer "What is a Transformer?"
    question = "What is a Transformer?"
    response = agent.step(question)
    answer = response.msg.content if response and response.msg else "No answer."
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()
