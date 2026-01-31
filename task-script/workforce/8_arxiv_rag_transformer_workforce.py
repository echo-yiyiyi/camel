from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import ArxivToolkit
from camel.retrievers import VectorRetriever
from camel.societies.workforce import Workforce
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage


def main():
    # Create a workforce
    workforce = Workforce(description="Arxiv and Vector Memory Workforce")

    # Create Arxiv worker
    arxiv_sys_msg = BaseMessage.make_assistant_message(
        role_name="ArxivWorker",
        content="You are an Arxiv worker who searches and downloads papers."
    )
    arxiv_toolkit = ArxivToolkit()
    arxiv_tools = arxiv_toolkit.get_tools()
    arxiv_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )
    arxiv_worker = ChatAgent(
        system_message=arxiv_sys_msg,
        model=arxiv_model,
        tools=arxiv_tools,
    )
    workforce.add_single_agent_worker(description="Arxiv Worker", worker=arxiv_worker)

    # Create Vector Memory worker
    vector_sys_msg = BaseMessage.make_assistant_message(
        role_name="VectorMemoryWorker",
        content="You are a vector memory worker who processes and retrieves information from vector memory."
    )
    vector_worker = ChatAgent(
        system_message=vector_sys_msg,
        model=arxiv_model,  # reuse the same model
    )
    workforce.add_single_agent_worker(description="Vector Memory Worker", worker=vector_worker)

    # Step 1: Use Arxiv worker to search and download the paper
    search_msg = "Search paper 'Attention Is All You Need'"
    search_response = arxiv_worker.step(search_msg)

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

    download_msg = f"Download paper 'Attention Is All You Need' with paper_ids {paper_ids[:1]}"
    download_response = arxiv_worker.step(download_msg)
    print("Download response:", download_response.msg.content)

    # Step 2: Process the paper content with VectorRetriever
    paper_content = "".join([paper.get('summary', '') for paper in search_response.info['tool_calls'][0].result])

    vector_retriever = VectorRetriever()
    vector_retriever.process(content=paper_content)

    # Step 3: Query the vector retriever for the question
    query = "What is a Transformer?"
    retrieved_info = vector_retriever.query(query=query, top_k=5)

    # Step 4: Use Vector Memory worker to answer the question based on retrieved context
    answer_sys_msg = "You are a helpful assistant. Answer the question based on the retrieved context."
    answer_agent = ChatAgent(system_message=answer_sys_msg, model=arxiv_model)

    user_msg = f"Context: {retrieved_info}\nQuestion: {query}"
    answer_response = answer_agent.step(user_msg)

    print("Answer:", answer_response.msg.content)


if __name__ == '__main__':
    main()
