"""
Workforce with LinkUp search worker and data keeper worker with retrieval tools
for LLM-based social simulation research.
"""

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.societies.workforce.single_agent_worker import SingleAgentWorker
from camel.toolkits import SearchToolkit


def create_linkup_search_worker():
    search_toolkit = SearchToolkit()
    system_msg = BaseMessage.make_assistant_message(
        role_name="LinkUp Search Worker",
        content="You are a worker that uses LinkUp search to retrieve information for social simulation research.",
    )
    agent = ChatAgent(system_message=system_msg, tools=[search_toolkit.search_linkup])
    worker = SingleAgentWorker(description="LinkUp Search Worker", worker=agent)
    return worker


def create_data_keeper_worker():
    # Placeholder for data keeper worker with Neo4j or other retrieval tools
    system_msg = BaseMessage.make_assistant_message(
        role_name="Data Keeper Worker",
        content="You are a worker that stores and retrieves social simulation research data using Neo4j.",
    )
    agent = ChatAgent(system_message=system_msg)
    worker = SingleAgentWorker(description="Data Keeper Worker", worker=agent)
    return worker


def main():
    workforce = Workforce(description="LLM-based Social Simulation Research Workforce")

    linkup_worker = create_linkup_search_worker()
    data_keeper_worker = create_data_keeper_worker()

    workforce.add_single_agent_worker(description="LinkUp Search Worker", worker=linkup_worker.worker)
    workforce.add_single_agent_worker(description="Data Keeper Worker", worker=data_keeper_worker.worker)

    print("Starting workforce MCP server...")
    mcp_server = workforce.to_mcp(
        name="LinkUp-Neo4j-Workforce",
        description="Workforce with LinkUp search and Neo4j data keeper",
        port=8006,
    )

    try:
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
