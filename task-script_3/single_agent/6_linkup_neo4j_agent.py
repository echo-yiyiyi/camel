"""
This script creates a knowledge graph agent that uses LinkUp tools to retrieve webpages related to LLM-based social simulation research,
extracts knowledge graph elements from the content, and stores them in a Neo4j graph database.
"""

from camel.agents.knowledge_graph_agent import KnowledgeGraphAgent
from camel.toolkits.search_toolkit import SearchToolkit
from camel.storages.graph_storages.neo4j_graph import Neo4jGraph


def main():
    # Define the query for LinkUp search
    query = "LLM-based social simulation research"

    # Initialize the search toolkit
    search_toolkit = SearchToolkit()

    # Perform LinkUp search to retrieve webpages
    print(f"Searching LinkUp for query: {query}")
    search_results = search_toolkit.search_linkup(query=query)

    # Initialize the knowledge graph agent
    kg_agent = KnowledgeGraphAgent()

    # Initialize Neo4j graph storage with credentials
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",  # Replace with your Neo4j URL
        username="neo4j",             # Replace with your Neo4j username
        password="password"           # Replace with your Neo4j password
    )

    # Process each search result
    for idx, result in enumerate(search_results):
        print(f"Processing result {idx + 1}/{len(search_results)}: {result['title']}")
        content = result.get('content', '')
        url = result.get('url', '')

        # Run the knowledge graph agent on the content
        kg_agent.run(content)

        # Extract graph elements from the agent's output
        graph_elements = kg_agent._parse_graph_elements(kg_agent.output)

        # Add graph elements to Neo4j
        neo4j_graph.add_graph_elements(graph_elements)

        # Optionally, add a node for the source URL
        if url:
            from camel.schema.graph_element import Node
            source_node = Node(id=url, name=url, type="WebPage")
            neo4j_graph.add_graph_elements([source_node])

    print("Finished processing and storing knowledge graph elements.")


if __name__ == "__main__":
    main()
