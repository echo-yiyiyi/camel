"""
Knowledge Graph Agent with LinkUp tools to retrieve webpages related to LLM-based social simulation research and store them in Neo4j.
"""

from camel.agents.knowledge_graph_agent import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from camel.toolkits import SearchToolkit


# Minimal compatible Element class for KnowledgeGraphAgent
class Element:
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text


def main():
    # Initialize Neo4jGraph connection
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",  # Replace with your Neo4j URL
        username="neo4j",             # Replace with your Neo4j username
        password="password"           # Replace with your Neo4j password
    )

    # Initialize SearchToolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"

    # Use LinkUp search to retrieve webpages
    search_results = search_toolkit.search_linkup(
        query=query,
        depth="standard",
        output_type="searchResults"
    )

    # Initialize KnowledgeGraphAgent
    kg_agent = KnowledgeGraphAgent()

    graph_elements = []

    # Process each search result
    for result in search_results.get("results", []):
        url = result.get("url")
        content = result.get("content", "")
        title = result.get("name", "")

        # Prepare text for graph extraction
        text_to_process = f"Title: {title}\nURL: {url}\nContent: {content}"

        # Wrap text in Element
        element = Element(text_to_process)

        # Extract graph elements from content
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        graph_elements.append(graph_element)

    # Add extracted graph elements to Neo4j
    neo4j_graph.add_graph_elements(graph_elements, include_source=True, base_entity_label=True)

    print("Graph elements have been added to Neo4j.")


if __name__ == "__main__":
    main()
