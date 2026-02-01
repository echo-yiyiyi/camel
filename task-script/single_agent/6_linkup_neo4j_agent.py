from dotenv import load_dotenv

from camel.agents import KnowledgeGraphAgent
from camel.toolkits.search_toolkit import SearchToolkit
from camel.storages import Neo4jGraph
from unstructured.documents.elements import Text

load_dotenv()


def main():
    # Initialize the LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"

    # Use LinkUp search to retrieve webpages
    search_results = search_toolkit.search_linkup(query=query, depth="standard", output_type="searchResults")

    if "error" in search_results:
        print(f"Error in LinkUp search: {search_results['error']}")
        return

    results = search_results.get("results", [])

    # Initialize the knowledge graph agent
    kg_agent = KnowledgeGraphAgent()

    # Initialize Neo4j graph storage
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",  # Adjust if needed
        username="neo4j",  # Adjust if needed
        password="password"  # Adjust if needed
    )

    # Process each search result
    graph_elements = []
    for result in results:
        title = result.get("title", "")
        snippet = result.get("description", "")
        url = result.get("url", "")

        # Combine title and snippet as content for extraction
        content = f"Title: {title}\nSnippet: {snippet}\nURL: {url}"

        # Create a Text element from content
        element = Text(text=content)

        # Extract nodes and relationships from content
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        graph_elements.append(graph_element)

    # Add extracted graph elements to Neo4j
    neo4j_graph.add_graph_elements(graph_elements)

    print("Knowledge graph has been updated with LinkUp search results.")


if __name__ == "__main__":
    main()
