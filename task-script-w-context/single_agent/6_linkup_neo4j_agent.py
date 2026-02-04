from dotenv import load_dotenv

from camel.agents import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from camel.toolkits import SearchToolkit
from camel.loaders import UnstructuredIO

load_dotenv()


def main():
    # Initialize LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"
    print(f"Searching LinkUp for query: {query}")
    search_results = search_toolkit.search_linkup(query=query, depth="standard", output_type="searchResults")

    if "error" in search_results:
        print(f"Error in LinkUp search: {search_results['error']}")
        return

    results = search_results.get("results", [])
    print(f"Found {len(results)} results")

    # Initialize KnowledgeGraphAgent
    kg_agent = KnowledgeGraphAgent()

    # Initialize Neo4jGraph connection
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="test",
        database="neo4j"
    )

    # Initialize UnstructuredIO for creating elements from text
    uio = UnstructuredIO()

    for idx, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("description", "")

        print(f"\nProcessing result {idx}: {title}")
        print(f"URL: {url}")

        # Create element from snippet text
        element = uio.create_element_from_text(text=snippet)

        # Extract graph elements from content
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        # Add extracted graph elements to Neo4j
        neo4j_graph.add_graph_elements([graph_element], include_source=True, base_entity_label=True)

        print(f"Added graph elements from result {idx} to Neo4j")

    print("All results processed and stored in Neo4j.")


if __name__ == "__main__":
    main()
