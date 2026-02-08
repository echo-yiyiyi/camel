from dotenv import load_dotenv
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.toolkits import SearchToolkit

load_dotenv()


def main():
    # Initialize the LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Initialize the Neo4j graph storage
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
    )

    # Initialize the knowledge graph agent
    kg_agent = KnowledgeGraphAgent()

    # Initialize UnstructuredIO for creating elements
    uio = UnstructuredIO()

    # Define the query for LLM-based social simulation research
    query = "LLM-based social simulation research"

    # Perform LinkUp search
    print(f"Searching LinkUp for query: {query}")
    search_results = search_toolkit.search_linkup(query=query, output_type="searchResults")

    if "error" in search_results:
        print(f"Error in LinkUp search: {search_results['error']}")
        return

    results = search_results.get("results", [])
    print(f"Found {len(results)} results")

    # Process each search result
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")

        print(f"Processing result {idx}: {title}")

        # Create a text element from the snippet or title
        text_content = snippet if snippet else title

        # Create an Element from text content
        element = uio.create_element_from_text(text=text_content)

        # Use the knowledge graph agent to extract graph elements
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        # Add the extracted graph elements to Neo4j
        neo4j_graph.add_graph_elements([graph_element], include_source=False, base_entity_label=True)

    print("Finished processing all results and storing in Neo4j.")


if __name__ == "__main__":
    main()
