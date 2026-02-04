from dotenv import load_dotenv

from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.storages.graph_storages.neo4j_graph import Neo4jGraph
from camel.toolkits.search_toolkit import SearchToolkit

load_dotenv()


def main():
    # Initialize the LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"
    search_results = search_toolkit.search_linkup(query=query, depth="standard", output_type="searchResults")

    # Initialize UnstructuredIO and KnowledgeGraphAgent
    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent()

    # Initialize Neo4jGraph connection
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j"
    )

    graph_elements = []

    # Process each search result
    for result in search_results.get("results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")

        # Combine title and snippet as text content
        text_content = f"Title: {title}\nSnippet: {snippet}\nURL: {url}"

        # Create element from text
        element = uio.create_element_from_text(text=text_content)

        # Extract graph elements from content
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        graph_elements.append(graph_element)

    # Add extracted graph elements to Neo4j
    neo4j_graph.add_graph_elements(graph_elements, include_source=True, base_entity_label=True)

    print(f"Added {len(graph_elements)} graph elements to Neo4j.")


if __name__ == "__main__":
    main()
