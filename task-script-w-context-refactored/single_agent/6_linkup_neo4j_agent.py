import os
import asyncio
from camel.models import ModelFactory
from camel.types import ModelType
from camel.toolkits.search_toolkit import SearchToolkit
from camel.agents import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from unstructured.documents.elements import Text

async def main():
    # Create model
    model = ModelFactory.create(model_type=ModelType.GPT_4O_MINI)

    # Create SearchToolkit
    search_toolkit = SearchToolkit()

    # Search LinkUp for webpages related to LLM-based social simulation research
    query = "LLM-based social simulation research"
    search_results = search_toolkit.search_linkup(query=query, output_type="searchResults")

    # Create KnowledgeGraphAgent
    kg_agent = KnowledgeGraphAgent(model=model)

    # Create Neo4jGraph instance
    neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)

    # Process each search result
    for result in search_results.get("results", []):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("description", "")

        # Compose content for graph extraction
        content = f"Title: {title}\nURL: {url}\nSnippet: {snippet}"

        # Create unstructured Text element
        element = Text(text=content)

        # Extract graph elements
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        # Add graph elements to Neo4j
        neo4j_graph.add_graph_elements([graph_element], include_source=True, base_entity_label=True)

        print(f"Added graph elements from: {title}")

if __name__ == "__main__":
    asyncio.run(main())
