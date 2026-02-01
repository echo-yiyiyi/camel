import os
from dotenv import load_dotenv
from camel.agents import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from camel.toolkits.search_toolkit import SearchToolkit
from camel.loaders import UnstructuredIO

load_dotenv()

# Load Neo4j credentials from environment variables
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

neo4j_graph = None
try:
    neo4j_graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    print("Connected to Neo4j successfully.")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")

# Initialize SearchToolkit for LinkUp web search
search_toolkit = SearchToolkit()

# Initialize KnowledgeGraphAgent with Neo4j storage if connected
kg_agent = KnowledgeGraphAgent(storage=neo4j_graph) if neo4j_graph else KnowledgeGraphAgent()

# Initialize UnstructuredIO for creating elements from text
uio = UnstructuredIO()

# Define the search query
query = "LLM-based social simulation research"

# Use DuckDuckGo search to retrieve webpages related to the query
search_results = search_toolkit.search_duckduckgo(query=query, number_of_result_pages=2)

if search_results:
    print(f"Retrieved {len(search_results)} search results from DuckDuckGo.")
else:
    print(f"Search failed or returned no results: {search_results}")

# Process each search result
for result in search_results:
    title = result.get('title', 'No Title')
    snippet = result.get('body', '')
    url = result.get('url', '')
    print(f"Processing: {title} - {url}")

    # Create a text element combining title and snippet
    text_content = f"Title: {title}\nURL: {url}\nDescription: {snippet}"
    element = uio.create_element_from_text(text=text_content)

    # Run the knowledge graph agent to extract and store knowledge
    kg_agent.run(element)

# Query Neo4j to confirm data storage if connected
if neo4j_graph:
    print("Sample nodes in Neo4j:")
    nodes = neo4j_graph.query("MATCH (n) RETURN n LIMIT 5")
    for node in nodes:
        print(node)
else:
    print("Neo4j graph storage not connected, skipping query.")
