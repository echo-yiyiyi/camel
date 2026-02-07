from dotenv import load_dotenv

from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.toolkits import SearchToolkit

load_dotenv()

# Initialize the LinkUp search toolkit
search_toolkit = SearchToolkit()

# Query for LLM-based social simulation research
query = "LLM-based social simulation research"

# Retrieve webpages related to the query using LinkUp
search_results = search_toolkit.search_linkup(
    query=query,
    depth="standard",
    output_type="searchResults",
)

# Initialize the KnowledgeGraphAgent
kg_agent = KnowledgeGraphAgent()

# Initialize UnstructuredIO for creating elements from text
uio = UnstructuredIO()

# Initialize Neo4jGraph with environment variables or placeholders
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",  # Replace with your Neo4j URL
    username="neo4j",             # Replace with your Neo4j username
    password="password"           # Replace with your Neo4j password
)

# Process each search result content
for result in search_results.get("results", []):
    content = result.get("content", "")
    if not content:
        continue

    # Create element from text content
    element = uio.create_element_from_text(text=content)

    # Use KnowledgeGraphAgent to extract graph elements
    graph_element = kg_agent.run(
        element=element,
        parse_graph_elements=True,
    )

    # Add extracted graph elements to Neo4j
    neo4j_graph.add_graph_elements([graph_element], include_source=True)

print("Finished processing and storing LLM-based social simulation research webpages into Neo4j.")
