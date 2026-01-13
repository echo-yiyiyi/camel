import os
"""
Knowledge Graph Agent with LinkUp Tools
This agent retrieves webpages related to LLM-based social simulation research using LinkUp tools and stores them in Neo4j.
"""

from camel.agents import ChatAgent
from camel.toolkits import FunctionTool, SearchToolkit
from camel.storages import Neo4jGraph

# Initialize Neo4j storage
neo4j_storage = Neo4jGraph(
    url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),  # Replace with your Neo4j URL
    username=os.getenv("NEO4J_USERNAME", "neo4j"),             # Replace with your Neo4j username
    password=os.getenv("NEO4J_PASSWORD", "password")           # Replace with your Neo4j password
)

# Define a function to retrieve webpages related to LLM-based social simulation research
search_toolkit = SearchToolkit()

def retrieve_webpages(query: str):
    # Use LinkUp or Brave search to get webpages
    results = search_toolkit.search_brave(q=query, search_lang="en")
    return results

# Define a function to store webpages in Neo4j
# Here we assume results is a list of dicts with keys: title, url, snippet

def store_in_neo4j(results):
    for item in results:
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        # Add nodes and relationships to Neo4j
        neo4j_storage.add_triplet(subj=title, obj=url, rel="HAS_URL")
        neo4j_storage.add_triplet(subj=title, obj=snippet, rel="HAS_SNIPPET")


# Create the knowledge graph agent
agent = ChatAgent(
    system_message="""
    You are a knowledge graph agent that retrieves webpages related to LLM-based social simulation research
    and stores the information in Neo4j graph database.
    """,
    tools=[FunctionTool(retrieve_webpages)],
)


if __name__ == "__main__":
    query = "LLM-based social simulation research"
    webpages = retrieve_webpages(query)
    store_in_neo4j(webpages)
    print("Webpages related to LLM-based social simulation research have been stored in Neo4j.")
