"""
Workforce with LinkUp Search Worker and Data Keeper Worker
This workforce retrieves and stores LLM-based social simulation research using retrieval tools.
"""

import os
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.toolkits import FunctionTool, SearchToolkit
from camel.storages.graph_storages.neo4j_graph import Neo4jGraph

# Initialize Neo4j storage
neo4j_storage = Neo4jGraph(
try:
    neo4j_storage.driver.verify_connectivity()
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    exit(1)
ntry:
    neo4j_storage.driver.verify_connectivity()
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    exit(1)
    url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password")
)

# Define retrieval function using LinkUp search
search_toolkit = SearchToolkit()

def retrieve_webpages(query: str):
    results = search_toolkit.search_brave(q=query, search_lang="en")
    return results

# Define function to store webpages in Neo4j
# Assuming results is a list of dicts with keys: title, url, snippet

def store_in_neo4j(results):
    for item in results:
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        neo4j_storage.add_triplet(subj=title, obj=url, rel="HAS_URL")
        neo4j_storage.add_triplet(subj=title, obj=snippet, rel="HAS_SNIPPET")


# Create LinkUp Search Worker
search_worker_msg = BaseMessage.make_assistant_message(
    role_name="LinkUp Search Worker",
    content="You are a worker that retrieves webpages related to LLM-based social simulation research using LinkUp search tools.",
)
search_worker = ChatAgent(system_message=search_worker_msg, tools=[FunctionTool(retrieve_webpages)])

# Create Data Keeper Worker
data_keeper_msg = BaseMessage.make_assistant_message(
    role_name="Data Keeper Worker",
    content="You are a worker that stores retrieved webpages in Neo4j graph database.",
)
# Data keeper worker will have a tool to store data
store_tool = FunctionTool(store_in_neo4j)

# For simplicity, data keeper worker is a ChatAgent with store tool
data_keeper_worker = ChatAgent(system_message=data_keeper_msg, tools=[store_tool])


# Create Workforce
workforce = Workforce(description="LLM-based Social Simulation Research Workforce")
workforce.add_single_agent_worker(description="LinkUp Search Worker", worker=search_worker)
workforce.add_single_agent_worker(description="Data Keeper Worker", worker=data_keeper_worker)


if __name__ == "__main__":
    query = "LLM-based social simulation research"
    webpages = retrieve_webpages(query)
    store_in_neo4j(webpages)
    print("Webpages related to LLM-based social simulation research have been stored in Neo4j.")

