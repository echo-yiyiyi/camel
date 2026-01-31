import os
from dotenv import load_dotenv
import asyncio

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.toolkits.search_toolkit import SearchToolkit
from camel.toolkits.retrieval_toolkit import RetrievalToolkit
from camel.storages.graph_storages.neo4j_graph import Neo4jGraph
from camel.tasks.task import Task
from camel.types import ModelPlatformType, ModelType

load_dotenv()

# Load Neo4j credentials from environment variables
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print(f"Using Neo4j URL: {NEO4J_URL}")

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

# Initialize SearchToolkit for LinkUp search
search_toolkit = SearchToolkit()

# Initialize RetrievalToolkit with Neo4j graph storage if available
retrieval_toolkit = RetrievalToolkit(auto_retriever=None)  # We will configure retriever later if needed

# Create ChatAgent for LinkUp Search Worker
search_agent = ChatAgent(
    model=ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT
    ),
    toolkits_to_register_agent=[search_toolkit],
)

# Create ChatAgent for Data Keeper Worker
# We assume retrieval_toolkit can be used as toolkit for data keeper
data_keeper_agent = ChatAgent(
    model=ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT
    ),
    toolkits_to_register_agent=[retrieval_toolkit],
)

# Create workforce
workforce = Workforce(description="Workforce for LLM-based social simulation research")

# Add agents as single agent workers to workforce
workforce.add_single_agent_worker("LinkUp Search Worker", worker=search_agent)
workforce.add_single_agent_worker("Data Keeper Worker", worker=data_keeper_agent)

# Async function to run example tasks
async def run_tasks():
    # Create search task
    search_task = Task(content="search task", additional_info={"query": "LLM-based social simulation research"})
    await workforce.process_task(search_task)
    print("Search results:", search_task.result)

    # Create store task
    store_task = Task(content="store task", additional_info={"action": "store", "data": search_task.result})
    await workforce.process_task(store_task)
    print(store_task.result)

    # Create retrieve task
    retrieve_task = Task(content="retrieve task", additional_info={"action": "retrieve", "query": "social simulation"})
    await workforce.process_task(retrieve_task)
    print("Retrieve results:", retrieve_task.result)


def main():
    asyncio.run(run_tasks())


if __name__ == "__main__":
    main()
