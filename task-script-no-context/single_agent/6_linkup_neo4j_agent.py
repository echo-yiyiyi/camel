# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import os
from dotenv import load_dotenv

from camel.agents import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from camel.toolkits import SearchToolkit
from camel.loaders import UnstructuredIO

load_dotenv()


def main():
    # Initialize Neo4jGraph with environment variables or placeholders
    neo4j_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    neo4j_graph = Neo4jGraph(
        url=neo4j_url,
        username=neo4j_username,
        password=neo4j_password,
    )

    # Initialize KnowledgeGraphAgent
    kg_agent = KnowledgeGraphAgent()

    # Initialize SearchToolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"

    print(f"Searching LinkUp for query: {query}")
    search_results = search_toolkit.search_linkup(query=query, output_type="searchResults")

    if "error" in search_results:
        print(f"Error in LinkUp search: {search_results['error']}")
        return

    results = search_results.get("results", [])
    print(f"Found {len(results)} results from LinkUp.")

    uio = UnstructuredIO()

    for idx, result in enumerate(results, start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        content = result.get("content", "")  # Some results may have content

        print(f"\nProcessing result {idx}: {title}")
        print(f"URL: {url}")

        # Use snippet or content if available, else just title
        text_to_process = content or snippet or title

        if not text_to_process:
            print("No textual content available to process.")
            continue

        # Create element from text
        element = uio.create_element_from_text(text=text_to_process)

        # Extract graph elements using KnowledgeGraphAgent
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        # Add extracted graph elements to Neo4j
        neo4j_graph.add_graph_elements([graph_element], include_source=True, base_entity_label=True)

        print(f"Added graph elements from result {idx} to Neo4j.")

    print("\nAll results processed and stored in Neo4j.")


if __name__ == "__main__":
    main()
