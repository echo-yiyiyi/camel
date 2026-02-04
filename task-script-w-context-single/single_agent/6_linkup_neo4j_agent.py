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

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.search_toolkit import SearchToolkit
from camel.storages import Neo4jGraph
from camel.types import ModelPlatformType, ModelType


def main():
    # Create model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create agent
    agent = ChatAgent(system_message="You are a knowledge graph agent.", model=model)

    # Initialize LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Initialize Neo4j graph storage
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password",
    )

    # Define the research topic query
    query = "LLM-based social simulation research"

    # Perform LinkUp search
    print(f"Performing LinkUp search for query: {query}")
    search_results = search_toolkit.search_linkup(query=query, depth="standard", output_type="searchResults")

    if "error" in search_results:
        print(f"Error during LinkUp search: {search_results['error']}")
        return

    results = search_results.get("results", [])
    print(f"Number of results retrieved: {len(results)}")

    # Store results in Neo4j
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "No Title")
        url = result.get("url", "No URL")
        snippet = result.get("snippet", "")

        # Add triplet: (query)-[RELATED_TO]->(title)
        neo4j_graph.add_triplet(subj=query, obj=title, rel="RELATED_TO")

        # Add triplet: (title)-[HAS_URL]->(url)
        neo4j_graph.add_triplet(subj=title, obj=url, rel="HAS_URL")

        # Optionally add snippet as a node and link it
        if snippet:
            snippet_node = f"Snippet_{idx}"
            neo4j_graph.add_triplet(subj=title, obj=snippet_node, rel="HAS_SNIPPET")

    print("Stored search results in Neo4j graph database.")


if __name__ == "__main__":
    main()
