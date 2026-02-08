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

from camel.toolkits import SearchToolkit
from camel.agents import KnowledgeGraphAgent
from camel.storages import Neo4jGraph
from camel.loaders import UnstructuredIO


def main():
    # Initialize LinkUp search toolkit
    search_toolkit = SearchToolkit()

    # Query for LLM-based social simulation research
    query = "LLM-based social simulation research"

    # Perform LinkUp search to get webpages
    search_results = search_toolkit.search_linkup(
        query=query,
        depth="standard",
        output_type="searchResults"
    )

    # Initialize KnowledgeGraphAgent
    kg_agent = KnowledgeGraphAgent()

    # Initialize Neo4jGraph storage
    neo4j_graph = Neo4jGraph(
        url="bolt://localhost:7687",  # Replace with your Neo4j URL
        username="neo4j",             # Replace with your Neo4j username
        password="password"           # Replace with your Neo4j password
    )

    # Initialize UnstructuredIO for creating elements from text
    uio = UnstructuredIO()

    graph_elements = []

    # Process each search result
    for result in search_results.get("results", []):
        url = result.get("url", "")
        content = result.get("content", "")

        # Create an element from the content
        element = uio.create_element_from_text(text=content)

        # Extract graph elements using the knowledge graph agent
        graph_element = kg_agent.run(element, parse_graph_elements=True)

        # Collect graph elements
        graph_elements.append(graph_element)

    # Add all extracted graph elements to Neo4j
    neo4j_graph.add_graph_elements(graph_elements, include_source=True, base_entity_label=True)

    print("Knowledge graph data from LinkUp search results has been stored in Neo4j.")


if __name__ == "__main__":
    main()
