# task-script/single_agent/5_deepwiki_mcp_agent.py

"""
Agent script using MCP tools with DeepWiki server to retrieve the architecture
of the camel-ai/oasis repository.
"""

import asyncio
import requests
from camel.agents import MCPAgent
from camel.toolkits.function_tool import FunctionTool

class DeepWikiTool(FunctionTool):
    def __init__(self, server_url: str):
        def query_deepwiki(query: str) -> str:
            response = requests.post(server_url, json={"query": query})
            response.raise_for_status()
            return response.text

        self.server_url = server_url
        super().__init__(func=query_deepwiki)
        self.name = "DeepWikiTool"
        self.description = "Tool to query DeepWiki server."


async def create_deepwiki_agent():
    # URL of the DeepWiki server
    deepwiki_server_url = "https://mcpservers.org/servers/devin/deepwiki"

    # Initialize the DeepWiki tool with the server URL
    deepwiki_tool = DeepWikiTool(server_url=deepwiki_server_url)

    # Create the MCP agent asynchronously with empty registry_configs
    agent = await MCPAgent.create(tools=[deepwiki_tool], registry_configs=[])

    return agent


async def main():
    agent = await create_deepwiki_agent()

    # Query to retrieve the architecture of the camel-ai/oasis repository
    query = "Retrieve the architecture of the camel-ai/oasis repository."

    # Use the agent to run the query
    result = await agent.arun(query)

    # Print the result
    print("Architecture of camel-ai/oasis repository:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
