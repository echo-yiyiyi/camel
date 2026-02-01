"""
This script creates an MCPAgent with MCP tools using a DeepWiki MCP server configuration
and queries the architecture of the camel-ai/oasis repository.
"""
import asyncio
from camel.agents.mcp_agent import MCPAgent
from camel.types.mcp_registries import BaseMCPRegistryConfig, MCPRegistryType
from typing import Dict, Any
from typing import Literal
from pydantic import Field

class DeepWikiRegistryConfig(BaseMCPRegistryConfig):
    type: Literal[MCPRegistryType.CUSTOM] = MCPRegistryType.CUSTOM
    url: str = Field(..., description="The URL of the MCP server")
    description: str = Field("", description="Description of the MCP server")

    def get_config(self) -> Dict[str, Any]:
        # Return the MCP server config dict expected by MCPToolkit
        return {
            "mcpServers": {
                "deepwiki": {
                    "url": self.url,
                    "description": self.description,
                }
            }
        }

async def main():
    # Define the DeepWiki MCP server configuration
    deepwiki_registry_config = DeepWikiRegistryConfig(
        type=MCPRegistryType.CUSTOM,
        os="linux",
        url="https://mcpservers.org/servers/devin/deepwiki",
        description="DeepWiki MCP server for camel-ai/oasis architecture retrieval"
    )

    # Create the MCPAgent with the DeepWiki registry config
    agent = MCPAgent(registry_configs=[deepwiki_registry_config])

    # Connect the agent to the MCP server
    await agent.connect()

    # Query to retrieve the architecture of camel-ai/oasis repository
    query = "Retrieve the architecture of the camel-ai/oasis repository."

    # Run the query asynchronously
    response = await agent.arun(query)

    # Print the response
    print("Response from DeepWiki MCP server:")
    print(response)

    # Disconnect the agent
    await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
