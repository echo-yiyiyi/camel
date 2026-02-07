"""
Script to create an MCPAgent with MCP tools using the DeepWiki server to retrieve the architecture of the camel-ai/oasis repository.
"""
import asyncio
from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ModelType

async def main():
    # Local config dictionary for MCPToolkit with DeepWiki server
    local_config = {
        "mcpServers": {
            "deepwiki": {
                "type": "streamable_http",
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "name": "deepwiki",
                "token": None,
            }
        }
    }

    # Create model
    model = ModelFactory.create(
        model_type=ModelType.DEFAULT,
    )

    # Create MCPAgent with local config and model
    agent = await MCPAgent.create(
        local_config=local_config,
        model=model,
        function_calling_available=True,
    )

    # Query to retrieve the architecture of the camel-ai/oasis repository
    query = "Retrieve the architecture of the camel-ai/oasis repository."

    # Run one step of the agent with the query
    response = await agent.astep(query)

    # Print the response
    print("Response from DeepWiki MCP agent:")
    print(response.msgs[0].content)

    # Disconnect the agent
    await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
