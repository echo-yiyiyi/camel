import asyncio
from camel.agents.mcp_agent import MCPAgent

async def main():
    # Configuration dict for DeepWiki MCP server with correct type and url
    config_dict = {
        "mcpServers": {
            "deepwiki": {
                "type": "web",
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "timeout": 30,
                "headers": {}
            }
        }
    }

    # Create and connect the MCPAgent with the DeepWiki server
    agent = await MCPAgent.create(
        config_dict=config_dict,
        function_calling_available=False
    )

    # Query to retrieve the architecture of the camel-ai/oasis repository
    query = "Retrieve the architecture of the camel-ai/oasis repository."

    # Run the agent with the query
    response = await agent.astep(query)

    # Print the response content
    print("Agent response:")
    print(response.msgs[0].content if response.msgs else "No response")

    # Disconnect the agent
    await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
