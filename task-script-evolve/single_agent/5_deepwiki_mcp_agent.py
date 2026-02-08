import asyncio
from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ModelType

async def main():
    # Define the DeepWiki MCP server config with remote URL
    deepwiki_config = {
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki"
            }
        },
        "mcpWebServers": {}
    }

    # Create a model
    model = ModelFactory.create(model_type=ModelType.DEFAULT)

    # Create MCPAgent with the DeepWiki config as local_config
    agent = MCPAgent(
        model=model,
        local_config=deepwiki_config,
    )

    # Use async context manager to connect and disconnect
    async with agent:
        # Ask about the architecture of camel-ai/oasis repository
        question = "Retrieve the architecture of the camel-ai/oasis repository."
        response = await agent.astep(question)
        print("Response:")
        print(response.msgs[0].content)

if __name__ == "__main__":
    asyncio.run(main())
