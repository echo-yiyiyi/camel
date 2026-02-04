import asyncio
from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

async def main():
    # DeepWiki MCP server config with correct type
    deepwiki_config = {
        "mcpServers": {
            "deepwiki": {
                "type": "streamable_http",
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "name": "deepwiki",
            }
        }
    }

    # Create model with default platform and type
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create MCPAgent with local config
    agent = await MCPAgent.create(
        local_config=deepwiki_config,
        model=model,
        function_calling_available=False,
    )

    query = "Retrieve the architecture of the camel-ai/oasis repository."
    response = await agent.astep(query)

    print("Response:")
    print(response.msgs[0].content)

    await agent.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
