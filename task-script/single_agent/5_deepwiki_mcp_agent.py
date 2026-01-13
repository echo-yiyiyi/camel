import asyncio
from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import BaseMCPRegistryConfig, MCPRegistryType, ModelPlatformType, ModelType

async def main():
    # DeepWiki MCP server configuration dictionary with streamable_http type
    config_dict = {
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "timeout": 30,
                "type": "streamable_http"
            }
        }
    }

    # Create MCPAgent using the factory method with config_dict
    agent = await MCPAgent.create(
        config_path=None,
        registry_configs=None,
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
        function_calling_available=False,
        local_config=config_dict,
    )

    async with agent:
        # Query to retrieve architecture of camel-ai/oasis repository
        query = "Retrieve the architecture of the camel-ai/oasis repository."
        response = await agent.astep(query)
        print("Response:")
        print(response.msgs[0].content)

if __name__ == '__main__':
    asyncio.run(main())
