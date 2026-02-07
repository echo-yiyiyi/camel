import asyncio
from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ModelType

async def main():
    # Define the DeepWiki MCP server configuration dictionary with correct type
    deepwiki_config_dict = {
        "mcpServers": {
            "deepwiki": {
                "type": "streamable_http",
                "url": "https://mcpservers.org/servers/devin/deepwiki"
            }
        }
    }

    # Create the model
    model = ModelFactory.create(model_type=ModelType.GPT_4O_MINI)

    # Create the MCPAgent with the DeepWiki config dict as local_config
    agent = MCPAgent(
        model=model,
        local_config=deepwiki_config_dict,
        function_calling_available=False,
    )

    # Use the agent asynchronously
    async with agent:
        query = "Retrieve the architecture of the camel-ai/oasis repository."
        response = await agent.astep(query)
        print("Response:")
        print(response.msgs[0].content)

if __name__ == "__main__":
    asyncio.run(main())
