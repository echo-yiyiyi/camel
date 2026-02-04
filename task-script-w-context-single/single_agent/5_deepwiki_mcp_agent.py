import asyncio
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import MCPToolkit
from camel.types import ModelPlatformType, ModelType

async def main():
    # MCP config for DeepWiki server
    config_path = "task-script/single_agent/deepwiki_mcp_config.json"

    # Create MCPToolkit instance and connect
    mcp_toolkit = MCPToolkit(config_path=config_path)
    await mcp_toolkit.connect()

    # Get MCP tools
    tools = list(mcp_toolkit.get_tools())

    # Create model instance
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict={"temperature": 0.3},
    )

    # Create ChatAgent with MCP tools
    system_message = "You are an assistant that retrieves the architecture of the camel-ai/oasis repository using DeepWiki MCP server."
    agent = ChatAgent(system_message=system_message, model=model, tools=tools)

    # Query prompt
    prompt = "Retrieve the architecture of the camel-ai/oasis repository."

    # Run agent step
    response = await agent.astep(prompt)

    # Disconnect MCP
    await mcp_toolkit.disconnect()

    # Print response
    print(response.msgs[0].content)

if __name__ == "__main__":
    asyncio.run(main())
