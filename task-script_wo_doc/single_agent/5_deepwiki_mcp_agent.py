from camel.agents.mcp_agent import MCPAgent
from camel.toolkits.mcp_toolkit import MCPToolkit
from camel.models import ModelFactory
from camel.configs.openai_config import ChatGPTConfig
import asyncio

async def main():
    # Configure the MCPToolkit with DeepWiki server using config_dict
    deepwiki_config = {
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki"
            }
        }
    }
    # Set timeout to 30 seconds
    mcp_toolkit = MCPToolkit(config_dict=deepwiki_config, timeout=30.0)

    # Connect to the MCP servers
    await mcp_toolkit.connect()

    # Create the MCPAgent with the MCPToolkit
    model = ModelFactory.create_model(ChatGPTConfig())
    agent = MCPAgent(model=model, toolkit=mcp_toolkit)

    # Define the task to retrieve the architecture of camel-ai/oasis repository
    task = "Retrieve the architecture of the camel-ai/oasis repository."

    # Run the agent to perform the task
    response = await agent.arun(task)

    # Print the retrieved architecture information
    print("Retrieved Architecture Information:")
    print(response)

    # Disconnect from the MCP servers
    await mcp_toolkit.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
