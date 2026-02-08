import asyncio
import json
import tempfile
from pathlib import Path

from camel.agents.mcp_agent import MCPAgent

async def main():
    # MCP server config for DeepWiki server with explicit transport type
    config_data = {
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "type": "streamable_http"
            }
        }
    }

    # Write config to a temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmpfile:
        json.dump(config_data, tmpfile)
        tmpfile_path = tmpfile.name

    # Load config dict from file and remove 'type' key from each server config
    with open(tmpfile_path, 'r') as f:
        config_dict = json.load(f)

    for server_name, server_cfg in config_dict.get("mcpServers", {}).items():
        if "type" in server_cfg:
            del server_cfg["type"]

    # Create and connect MCPAgent with DeepWiki MCP server using cleaned config dict
    agent = await MCPAgent.create(
        registry_configs=None,
        local_config=config_dict,
        function_calling_available=False,
    )

    # Query the architecture of camel-ai/oasis repository
    user_message = "Retrieve the architecture of the camel-ai/oasis repository."
    response = await agent.astep(user_message)

    print("Agent response:")
    print(response.msgs[0].content)

    await agent.disconnect()

    # Clean up temporary config file
    Path(tmpfile_path).unlink()

if __name__ == '__main__':
    asyncio.run(main())
