#!/usr/bin/env python3
import asyncio
import os

from rich import print as rprint

from camel.agents import MCPAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelType


async def main():
    try:
        rprint("[green]CAMEL AI MCP Agent with DeepWiki MCP Server[/green]")

        # MCP config dict for DeepWiki server
        mcp_config = {
            "mcpServers": {
                "deepwiki": {
                    "url": "https://mcpservers.org/servers/devin/deepwiki"
                }
            }
        }

        # Create MCPAgent with MCP config
        agent = await MCPAgent.create(
            registry_configs=None,
            model=ModelFactory.create(
                model_type=ModelType.GEMINI_2_5_PRO,
                api_key=os.getenv("GEMINI_API_KEY"),
                model_config_dict={"temperature": 0.7, "max_tokens": 40000},
            ),
            function_calling_available=True,
            local_config=mcp_config,
        )

        rprint("[green]Agent connected and ready[/green]")

        # User query to retrieve architecture of camel-ai/oasis repo
        user_query = "Retrieve the architecture of the camel-ai/oasis repository."
        user_message = BaseMessage.make_user_message(
            role_name="User", content=user_query
        )

        rprint(f"\n[yellow]Querying: {user_query}[/yellow]")

        response = await agent.astep(user_message)

        # Print response messages
        if response and hasattr(response, "msgs") and response.msgs:
            rprint(f"\nFound [cyan]{len(response.msgs)}[/cyan] messages:")
            for i, msg in enumerate(response.msgs):
                rprint(f"Message {i + 1}: {msg.content}")
        elif response:
            rprint(f"Response content: {response}")
        else:
            rprint("[red]No response received[/red]")

        await agent.disconnect()
        rprint("\n[green]Done[/green]")

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        import traceback
        rprint(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
