import asyncio
import os

from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ACIRegistryConfig, ModelPlatformType, ModelType

async def main():
    # Check environment variables
    api_key = os.getenv("ACI_API_KEY")
    linked_account_owner_id = os.getenv("ACI_LINKED_ACCOUNT_OWNER_ID")

    if not api_key or not linked_account_owner_id:
        raise ValueError(
            "Environment variables ACI_API_KEY and ACI_LINKED_ACCOUNT_OWNER_ID must be set."
        )

    # Create the ACI registry config for DeepWiki server
    deepwiki_registry_config = ACIRegistryConfig(
        api_key=api_key,
        linked_account_owner_id=linked_account_owner_id,
    )

    # Create a model (using default or OpenAI GPT-4o for better results)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    # Create MCPAgent with the DeepWiki registry config
    agent = MCPAgent(
        model=model,
        registry_configs=[deepwiki_registry_config],
    )

    # Use async context manager to connect and disconnect
    async with agent:
        # Query about the architecture of camel-ai/oasis repository
        query = "Retrieve the architecture of the camel-ai/oasis repository."
        response = await agent.astep(query)
        print("Response:")
        print(response.msgs[0].content)

if __name__ == "__main__":
    asyncio.run(main())
