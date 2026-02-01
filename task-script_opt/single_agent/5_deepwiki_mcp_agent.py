# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import asyncio
import os
import sys

from camel.agents import MCPAgent
from camel.models import ModelFactory
from camel.types import ACIRegistryConfig, ModelPlatformType, ModelType


def check_env_vars():
    aci_api_key = os.getenv("ACI_API_KEY")
    linked_account_owner_id = os.getenv("ACI_LINKED_ACCOUNT_OWNER_ID")
    if not aci_api_key:
        print("Error: ACI_API_KEY environment variable is not set.")
        sys.exit(1)
    if not linked_account_owner_id:
        print("Error: ACI_LINKED_ACCOUNT_OWNER_ID environment variable is not set.")
        sys.exit(1)
    return aci_api_key, linked_account_owner_id


async def main():
    aci_api_key, linked_account_owner_id = check_env_vars()

    # Create the ACI registry config for DeepWiki server
    aci_config = ACIRegistryConfig(
        api_key=aci_api_key,
        linked_account_owner_id=linked_account_owner_id,
    )

    # Create a model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
    )

    # Create MCPAgent with the ACI registry config
    agent = MCPAgent(
        model=model,
        registry_configs=[aci_config],
    )

    # Message to retrieve the architecture of camel-ai/oasis repository
    message = "Retrieve the architecture of the camel-ai/oasis repository."

    # Use agent with async context manager
    async with agent:
        response = await agent.astep(message)
        print(f"\nResponse to message '{message}':")
        print(response.msgs[0].content)


if __name__ == "__main__":
    asyncio.run(main())
