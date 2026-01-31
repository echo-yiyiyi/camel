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

"""
Workforce with one MCP worker using DeepWiki server and one PPT worker using PPTXToolkit.
"""

import asyncio
import os
from camel.agents import ChatAgent, MCPAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.toolkits.mcp_toolkit import MCPToolkit
from camel.toolkits.pptx_toolkit import PPTXToolkit
from camel.types import ACIRegistryConfig, ModelPlatformType, ModelType


async def main():
    # Create the workforce
    workforce = Workforce(description="MCP + PPT Workforce")

    # MCP worker setup using DeepWiki server
    # Configure ACI registry for DeepWiki MCP server
    aci_config = ACIRegistryConfig(
        api_key=os.getenv("ACI_API_KEY", ""),
        linked_account_owner_id=os.getenv("ACI_LINKED_ACCOUNT_OWNER_ID", ""),
    )

    # MCPToolkit requires config_dict for initialization
    mcp_toolkit = MCPToolkit(
        config_dict=aci_config.get_config(),
        timeout=30,
    )

    mcp_agent = MCPAgent(
        toolkit=mcp_toolkit,
        description="MCP worker using DeepWiki server",
    )

    workforce.add_single_agent_worker(description="MCP DeepWiki Worker", worker=mcp_agent)

    # PPT worker setup
    ppt_toolkit = PPTXToolkit()
    ppt_agent = ChatAgent(
        toolkit=ppt_toolkit,
        description="PPT worker to generate slides",
    )

    workforce.add_single_agent_worker(description="PPT Slide Generator Worker", worker=ppt_agent)

    # Run the workforce as MCP server
    mcp_server = workforce.to_mcp(
        name="MCP-PPT-Workforce",
        description="Workforce with MCP and PPT workers",
        port=8002,
    )

    print("Starting MCP-PPT Workforce server at http://localhost:8002")
    try:
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    asyncio.run(main())
