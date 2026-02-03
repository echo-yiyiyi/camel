"""
Create a workforce with two workers:
- One worker using MCP tools connected to the DeepWiki MCP server to retrieve CAMEL-AI info
- Another worker using PPT tools to generate slides

Run the workforce as an MCP server
"""

from camel.agents import ChatAgent
from camel.societies.workforce.workforce import Workforce
from camel.toolkits.mcp_toolkit import MCPToolkit
from camel.toolkits.pptx_toolkit import PPTXToolkit
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType


def main():
    # Create MCPToolkit connected to DeepWiki MCP server using correct config dict
    mcp_toolkit = MCPToolkit(config_dict={
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "client_type": "http"
            }
        }
    })

    # Create a ChatAgent worker using MCP tools
    mcp_worker = ChatAgent(
        name="MCPWorker",
        model=ModelFactory.create(
            model_platform=ModelPlatformType.MCP,
            model_type=ModelType.LLAMA_3_1_8B
        ),
        toolkit=mcp_toolkit
    )

    # Create PPTXToolkit for slide generation
    ppt_toolkit = PPTXToolkit()

    # Create a ChatAgent worker using PPT tools
    ppt_worker = ChatAgent(
        name="PPTWorker",
        model=ModelFactory.create(
            model_platform=ModelPlatformType.LOCAL,
            model_type=ModelType.LLAMA_3_1_8B
        ),
        toolkit=ppt_toolkit
    )

    # Create workforce and add both workers
    workforce = Workforce()
    workforce.add_worker(mcp_worker)
    workforce.add_worker(ppt_worker)

    # Run the workforce as an MCP server
    workforce.to_mcp(host="0.0.0.0", port=7860).run()


if __name__ == "__main__":
    main()
