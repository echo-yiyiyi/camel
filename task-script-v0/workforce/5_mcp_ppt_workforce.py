"""
Workforce with one worker using MCP tools (DeepWiki server) to retrieve CAMEL-AI info,
and another worker with PPT tools to generate slides.
"""

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.toolkits import PPTXToolkit
from camel.agents.tool_agents.mcp_tool_agent import MCPToolAgent


def main():
    # Create workforce
    workforce = Workforce(description="MCP and PPT Workforce")

    # Worker 1: MCP Tool Agent connected to DeepWiki server
    deepwiki_mcp_url = "https://mcpservers.org/servers/devin/deepwiki"
    mcp_agent = MCPToolAgent(
        mcp_server_url=deepwiki_mcp_url,
        description="MCP Agent for DeepWiki CAMEL-AI info retrieval",
    )
    workforce.add_single_agent_worker(
        description="DeepWiki MCP Worker",
        worker=mcp_agent,
    )

    # Worker 2: PPTX Toolkit Agent
    ppt_toolkit = PPTXToolkit()
    ppt_agent_msg = BaseMessage.make_assistant_message(
        role_name="PPT Generator",
        content="You are a helpful assistant that generates slides using PPTXToolkit.",
    )
    ppt_agent = ChatAgent(system_message=ppt_agent_msg, toolkits=[ppt_toolkit])
    workforce.add_single_agent_worker(
        description="PPT Generator Worker",
        worker=ppt_agent,
    )

    print("Starting MCP and PPT Workforce MCP server...")
    print("Server will be available at: http://localhost:8002")

    # Run workforce as MCP server
    mcp_server = workforce.to_mcp(
        name="MCP-PPT-Workforce",
        description="Workforce with MCP and PPT workers",
        port=8002,
    )
    mcp_server.run()


if __name__ == "__main__":
    main()
