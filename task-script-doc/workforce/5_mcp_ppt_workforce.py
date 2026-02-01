import asyncio
from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.toolkits.mcp_toolkit import MCPToolkit
from camel.toolkits.pptx_toolkit import PPTXToolkit

async def main():
    # Create workforce
    workforce = Workforce(description="MCP and PPT Slide Generation Workforce")

    # MCP worker using DeepWiki MCP server
    mcp_config = {
        "mcpServers": {
            "deepwiki": {
                "url": "https://mcpservers.org/servers/devin/deepwiki",
                "timeout": 60
            }
        }
    }
    mcp_toolkit = await MCPToolkit.create(config_dict=mcp_config)
    mcp_tools = mcp_toolkit.get_tools()

    mcp_msg = BaseMessage.make_assistant_message(
        role_name="DeepWiki Retriever",
        content="You are a worker that retrieves CAMEL-AI information using DeepWiki MCP tools."
    )
    mcp_worker = ChatAgent(system_message=mcp_msg, tools=mcp_tools)
    workforce.add_single_agent_worker(description="DeepWiki MCP Worker", worker=mcp_worker)

    # PPT worker using PPTXToolkit
    ppt_toolkit = PPTXToolkit()
    ppt_msg = BaseMessage.make_assistant_message(
        role_name="PPT Slide Generator",
        content="You are a worker that generates presentation slides using PPTXToolkit."
    )
    ppt_worker = ChatAgent(system_message=ppt_msg, tools=[ppt_toolkit])
    workforce.add_single_agent_worker(description="PPT Slide Generator Worker", worker=ppt_worker)

    # Task content
    task_content = (
        "Retrieve detailed information about CAMEL-AI from the DeepWiki server. "
        "Then generate a presentation with a title slide and 5 content slides summarizing the key points."
    )

    # Process the task asynchronously
    result = await workforce.process_task_async(task_content)
    print("Workforce task result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
