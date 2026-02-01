from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.toolkits import MCPToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import asyncio
import os
from camel.tasks.task import Task

# Create a minimal MCP config for testing
mcp_config = {
    "mcpServers": {
        "local_filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
        }
    }
}

async def create_workforce():
    # Load MCP toolkit and connect to MCP servers
    mcp_toolkit = MCPToolkit(config_dict=mcp_config)
    await mcp_toolkit.connect()
    tools = mcp_toolkit.get_tools()

    # Setup model for agents
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_5_PRO,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_config_dict={"temperature": 0.7, "max_tokens": 40000},
    )

    # Create workforce
    workforce = Workforce(description="Clinical Trial Enrollment Data Team")

    # Create workers with different tools
    # Worker 1: MCP tools worker
    mcp_worker_msg = BaseMessage.make_assistant_message(
        role_name="MCP Worker",
        content="You are a worker with access to MCP tools to fetch clinical trial data.",
    )
    mcp_worker_agent = ChatAgent(system_message=mcp_worker_msg, model=model, tools=tools)
    workforce.add_single_agent_worker(description="MCP Tools Worker", worker=mcp_worker_agent)

    # Worker 2: Search and reasoning worker
    search_worker_msg = BaseMessage.make_assistant_message(
        role_name="Search Worker",
        content="You are a worker specialized in searching and reasoning to find clinical trial enrollment info.",
    )
    search_worker_agent = ChatAgent(system_message=search_worker_msg, model=model)
    workforce.add_single_agent_worker(description="Search and Reasoning Worker", worker=search_worker_agent)

    return workforce

async def main():
    workforce = await create_workforce()

    # Define the task
    task_content = (
        "What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
        "from Jan-May 2018 as listed on the NIH website?"
    )

    # Wrap the task content in a Task object
    task = Task(content=task_content)

    # Process the task
    result_task = await workforce.process_task_async(task)

    # Print the result
    print("Task Result:", result_task.result)

if __name__ == "__main__":
    asyncio.run(main())
