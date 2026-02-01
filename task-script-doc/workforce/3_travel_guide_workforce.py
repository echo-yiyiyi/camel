from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce


def main():
    """Create and run a travel guide workforce MCP server."""

    # Create a workforce
    workforce = Workforce(description="Travel Guide Workforce")

    # Weather search worker
    weather_msg = BaseMessage.make_assistant_message(
        role_name="Weather Search Worker",
        content="You are a worker specialized in searching and providing weather information.",
    )
    weather_worker = ChatAgent(system_message=weather_msg)
    workforce.add_single_agent_worker(
        description="Worker for weather search", worker=weather_worker
    )

    # Historical information worker
    historical_msg = BaseMessage.make_assistant_message(
        role_name="Historical Information Worker",
        content="You are a worker specialized in providing historical information and answering history-related questions.",
    )
    historical_worker = ChatAgent(system_message=historical_msg)
    workforce.add_single_agent_worker(
        description="Worker for historical information", worker=historical_worker
    )

    # Tourist worker
    tourist_msg = BaseMessage.make_assistant_message(
        role_name="Tourist Worker",
        content="You are a worker specialized in providing tourist information and travel advice.",
    )
    tourist_worker = ChatAgent(system_message=tourist_msg)
    workforce.add_single_agent_worker(
        description="Worker for tourist information", worker=tourist_worker
    )

    print("Creating Travel Guide Workforce MCP server...")
    print("Server will be available at: http://localhost:8001")
    print("Press Ctrl+C to stop the server")

    # Convert to MCP server and run
    mcp_server = workforce.to_mcp(
        name="Travel-Guide-Workforce",
        description="A workforce for travel guide tasks",
        port=8001,
    )

    try:
        mcp_server.run()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
