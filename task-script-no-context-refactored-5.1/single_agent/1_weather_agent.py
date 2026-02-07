from camel.agents import ChatAgent
from camel.toolkits import WeatherToolkit
from camel.models.qwen_model import QwenModel


def main():
    sys_msg = "You are a helpful agent with the weather tool."

    # Use the model type string that might be recognized by the backend
    model = QwenModel(model_type="qwen-2.5b-instruct")

    # Create the weather toolkit instance
    weather_toolkit = WeatherToolkit()

    # Create the chat agent with the system message, model, and weather tools
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=weather_toolkit.get_tools(),
    )

    # Example user message about weather
    usr_msg = "What's the weather like in New York today?"

    # Get the agent's response
    response = agent.step(usr_msg)

    # Print the tool calls (if any)
    for tool_call in response.info.get("tool_calls", []):
        print(f"Tool call: {tool_call}")

    # Print the agent's reply
    print(f"Agent reply: {response.msg.content}")


if __name__ == '__main__':
    main()
