from camel.agents import ChatAgent
from camel.models.qwen_model import QwenModel
from camel.toolkits.weather_toolkit import WeatherToolkit


def main():
    system_message = "You are a helpful assistant with access to weather information."

    # Initialize the Qwen2.5-14B-Instruct model
    model = QwenModel("qwen-2.5-14b-instruct")

    # Initialize the weather toolkit
    weather_toolkit = WeatherToolkit()

    # Create the chat agent with the weather tool
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=weather_toolkit.get_tools(),
    )

    # Example question about weather
    user_message = "What's the weather like in New York today?"

    response = agent.step(user_message)
    print(response.msg.content)


if __name__ == '__main__':
    main()
