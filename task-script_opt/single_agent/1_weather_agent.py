from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.weather_toolkit import WeatherToolkit


def main():
    # Create Qwen 3 Coder Plus model (closest available to Qwen2.5-14B-Instruct)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_3_CODER_PLUS,
    )

    # Create agent with weather tool
    weather_toolkit = WeatherToolkit()
    agent = ChatAgent(
        system_message="You are a helpful assistant with access to weather information.",
        model=model,
        tools=weather_toolkit.get_tools(),
    )

    # Ask a weather question
    user_message = "What's the weather like in New York today?"
    response = agent.step(user_message)

    # Print the agent's response
    print(response.msgs[0].content)


if __name__ == '__main__':
    main()
