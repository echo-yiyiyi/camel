from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types.enums import ModelType, ModelPlatformType
from camel.configs import QwenConfig


def main():
    # Create Qwen2.5-14B-Instruct model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B,
        model_config_dict=QwenConfig(temperature=0.0).as_dict(),
    )

    # Create WeatherToolkit
    weather_toolkit = WeatherToolkit()

    # Create ChatAgent with the model and weather tool
    agent = ChatAgent(
        system_message="You are a helpful assistant with access to weather information.",
        model=model,
        tools=weather_toolkit.get_tools(),
    )

    # Example question about weather
    user_message = "What's the weather like in New York City today?"

    # Get response from agent
    response = agent.step(user_message)

    print("Agent response:")
    print(response.msg.content)


if __name__ == '__main__':
    main()
