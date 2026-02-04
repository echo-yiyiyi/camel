from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.weather_toolkit import WeatherToolkit


def main():
    # Create Qwen2.5-14B-Instruct model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_PLUS_LATEST,  # closest available Qwen 2.5 variant
        model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    )

    # Create WeatherToolkit and get tools
    weather_toolkit = WeatherToolkit()
    weather_tools = weather_toolkit.get_tools()

    # Define system message
    system_message = "You are a helpful assistant with access to weather information."

    # Create agent with model and weather tools
    agent = ChatAgent(system_message=system_message, model=model, tools=weather_tools)

    # Example question about weather
    user_message = "What's the weather like in New York City today?"

    # Get response
    response = agent.step(user_message)

    # Print the answer
    print(response.msgs[0].content)


if __name__ == '__main__':
    main()
