from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.models import ModelFactory
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # Create the Qwen 2.5 14B Instruct model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B,
        model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    )

    # Create the weather toolkit
    weather_toolkit = WeatherToolkit()

    # Define system message
    sys_msg = "You are a helpful assistant with access to weather information."

    # Create the agent with the weather tool
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=weather_toolkit.get_tools(),
    )

    # Example user question about weather
    user_msg = "What's the weather like in New York City today?"

    # Get response from the agent
    response = agent.step(user_msg)

    # Print the response content
    print(response.msgs[0].content)


if __name__ == '__main__':
    main()
