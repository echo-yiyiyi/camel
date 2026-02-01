from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types.enums import ModelType, ModelPlatformType
from camel.toolkits.weather_toolkit import WeatherToolkit


def main():
    # Create the weather toolkit and get tools
    weather_tools = WeatherToolkit().get_tools()

    # Create the Qwen2.5-14B-Instruct model with platform
    model = ModelFactory.create(model_platform=ModelPlatformType.QWEN, model_type=ModelType.QWEN_2_5_14B)

    # Create the chat agent with the model and the weather tools
    agent = ChatAgent(
        model=model,
        tools=weather_tools,
    )

    # Example question about weather
    question = "What's the weather like in New York today?"

    # Get the agent's response
    response = agent.step(question)

    print("Question:", question)
    print("Answer:", response.msg.content)


if __name__ == '__main__':
    main()
