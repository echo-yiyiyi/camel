from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types.enums import ModelType, ModelPlatformType
from camel.toolkits.weather_toolkit import WeatherToolkit


def main():
    # System message for the agent
    sys_msg = "You are a helpful agent with the weather tool to answer weather questions."

    # Create the weather toolkit and get its tools
    weather_tools = WeatherToolkit().get_tools()

    # Create the Qwen2.5-14B-Instruct model using the correct enum
    model = ModelFactory.create(
        model_platform=ModelPlatformType.MODELSCOPE,
        model_type=ModelType.MODELSCOPE_QWEN_2_5_14B_INSTRUCT
    )

    # Create the chat agent with the model and the weather tools
    agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=weather_tools
    )

    # Example question about weather
    question = "What's the weather like in New York today?"
    response = agent.step(question)
    print(f"Q: {question}\nA: {response.msg.content}")


if __name__ == '__main__':
    main()
