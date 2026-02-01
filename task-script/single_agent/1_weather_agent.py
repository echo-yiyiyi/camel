from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.models import ModelFactory
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # Create the weather toolkit
    weather_toolkit = WeatherToolkit()
    weather_tools = weather_toolkit.get_tools()

    # Create the Qwen 2.5 14B Instruct model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B,
        model_config_dict=QwenConfig(tools=weather_tools).as_dict(),
    )

    # Create the chat agent with the model and weather tools
    agent = ChatAgent(model=model, tools=weather_tools)

    # Example question about weather
    question = "What is the weather like in New York City today?"

    # Get the agent's answer
    answer = agent.step(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
