from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelType


def main():
    # Create Qwen model with Qwen2.5-14B-Instruct
    model = ModelFactory.create(
        model_type=ModelType.QWEN_2_5_14B,
        model_config_dict=QwenConfig().as_dict()
    )

    # Get weather tools
    weather_toolkit = WeatherToolkit()
    tools = weather_toolkit.get_tools()

    # Create ChatAgent with weather tools
    agent = ChatAgent(
        system_message="You are a helpful assistant that can answer questions about the weather.",
        model=model,
        tools=tools
    )

    # Example question about weather
    question = "What is the weather like in New York today?"
    response = agent.step(question)
    print("Question:", question)
    print("Answer:", response.msgs[0].content)


if __name__ == "__main__":
    main()
