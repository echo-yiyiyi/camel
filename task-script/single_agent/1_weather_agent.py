# task-script/single_agent/1_weather_agent.py
# Agent with weather tool using Qwen2.5-14B-Instruct to answer weather questions

import os
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # Create weather toolkit instance
    weather_toolkit = WeatherToolkit()
    tools = weather_toolkit.get_tools()

    # Create Qwen2.5-14B-Instruct model instance
    model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_2_5_14B,
        api_key=os.getenv("QWEN_API_KEY"),
        model_config_dict={"temperature": 0.3, "max_tokens": 2048},
    )

    # Create chat agent with model and weather tools
    agent = ChatAgent(model=model, tools=tools)

    print("Agent with weather tool is ready to answer your questions about weather.")

    while True:
        query = input("Enter your weather question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = agent.step(query)
        print(f"Answer: {response.msg}")


if __name__ == "__main__":
    main()
