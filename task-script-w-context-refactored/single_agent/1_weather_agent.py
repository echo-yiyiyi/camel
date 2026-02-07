from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import WeatherToolkit
from camel.types import ModelType


def main():
    # Create the Qwen2.5-14B-Instruct model
    model = ModelFactory.create(model_type=ModelType.QWEN_2_5_14B)

    # Create the WeatherToolkit and get its tools
    weather_toolkit = WeatherToolkit()
    weather_tools = weather_toolkit.get_tools()

    # Create the agent with a system message and the weather tools
    system_message = "You are a helpful assistant with access to weather information."
    agent = ChatAgent(system_message=system_message, model=model, tools=weather_tools)

    # Ask a weather question
    user_message = "What is the weather like in New York City today?"
    response = agent.step(user_message)

    # Print the agent's response
    print(response.msg.content)


if __name__ == '__main__':
    main()
