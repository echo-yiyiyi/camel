from camel.agents.chat_agent import ChatAgent
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.configs.qwen_config import QwenConfig


def main():
    # Initialize the weather toolkit
    weather_toolkit = WeatherToolkit()

    # Use the get_weather_data method as the callable for FunctionTool
    weather_function_tool = FunctionTool(weather_toolkit.get_weather_data)

    # Configure the Qwen2.5-14B-Instruct model
    qwen_config = QwenConfig(
        temperature=0.7,
        max_tokens=512,
        stop=["\n"],
        # Add the wrapped weather function as a tool for the model
        tools=[weather_function_tool],
    )

    # Create the chat agent with the Qwen model and weather toolkit
    agent = ChatAgent(
        system_message="You are a helpful assistant that can answer questions about the weather.",
        model="Qwen2.5-14B-Instruct",
    )

    # Manually set the config attribute
    agent.config = qwen_config

    agent.reset()

    # Example question about weather
    question = "What is the weather like in New York today?"

    # Get the agent's response
    response = agent.step(question)
    print("Question:", question)
    print("Answer:", response.msg.content)


if __name__ == '__main__':
    main()