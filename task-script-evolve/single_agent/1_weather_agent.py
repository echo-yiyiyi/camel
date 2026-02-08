from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.types import ModelType
from camel.toolkits import WeatherToolkit


def main():
    # Create the Qwen2.5-14B-Instruct model
    model = ModelFactory.create(
        model_type=ModelType.QWEN_2_5_14B,
        model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    )

    # Define system message
    sys_msg = "You are a helpful assistant with access to weather information."

    # Create the WeatherToolkit instance and get its tools
    weather_tools = WeatherToolkit().get_tools()

    # Create the ChatAgent with the model and weather tools
    agent = ChatAgent(system_message=sys_msg, model=model, tools=weather_tools)

    # Example user question about weather
    user_question = "What's the weather like in New York today?"

    # Get the agent's response
    response = agent.step(user_question)

    print("User question:", user_question)
    print("Agent response:", response.msg.content)


if __name__ == '__main__':
    main()
