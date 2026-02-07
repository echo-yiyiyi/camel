from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelType

# Create the Qwen 2.5 14B Instruct model
model = ModelFactory.create(
    model_type=ModelType.QWEN_2_5_14B,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
)

# Create the weather toolkit
weather_toolkit = WeatherToolkit()

# Create the agent with system message and weather tools
system_message = "You are a helpful assistant with access to weather information."
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=weather_toolkit.get_tools(),
)

# Example usage function

def ask_weather_question(question: str) -> str:
    response = agent.step(question)
    return response.msgs[0].content


if __name__ == "__main__":
    # Example question
    question = "What's the weather like in New York today?"
    answer = ask_weather_question(question)
    print(answer)
