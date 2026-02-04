from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.configs import QwenConfig
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.types import ModelPlatformType, ModelType

# Create the Qwen 2.5 14B Instruct model
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_2_5_14B,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
)

# Create the weather toolkit instance
weather_toolkit = WeatherToolkit()

# Create the agent with system message and weather tools
system_message = "You are a helpful assistant that can answer questions about the weather."
agent = ChatAgent(
    system_message=system_message,
    model=model,
    tools=weather_toolkit.get_tools(),
)

# Example usage function
if __name__ == '__main__':
    question = "What's the weather like in New York today?"
    response = agent.step(question)
    print(response.msg.content)
