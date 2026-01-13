from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.configs.qwen_config import QwenConfig

def main():
    # Initialize the weather toolkit
    weather_toolkit = FunctionTool(WeatherToolkit)

    # Create the Qwen2.5-14B-Instruct model with the weather toolkit as a tool
    model = ModelFactory.create(
    model_config=QwenConfig(tools=[weather_toolkit]),
    model_config=QwenConfig(tools=[weather_toolkit])
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.MODELSCOPE_QWEN_2_5_14B_INSTRUCT,
        
    )

    # Create the chat agent with a system prompt and the model
    system_prompt = "You are a helpful assistant with access to weather information."
    agent = ChatAgent(system_prompt, model=model)

    # Reset the agent to start fresh
    agent.reset()

    # Example interaction: ask about the weather
    user_input = "What is the weather like in New York today?"
    response = agent.step(user_input)

    print("Agent response:")
    print(response.msgs[0].content)

if __name__ == "__main__":
    main()
