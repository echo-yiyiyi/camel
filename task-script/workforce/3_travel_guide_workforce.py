from camel.societies.workforce.workforce import Workforce
from camel.agents.chat_agent import ChatAgent
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.toolkits.openbb_toolkit import OpenBBToolkit
from camel.toolkits.context_summarizer_toolkit import ContextSummarizerToolkit
from camel.types.enums import ModelPlatformType, ModelType


def main():
    # Create the workforce with a description
    workforce = Workforce(description="Travel guide workforce with weather, historical, and tourist workers")

    # Create the weather search worker
    weather_toolkit = WeatherToolkit()
    weather_worker = ChatAgent(
        system_message="You are a helpful assistant specialized in weather information.",
        model=(ModelPlatformType.GROQ, ModelType.GROQ_LLAMA_3_1_8B),
    )
    weather_worker.add_tools(weather_toolkit.get_tools())

    # Create the historical information worker
    openbb_toolkit = OpenBBToolkit()
    historical_worker = ChatAgent(
        system_message="You are a knowledgeable assistant specialized in historical and financial data.",
        model=(ModelPlatformType.GROQ, ModelType.GROQ_LLAMA_3_1_8B),
    )
    historical_worker.add_tools(openbb_toolkit.get_tools())

    # Create the tourist information worker
    context_summarizer_toolkit = ContextSummarizerToolkit()
    tourist_worker = ChatAgent(
        system_message="You are a friendly tourist guide assistant.",
        model=(ModelPlatformType.GROQ, ModelType.GROQ_LLAMA_3_1_8B),
    )
    tourist_worker.add_tools(context_summarizer_toolkit.get_tools())

    # Add workers to the workforce
    workforce.add_worker(weather_worker)
    workforce.add_worker(historical_worker)
    workforce.add_worker(tourist_worker)

    # Example main task for the workforce
    main_task = "Assist users with weather, historical information, and tourist guidance."

    # Run the workforce on the main task
    result = workforce.run(main_task)

    print("Workforce result:")
    print(result)


if __name__ == "__main__":
    main()
