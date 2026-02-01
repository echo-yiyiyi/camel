from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import WeatherToolkit, SearchToolkit
from camel.messages.base import BaseMessage
from camel.types import ModelPlatformType, ModelType


def main():
    # Worker 1: Weather Search Specialist
    weather_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Weather Specialist",
            content="You provide accurate and up-to-date weather information.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
        tools=[WeatherToolkit().get_tools()[0]],  # Use the main weather search tool
    )

    # Worker 2: Historical Information Specialist
    historical_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Historical Information Specialist",
            content="You provide detailed historical information and context using search tools.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
        tools=[SearchToolkit().search_wiki],
    )

    # Worker 3: Tourist Guide
    tourist_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Tourist Guide",
            content="You provide travel advice, sightseeing recommendations, and local tips.",
        ),
        model=ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        ),
    )

    workforce = Workforce(
        "Travel Guide Workforce",
        graceful_shutdown_timeout=30.0,
    )

    workforce.add_single_agent_worker(
        "A worker specialized in weather search.",
        worker=weather_agent,
    ).add_single_agent_worker(
        "A worker specialized in historical information search.",
        worker=historical_agent,
    ).add_single_agent_worker(
        "A worker specialized as a tourist guide.",
        worker=tourist_agent,
    )

    # Define a sample task for the workforce with specified travel destination
    task = Task(
        content="Plan a travel itinerary for Paris including weather forecast, historical sites, and tourist attractions.",
        id="travel_guide_task",
    )

    workforce.process_task(task)


if __name__ == "__main__":
    main()
