from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.societies.workforce import Workforce
from camel.toolkits.weather_toolkit import WeatherToolkit
from camel.toolkits import ACIToolkit


def main():
    """Create and run a travel guide workforce with three workers."""

    # Create the workforce
    workforce = Workforce(description="Travel Guide Workforce")

    # Weather search worker
    weather_msg = BaseMessage.make_assistant_message(
        role_name="WeatherSearcher",
        content="You are a helpful assistant who provides weather information.",
    )
    weather_agent = ChatAgent(system_message=weather_msg, tools=WeatherToolkit().get_tools())
    workforce.add_single_agent_worker(description="Weather Search Worker", worker=weather_agent)

    # Historical information worker
    historical_msg = BaseMessage.make_assistant_message(
        role_name="Historian",
        content="You are a knowledgeable assistant who provides historical information.",
    )
    aci_toolkit = ACIToolkit(linked_account_owner_id=None)  # Adjust if needed
    historical_agent = ChatAgent(system_message=historical_msg, tools=aci_toolkit.get_tools())
    workforce.add_single_agent_worker(description="Historical Information Worker", worker=historical_agent)

    # Tourist worker
    tourist_msg = BaseMessage.make_assistant_message(
        role_name="TouristGuide",
        content="You are a friendly tourist guide who helps with travel and sightseeing.",
    )
    tourist_agent = ChatAgent(system_message=tourist_msg)
    workforce.add_single_agent_worker(description="Tourist Worker", worker=tourist_agent)

    print("Travel Guide Workforce created with 3 workers.")


if __name__ == "__main__":
    main()
