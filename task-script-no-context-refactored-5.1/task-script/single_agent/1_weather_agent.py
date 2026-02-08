from camel.agents import ChatAgent
from camel.toolkits import WeatherToolkit


def main():
    sys_msg = "You are a helpful assistant with weather knowledge."

    agent = ChatAgent(
        system_message=sys_msg,
        tools=WeatherToolkit().get_tools(),
        model="qwen2.5-14b-instruct",
    )

    user_question = "What's the weather like in New York today?"

    response = agent.step(user_question)

    print("Agent response:")
    print(response.msg.content)


if __name__ == '__main__':
    main()
