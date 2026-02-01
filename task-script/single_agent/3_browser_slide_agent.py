from camel.agents.chat_agent import ChatAgent
from camel.models.model_factory import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types.enums import ModelPlatformType, ModelType


def main():
    # Create the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT
    )

    # Create the browser toolkit
    browser_toolkit = BrowserToolkit()

    # Create the PPTX toolkit
    pptx_toolkit = PPTXToolkit(working_directory="./pptx_outputs")

    # Create the agent with both browser and pptx toolkits
    system_message = "You are an intelligent assistant that searches information about CAMEL-AI and generates PowerPoint slides."
    agent = ChatAgent(
        system_message=system_message,
        model=model,
        tools=browser_toolkit.get_tools() + pptx_toolkit.get_tools(),
    )

    # Define the task prompt
    task_prompt = "Search for information about CAMEL-AI and generate a PowerPoint presentation with the key points."

    # Reset the agent before starting
    agent.reset()

    # Step the agent with the task prompt
    response = agent.step(task_prompt)

    # Print the response content
    print(response.msgs[0].content)


if __name__ == "__main__":
    main()
