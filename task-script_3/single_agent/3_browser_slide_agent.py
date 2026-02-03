from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types import ModelType, ModelPlatformType


def main():
    # Create the model
    model = ModelFactory.create(
        model_platform=ModelPlatformType.TOGETHER,
        model_type=ModelType.TOGETHER_LLAMA_3_1_8B
    )

    # Create the browser toolkit
    browser_toolkit = BrowserToolkit()

    # Create the PPTX toolkit
    pptx_toolkit = PPTXToolkit(working_directory="./pptx_outputs")

    # Combine tools from both toolkits
    tools = browser_toolkit.get_tools() + pptx_toolkit.get_tools()

    # Create the agent with combined tools
    agent = ChatAgent(model=model, tools=tools, max_iteration=10)

    # Define the task prompt
    prompt = (
        "Search for information about CAMEL-AI using the browser tools, "
        "then generate a PowerPoint presentation summarizing the key points. "
        "Save the presentation as 'camel_ai_presentation.pptx'."
    )

    # Run the agent synchronously with step()
    response = agent.step(prompt)

    print("Agent response:")
    if response.msgs:
        print(response.msgs[0].content)
    else:
        print("No response message.")


if __name__ == "__main__":
    main()
