"""
Agent script to generate slides about CAMEL-AI using the agent's model and PPT tools.
"""
import json
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import PPTXToolkit
from camel.types import ModelPlatformType, ModelType


def main():
    # Define the task prompt
    task_prompt = "Generate a slide presentation summarizing key information about CAMEL-AI in JSON format."

    # Create the PPTX toolkit instance
    pptx_toolkit = PPTXToolkit()

    # Create the model for the agent
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4_1,
        model_config_dict={"temperature": 0.3, "max_tokens": 4000},
    )

    # Create the agent with PPTX toolkit only
    agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Assistant",
            content="You are an assistant with PPTX tools to generate slides in JSON format.",
        ),
        model=model,
        tools=pptx_toolkit.get_tools(),
    )

    # Ask the agent to generate slide content
    user_message = BaseMessage.make_user_message(
        role_name="User",
        content=task_prompt,
    )

    response = agent.step(user_message)

    # Extract slide content JSON from response
    slides_json = response.msgs[0].content

    # Validate and parse JSON
    import json
    try:
        slides_data = json.loads(slides_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(slides_json)
        return

    # Convert back to JSON string for PPTX toolkit
    slides_json_str = json.dumps(slides_data)

    # Generate PPTX file
    pptx_filename = "CAMEL_AI_Presentation.pptx"
    pptx_result = pptx_toolkit.create_presentation(slides_json_str, pptx_filename)

    print(pptx_result)


if __name__ == "__main__":
    main()
