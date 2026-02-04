import os
import json
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types import ModelPlatformType, ModelType

def create_browser_slide_agent(openai_api_key: str, pexels_api_key: str = ""):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PEXELS_API_KEY"] = pexels_api_key

    base_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={"temperature": 0.0},
    )

    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={"temperature": 0.0},
    )

    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict={"temperature": 0.0},
    )

    browser_toolkit = BrowserToolkit(
        headless=True,
        web_agent_model=web_agent_model,
        planning_agent_model=planning_agent_model,
        channel="chromium",
    )

    pptx_toolkit = PPTXToolkit()

    agent = ChatAgent(
        system_message="You are a helpful assistant that can browse the web and create presentation slides.",
        model=base_model,
        tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
    )

    search_prompt = (
        "Search for detailed information about CAMEL-AI, including its features, "
        "capabilities, and usage examples. Summarize the key points clearly."
    )
    search_response = agent.step(search_prompt)
    print("Search Result:")
    print(search_response.msgs[0].content)

    slide_prompt = (
        f"Create a presentation about CAMEL-AI based on the following information:\n" 
        f"{search_response.msgs[0].content}\n" 
        "Include a title slide, bullet point slides, and a table slide with relevant data. "
        "Use Markdown formatting and include image keywords where appropriate. "
        "Generate the slide content in JSON format suitable for PPTXToolkit."
    )
    slide_response = agent.step(slide_prompt)
    print("Slide JSON:")
    print(slide_response.msgs[0].content)

    pptx_result = pptx_toolkit.create_presentation(slide_response.msgs[0].content, "camel_ai_presentation.pptx")
    print(f"PPTX creation result: {pptx_result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent with browser and PPT tools for CAMEL-AI info and slides")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--pexels_api_key", type=str, default="", help="Pexels API key for images (optional)")
    args = parser.parse_args()

    create_browser_slide_agent(args.openai_api_key, args.pexels_api_key)
