from camel.agents import ChatAgent
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

def create_browser_slide_agent():
    # Create models for browser agents
    web_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
    )
    planning_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
    )
    chat_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4,
    )

    # Initialize toolkits
    browser_toolkit = BrowserToolkit(
        web_agent_model=web_agent_model,
        planning_agent_model=planning_agent_model,
    )
    pptx_toolkit = PPTXToolkit()

    # Create the chat agent with both toolkits
    agent = ChatAgent(
        model=chat_agent_model,
        tools=browser_toolkit.get_tools() + pptx_toolkit.get_tools(),
        
        
    )
    return agent

def main():
    browser_toolkit = BrowserToolkit()
    agent = create_browser_slide_agent()

    # Task prompt to search for CAMEL-AI information and generate slides
    task_prompt = (
        "Search the web for information about CAMEL-AI, "
        "summarize the key points, and generate a PowerPoint presentation "
        "with a title slide and content slides covering the main features, "
        "use cases, and benefits of CAMEL-AI."
    )

    # Use browser toolkit to search starting from Google
    start_url = "https://www.google.com"

    # Agent performs browsing and information gathering
    
    # Use browse_url method of browser_toolkit directly
    search_result = browser_toolkit.browse_url(
        "browse_url",
        
        start_url=start_url,
        
    )

    # Prepare content for PPTX creation
    pptx_content = f"""
    [
        {{
            "title": "CAMEL-AI Overview",
            "subtitle": "Generated presentation about CAMEL-AI"
        }},
        {{
            "heading": "Summary",
            "bullet_points": {search_result.get('summary', ['Information about CAMEL-AI not found'])}
        }}
    ]
    """

    # Create presentation slides from the search result
    pptx_toolkit = agent.get_toolkit("PPTXToolkit")
    pptx_filename = "camel_ai_presentation.pptx"
    pptx_toolkit.create_presentation(content=pptx_content, filename=pptx_filename)

    print(f"Presentation saved to {pptx_filename}")

if __name__ == "__main__":
    main()
