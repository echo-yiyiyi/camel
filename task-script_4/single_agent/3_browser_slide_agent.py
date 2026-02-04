from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.types import ModelPlatformType, ModelType

# Create models for the agent and browser/planning agents
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

web_agent_model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

planning_agent_model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)

# Initialize BrowserToolkit with the browser models
browser_toolkit = BrowserToolkit(
    headless=True,
    web_agent_model=web_agent_model,
    planning_agent_model=planning_agent_model,
)

# Initialize PPTXToolkit for slide generation
pptx_toolkit = PPTXToolkit(
    working_directory="./pptx_outputs",
)

# Create a ChatAgent with both BrowserToolkit and PPTXToolkit tools
agent = ChatAgent(
    system_message="You are a helpful assistant that can browse the web and create PowerPoint presentations.",
    model=model,
    tools=[*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()],
)

# Step 1: Use the agent to browse and gather information about CAMEL-AI
browse_task = "Search for detailed information about CAMEL-AI, including its purpose, features, community, and impact."
start_url = "https://www.camel-ai.org/"  # Assuming this is the official site

print("Browsing and gathering information about CAMEL-AI...")
browse_response = agent.step(f"browse_url('{browse_task}', '{start_url}')")
print(browse_response.msgs[0].content)

# Step 2: Use the agent to generate a PowerPoint presentation about CAMEL-AI
presentation_prompt = (
    "Create a PowerPoint presentation about CAMEL-AI based on the information you gathered. "
    "Include slides on introduction, key features, applications, community, and future outlook. "
    "Use appropriate templates and formatting."
)

print("Generating PowerPoint presentation about CAMEL-AI...")
pptx_response = agent.step(presentation_prompt)
print(pptx_response.msgs[0].content)

print("Script execution completed. Check the pptx_outputs directory for the generated presentation.")
