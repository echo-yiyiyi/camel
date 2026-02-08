from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import BrowserToolkit, PPTXToolkit
from camel.configs import ChatGPTConfig
from camel.types import ModelType

# Create the main model for the agent
model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Create models for the browser toolkit
web_agent_model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)
planning_agent_model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
    model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
)

# Initialize the BrowserToolkit
browser_toolkit = BrowserToolkit(
    headless=True,
    web_agent_model=web_agent_model,
    planning_agent_model=planning_agent_model,
    channel="chromium",
)

# Initialize the PPTXToolkit
pptx_toolkit = PPTXToolkit(working_directory="outputs")

# Combine tools from both toolkits
tools = [*browser_toolkit.get_tools(), *pptx_toolkit.get_tools()]

# Create the agent with both browser and pptx tools
agent = ChatAgent(
    system_message="You are a helpful assistant that searches for information about CAMEL-AI and generates slides in JSON format suitable for PPTXToolkit.",
    model=model,
    tools=tools,
)

# Step 1: Search for information about CAMEL-AI using browser tools
search_query = "Search for information about CAMEL-AI"
search_response = agent.step(search_query)
print("Search Response:")
print(search_response.msgs[0].content)

# Step 2: Generate slides based on the search results
slide_generation_prompt = f"Generate a presentation about CAMEL-AI based on the following information:\n{search_response.msgs[0].content}\n" \
                        "Create a title slide and several content slides with bullet points and tables as appropriate. " \
                        "Output the slides as a JSON list of dictionaries following the PPTXToolkit format."
slide_response = agent.step(slide_generation_prompt)
print("Slide Generation Response:")
print(slide_response.msgs[0].content)

# Optionally, you could save the generated slides to a PPTX file using the PPTXToolkit
# This depends on the format of the slide_response content and may require parsing JSON

