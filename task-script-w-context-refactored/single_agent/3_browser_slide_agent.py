from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelType
from camel.toolkits.browser_toolkit import BrowserToolkit
from camel.toolkits.pptx_toolkit import PPTXToolkit

# Create model
model = ModelFactory.create(model_type=ModelType.DEFAULT)

# Create toolkits
browser_toolkit = BrowserToolkit(headless=True, web_agent_model=model)
pptx_toolkit = PPTXToolkit(working_directory="./pptx_outputs")

# Combine tools from both toolkits
combined_tools = browser_toolkit.get_tools() + pptx_toolkit.get_tools()

# System message for the agent
system_message = '''You are an intelligent assistant with browser and PowerPoint presentation creation capabilities.
Your task is to search for information about CAMEL-AI using the browser tools and then generate a PowerPoint presentation summarizing the information using the PPT tools.
Use the browser tools to gather detailed and accurate information.
Then use the PPT tools to create slides with appropriate titles, bullet points, and images if relevant.
'''

# Create the agent
agent = ChatAgent(system_message=system_message, model=model, tools=combined_tools)

# Task content
task_content = "Search for information about CAMEL-AI and generate a PowerPoint presentation summarizing the key points."

# Run the agent
response = agent.step(task_content)

# Print the response content
print(response.msgs[0].content)
