import os
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.types import ModelType

# Define workspace directory
workspace_dir = os.path.abspath("./workspace")

# System message for the agent
sys_msg = (
    "You are a helpful assistant with access to terminal and Python interpreter tools. "
    "You can retrieve system information using terminal commands and then print it using Python code execution."
)

# Initialize TerminalToolkit
terminal_toolkit = TerminalToolkit(working_directory=workspace_dir)
terminal_tools = terminal_toolkit.get_tools()

# Initialize CodeExecutionToolkit with Jupyter interpreter
code_exec_toolkit = CodeExecutionToolkit(sandbox="jupyter", verbose=True)
code_exec_tools = code_exec_toolkit.get_tools()

# Combine tools
tools = terminal_tools + code_exec_tools

# Model config
model_config_dict = ChatGPTConfig(temperature=0.0).as_dict()

# Create model
model = ModelFactory.create(model_type=ModelType.DEFAULT, model_config_dict=model_config_dict)

# Create agent
agent = ChatAgent(system_message=sys_msg, model=model, tools=tools)
agent.reset()

# Step 1: Retrieve system information using terminal command
usr_msg_1 = "Retrieve system information such as OS name, version, and kernel info using terminal commands."
response_1 = agent.step(usr_msg_1)
print("Terminal tool calls and results:")
print(response_1.info['tool_calls'])

# Extract the system info from the tool call results
# We assume the last shell_exec call result contains the system info
sys_info = ""
for call in response_1.info['tool_calls']:
    if call.tool_name == 'shell_exec' and call.result.strip():
        sys_info = call.result.strip()

# Step 2: Print the retrieved system information in Python interpreter
usr_msg_2 = f"Print the following system information in Python interpreter:\n\n{sys_info}"
response_2 = agent.step(usr_msg_2)
print("Python interpreter tool calls and results:")
print(response_2.info['tool_calls'])

