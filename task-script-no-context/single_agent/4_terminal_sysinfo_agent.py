import os
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.toolkits import TerminalToolkit
from camel.types import ModelPlatformType, ModelType
from camel.interpreters import JupyterKernelInterpreter

# Setup workspace directory
workspace_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

# Create terminal tools
terminal_tools = TerminalToolkit(working_directory=workspace_dir).get_tools()

# Create model config and model
model_config = ChatGPTConfig(temperature=0.0).as_dict()
model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
    model_config_dict=model_config,
)

# Create ChatAgent with terminal tools
sys_msg = "You are a system assistant with terminal access."
terminal_agent = ChatAgent(system_message=sys_msg, model=model, tools=terminal_tools)
terminal_agent.reset()

# Ask the agent to retrieve system information
user_prompt = "Retrieve system information such as kernel version and OS details."
response = terminal_agent.step(user_prompt)

# Extract the terminal command output from tool calls
tool_calls = response.info.get('tool_calls', [])
sys_info = ""
for call in tool_calls:
    if call.tool_name == 'shell_exec':
        sys_info += call.result + "\n"

# Create a Python interpreter agent
interpreter = JupyterKernelInterpreter(require_confirm=False, print_stdout=True, print_stderr=True)

# Python code to print the retrieved system information
python_code = f"""
sys_info = '''{sys_info.strip()}'''
print('System Information Retrieved from Terminal Agent:')
print(sys_info)
"""

# Run the Python code in the interpreter
result = interpreter.run(python_code, "python")
print(result)
