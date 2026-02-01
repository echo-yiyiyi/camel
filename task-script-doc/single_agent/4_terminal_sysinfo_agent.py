# Agent script to retrieve system information using terminal and code execution tools
from camel.agents import ChatAgent
from camel.toolkits.terminal_toolkit.terminal_toolkit import TerminalToolkit
from camel.toolkits.code_execution import CodeExecutionToolkit
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Initialize terminal toolkit
terminal_toolkit = TerminalToolkit()

# Initialize code execution toolkit with internal python interpreter
code_execution_toolkit = CodeExecutionToolkit(sandbox="internal_python")

# Create the agent with both toolkits
agent = ChatAgent(
    
    "You are a system info retrieval agent.",
    model=ModelType.GPT_4O_MINI,
)

# Prompt to retrieve system information and print it in Python interpreter
prompt = '''
Use the terminal toolkit to run system commands to retrieve system information such as OS details, CPU info, memory usage, and disk usage.
Then use the code execution toolkit to print the retrieved information in the Python interpreter.
'''

# Run the agent with the prompt
response = agent.step(prompt)
print(response.msgs[0].content)
