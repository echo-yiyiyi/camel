# CAMEL Framework Context

This file contains CAMEL-specific context information for the explore and code agents.  
Load this file before running tasks on the CAMEL repository.

## Repository Structure

### Core Modules
- `camel/agents/` - Agent implementations (ChatAgent is the base)
- `camel/models/` - LLM provider integrations (50+)
- `camel/memories/` - Memory systems (ChatHistoryMemory, LongtermAgentMemory, VectorDBMemory)
- `camel/configs/` - Model configuration classes
- `camel/toolkits/` - Tool integrations (50+)
- `camel/types/` - Enums (ModelType, RoleType)

### Other Directories
- `examples/` - Usage examples
- `examples/models/` - Model usage examples
- `examples/toolkits/` - Toolkit examples
- `examples/memories/` - Memory usage examples
- `test/` - Test files
- `docs/` - Documentation


### Import Patterns
```python
# Standard imports
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelType

# Create model (platform is auto-inferred from ModelType)
model = ModelFactory.create(
    model_type=ModelType.GPT_4O_MINI,
)

# Create agent
agent = ChatAgent(system_message="You are helpful.", model=model)
response = agent.step("Hello")
```

### Common Classes and Their Locations
| Class | File |
|-------|------|
| ChatAgent | camel/agents/chat_agent.py |
| ModelFactory | camel/models/model_factory.py |
| FunctionTool | camel/toolkits/function_tool.py |
| ChatHistoryMemory | camel/memories/chat_history_memory.py |
| LongtermAgentMemory | camel/memories/agent_memories.py |
| VectorDBMemory | camel/memories/vector_db_memory.py |

## Code Writing Rules

### Tool Selection
- When a task explicitly requires a specific tool (e.g., "tools to search duckduckgo"), use the exact tool attribute or method provided by the toolkit (e.g., `SearchToolkit().search_duckduckgo`) rather than generic methods that return multiple tools (e.g., `get_tools()`).
- Avoid using generic tool retrieval methods like `get_tools()` when the task specifies a particular tool to ensure precise compliance with task requirements.

### Model Creation
- Use `ModelFactory.create()` with the appropriate `ModelType` enum to instantiate models.
- When needed, pass model-specific configuration dictionaries (e.g., `model_config_dict=GeminiConfig(temperature=0.2).as_dict()`).

### Agent Creation
- Instantiate `ChatAgent` with required parameters such as `model`, optional `tools` list, and `system_message` to set context.
- Pass tools as a list of tool instances, not as a generic toolkit object.

## Common Mistakes to Avoid

- Do NOT use `SearchToolkit().get_tools()` when the task requires a specific search tool; this returns all tools and violates task specificity.
- Do NOT ignore explicit tool requirements mentioned in the task description; always prioritize exact tool usage.
- Avoid mixing multiple tools when only one specific tool is requested.

## Code Examples

### Correct Usage of a Specific Tool

```python
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import SearchToolkit
from camel.types import ModelType
from camel.configs import GeminiConfig

# Instantiate Gemini model with configuration
model = ModelFactory.create(
    model_type=ModelType.GEMINI_2_5_PRO,
    model_config_dict=GeminiConfig(temperature=0.0).as_dict()
)

# Get the specific DuckDuckGo search tool
search_tool = SearchToolkit().search_duckduckgo

# Create agent with the specific tool only
agent = ChatAgent(model=model, tools=[search_tool])

# Ask a question
response = agent.step("Search CAMEL-AI framework on DuckDuckGo and provide summary.")
print(response.msgs[0].content)
```

### Incorrect Usage to Avoid

```python
# Avoid this when a specific tool is required
search_toolkit = SearchToolkit()
tools = search_toolkit.get_tools()  # returns all tools, not just DuckDuckGo

agent = ChatAgent(model=model, tools=tools)  # Violates task requirement
```

## Search Techniques for Tool Identification

- When a task mentions a specific tool by name, perform targeted searches (e.g., grep) for function or attribute definitions matching that tool name within the `camel/toolkits/` directory.
- Prioritize reading example scripts that demonstrate the exact tool usage rather than generic toolkit usage to understand correct usage patterns.

---

This will help ensure generated code precisely matches task requirements, especially regarding tool usage.