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


## New Technique: Prefer Specific Toolkit Functions Over Generic get_tools()

When a task requires a specific tool or search function (e.g., "search duckduckgo"), the agent should:

- Identify and use the exact specific function from the relevant toolkit (e.g., `SearchToolkit().search_duckduckgo`).
- Avoid using the generic `get_tools()` method that returns all tools from the toolkit.
- Always prefer the most specific tool relevant to the task to improve precision and efficiency.

### Rationale
Using the specific function directly ensures that the agent uses the intended tool with the correct interface and behavior. It avoids unnecessary overhead and ambiguity caused by including all tools via `get_tools()`.

### Pattern
- When the task or ground truth mentions a specific search or tool (e.g., DuckDuckGo), search for and use the exact function (e.g., `search_duckduckgo`) from the toolkit.
- Do NOT use the generic `get_tools()` that returns all tools.

### Example
```python
from camel.toolkits.search_toolkit import SearchToolkit

search_toolkit = SearchToolkit()
result = search_toolkit.search_duckduckgo(query="example query")
```

This pattern should be followed in code agents and task implementations to ensure systematic improvement and adherence to task requirements.

---

## New Technique: Systematic Improvement for Human Interaction Tools

- GENERAL rule for code agent's system prompt:
  - "When a task requests 'human interaction tools', search for toolkits with 'human' in their name (e.g., `HumanToolkit`), not just memory or generic toolkits."
- Pattern to learn:
  - "If the task mentions a specific type of tool (e.g., 'human interaction'), always search for that keyword in toolkit files and examples, not just the word 'toolkit' or the most common toolkit."

## New Technique: General Search Technique for CAMEL Codebase

- GENERAL search technique to document in CAMEL.md:
  - "When a task requires a specific type of toolkit (e.g., 'human interaction'), always perform a glob/grep search for that keyword (e.g., 'human') in the toolkits directory and in example scripts."
- Pattern to add for similar use cases:
  - "Add example scripts and documentation for each major toolkit (e.g., `HumanToolkit`) in the examples/toolkits/ directory, and ensure they are discoverable by keyword."

This technique helps ensure that code agents and developers systematically find and use the correct toolkits and examples for specialized tool types.

---
