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
- `camel/types/` - Enums (ModelPlatformType, ModelType, RoleType)

### Other Directories
- `examples/` - Usage examples
- `examples/models/` - Model usage examples
- `examples/toolkits/` - Toolkit examples
- `examples/memories/` - Memory usage examples
- `test/` - Test files
- `docs/` - Documentation

## CAMEL-Specific Search Techniques

### Technique: Find EXACT enum values for models/tools
When task specifies a model like "Llama-3.1-8B-Instruct", search for the exact ModelType enum:
```python
# Search with partial match (enum names use underscores, not hyphens)
grep_search("LLAMA_3_1.*8B", path="camel/types/enums.py")
grep_search("LLAMA_3_1", path="camel/types/enums.py")
```

### ModelType Naming Convention
- No "_INSTRUCT" suffix usually (e.g., LLAMA_3_1_8B not LLAMA_3_1_8B_INSTRUCT)
- Use underscores not hyphens (e.g., LLAMA_3_1 not LLAMA-3.1)
- ModelPlatformType and ModelType MUST match according to enums.py

### Technique: Find SPECIFIC tool methods
When task specifies a tool like "brave search", find the exact method:
```python
grep_search("search_brave", path="camel/toolkits")
grep_search("def search_brave")
```
Don't use get_tools() if task asks for a specific tool - use the specific method directly.

### Technique: Read __init__.py to understand module exports
```python
read_file("camel/toolkits/__init__.py")
read_file("camel/models/__init__.py")
read_file("camel/memories/__init__.py")
```
Class names often don't match file names (e.g., `LongtermAgentMemory` is in `agent_memories.py`, not `longterm_memory.py`).

### Technique: Read documentation for API usage and tutorials
CAMEL has extensive documentation in the `docs/` directory:
```python
# Find all documentation files
glob_search("**/*.md", path="docs")

# Search for specific topics
grep_search("ChatAgent", glob_filter="*.md", path="docs")
grep_search("memory", glob_filter="*.md", path="docs", ignore_case=True)

# Key documentation files
read_file("docs/key_modules/agents.md")      # Agent usage guide
read_file("docs/key_modules/models.md")      # Model configuration
read_file("docs/key_modules/memories.md")    # Memory systems
read_file("docs/key_modules/tools.md")       # Toolkit usage
read_file("docs/cookbooks/create_your_first_agent.md")  # Getting started
```

### Search Examples for CAMEL
```python
# Find model examples
glob_search("**/*llama*.py")
find_imports("LlamaModel")

# Find toolkit examples
glob_search("**/*weather*.py")
find_imports("WeatherToolkit")

# Find memory implementations
grep_search("class.*Memory", path="camel/memories")
```

## Code Writing Rules for CAMEL

### Model Selection
- If task does NOT specify a model -> Use `ModelPlatformType.DEFAULT` and `ModelType.DEFAULT`
- Task says "Llama-3.1-8B-Instruct" -> Use `ModelType.LLAMA_3_1_8B` EXACTLY
- Do NOT use LLAMA_3_2_3B as "closest alternative"
- Do NOT guess enum names - verify from enums.py or examples

### Tool Selection
- Task says "brave search" -> Use `search_toolkit.search_brave` EXACTLY
- Do NOT use `get_tools()` which returns ALL tools
- Do NOT substitute with other search engines

### Import Patterns
```python
# Standard imports
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Create model
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
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

## Output Format for Explore Agent

For each file, list the core classes/functions inside:

```
## Documentation (check for API usage and tutorials)
- docs/key_modules/agents.md
  - Explains ChatAgent usage and configuration
- docs/cookbooks/create_your_first_agent.md
  - Step-by-step tutorial for creating agents

## Examples (MOST IMPORTANT)
- examples/models/llama_model_example.py
  - Shows: ModelFactory.create() with LLAMA model
  - Key usage: model_platform=ModelPlatformType.TOGETHER, model_type=ModelType.LLAMA_3_1_8B

## Implementation
- camel/memories/agent_memories.py
  - Classes: AgentMemory, ChatHistoryMemory, VectorDBMemory, **LongtermAgentMemory**
  - Note: LongtermAgentMemory is for persistent memory across sessions

- camel/models/togetherai_model.py
  - Classes: TogetherAIModel
  - Config: TogetherAIConfig

## Enums (for exact values)
- camel/types/enums.py:317 - LLAMA_3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

## Tests
- test/models/test_qwen_model.py - test cases showing parameter usage
```
