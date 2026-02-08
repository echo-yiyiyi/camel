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

## Diagnosis Reports

---

### Technique: Find SPECIFIC tool methods  
When a task specifies a particular tool like "brave search", locate the exact method or attribute by searching for its name in the toolkit directory, e.g.,  
`grep_search("search_brave", path="camel/toolkits")`  
This helps ensure the code agent uses the precise tool required rather than a generic collection of tools. For example, for a task requiring "bing search", search for `search_bing` in `camel/toolkits/search_toolkit.py` to find the exact tool method.

---

### Technique: Find MCPToolkit usage patterns  
To correctly use MCP tools, search for `MCPToolkit.create()` and `get_tools()` usage in `camel/toolkits/mcp_toolkit.py` or example scripts, and prefer `ChatAgent` instantiation with these tools rather than direct `MCPAgent` usage. This ensures valid config and proper connection handling. For example, for a task involving DeepWiki MCP server, search for `MCPToolkit.create` and `ChatAgent` usage in `examples/` or `camel/toolkits/mcp_toolkit.py`.

---

### Technique: Find dynamic computation of gold answers  
When a task requires computing a "gold answer" or "ground truth" using a specific library or method (e.g., sympy), search for examples that import and use that library to dynamically compute answers rather than hardcoding them. For instance, search for `import sympy` or `sp.solve` in datagen scripts or example directories to find canonical patterns of symbolic computation.

---

### Technique: Find explicit document download and vector retrieval patterns  
When a task requires working with documents (e.g., papers, datasets), explicitly locate usage of the toolkit's download methods (e.g., `download_papers`) rather than relying on search results or metadata. Then, build vector retrieval indexes from the actual downloaded document content (e.g., PDF text) using embedding and vector retriever classes.  
Search for examples combining the download method with vector retrieval classes like `VectorRetriever` and embedding models to ensure correct workflow.  
Example search commands:  
```
grep -r --include="*.py" "download_papers" .
grep -r --include="*.py" "VectorRetriever" .
grep -r --include="*.py" "ArxivToolkit" . | grep "download_papers"
```

---

### Technique: Find HUMAN interaction tools usage  
When a task requires "human interaction tools," explicitly search for and use the `HumanToolkit` rather than `MemoryToolkit`.  
Ensure example queries demonstrate triggering human interaction tools (e.g., prompting user input) instead of only memory management commands.  
Search for `HumanToolkit` usage examples and example queries that invoke human interaction tools to guide correct implementation.

Example search commands:  
```
grep -r --include="*.py" "HumanToolkit" examples/
grep -r --include="*.py" "human_toolkit" camel/toolkits/
```

---

# Summary of Added Techniques

### Technique: Find SPECIFIC tool methods  
When a task specifies a tool like "brave search", find the exact method:  
`grep_search("search_brave", path="camel/toolkits")`  

### Technique: Find MCPToolkit usage patterns  
To correctly use MCP tools, search for `MCPToolkit.create()` and `get_tools()` usage in `camel/toolkits/mcp_toolkit.py` or example scripts, and prefer `ChatAgent` instantiation with these tools rather than direct `MCPAgent` usage.

### Technique: Find dynamic computation of gold answers  
When a task requires computing a "gold answer" or "ground truth" using a specific library or method (e.g., sympy), search for examples that import and use that library to dynamically compute answers rather than hardcoding them. For example, search for `import sympy` or `sp.solve` in datagen or example scripts.

### Technique: Find explicit document download and vector retrieval patterns  
When a task involves document-based retrieval, locate usage of the toolkit's download methods (e.g., `download_papers`) and vector retrieval classes (e.g., `VectorRetriever`, `OpenAIEmbedding`) to build vector stores from actual document content rather than search metadata.  
Search for combined usage examples to ensure correct workflow.

### Technique: Find HUMAN interaction tools usage  
When a task requires "human interaction tools," explicitly search for and use the `HumanToolkit` rather than `MemoryToolkit`.  
Ensure example queries demonstrate triggering human interaction tools (e.g., prompting user input) instead of only memory management commands.  
Search for `HumanToolkit` usage examples and example queries that invoke human interaction tools to guide correct implementation.

## Diagnosis Reports