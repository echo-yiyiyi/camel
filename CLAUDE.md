# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAMEL (Communicative Agents for AI Society Study) is an open-source framework for building multi-agent AI systems. It supports large-scale agent simulations (up to 1M agents), stateful memory, and 50+ model providers.

## Development Setup

```bash
# Clone and setup
git clone https://github.com/camel-ai/camel.git
cd camel
pip install uv
uv venv .venv --python=3.10
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[all, dev, docs]"
pre-commit install
```

## Common Commands

### Testing
```bash
pytest .                           # Default mode (skips very slow tests)
pytest --fast-test-mode .          # Fast: no LLM inference tests
pytest --llm-test-only .           # LLM tests only
pytest --full-test-mode .          # All tests including very slow
pytest --very-slow-test-only .     # Only very slow tests
```

### Linting & Formatting
```bash
make ruff                          # Run ruff linter
make ruff-fix                      # Auto-fix with ruff
make mypy                          # Type checking
make pre-commit                    # Run all pre-commit hooks
make format                        # Format code (yapf + isort + ruff)
```

### Build
```bash
make install                       # Standard install
make install-editable              # Editable install for development
make build                         # Build distribution
```

### Dependencies
```bash
uv lock                            # Sync dependencies after modifying pyproject.toml
```

## Architecture

### Core Modules

- **`camel/agents/`**: Agent implementations. `ChatAgent` is the foundation; others inherit from it.
- **`camel/models/`**: 50+ LLM provider integrations. Use `ModelFactory.create()` to instantiate.
- **`camel/configs/`**: Model configuration classes (e.g., `ChatGPTConfig`, `AnthropicConfig`).
- **`camel/toolkits/`**: 50+ tool integrations. Wrap functions with `FunctionTool`.
- **`camel/messages/`**: Message types (`BaseMessage`, `FunctionCallingMessage`).
- **`camel/types/`**: Enums (`ModelPlatformType`, `ModelType`, `RoleType`).
- **`camel/memories/`**: Memory systems for agent state.
- **`camel/runtimes/`**: Execution environments (Docker, HTTP, Daytona).
- **`camel/societies/`**: Multi-agent coordination (role-playing, workforce).
- **`camel/datagen/`**: Synthetic data generation (CoT, self-instruct).

### Key Patterns

**Creating an agent:**
```python
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)
agent = ChatAgent(system_message="You are helpful.", model=model)
response = agent.step("Hello")  # or await agent.astep("Hello")
```

**Adding tools to an agent:**
```python
from camel.agents import ChatAgent
from camel.toolkits import FunctionTool

tool = FunctionTool(func=my_function, description="What it does")
agent = ChatAgent(model=model, tools=[tool])
```

### Test Markers
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.very_slow`: Skipped by default
- `@pytest.mark.model_backend`: Requires LLM API access
- `@pytest.mark.heavy_dependency`: Requires heavy optional dependencies

## Code Conventions

- **Line length**: 79 characters (ruff)
- **Docstrings**: Google style (`r"""` prefix)
- **Naming**: No abbreviations (e.g., `message_window_size` not `msg_win_sz`)
- **Toolkit functions**: Use prefix pattern (`github_create_issue()` not `create_issue()`)
- **Logging**: Use `logger` from `camel.logger`, not `print`
- **Async**: Always provide both `step()` and `astep()` methods for agents
- **License**: Apache 2.0 header required on all files

## Environment Variables

Copy `.env.example` to `.env` and configure API keys:
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, etc.
- See `.env.example` for the full list of supported providers and services.
