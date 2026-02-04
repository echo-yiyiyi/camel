# Example Project Context Template

This is a template for creating project-specific context files.
Copy this file and customize it for your project.

## Repository Structure

### Core Modules
- `src/` - Main source code
- `lib/` - Library code
- `tests/` - Test files
- `examples/` - Usage examples
- `docs/` - Documentation

### Key Directories
- `src/api/` - API handlers
- `src/models/` - Data models
- `src/services/` - Business logic
- `src/utils/` - Utility functions

## Project-Specific Search Techniques

### Finding Configuration
```python
# Find environment configuration
grep_search("API_KEY|SECRET", glob_filter="*.env*")
grep_search("config", glob_filter="*.json")
```

### Finding Entry Points
```python
# Find main entry points
grep_search("if __name__.*main", glob_filter="*.py")
grep_search("def main", glob_filter="*.py")
```

### Finding Tests
```python
# Find test files
glob_search("**/test_*.py")
glob_search("**/*_test.py")
```

### Reading Documentation
```python
# Find documentation files
glob_search("**/*.md", path="docs")
glob_search("**/README.md")

# Search for specific topics in docs
grep_search("authentication", glob_filter="*.md", path="docs")
grep_search("api", glob_filter="*.md", path="docs", ignore_case=True)

# Read specific documentation
read_file("docs/getting_started.md")
read_file("docs/api_reference.md")
```
Documentation is especially useful for understanding API usage patterns and configuration options.

## Code Writing Rules

### Import Patterns
```python
# Standard project imports
from src.models import User
from src.services import UserService
from src.utils import helper
```

### Configuration
```python
# Load configuration
import os
API_KEY = os.environ.get("API_KEY")
```

### Error Handling
```python
# Standard error handling pattern
try:
    result = do_something()
except CustomError as e:
    logger.error(f"Error: {e}")
    raise
```

## Common Classes and Their Locations
| Class | File |
|-------|------|
| User | src/models/user.py |
| UserService | src/services/user_service.py |
| APIHandler | src/api/handler.py |

## Output Format for Explore Agent

```
## Documentation
- docs/getting_started.md
  - Project overview and setup instructions
- docs/api/users.md
  - User API documentation with examples

## Examples
- examples/basic_usage.py
  - Shows: Basic API usage
  - Key functions: create_user(), get_user()

## Implementation
- src/services/user_service.py
  - Classes: UserService
  - Methods: create, get, update, delete

## Tests
- tests/test_user_service.py
  - Test cases for UserService
```
