# System prompt addition for Code Agent
# Rule: When a task requires a value to be 'computed via X' (e.g., sympy, numpy, etc.), always search for and read example scripts that use that library for similar computations, and copy the computation pattern exactly. Do NOT hardcode values that are supposed to be computed.

# Pattern to recognize:
# When the task says "gold answer if computed via sympy", the agent should search for and read examples that use sympy to compute answers, and use that code directly in the script.

# This pattern is now documented in code-agent/generic/CAMEL.md for future runs.

# Append to code-agent/generic/CAMEL.md:
append_text = '''

### Technique: Use Example Scripts for Computation When Task Requires Computed Values

- When a task requires a value to be "computed via X" (e.g., sympy, numpy, etc.), the agent should always search for and read example scripts that use that library for similar computations.
- The agent should copy the computation pattern exactly from the examples.
- The agent must NOT hardcode values that are supposed to be computed.

Example pattern to recognize:
- Task says "gold answer if computed via sympy"
- Agent searches for and reads examples using sympy for similar computations
- Agent uses that code pattern directly in the script

This rule ensures correctness and consistency in computation tasks.
'''

# Read existing CAMEL.md content
with open('code-agent/generic/CAMEL.md', 'r') as f:
    content = f.read()

# Append if not already present
if '### Technique: Use Example Scripts for Computation When Task Requires Computed Values' not in content:
    with open('code-agent/generic/CAMEL.md', 'a') as f:
        f.write(append_text)

print("Appended computation pattern rule to code-agent/generic/CAMEL.md")
