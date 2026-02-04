# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2025 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Generic code search toolkit for fast and precise file exploration.
All operations use TerminalToolkit shell commands for consistency.
This is a framework-agnostic version that can be used with any codebase.
"""

from pathlib import Path
from typing import List, Optional, Literal
from camel.toolkits import FunctionTool, TerminalToolkit


class GenericCodeSearchToolkit:
    """A toolkit optimized for fast code exploration and file search.

    This toolkit provides core capabilities using shell commands:
    1. glob_search: Fast file pattern matching using fd/find
    2. grep_search: Content search using ripgrep (rg)
    3. read_file: Read file contents using cat/head
    4. list_directory: List directory contents using ls

    Key design principles:
    - All operations use TerminalToolkit for consistency
    - Uses fast tools: fd for glob, rg for grep
    - Structured output for LLM consumption
    - Built-in exclusion of common noise directories
    - Framework-agnostic: works with any codebase
    """

    # Directories to always exclude from search
    DEFAULT_EXCLUDE_DIRS = {
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build', '.eggs',
        '*.egg-info', '.initial_env', 'venv', 'env',
        'task-script*',
    }

    def __init__(
        self,
        working_directory: str,
        exclude_dirs: Optional[set] = None,
        max_results: int = 100,
    ):
        """Initialize the GenericCodeSearchToolkit.

        Args:
            working_directory: The root directory for all searches.
            exclude_dirs: Additional directories to exclude from search.
            max_results: Maximum number of results to return per search.
        """
        self.working_dir = Path(working_directory).resolve()
        self.exclude_dirs = self.DEFAULT_EXCLUDE_DIRS.copy()
        if exclude_dirs:
            self.exclude_dirs.update(exclude_dirs)
        self.max_results = max_results

        # Initialize terminal toolkit
        self._terminal_toolkit = TerminalToolkit(
            working_directory=working_directory,
            clone_current_env=True,
            timeout=60.0,
        )
        self._shell_exec = self._terminal_toolkit.get_tools()[0]

    def _build_exclude_args(self, tool: str = 'fd') -> str:
        """Build exclude arguments for fd or rg commands."""
        args = []
        for excl in self.exclude_dirs:
            if tool == 'fd':
                # Quotes needed to prevent shell expansion of glob patterns
                args.append(f"-E '{excl}'")
            else:  # rg
                args.append(f"--glob '!{excl}' --glob '!**/{excl}/**'")
        return ' '.join(args)

    def glob_search(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_type: Optional[Literal['file', 'dir', 'any']] = 'file',
        max_results: Optional[int] = None,
    ) -> str:
        r"""Fast file pattern matching using fd command.

        Supports patterns like "**/*.py", "src/**/*.ts", "test_*.py".

        Args:
            pattern: Glob pattern to match (e.g., "**/*.py", "*.md").
            path: Subdirectory to search in (relative to working_dir).
                  If None, searches from working_dir.
            file_type: Filter by type - 'file', 'dir', or 'any'.
            max_results: Override default max results limit.

        Returns:
            A formatted string with matching file paths.

        Examples:
            - glob_search("**/*.py") - Find all Python files
            - glob_search("**/test_*.py") - Find all test files
            - glob_search("**/*config*.json") - Find config files
            - glob_search("*.md", path="docs") - Find markdown in docs/
        """
        search_dir = str(self.working_dir)
        if path:
            search_dir = str(self.working_dir / path)

        limit = max_results or self.max_results
        exclude_args = self._build_exclude_args('fd')

        # Build fd command
        # fd uses regex by default, convert glob pattern to fd-compatible
        # Remove leading **/ as fd searches recursively by default
        fd_pattern = pattern.replace('**/', '').replace('*', '.*')

        type_flag = ''
        if file_type == 'file':
            type_flag = '-t f'
        elif file_type == 'dir':
            type_flag = '-t d'

        # Build find exclude args (for fallback)
        find_exclude_args = ' '.join(
            f"-not -path '*/{excl}/*'" for excl in self.exclude_dirs
        )

        # Try fd/fdfind first (fdfind is the name on Debian/Ubuntu), fall back to find
        command = f'''
if command -v fd &> /dev/null; then
    fd {type_flag} {exclude_args} "{fd_pattern}" "{search_dir}" 2>/dev/null | head -n {limit}
elif command -v fdfind &> /dev/null; then
    fdfind {type_flag} {exclude_args} "{fd_pattern}" "{search_dir}" 2>/dev/null | head -n {limit}
else
    find "{search_dir}" -name "{pattern.replace('**/', '')}" {"-type f" if file_type == "file" else "-type d" if file_type == "dir" else ""} {find_exclude_args} 2>/dev/null | head -n {limit}
fi
'''

        try:
            result = self._shell_exec(
                id=f"glob_search_{pattern}",
                command=command.strip(),
            )

            # Check for empty result or TerminalToolkit's "no output" message
            if (not result or
                result.strip() == '' or
                'Command executed successfully (no output)' in result):
                return f"No files found matching pattern '{pattern}'"

            # Convert to relative paths and format output
            lines = result.strip().split('\n')
            rel_paths = []
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        if line.startswith(str(self.working_dir)):
                            rel_path = line[len(str(self.working_dir)):].lstrip('/')
                        else:
                            rel_path = line
                        rel_paths.append(rel_path)
                    except:
                        rel_paths.append(line)

            if not rel_paths:
                return f"No files found matching pattern '{pattern}'"

            # Sort alphabetically
            rel_paths.sort(key=str.lower)

            result_lines = [f"Found {len(rel_paths)} file(s) matching '{pattern}':"]
            result_lines.extend(rel_paths)
            return "\n".join(result_lines)

        except Exception as e:
            return f"Error during glob search: {e}"

    def grep_search(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob_filter: Optional[str] = None,
        ignore_case: bool = False,
        output_mode: Literal['files', 'content', 'count'] = 'files',
        context_lines: int = 0,
        max_results: Optional[int] = None,
    ) -> str:
        r"""Search file contents using ripgrep (rg) command.

        Args:
            pattern: Regex pattern to search for in file contents.
            path: Subdirectory to search in (relative to working_dir).
            glob_filter: Only search files matching this glob (e.g., "*.py").
            ignore_case: Whether to ignore case in pattern matching.
            output_mode:
                - 'files': Return only file paths (fastest)
                - 'content': Return matching lines with context
                - 'count': Return match counts per file
            context_lines: Lines of context around matches (for 'content' mode).
            max_results: Override default max results limit.

        Returns:
            Formatted search results based on output_mode.

        Examples:
            - grep_search("class.*Handler") - Find class definitions
            - grep_search("def process", glob_filter="*.py") - Find process functions
            - grep_search("import os", output_mode="files") - Files importing os
            - grep_search("TODO", output_mode="content", context_lines=2)
        """
        search_dir = str(self.working_dir)
        if path:
            search_dir = str(self.working_dir / path)

        limit = max_results or self.max_results
        exclude_args = self._build_exclude_args('rg')

        # Build rg command
        cmd_parts = ['rg', '--no-heading', '--with-filename']
        cmd_parts.append(exclude_args)

        if ignore_case:
            cmd_parts.append('-i')

        if glob_filter:
            cmd_parts.append(f'--glob "{glob_filter}"')

        if output_mode == 'files':
            cmd_parts.append('-l')
        elif output_mode == 'count':
            cmd_parts.append('-c')
        elif output_mode == 'content':
            cmd_parts.append('-n')
            if context_lines > 0:
                cmd_parts.append(f'-C {context_lines}')

        cmd_parts.append(f'-m {limit}')

        # Escape pattern for shell
        escaped_pattern = pattern.replace('"', '\\"')
        cmd_parts.append(f'"{escaped_pattern}"')
        cmd_parts.append(f'"{search_dir}"')

        command = ' '.join(cmd_parts) + f' 2>/dev/null | head -n {limit}'

        try:
            result = self._shell_exec(
                id=f"grep_search_{pattern}",
                command=command,
            )

            # Check for empty result or TerminalToolkit's "no output" message
            if (not result or
                result.strip() == '' or
                'Command executed successfully (no output)' in result):
                return f"No matches found for pattern '{pattern}'"

            # Convert to relative paths
            lines = result.strip().split('\n')
            rel_lines = []
            for line in lines:
                try:
                    if str(search_dir) in line:
                        line = line.replace(str(search_dir) + '/', '')
                    elif str(self.working_dir) in line:
                        line = line.replace(str(self.working_dir) + '/', '')
                except:
                    pass
                rel_lines.append(line)

            header = f"Search results for '{pattern}'"
            if glob_filter:
                header += f" in {glob_filter} files"
            header += f" ({len(rel_lines)} matches):"

            return header + "\n" + "\n".join(rel_lines)

        except Exception as e:
            return f"Error during grep search: {e}"

    def read_file(
        self,
        file_path: str,
    ) -> str:
        r"""Read file contents using shell commands.

        If the file is less than 200 lines, reads the whole file.
        Otherwise, reads the first 2000 lines.

        Args:
            file_path: Path to the file (relative to working_dir or absolute).

        Returns:
            File contents, or error message.

        Examples:
            - read_file("src/main.py") - Read source file
            - read_file("README.md") - Read markdown file
        """
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path

        # Use shell command to read file
        command = f'''
if [ ! -e "{path}" ]; then
    echo "Error: File '{file_path}' does not exist."
elif [ ! -f "{path}" ]; then
    echo "Error: '{file_path}' is not a file."
else
    lines=$(wc -l < "{path}")
    if [ $lines -lt 200 ]; then
        cat "{path}"
    else
        head -2000 "{path}"
    fi
fi
'''

        try:
            result = self._shell_exec(
                id=f"read_file_{file_path}",
                command=command.strip(),
            )
            return result
        except Exception as e:
            return f"Error reading file: {e}"

    def list_directory(
        self,
        path: Optional[str] = None,
        show_hidden: bool = False,
        max_items: int = 200,
    ) -> str:
        r"""List directory contents using ls command.

        Args:
            path: Directory path (relative to working_dir). If None, lists working_dir.
            show_hidden: Whether to show hidden files (starting with .).
            max_items: Maximum items to list.

        Returns:
            Formatted directory listing with [DIR] and [FILE] markers.
        """
        target_dir = str(self.working_dir)
        if path:
            target_dir = str(self.working_dir / path)

        hidden_flag = '-a' if show_hidden else ''

        # Use ls with -F flag to mark directories with /
        command = f'''
if [ ! -e "{target_dir}" ]; then
    echo "Error: Directory '{path or "."}' does not exist."
elif [ ! -d "{target_dir}" ]; then
    echo "Error: '{path or "."}' is not a directory."
else
    echo "Contents of {path or '.'}:"
    echo "----------------------------------------"
    ls -1F {hidden_flag} "{target_dir}" 2>/dev/null | head -n {max_items} | while read item; do
        if [[ "$item" == */ ]]; then
            echo "[DIR]  $item"
        else
            echo "[FILE] $item"
        fi
    done
fi
'''

        try:
            result = self._shell_exec(
                id=f"list_directory_{path}",
                command=command.strip(),
            )
            return result
        except Exception as e:
            return f"Error listing directory: {e}"

    def find_imports(
        self,
        module_name: str,
        ignore_case: bool = True,
    ) -> str:
        r"""Find files that import a specific module or class.

        This is useful for finding real usage examples of a module.
        Searches for both 'from X import' and 'import X' patterns.

        Args:
            module_name: Module or class name to search for.
            ignore_case: Whether to ignore case in matching.

        Returns:
            List of files that import the specified module.

        Examples:
            - find_imports("requests") - Find files using requests library
            - find_imports("json") - Find files importing json
        """
        # Build regex pattern for import statements
        pattern = f"(from\\s+\\S*{module_name}\\S*\\s+import|import\\s+\\S*{module_name})"

        return self.grep_search(
            pattern=pattern,
            glob_filter="*.py",
            ignore_case=ignore_case,
            output_mode='content',
        )

    def find_definition(
        self,
        name: str,
        definition_type: Literal['class', 'function', 'any'] = 'any',
        glob_filter: str = "**/*.py",
    ) -> str:
        r"""Find class or function definitions by name.

        This is a convenience method that constructs the appropriate regex
        pattern for finding Python definitions.

        Args:
            name: Name of the class/function to find (supports partial match).
            definition_type: 'class', 'function', or 'any'.
            glob_filter: File pattern to search in.

        Returns:
            File paths and line numbers where definition is found.

        Examples:
            - find_definition("UserHandler") - Find UserHandler class
            - find_definition("process", definition_type="function") - Find process functions
        """
        if definition_type == 'class':
            pattern = f"^class\\s+{name}"
        elif definition_type == 'function':
            pattern = f"^\\s*def\\s+{name}"
        else:
            pattern = f"(^class\\s+{name}|^\\s*def\\s+{name})"

        return self.grep_search(
            pattern=pattern,
            glob_filter=glob_filter,
            output_mode='content',
            context_lines=0,
        )

    def get_tools(self) -> List[FunctionTool]:
        """Get all tools as FunctionTool instances for use with ChatAgent.

        Returns:
            List of FunctionTool instances.
        """
        return [
            FunctionTool(self.glob_search),
            FunctionTool(self.grep_search),
            FunctionTool(self.read_file),
            FunctionTool(self.list_directory),
            FunctionTool(self.find_definition),
            FunctionTool(self.find_imports),
        ]


# Example usage and quick test
if __name__ == "__main__":
    # Use current directory as working directory
    import sys
    working_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    toolkit = GenericCodeSearchToolkit(working_directory=working_dir)

    print("=== Testing glob_search ===")
    print(toolkit.glob_search("**/*.py", max_results=10))
    print()

    print("=== Testing grep_search ===")
    print(toolkit.grep_search("class.*", glob_filter="*.py", output_mode='files', max_results=10))
    print()

    print("=== Testing find_definition ===")
    print(toolkit.find_definition("main", definition_type='function'))
    print()

    print("=== Testing list_directory ===")
    print(toolkit.list_directory())
