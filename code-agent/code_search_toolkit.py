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
Optimized code search toolkit for fast and precise file exploration.
Inspired by Claude Code's internal Glob/Grep/Read tools.
"""

import re
import subprocess
import fnmatch
from pathlib import Path
from typing import List, Optional, Literal
from camel.toolkits import FunctionTool


class CodeSearchToolkit:
    """A toolkit optimized for fast code exploration and file search.

    This toolkit provides three core capabilities:
    1. glob_search: Fast file pattern matching (like `find` but faster)
    2. grep_search: Content search with regex support (uses ripgrep if available)
    3. read_file: Read file contents with line limits

    Key design principles:
    - Native Python operations where possible (faster than shell)
    - Ripgrep for content search (much faster than grep)
    - Structured output for LLM consumption
    - Built-in exclusion of common noise directories
    """

    # Directories to always exclude from search
    DEFAULT_EXCLUDE_DIRS = {
        'node_modules', '.venv', '.git', '__pycache__', '.tox',
        '.mypy_cache', '.pytest_cache', 'dist', 'build', '.eggs',
        '*.egg-info', '.initial_env', 'venv', 'env'
    }

    def __init__(
        self,
        working_directory: str,
        exclude_dirs: Optional[set] = None,
        max_results: int = 100,
    ):
        """Initialize the CodeSearchToolkit.

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

        # Check if ripgrep is available
        self._has_ripgrep = self._check_ripgrep()

    def _check_ripgrep(self) -> bool:
        """Check if ripgrep (rg) is available on the system."""
        try:
            subprocess.run(['rg', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded based on exclude patterns."""
        parts = path.parts
        for part in parts:
            for pattern in self.exclude_dirs:
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False

    def glob_search(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_type: Optional[Literal['file', 'dir', 'any']] = 'file',
        max_results: Optional[int] = None,
    ) -> str:
        r"""Fast file pattern matching using glob patterns.

        This is much faster than shell `find` for most use cases.
        Supports patterns like "**/*.py", "src/**/*.ts", "test_*.py".

        Args:
            pattern: Glob pattern to match (e.g., "**/*.py", "*.md").
            path: Subdirectory to search in (relative to working_dir).
                  If None, searches from working_dir.
            file_type: Filter by type - 'file', 'dir', or 'any'.
            max_results: Override default max results limit.

        Returns:
            A formatted string with matching file paths, sorted by
            modification time (most recent first).

        Examples:
            - glob_search("**/*.py") - Find all Python files
            - glob_search("**/test_*.py") - Find all test files
            - glob_search("**/*agent*.py") - Find files with 'agent' in name
            - glob_search("*.md", path="docs") - Find markdown in docs/
        """
        search_dir = self.working_dir
        if path:
            search_dir = self.working_dir / path

        if not search_dir.exists():
            return f"Error: Directory '{search_dir}' does not exist."

        limit = max_results or self.max_results
        matches = []

        try:
            for match in search_dir.glob(pattern):
                # Skip excluded directories
                if self._should_exclude(match.relative_to(self.working_dir)):
                    continue

                # Filter by type
                if file_type == 'file' and not match.is_file():
                    continue
                elif file_type == 'dir' and not match.is_dir():
                    continue

                matches.append(match)

                if len(matches) >= limit:
                    break

            # Sort alphabetically for consistent results
            matches.sort(key=lambda p: str(p).lower())

            if not matches:
                return f"No files found matching pattern '{pattern}'"

            # Format output
            result_lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
            for match in matches:
                rel_path = match.relative_to(self.working_dir)
                result_lines.append(str(rel_path))

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
        r"""Search file contents using regex patterns.

        Uses ripgrep (rg) if available for best performance,
        falls back to Python regex otherwise.

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
            - grep_search("class.*Agent") - Find class definitions
            - grep_search("def search", glob_filter="*.py") - Find search functions
            - grep_search("import re", output_mode="files") - Files importing re
            - grep_search("TODO", output_mode="content", context_lines=2)
        """
        search_dir = self.working_dir
        if path:
            search_dir = self.working_dir / path

        if not search_dir.exists():
            return f"Error: Directory '{search_dir}' does not exist."

        limit = max_results or self.max_results

        if self._has_ripgrep:
            return self._grep_with_ripgrep(
                pattern, search_dir, glob_filter, ignore_case,
                output_mode, context_lines, limit
            )
        else:
            return self._grep_with_python(
                pattern, search_dir, glob_filter, ignore_case,
                output_mode, context_lines, limit
            )

    def _grep_with_ripgrep(
        self,
        pattern: str,
        search_dir: Path,
        glob_filter: Optional[str],
        ignore_case: bool,
        output_mode: str,
        context_lines: int,
        limit: int,
    ) -> str:
        """Use ripgrep for fast content search."""
        cmd = ['rg', '--no-heading', '--with-filename']

        # Add exclude patterns
        for excl in self.exclude_dirs:
            cmd.extend(['--glob', f'!{excl}', '--glob', f'!**/{excl}/**'])

        if ignore_case:
            cmd.append('-i')

        if glob_filter:
            cmd.extend(['--glob', glob_filter])

        if output_mode == 'files':
            cmd.append('-l')  # Only print file names
        elif output_mode == 'count':
            cmd.append('-c')  # Print count per file
        elif output_mode == 'content':
            cmd.append('-n')  # Print line numbers
            if context_lines > 0:
                cmd.extend(['-C', str(context_lines)])

        cmd.extend(['-m', str(limit)])  # Max matches per file
        cmd.append(pattern)
        cmd.append(str(search_dir))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.working_dir)
            )

            output = result.stdout.strip()
            if not output:
                return f"No matches found for pattern '{pattern}'"

            # Convert absolute paths to relative
            lines = output.split('\n')[:limit]
            rel_lines = []
            for line in lines:
                try:
                    # Try to make paths relative
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

        except subprocess.TimeoutExpired:
            return "Error: Search timed out after 30 seconds"
        except Exception as e:
            return f"Error during ripgrep search: {e}"

    def _grep_with_python(
        self,
        pattern: str,
        search_dir: Path,
        glob_filter: Optional[str],
        ignore_case: bool,
        output_mode: str,
        context_lines: int,
        limit: int,
    ) -> str:
        """Fallback to Python regex for content search."""
        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        file_pattern = glob_filter or '**/*'
        results = []
        files_searched = 0

        for file_path in search_dir.glob(file_pattern):
            if not file_path.is_file():
                continue
            if self._should_exclude(file_path.relative_to(self.working_dir)):
                continue

            files_searched += 1
            if files_searched > 1000:  # Safety limit
                break

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')

                matches_in_file = []
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches_in_file.append((i, line.strip()))

                if matches_in_file:
                    rel_path = file_path.relative_to(self.working_dir)
                    if output_mode == 'files':
                        results.append(str(rel_path))
                    elif output_mode == 'count':
                        results.append(f"{rel_path}: {len(matches_in_file)}")
                    else:  # content
                        for line_num, line_text in matches_in_file[:5]:
                            results.append(f"{rel_path}:{line_num}: {line_text[:200]}")

                if len(results) >= limit:
                    break

            except Exception:
                continue

        if not results:
            return f"No matches found for pattern '{pattern}'"

        header = f"Search results for '{pattern}' ({len(results)} matches):"
        return header + "\n" + "\n".join(results[:limit])

    def read_file(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        max_lines: int = 500,
    ) -> str:
        r"""Read file contents with optional line range.

        Args:
            file_path: Path to the file (relative to working_dir or absolute).
            start_line: Starting line number (1-indexed). If None, starts from beginning.
            end_line: Ending line number (inclusive). If None, reads to max_lines.
            max_lines: Maximum lines to read if no end_line specified.

        Returns:
            File contents with line numbers, or error message.

        Examples:
            - read_file("camel/agents/chat_agent.py") - Read first 500 lines
            - read_file("README.md", start_line=1, end_line=50) - Read lines 1-50
            - read_file("long_file.py", start_line=100, end_line=200) - Read specific range
        """
        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path

        if not path.exists():
            return f"Error: File '{file_path}' does not exist."

        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            total_lines = len(lines)

            # Determine line range
            start = (start_line - 1) if start_line else 0
            start = max(0, start)

            if end_line:
                end = min(end_line, total_lines)
            else:
                end = min(start + max_lines, total_lines)

            # Extract lines
            selected_lines = lines[start:end]

            # Format with line numbers
            result_lines = [f"File: {file_path} (lines {start+1}-{end} of {total_lines})"]
            result_lines.append("-" * 60)

            for i, line in enumerate(selected_lines, start + 1):
                # Truncate very long lines
                line_content = line.rstrip('\n\r')
                if len(line_content) > 500:
                    line_content = line_content[:500] + "..."
                result_lines.append(f"{i:6d}\t{line_content}")

            if end < total_lines:
                result_lines.append(f"\n... ({total_lines - end} more lines)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"Error reading file: {e}"

    def list_directory(
        self,
        path: Optional[str] = None,
        show_hidden: bool = False,
        max_items: int = 200,
    ) -> str:
        r"""List directory contents with file types.

        Args:
            path: Directory path (relative to working_dir). If None, lists working_dir.
            show_hidden: Whether to show hidden files (starting with .).
            max_items: Maximum items to list.

        Returns:
            Formatted directory listing with [DIR] and [FILE] markers.
        """
        target_dir = self.working_dir
        if path:
            target_dir = self.working_dir / path

        if not target_dir.exists():
            return f"Error: Directory '{path}' does not exist."

        if not target_dir.is_dir():
            return f"Error: '{path}' is not a directory."

        try:
            items = list(target_dir.iterdir())

            # Filter hidden files
            if not show_hidden:
                items = [i for i in items if not i.name.startswith('.')]

            # Sort: directories first, then files, alphabetically
            dirs = sorted([i for i in items if i.is_dir()], key=lambda x: x.name.lower())
            files = sorted([i for i in items if i.is_file()], key=lambda x: x.name.lower())

            result_lines = [f"Contents of {path or '.'}:"]
            result_lines.append("-" * 40)

            count = 0
            for d in dirs:
                if count >= max_items:
                    break
                if not self._should_exclude(d.relative_to(self.working_dir)):
                    result_lines.append(f"[DIR]  {d.name}/")
                    count += 1

            for f in files:
                if count >= max_items:
                    break
                result_lines.append(f"[FILE] {f.name}")
                count += 1

            if len(items) > max_items:
                result_lines.append(f"\n... and {len(items) - max_items} more items")

            return "\n".join(result_lines)

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
            module_name: Module or class name to search for (e.g., 'weather', 'QwenModel').
            ignore_case: Whether to ignore case in matching.

        Returns:
            List of files that import the specified module.

        Examples:
            - find_imports("WeatherToolkit") - Find files using WeatherToolkit
            - find_imports("qwen") - Find files importing anything with 'qwen'
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
            - find_definition("ChatAgent") - Find ChatAgent class
            - find_definition("search", definition_type="function") - Find search functions
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
            FunctionTool(self.find_examples),
        ]


# Example usage and quick test
if __name__ == "__main__":
    # Use parent directory of this script as working directory
    script_dir = Path(__file__).parent
    camel_dir = script_dir.parent

    toolkit = CodeSearchToolkit(working_directory=str(camel_dir))

    print("=== Testing glob_search ===")
    print(toolkit.glob_search("**/*agent*.py", max_results=10))
    print()

    print("=== Testing grep_search ===")
    print(toolkit.grep_search("class.*Agent", glob_filter="*.py", output_mode='files', max_results=10))
    print()

    print("=== Testing find_definition ===")
    print(toolkit.find_definition("ChatAgent", definition_type='class'))
    print()

    print("=== Testing list_directory ===")
    print(toolkit.list_directory("camel"))
