import glob
import logging
import mimetypes
import os
import re
import stat
from datetime import datetime
from typing import Type, Any, Optional, List, Literal

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException, BaseToolkit
from pydantic import BaseModel, Field

from llm_workers.api import ConfirmationRequestToolCallDescription
from llm_workers.api import ExtendedBaseTool

logger = logging.getLogger(__name__)


def _not_in_working_directory(file_path) -> bool:
    return file_path.startswith("/") or ".." in file_path.split("/")


class ReadFileToolSchema(BaseModel):
    path: str = Field(..., description="Path to the file to read")
    offset: int = Field(0, description="Offset in lines. >=0 means from the start of the file, <0 means from the end of the file.")
    lines: int = Field(..., description="Number of lines to read.")
    show_line_numbers: bool = Field(False, description="If true, prefix each line with its line number.")

class ReadFileTool(BaseTool, ExtendedBaseTool):
    name: str = "read_file"
    description: str = "Reads a file and returns its content. Output is limited to `lines` parameter."
    args_schema: Type[ReadFileToolSchema] = ReadFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['path'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        path = input['path']
        return ConfirmationRequestToolCallDescription(
            action = f"read file \"{path}\" outside working directory" if _not_in_working_directory(path)
            else f"read file \"{path}\"",
            params = [ ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        offset = input.get('offset', 0)
        lines = input['lines']
        path_ = input['path']
        if offset >= 0:
            return f"Reading file \"{path_}\" (lines {offset + 1}-{offset + lines})"
        elif lines == -offset:
            return f"Reading last {lines} lines of file \"{path_}\""
        else:
            return f"Reading {lines} lines starting {-offset} lines from the end of file \"{path_}\""

    def _run(self, path: str, lines: int, offset: int = 0, show_line_numbers: bool = False) -> str:
        try:
            with open(path, 'r') as file:
                file_lines: list[str] = file.readlines()

            total_lines = len(file_lines)

            if offset >= 0:
                # Offset from start
                start = offset
                end = start + lines
            else:
                # Offset from end (negative offset)
                start = max(0, total_lines + offset)
                end = start + lines

            selected_lines = file_lines[start:end]

            if show_line_numbers:
                # Line numbers are 1-based
                result_lines = [
                    f"{start + i + 1}: {line.rstrip('\n')}"
                    for i, line in enumerate(selected_lines)
                ]
                return '\n'.join(result_lines)
            else:
                return ''.join(selected_lines)
        except Exception as e:
            raise ToolException(f"Error reading file {path}: {e}")


class WriteFileToolSchema(BaseModel):
    path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write")
    if_exists: Literal["fail", "append", "overwrite"] = Field("fail", description="What to do if the file already exists: 'fail' (default), 'append', or 'overwrite'")


class WriteFileTool(BaseTool, ExtendedBaseTool):
    name: str = "write_file"
    description: str = "Writes content to a file"
    args_schema: Type[WriteFileToolSchema] = WriteFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['path'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        path = input['path']
        return ConfirmationRequestToolCallDescription(
            action = f"write to the file \"{path}\" outside working directory" if _not_in_working_directory(path)
            else f"write to the file \"{path}\"",
            params = []
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Writing file \"{input['path']}\""

    def _run(self, path: str, content: str, if_exists: str = "fail"):
        try:
            if if_exists == "fail" and os.path.exists(path):
                raise ToolException(f"File {path} already exists")

            mode = 'a' if if_exists == "append" else 'w'
            with open(path, mode) as file:
                file.write(content)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error writing file {path}: {e}")



class EditFileToolSchema(BaseModel):
    path: str = Field(..., description="Path to the file to edit")
    old_string: str = Field(..., description="The exact text to find and replace")
    new_string: str = Field(..., description="The text to replace with")
    replace_all: bool = Field(False, description="Replace all occurrences (default: only first)")
    expected_count: Optional[int] = Field(None, description="Expected number of replacements. Fails if mismatch (safety check).")


class EditFileTool(BaseTool, ExtendedBaseTool):
    name: str = "edit_file"
    description: str = "Make targeted replacements in a file. More efficient than read+write for small changes."
    args_schema: Type[EditFileToolSchema] = EditFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input.get('path', ''))

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        path = input['path']
        return ConfirmationRequestToolCallDescription(
            action=f"edit file \"{path}\" outside working directory" if _not_in_working_directory(path)
            else f"edit file \"{path}\"",
            params=[]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Updating file \"{input['path']}\""

    def _run(self, path: str, old_string: str, new_string: str,
             replace_all: bool = False, expected_count: Optional[int] = None) -> dict:
        try:
            with open(path, 'r') as file:
                content = file.read()

            count = content.count(old_string)

            if count == 0:
                raise ToolException(f"old_string not found in file {path}")

            if not replace_all and count > 1:
                raise ToolException(f"Multiple matches ({count}) found but replace_all=false. "
                                    f"Set replace_all=true to replace all occurrences.")

            if expected_count is not None and count != expected_count:
                raise ToolException(f"Expected {expected_count} replacements but found {count}")

            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements_made = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements_made = 1

            with open(path, 'w') as file:
                file.write(new_content)

            return { "replacements": replacements_made}

        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error editing file {path}: {e}")


class GlobFilesToolSchema(BaseModel):
    pattern: str = Field(..., description="Glob pattern (e.g., '**/*.py', 'src/**/*.ts')")
    path: str = Field(".", description="Base directory to search from")
    max_results: int = Field(100, description="Maximum number of results to return")
    include_hidden: bool = Field(False, description="Include hidden files (starting with .)")


class GlobFilesTool(BaseTool, ExtendedBaseTool):
    name: str = "glob_files"
    description: str = "Find files matching glob patterns. Efficient for locating files by name or extension."
    args_schema: Type[GlobFilesToolSchema] = GlobFilesToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input.get('path', '.'))

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        path = input.get('path', '.')
        pattern = input['pattern']
        return ConfirmationRequestToolCallDescription(
            action=f"search for files matching \"{pattern}\" at \"{path}\" outside working directory"
            if _not_in_working_directory(path)
            else f"search for files matching \"{pattern}\"",
            params=[]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Searching for files matching \"{input['pattern']}\""

    def _run(self, pattern: str, path: str = ".", max_results: int = 100,
             include_hidden: bool = False) -> List[str]:
        try:
            # Construct full pattern
            if path != ".":
                full_pattern = os.path.join(path, pattern)
            else:
                full_pattern = pattern

            # Use glob with recursive support
            matches = glob.glob(full_pattern, recursive=True)

            # Filter out hidden files if not requested
            if not include_hidden:
                matches = [m for m in matches if not any(
                    part.startswith('.') and part not in ('.', '..')
                    for part in m.split(os.sep)
                )]

            # Filter to only files (not directories)
            matches = [m for m in matches if os.path.isfile(m)]

            # Sort by modification time (newest first)
            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            # Limit results
            return matches[:max_results]

        except Exception as e:
            raise ToolException(f"Error searching for files with pattern {pattern}: {e}")


class GrepFilesToolSchema(BaseModel):
    pattern: str = Field(..., description="Regular expression pattern to search for")
    files_glob: str = Field(..., description="File path, directory, or glob pattern (e.g., '*.py', 'src/**/*.ts')")
    lines_before: int = Field(0, description="Number of lines to show before each match")
    lines_after: int = Field(0, description="Number of lines to show after each match")
    case_insensitive: bool = Field(False, description="Ignore case when matching")
    max_results: int = Field(50, description="Maximum number of matches to return")
    output_mode: str = Field("content", description="Output format: 'content' (matching lines), 'files_only' (just filenames), or 'count' (match counts)")


class GrepFilesTool(BaseTool, ExtendedBaseTool):
    name: str = "grep_files"
    description: str = "Search for regex patterns within files. Returns matching lines with optional context."
    args_schema: Type[GrepFilesToolSchema] = GrepFilesToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['files_glob'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        files_glob = input['files_glob']
        pattern = input['pattern']
        return ConfirmationRequestToolCallDescription(
            action=f"search for \"{pattern}\" at \"{files_glob}\" outside working directory"
            if _not_in_working_directory(files_glob)
            else f"search for \"{pattern}\"",
            params=[]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        files_glob = input['files_glob']
        return f"Searching for \"{input['pattern']}\" in \"{files_glob}\""

    def _is_glob_pattern(self, pattern: str) -> bool:
        """Check if pattern contains glob special characters."""
        return any(char in pattern for char in '*?[]')

    def _get_files_to_search(self, files_glob: str) -> List[str]:
        """Get list of files based on files_glob (file, directory, or glob pattern)."""
        if self._is_glob_pattern(files_glob):
            # Treat as glob pattern
            files = [f for f in glob.glob(files_glob, recursive=True) if os.path.isfile(f)]
        elif os.path.isfile(files_glob):
            # Literal file path
            return [files_glob]
        elif os.path.isdir(files_glob):
            # Directory - walk recursively
            files = []
            for root, _, filenames in os.walk(files_glob):
                # Skip hidden directories
                if any(part.startswith('.') for part in root.split(os.sep) if part):
                    continue
                for filename in filenames:
                    if not filename.startswith('.'):
                        files.append(os.path.join(root, filename))
        else:
            # Try as glob anyway (user may expect glob behavior)
            files = [f for f in glob.glob(files_glob, recursive=True) if os.path.isfile(f)]

        return files

    def _run(self, pattern: str, files_glob: str,
             lines_before: int = 0, lines_after: int = 0,
             case_insensitive: bool = False,
             max_results: int = 50, output_mode: str = "content") -> dict:
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)

            files = self._get_files_to_search(files_glob)
            matches = []
            files_with_matches = set()
            total_matches = 0

            for filepath in files:
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            total_matches += 1
                            files_with_matches.add(filepath)

                            if output_mode == "content" and len(matches) < max_results:
                                match_entry: dict[str, Any] = {
                                    "file": filepath,
                                    "line_number": line_num,
                                    "content": line.rstrip('\n')
                                }

                                if lines_before > 0:
                                    start = max(0, line_num - 1 - lines_before)
                                    match_entry["context_before"] = [
                                        lines[i].rstrip('\n') for i in range(start, line_num - 1)
                                    ]
                                if lines_after > 0:
                                    end = min(len(lines), line_num + lines_after)
                                    match_entry["context_after"] = [
                                        lines[i].rstrip('\n') for i in range(line_num, end)
                                    ]

                                matches.append(match_entry)

                except (IOError, UnicodeDecodeError):
                    # Skip files that can't be read
                    continue

            result = {
                "total_matches": total_matches,
                "files_searched": len(files)
            }

            if output_mode == "content":
                result["matches"] = matches
            elif output_mode == "files_only":
                result["files"] = list(files_with_matches)[:max_results]
            elif output_mode == "count":
                result["files_with_matches"] = len(files_with_matches)

            return result

        except re.error as e:
            raise ToolException(f"Invalid regex pattern: {e}")
        except Exception as e:
            raise ToolException(f"Error searching files: {e}")


class FileInfoToolSchema(BaseModel):
    path: str = Field(..., description="Path to the file or directory")


class FileInfoTool(BaseTool, ExtendedBaseTool):
    name: str = "file_info"
    description: str = "Get detailed information about a file or directory including size, permissions, and timestamps."
    args_schema: Type[FileInfoToolSchema] = FileInfoToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        # FileInfo is read-only, no confirmation needed
        return False

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Getting info for \"{input['path']}\""

    def _run(self, path: str) -> dict:
        result: dict[str, Any] = {"exists": os.path.exists(path)}

        if not result["exists"]:
            return result

        try:
            stat_info = os.stat(path)

            # Determine type
            if os.path.islink(path):
                result["type"] = "symlink"
            elif os.path.isfile(path):
                result["type"] = "file"
            elif os.path.isdir(path):
                result["type"] = "directory"
            else:
                result["type"] = "other"

            # Size
            result["size"] = stat_info.st_size

            # Permissions
            mode = stat_info.st_mode
            perms = ""
            for who in "USR", "GRP", "OTH":
                for what in "R", "W", "X":
                    perm = getattr(stat, f"S_I{what}{who}")
                    perms += what.lower() if mode & perm else "-"
            result["permissions"] = perms

            # Owner and group
            try:
                import pwd
                result["owner"] = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError, AttributeError):
                result["owner"] = str(stat_info.st_uid)

            try:
                import grp
                result["group"] = grp.getgrgid(stat_info.st_gid).gr_name
            except (ImportError, KeyError, AttributeError):
                result["group"] = str(stat_info.st_gid)

            # Timestamps
            result["created"] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result["modified"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            result["accessed"] = datetime.fromtimestamp(stat_info.st_atime).isoformat()

            # Access checks
            result["is_readable"] = os.access(path, os.R_OK)
            result["is_writable"] = os.access(path, os.W_OK)

            # MIME type (for files)
            if result["type"] == "file":
                mime_type, _ = mimetypes.guess_type(path)
                result["mime_type"] = mime_type or "application/octet-stream"

            return result

        except Exception as e:
            raise ToolException(f"Error getting file info for {path}: {e}")


class ListFilesToolSchema(BaseModel):
    path: str = Field(".", description="Path to directory to list")


class ListFilesTool(BaseTool, ExtendedBaseTool):
    name: str = "list_files"
    description: str = "Lists files and directories in the specified directory"
    args_schema: Type[ListFilesToolSchema] = ListFilesToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input.get('path', '.'))

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        path = input.get('path', '.')
        return ConfirmationRequestToolCallDescription(
            action=f"list files at \"{path}\" outside working directory" if _not_in_working_directory(path)
            else f"list files at \"{path}\"",
            params=[]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Listing files at \"{input.get('path', '.')}\""

    def _run(self, path: str = ".") -> str:
        try:
            if not os.path.exists(path):
                raise ToolException(f"Path {path} does not exist")

            if not os.path.isdir(path):
                raise ToolException(f"Path {path} is not a directory")

            files = []
            dirs = []

            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    dirs.append(item)
                else:
                    files.append(item)

            files.sort()
            dirs.sort()

            lines = ["files:"]
            for f in files:
                lines.append(f"- {f}")
            lines.append(f"total {len(files)} entries")
            lines.append("")
            lines.append("dirs:")
            for d in dirs:
                lines.append(f"- {d}")
            lines.append(f"total {len(dirs)} entries")

            return "\n".join(lines)

        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error listing files at {path}: {e}")


class FilesystemToolkit(BaseToolkit):
    """Toolkit containing filesystem operation tools."""

    def get_tools(self) -> list[BaseTool]:
        return [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GlobFilesTool(),
            GrepFilesTool(),
            FileInfoTool(),
            ListFilesTool(),
        ]
