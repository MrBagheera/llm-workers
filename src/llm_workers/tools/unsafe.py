import json
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Type, Any, Dict, Union, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from llm_workers.api import ConfirmationRequest, ConfirmationRequestParam
from llm_workers.api import ExtendedBaseTool
from llm_workers.utils import LazyFormatter, open_file_in_default_app, is_safe_to_open

logger = logging.getLogger(__name__)


def _not_in_working_directory(file_path) -> bool:
    return file_path.startswith("/") or ".." in file_path.split("/")


class ReadFileToolSchema(BaseModel):
    filename: str = Field(..., description="Path to the file to read")
    lines: int = Field(0, description="Number of lines to read. If 0 (default), read the entire file. If negative, read from the end of file (tail).")

class ReadFileTool(BaseTool, ExtendedBaseTool):
    name: str = "read_file"
    description: str = "Reads a file and returns its content"
    args_schema: Type[ReadFileToolSchema] = ReadFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['filename'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        filename = input['filename']
        return ConfirmationRequest(
            action = f"read file \"{filename}\" outside working directory" if _not_in_working_directory(filename)
            else f"read file \"{filename}\"",
            params = [ ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Reading file {input['filename']}"

    def _run(self, filename: str, lines: int = 0) -> str:
        try:
            with open(filename, 'r') as file:
                if lines == 0:
                    return file.read()
                else:
                    file_lines: list[str] = file.readlines()
                    if lines > 0:
                        return '\n'.join(file_lines[:lines])
                    else:
                        return '\n'.join(file_lines[lines:])
        except Exception as e:
            raise ToolException(f"Error reading file {filename}: {e}")


class WriteFileToolSchema(BaseModel):
    filename: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write")
    append: bool = Field(False, description="If true, append to the file instead of overwriting it")


class WriteFileTool(BaseTool, ExtendedBaseTool):
    name: str = "write_file"
    description: str = "Writes content to a file"
    args_schema: Type[WriteFileToolSchema] = WriteFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['filename'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        filename = input['filename']
        return ConfirmationRequest(
            action = f"write to the file \"{filename}\" outside working directory" if _not_in_working_directory(filename)
                else f"write to the file \"{filename}\"",
            params = []
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Writing file {input['filename']}"

    def _run(self, filename: str, content: str, append: bool = False):
        try:
            if append:
                with open(filename, 'a') as file:
                    file.write(content)
            else:
                with open(filename, 'w') as file:
                    file.write(content)
        except Exception as e:
            raise ToolException(f"Error writing file {filename}: {e}")



class RunPythonScriptToolSchema(BaseModel):
    """
    Schema for the RunPythonScriptTool.
    """

    script: str = Field(
        ...,
        description="Python script to run. Must be a valid Python code."
    )

class RunPythonScriptTool(BaseTool, ExtendedBaseTool):
    """
    Tool to run Python scripts. This tool is not safe to use with untrusted code.
    """

    name: str = "run_python_script"
    description: str = "Run a Python script and return its output."
    args_schema: Type[RunPythonScriptToolSchema] = RunPythonScriptToolSchema
    require_confirmation: bool = True

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return self.require_confirmation

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        return ConfirmationRequest(
            action = "run Python script",
            params = [ ConfirmationRequestParam(name = "script", value = input["script"], format = "python" ) ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return "Running Python script"

    def _run(self, script: str) -> str:
        file_path = f"script_{time.strftime('%Y%m%d_%H%M%S')}.py"
        with open(file_path, 'w') as file:
            file.write(script)
        try:
            cmd = ["python3", file_path]
            cmd_str = LazyFormatter(cmd, custom_formatter = lambda x: " ".join(x))
            logger.debug("Running %s", cmd_str)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            (result, stderr) = process.communicate()
            exit_code = process.wait()

            if exit_code != 0:
                raise ToolException(f"Running Python script returned code {exit_code}:\n{stderr}")
            return result
        except ToolException as e:
            raise e
        except Exception as e:
            raise ToolException(f"Error running Python script: {e}", e)
        finally:
            if file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete script file {file_path}: {e}")


class ShowFileToolSchema(BaseModel):
    filename: str = Field(..., description="Path to the file")

class ShowFileTool(BaseTool, ExtendedBaseTool):
    name: str = "show_file"
    description: str = "Show file to the user using OS-default application"
    args_schema: Type[ShowFileToolSchema] = ShowFileToolSchema

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['filename'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        filename = input['filename']
        return ConfirmationRequest(
            action=f"open the file \"{filename}\" outside working directory in OS-default application" if _not_in_working_directory(
                filename)
            else f"open the file \"{filename}\" in OS-default application",
            params=[]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Opening file {input['filename']}"

    def _run(self, filename: str):
        if not is_safe_to_open(filename):
            raise ToolException(f"File {filename} is not safe to open")
        open_file_in_default_app(filename)


class BashToolSchema(BaseModel):
    script: str = Field(..., description="Bash script to execute")
    timeout: int = Field(30, description="Timeout in seconds. Default is 30 seconds.")


class BashTool(BaseTool, ExtendedBaseTool):
    name: str = "bash"
    description: str = "Execute a bash script and return its output"
    args_schema: Type[BashToolSchema] = BashToolSchema
    require_confirmation: bool = True
    
    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return self.require_confirmation
    
    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        return ConfirmationRequest(
            action="execute bash script",
            params=[ConfirmationRequestParam(name="script", value=input["script"], format="bash")]
        )
    
    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return "Executing bash script"
    
    def _run(self, script: str, timeout: int = 30) -> str:
        file_path = f"script_{time.strftime('%Y%m%d_%H%M%S')}.sh"
        try:
            with open(file_path, 'w') as file:
                file.write(script)
            os.chmod(file_path, 0o755)  # Make the script executable
            
            logger.debug("Running bash script from %s", file_path)
            process = subprocess.Popen(
                ["bash", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            (result, stderr) = process.communicate(timeout=timeout)
            exit_code = process.wait()
            
            if exit_code != 0:
                raise ToolException(f"Bash script returned code {exit_code}:\n{stderr}")
            
            return result
        except subprocess.TimeoutExpired:
            process.kill()
            raise ToolException(f"Bash script execution timed out after {timeout} seconds")
        except Exception as e:
            raise ToolException(f"Error executing bash script: {e}")
        finally:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete script file {file_path}: {e}")


class ListFilesToolSchema(BaseModel):
    path: str = Field(..., description="Path to directory to list or file to show info")
    depth: int = Field(0, description="Set to a positive number to show nested directories (e.g., 1 for immediate subdirectories, 2 for subdirectories and their subdirectories, etc.). Default 0 - no recursion.")
    permissions: bool = Field(False, description="Whether to show permissions")
    times: bool = Field(False, description="Whether to show creation and modification times")
    sizes: bool = Field(False, description="Whether to show file size in bytes")


class ListFilesTool(BaseTool, ExtendedBaseTool):
    name: str = "list_files"
    description: str = "Lists files and directories with optional detailed information"
    args_schema: Type[ListFilesToolSchema] = ListFilesToolSchema
    _max_entries = 1024  # Maximum number of entries to process

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return _not_in_working_directory(input['path'])

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        path = input['path']
        return ConfirmationRequest(
            action = f"list files at \"{path}\" outside working directory" if _not_in_working_directory(path)
            else f"list files at \"{path}\"",
            params = []
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Listing files at {input['path']}"

    def _get_file_info(self, path: str, show_permissions: bool, show_times: bool, show_sizes: bool) -> Dict:
        """Get information about a file."""
        result = {}
        
        if show_permissions:
            stat_info = os.stat(path)
            # Convert mode to string format like "rwxr-xr--"
            mode = stat_info.st_mode
            perms = ""
            for who in "USR", "GRP", "OTH":
                for what in "R", "W", "X":
                    perm = getattr(os, f"{what}_{who}")
                    perms += what.lower() if mode & perm else "-"
            result["permissions"] = perms
        
        if show_times:
            stat_info = os.stat(path)
            result["ctime"] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result["mtime"] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
        
        if os.path.isfile(path) and show_sizes:
            result["size"] = os.path.getsize(path)
        
        return result

    def _is_hidden(self, name: str) -> bool:
        """Check if a file or directory is hidden (starts with a dot)."""
        return name != '.' and name != '..' and name.startswith('.')
        
    def _process_directory(self, path: str, remaining_depth: int, show_permissions: bool,
                           show_times: bool, show_sizes: bool, entry_count: int) -> tuple[Dict, int]:
        """Process a directory, returning its description and contents if needed along with total entry count."""
        result = self._get_file_info(path, show_permissions, show_times, show_sizes)
        
        # First pass - just count files and directories if we're not recursing
        files = []
        dirs = []
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            else:
                files.append(item)
        
        if remaining_depth == 0 or entry_count > self._max_entries or self._is_hidden(os.path.basename(path)):
            result["n_files"] = len(files)
            result["n_dirs"] = len(dirs)
            return result, entry_count
        
        # If we're recursing, process each item
        contents = {}
        
        # Process all files first
        for item in files:
            item_path = os.path.join(path, item)
            contents[item] = self._get_file_info(item_path, show_permissions, show_times, show_sizes)
            entry_count += 1
        
        # Then process directories
        for item in dirs:
            item_path = os.path.join(path, item)

            # Regular directory - recurse into it
            subdir_result, new_entry_count = self._process_directory(
                item_path, remaining_depth - 1, show_permissions, show_times, show_sizes, entry_count + 1
            )
            contents[item] = subdir_result
            entry_count = new_entry_count

        result["contents"] = contents
        return result, entry_count

    def _run(self, path: str, depth: int = 0, permissions: bool = False,
             times: bool = False, sizes: bool = False) -> any:
        try:
            if not os.path.exists(path):
                raise ToolException(f"Path {path} does not exist")
            
            if os.path.isfile(path):
                return self._get_file_info(path, permissions, times, sizes)
            else:
                result, _ = self._process_directory(
                    path, depth + 1, permissions, times, sizes, 0
                )
                return result

        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error listing files at {path}: {e}")


class RunProcessToolSchema(BaseModel):
    """
    Schema for the RunProcessTool.
    """
    
    command: str = Field(
        ...,
        description="Command to run as a subprocess."
    )
    args: list[str] = Field(
        default=[],
        description="Arguments for the command."
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds. Default is 30 seconds."
    )

class RunProcessTool(BaseTool, ExtendedBaseTool):
    """
    Tool to run arbitrary system processes. This tool is not safe to use with untrusted commands.
    """

    name: str = "run_process"
    description: str = "Run a system process and return its output."
    args_schema: Type[RunProcessToolSchema] = RunProcessToolSchema
    require_confirmation: bool = True

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return self.require_confirmation

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequest:
        command = input["command"]
        args = input.get("args", [])
        cmd_display = f"{command} {' '.join(args)}" if args else command
        return ConfirmationRequest(
            action = "run system process",
            params = [ ConfirmationRequestParam(name = "command", value = cmd_display, format = "bash" ) ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Running process {input['command']}"

    def _run(self, command: str, args: list[str] = [], timeout: int = 30) -> str:
        try:
            cmd = [command] + args
            cmd_str = LazyFormatter(cmd, custom_formatter = lambda x: " ".join(x))
            logger.debug("Running process %s", cmd_str)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            (result, stderr) = process.communicate(timeout=timeout)
            exit_code = process.wait()
            
            if exit_code != 0:
                raise ToolException(f"Process returned code {exit_code}:\n{stderr}")
                
            return result
        except subprocess.TimeoutExpired:
            process.kill()
            raise ToolException(f"Process execution timed out after {timeout} seconds")
        except Exception as e:
            raise ToolException(f"Error running process: {e}")
