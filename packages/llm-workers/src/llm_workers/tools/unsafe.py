import logging
import os
import subprocess
import sys
import time
from typing import Type, Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from llm_workers.api import ConfirmationRequestToolCallDescription, ConfirmationRequestParam
from llm_workers.api import ExtendedBaseTool
from llm_workers.cache import get_cache_filename
from llm_workers.utils import LazyFormatter, open_file_in_default_app, is_safe_to_open

logger = logging.getLogger(__name__)


def _not_in_working_directory(file_path) -> bool:
    return file_path.startswith("/") or ".." in file_path.split("/")


class RunPythonScriptToolSchema(BaseModel):
    script: str = Field(..., description="Python script to run. Must be a valid Python code.")

class RunPythonScriptTool(BaseTool, ExtendedBaseTool):
    name: str = "run_python_script"
    description: str = "Run a Python script and return its output."
    args_schema: Type[RunPythonScriptToolSchema] = RunPythonScriptToolSchema
    delete_after_run: bool = False  # Whether to delete the script file after running
    require_confirmation: bool = True

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return self.require_confirmation

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        return ConfirmationRequestToolCallDescription(
            action = "run Python script",
            params = [ ConfirmationRequestParam(name = "script", value = input["script"], format = "python" ) ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Running generated Python script ({get_cache_filename(input['script'], ".py")})"

    def _run(self, script: str) -> str:
        file_path = get_cache_filename(script, ".py")
        with open(file_path, 'w') as file:
            file.write(script)
        try:
            cmd = [sys.executable, file_path]
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
            if file_path and self.delete_after_run:
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

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        filename = input['filename']
        return ConfirmationRequestToolCallDescription(
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
    
    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        return ConfirmationRequestToolCallDescription(
            action="execute bash script",
            params=[ConfirmationRequestParam(name="script", value=input["script"], format="bash")]
        )
    
    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return "Executing bash script"
    
    def _run(self, script: str, timeout: int = 30) -> str:
        file_path = f"script_{time.strftime('%Y%m%d_%H%M%S')}.sh"
        process: Optional[subprocess.Popen[str]] = None
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
            if process is not None:
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


class RunProcessToolSchema(BaseModel):
    command: str = Field(..., description="Command to run as a subprocess.")
    args: list[str] = Field(default=[], description="Arguments for the command.")
    timeout: int = Field(default=30, description="Timeout in seconds. Default is 30 seconds.")

class RunProcessTool(BaseTool, ExtendedBaseTool):
    name: str = "run_process"
    description: str = "Run a system process and return its output."
    args_schema: Type[RunProcessToolSchema] = RunProcessToolSchema
    require_confirmation: bool = True

    def needs_confirmation(self, input: dict[str, Any]) -> bool:
        return self.require_confirmation

    def make_confirmation_request(self, input: dict[str, Any]) -> ConfirmationRequestToolCallDescription:
        command = input["command"]
        args = input.get("args", [])
        cmd_display = f"{command} {' '.join(args)}" if args else command
        return ConfirmationRequestToolCallDescription(
            action = "run system process",
            params = [ ConfirmationRequestParam(name = "command", value = cmd_display, format = "bash" ) ]
        )

    def get_ui_hint(self, input: dict[str, Any]) -> str:
        return f"Running process {input['command']}"

    def _run(self, command: str, args: Optional[list[str]] = None, timeout: int = 30) -> str:
        process: Optional[subprocess.Popen[str]] = None
        try:
            if args is None:
                args = []
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
            logger.debug("Process %s exited with code %d and following output:\n%s", cmd_str, exit_code, result)
            if len(stderr) > 0:
                logger.debug("Process %s stderr:\n%s", cmd_str, exit_code, stderr)

            if exit_code != 0:
                raise ToolException(f"Process returned code {exit_code}:\n{stderr}")
                
            return result
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
            raise ToolException(f"Process execution timed out after {timeout} seconds")
        except Exception as e:
            raise ToolException(f"Error running process: {e}")
