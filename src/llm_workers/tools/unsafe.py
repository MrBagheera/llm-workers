import logging
import os
import subprocess
import time
from typing import Type

from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field

from llm_workers.api import WorkerException
from llm_workers.utils import LazyFormatter

logger = logging.getLogger(__name__)


def _read_file(filename: str) -> str:
    """Read a file and return its content. File should be under working directory.

    Args:
        filename: path to the file to read
    """
    _verify_file_in_working_directory(filename)

    try:
        with open(filename, 'r') as file:
            return file.read()
    except Exception as e:
        raise WorkerException(f"Error reading file {filename}: {e}")


def _write_file(filename: str, content: str):
    """Write content to a file. File should be under working directory.

    Args:
        filename: path to the file to write
        content: content to write to the file
    """
    _verify_file_in_working_directory(filename)

    try:
        with open(filename, 'w') as file:
            file.write(content)
    except Exception as e:
        raise WorkerException(f"Error writing file {filename}: {e}")


def _verify_file_in_working_directory(file_path):
    if file_path.startswith("/"):
        raise WorkerException("File path should be relative to working directory")

    if ".." in file_path.split("/"):
        raise WorkerException("File path should be within working directory")

read_file_tool = StructuredTool.from_function(
    _read_file,
    name="read_file",
    parse_docstring=True,
    error_on_invalid_docstring=True
)

write_file_tool = StructuredTool.from_function(
    _write_file,
    name="write_file",
    parse_docstring=True,
    error_on_invalid_docstring=True
)


class RunPythonScriptToolSchema(BaseModel):
    """
    Schema for the RunPythonScriptTool.
    """

    script: str = Field(
        ...,
        description="Python script to run. Must be a valid Python code."
    )

class RunPythonScriptTool(BaseTool):
    """
    Tool to run Python scripts. This tool is not safe to use with untrusted code.
    """

    name: str = "run_python_script"
    description: str = "Run a Python script and return its output."
    args_schema: Type[RunPythonScriptToolSchema] = RunPythonScriptToolSchema

    def _run(self, script: str) -> str:
        # TODO change to more general confirmation UI
        # Show script to user and get confirmation
        print("\n=== AI Assistant wants to execute the following code ===")
        print(script)
        print("=== End of code ===\n")

        confirmation = input("Allow execution? (Yes/No): ").strip().lower()
        if confirmation not in ["yes", "y"]:
            raise WorkerException("Script execution cancelled by user")

        # run the script
        file_path = f"script_{time.strftime('%Y%m%d_%H%M%S')}.py"
        _write_file(file_path, script)
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
                raise WorkerException(f"Running Python script returned code {exit_code}:\n{stderr}")
            return result
        except WorkerException as e:
            raise e
        except Exception as e:
            raise WorkerException(f"Error running Python script: {e}")
        finally:
            if file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete script file {file_path}: {e}")
