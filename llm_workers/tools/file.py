from langchain_core.tools import StructuredTool

from llm_workers.api import WorkerException


def _read_file(file_path: str) -> str:
    """Read a file and return its content. File should be under working directory.

    Args:
        file_path: path to the file to read
    """
    _verify_file_in_working_directory(file_path)

    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        raise WorkerException(f"Error reading file {file_path}: {e}")


def _write_file(file_path: str, content: str):
    """Write content to a file. File should be under working directory.

    Args:
        file_path: path to the file to write
        content: content to write to the file
    """
    _verify_file_in_working_directory(file_path)

    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except Exception as e:
        raise WorkerException(f"Error writing file {file_path}: {e}")


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