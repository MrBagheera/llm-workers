# AGENTS.md

This file provides guidance to AI Coding Assistants when working with code in this repository.

## Project Overview

LLM Workers is a Python library and command-line tool for experimenting with Large Language Models (LLMs). It provides a YAML-based configuration system for defining LLM interactions and custom tools.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
poetry install

# Install with dev dependencies for AWS support
poetry install --extras dev
```

### Testing
```bash
# Run all tests
poetry run python -m unittest discover tests/

# Run specific test file
poetry run python -m unittest tests.test_worker

# Run tests with verbose output
poetry run python -m unittest discover tests/ -v
```

### Building and Packaging
```bash
# Build the package
poetry build

# Publish to PyPI (after version bump)
poetry publish
```

### Running the Tools
```bash
# CLI tool for batch processing
llm-workers-cli [--verbose] [--debug] <script_file> [prompts...]

# Interactive chat interface
llm-workers-chat [--verbose] [--debug] <script_file>

# Read from stdin
llm-workers-cli <script_file> --
```

## Architecture

### Core Components

- **Worker** (`src/llm_workers/worker.py`): Central orchestrator that manages LLM interactions and tool execution
- **Context** (`src/llm_workers/context.py`): Manages configuration loading and provides access to models and tools
- **Configuration** (`src/llm_workers/config.py`): Defines the YAML configuration schema for LLM scripts
- **Tools**: Extensible tool system supporting both built-in and custom tools

### LLM Scripts

The system uses YAML configuration files called "LLM scripts" that define:
- **Models**: LLM providers and configurations (OpenAI, Bedrock, etc.)
- **Tools**: Available tools for the LLM (web fetching, file operations, custom tools)
- **Chat/CLI**: Interface configurations for interactive or batch processing

### Tool System

Tools can be:
1. **Python classes** extending `langchain_core.tools.base.BaseTool`
2. **Factory functions** returning tool instances
3. **Custom tools** defined in YAML using statement composition

Built-in tool categories:
- **Fetch tools**: Web scraping and content retrieval
- **File tools**: Read/write operations (marked as "unsafe")
- **LLM tools**: Nested LLM calls within workflows
- **System tools**: Process execution and file system operations

### Configuration Loading

The system supports:
- Loading scripts from files or module resources (`module_name:resource.yaml`)
- Environment variable configuration via `~/.config/llm-workers/.env`
- Model-specific parameters and rate limiting
- Tool confirmation and security controls

### Safety Features

- Tools requiring confirmation (especially "unsafe" file/system operations)
- Confidential tool results that don't get passed to subsequent LLM calls
- Private tools (prefixed with `_`) not exposed to LLMs by default
- Rate limiting for API calls

### Execution Flow

1. Load YAML configuration and create context
2. Initialize Worker with specified model and tools
3. Process input through statement composition
4. Execute tool calls with confirmation if required
5. Return results with token usage statistics