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

### Package Structure

The project is split into three namespace packages:

```
packages/
├── core/           # llm_workers.core - Core library
├── console_chat/   # llm_workers.console_chat - TTY/console chat UI
└── cli/            # llm_workers.cli - CLI entry points
```

### Core Components (`llm_workers.core`)

- **Worker** (`packages/core/src/llm_workers/core/worker.py`): Central orchestrator that manages LLM interactions and tool execution
- **WorkersContext** (`packages/core/src/llm_workers/core/workers_context.py`): Manages tool registration and provides access to models and tools
- **UserContext** (`packages/core/src/llm_workers/core/user_context.py`): Manages user configuration and model initialization
- **Configuration** (`packages/core/src/llm_workers/core/config.py`): Defines the YAML configuration schema for LLM scripts
- **Tools** (`packages/core/src/llm_workers/core/tools/`): Extensible tool system supporting both built-in and custom tools

### Console Chat Components (`llm_workers.console_chat`)

- **ChatSession** (`packages/console_chat/src/llm_workers/console_chat/chat_session.py`): Interactive chat session management
- **ConsoleController** (`packages/console_chat/src/llm_workers/console_chat/console.py`): Rich console rendering
- **ChatCompleter** (`packages/console_chat/src/llm_workers/console_chat/chat_completer.py`): Command/file completion

### CLI Components (`llm_workers.cli`)

- **batch.py**: CLI batch processing entry point (`llm-workers-cli`)
- **chat_main.py**: Chat interface entry point (`llm-workers-chat`)

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

Built-in tool categories (in `llm_workers.core.tools`):
- **Fetch tools** (`fetch.py`): Web scraping and content retrieval
- **File tools** (`unsafe.py`): Read/write operations (marked as "unsafe")
- **LLM tools** (`llm_tool.py`): Nested LLM calls within workflows
- **Custom tools** (`custom_tool.py`): YAML-based statement composition
- **Misc tools** (`misc.py`): User input, approval tokens

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