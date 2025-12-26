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
- **Worker Utils** (`src/llm_workers/worker_utils.py`): Utility functions for worker operations
- **API** (`src/llm_workers/api.py`): Abstract base classes for UserContext and WorkersContext interfaces
- **User Context** (`src/llm_workers/user_context.py`): Manages user configuration, environment variables, and model initialization
- **Workers Context** (`src/llm_workers/workers_context.py`): Manages script configuration, tool loading, and MCP server connections
- **Configuration** (`src/llm_workers/config.py`): Defines the YAML configuration schema for LLM scripts and user settings
- **Expressions** (`src/llm_workers/expressions.py`): Expression evaluation system for dynamic values in YAML scripts
- **Starlark** (`src/llm_workers/starlark.py`): Starlark-like expression evaluation for advanced scripting
- **Utils** (`src/llm_workers/utils.py`): General utility functions and helpers
- **CLI** (`src/llm_workers/cli.py`): Command-line interface for batch processing
- **Chat** (`src/llm_workers/chat.py`): Interactive chat interface
- **Chat Completer** (`src/llm_workers/chat_completer.py`): Chat completion logic
- **Console** (`src/llm_workers/console.py`): Console output formatting and display
- **Token Tracking** (`src/llm_workers/token_tracking.py`): Token usage tracking and reporting
- **Cost Calculation** (`src/llm_workers/cost_calculation.py`): Cost estimation for LLM API calls

### Tool System (`src/llm_workers/tools/`)

- **Custom Tool** (`custom_tool.py`): Custom tool builder for YAML-defined tools
- **Fetch** (`fetch.py`): Web scraping and content retrieval tools
- **FS** (`fs.py`): File system operations (reading, writing, listing)
- **LLM Tool** (`llm_tool.py`): Nested LLM calls within workflows
- **Misc** (`misc.py`): Miscellaneous utility tools
- **Unsafe** (`unsafe.py`): Tools requiring confirmation (file operations, process execution)

### LLM Scripts

The system uses YAML configuration files called "LLM scripts" that define:
- **Models**: LLM providers and configurations (OpenAI, Bedrock, etc.)
- **Tools**: Available tools for the LLM (web fetching, file operations, custom tools)
- **Chat/CLI**: Interface configurations for interactive or batch processing

### Tool Implementation

Tools can be defined in three ways:
1. **Python classes** extending `langchain_core.tools.base.BaseTool`
2. **Factory functions** returning tool instances
3. **Custom tools** defined in YAML using statement composition (see `src/llm_workers/tools/custom_tool.py`)

Built-in tools are organized in `src/llm_workers/tools/` (see Tool System section above for details)

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

## Project Structure

```
llm-workers/
├── src/llm_workers/               # Main source code
│   ├── worker.py                  # Core worker implementation
│   ├── worker_utils.py            # Worker utility functions
│   ├── workers_context.py         # Script context management
│   ├── user_context.py            # User configuration management
│   ├── config.py                  # Configuration schemas
│   ├── expressions.py             # Expression evaluation
│   ├── starlark.py                # Starlark-like evaluation
│   ├── api.py                     # Abstract interfaces
│   ├── cli.py                     # CLI tool
│   ├── chat.py                    # Interactive chat interface
│   ├── chat_completer.py          # Chat completion logic
│   ├── console.py                 # Console output
│   ├── utils.py                   # General utilities
│   ├── token_tracking.py          # Token usage tracking
│   ├── cost_calculation.py        # Cost estimation
│   ├── generic-assistant.yaml     # Generic assistant LLM script
│   ├── default-anthropic-models.yaml   # Default Anthropic models configuration
│   ├── default-openai-models.yaml      # Default OpenAI models configuration
│   ├── default-openai-old-models.yaml  # Default OpenAI old (4-series) models configuration
│   └── tools/                     # Tool implementations
│       ├── custom_tool.py         # Custom tool builder
│       ├── fetch.py               # Web fetching tools
│       ├── fs.py                  # File system tools
│       ├── llm_tool.py            # Nested LLM tools
│       ├── misc.py                # Utility tools
│       └── unsafe.py              # Confirmed-action tools
├── tests/                         # Unit and integration tests
├── docs/                          # Documentation
│   ├── index.md                   # Main documentation
│   ├── llm-script.md              # Script format documentation
│   ├── examples.md                # Examples documentation
│   ├── release-notes.md           # Release notes
│   └── examples/                  # Example scripts and configurations
└── workspace/                     # Working directory for local tests
```