# AGENTS.md

This file provides guidance to AI Coding Assistants when working with code in this repository.

## Project Overview

LLM Workers is a Python library and command-line tools for experimenting with Large Language Models (LLMs). It provides a YAML-based configuration system for defining LLM interactions and custom tools.

**Architecture:** Monorepo with four separate packages:
- **llm-workers** - Core library (worker, config, tools)
- **llm-workers-console** - Console UI components (chat session, rich terminal output)
- **llm-workers-tools** - CLI entry points (llm-workers-cli, llm-workers-chat)
- **llm-workers-evaluation** - Evaluation framework (llm-workers-evaluate)

## Development Commands

### Setup and Installation
```bash
# Install all packages in development mode (from repository root)
poetry install

# This installs all four packages via path dependencies
```

### Testing
```bash
# Run all tests (from repository root)
poetry run python -m unittest discover tests/

# Run specific test file
poetry run python -m unittest tests.test_worker

# Run tests with verbose output
poetry run python -m unittest discover tests/ -v
```

### Building and Packaging

Each package must be built and published separately:

```bash
# Build core package
cd packages/llm-workers
poetry build

# Build console package
cd ../llm-workers-console
poetry build

# Build tools package
cd ../llm-workers-tools
poetry build

# Build evaluation package
cd ../llm-workers-evaluation
poetry build

cd ../..
```

### Publishing to PyPI

Publish in dependency order (after version bump in all four pyproject.toml files):

```bash
# 1. Publish core (no dependencies on other packages)
cd packages/llm-workers
poetry publish

# 2. Publish console (depends on llm-workers)
cd ../llm-workers-console
poetry publish

# 3. Publish tools (depends on llm-workers + llm-workers-console)
cd ../llm-workers-tools
poetry publish

# 4. Publish evaluation (depends on llm-workers)
cd ../llm-workers-evaluation
poetry publish

cd ../..
```

### Running the Tools
```bash
# CLI tool for batch processing
poetry run llm-workers-cli [--verbose] [--debug] <script_file> [prompts...]

# Interactive chat interface
poetry run llm-workers-chat [--verbose] [--debug] <script_file>

# Read from stdin
poetry run llm-workers-cli <script_file> -

# Run evaluation suites
poetry run llm-workers-evaluate [--verbose] [--debug] [-n iterations] <script_file> <evaluation_suite>
```

## Architecture

### Package Structure

The project is organized as a monorepo with four packages:

#### 1. llm-workers (Core Library)
Location: `packages/llm-workers/src/llm_workers/`

Core functionality with no UI dependencies:
- **Worker** (`worker.py`): Central orchestrator for LLM interactions and tool execution
- **Worker Utils** (`worker_utils.py`): Utility functions for worker operations
- **API** (`api.py`): Abstract base classes for UserContext and WorkersContext interfaces
- **User Context** (`user_context.py`): Manages user configuration, environment variables, and model initialization
- **Workers Context** (`workers_context.py`): Manages script configuration, tool loading, and MCP server connections
- **Configuration** (`config.py`): Defines the YAML configuration schema for LLM scripts and user settings
- **Expressions** (`expressions.py`): Expression evaluation system for dynamic values in YAML scripts
- **Starlark** (`starlark.py`): Starlark-like expression evaluation for advanced scripting
- **Utils** (`utils.py`): General utility functions and helpers
- **CLI Library** (`cli_lib.py`): Library functions for CLI batch processing (without main() entry point)
- **Token Tracking** (`token_tracking.py`): Token usage tracking and reporting
- **Cost Calculation** (`cost_calculation.py`): Cost estimation for LLM API calls
- **Cache** (`cache.py`): Caching functionality with TTL support
- **Chat History** (`chat_history.py`): Save/load chat sessions to YAML
- **Tool System** (`tools/`): Built-in tool implementations
  - `custom_tool.py` - Custom tool builder for YAML-defined tools
  - `fetch.py` - Web scraping and content retrieval tools
  - `fs.py` - File system operations (reading, writing, listing)
  - `llm_tool.py` - Nested LLM calls within workflows
  - `misc.py` - Miscellaneous utility tools
  - `unsafe.py` - Tools requiring confirmation (file operations, process execution)
- **Bundled YAML configs**: generic-assistant.yaml, default-*-models.yaml

Dependencies: langchain, pydantic, PyYAML, RestrictedPython

#### 2. llm-workers-console (Console UI)
Location: `packages/llm-workers-console/src/llm_workers_console/`

Rich terminal interface and chat functionality:
- **Console Controller** (`console.py`): Rich terminal output with markdown rendering, syntax highlighting, streaming
- **Chat Session** (`chat.py`): Interactive chat interface with command system (without main() entry point)
- **Chat Completer** (`chat_completer.py`): Auto-completion for slash commands and file paths

Dependencies: llm-workers, rich, prompt-toolkit, langchain-core

#### 3. llm-workers-tools (CLI Entry Points)
Location: `packages/llm-workers-tools/src/llm_workers_tools/`

Command-line entry points only:
- **CLI Main** (`cli_main.py`): main() entry point for llm-workers-cli
- **Chat Main** (`chat_main.py`): main() entry point for llm-workers-chat

Dependencies: llm-workers, llm-workers-console, model integrations (langchain-openai, langchain-anthropic, langchain-google-genai)

#### 4. llm-workers-evaluation (Evaluation Framework)
Location: `packages/llm-workers-evaluation/src/llm_workers_evaluation/`

Evaluation framework for running test suites against LLM scripts:
- **Config** (`config.py`): Pydantic models for evaluation suite YAML
- **Evaluation Library** (`evaluation_lib.py`): Core evaluation logic
- **Evaluation Main** (`evaluation_main.py`): main() entry point for llm-workers-evaluate

Dependencies: llm-workers, model integrations (langchain-openai, langchain-anthropic, langchain-google-genai)

### LLM Scripts

The system uses YAML configuration files called "LLM scripts" that define:
- **Models**: LLM providers and configurations (OpenAI, Anthropic, Google, etc.)
- **Tools**: Available tools for the LLM (web fetching, file operations, custom tools)
- **Chat/CLI**: Interface configurations for interactive or batch processing

### Tool Implementation

Tools can be defined in three ways:
1. **Python classes** extending `langchain_core.tools.base.BaseTool`
2. **Factory functions** returning tool instances
3. **Custom tools** defined in YAML using statement composition (see `packages/llm-workers/src/llm_workers/tools/custom_tool.py`)

Built-in tools are organized in `packages/llm-workers/src/llm_workers/tools/`

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
llm-workers/                                    # Repository root
├── pyproject.toml                              # Meta-project (llm-workers-dev)
├── poetry.lock                                 # Single lock file for development
├── README.md
├── AGENTS.md
├── tests/                                      # All tests (not moved to packages)
│   ├── test_worker.py
│   ├── test_custom_tool.py
│   ├── test_chat_history.py
│   └── ...
├── docs/                                       # Documentation
│   ├── index.md
│   ├── llm-script.md
│   ├── custom-tools.md
│   ├── built-in-tools.md
│   ├── evaluation.md
│   ├── examples.md
│   └── release-notes.md
├── workspace/                                  # Working directory for local tests
│
└── packages/                                   # Monorepo packages
    │
    ├── llm-workers/                            # Core library package
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── src/llm_workers/
    │       ├── __init__.py
    │       ├── api.py
    │       ├── worker.py
    │       ├── worker_utils.py
    │       ├── config.py
    │       ├── user_context.py
    │       ├── workers_context.py
    │       ├── expressions.py
    │       ├── starlark.py
    │       ├── utils.py
    │       ├── token_tracking.py
    │       ├── cost_calculation.py
    │       ├── cache.py
    │       ├── chat_history.py
    │       ├── cli_lib.py                      # CLI library (no main)
    │       ├── generic-assistant.yaml
    │       ├── default-*.yaml
    │       └── tools/
    │           ├── custom_tool.py
    │           ├── fetch.py
    │           ├── fs.py
    │           ├── llm_tool.py
    │           ├── misc.py
    │           └── unsafe.py
    │
    ├── llm-workers-console/                    # Console UI package
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── src/llm_workers_console/
    │       ├── __init__.py
    │       ├── console.py
    │       ├── chat.py                         # ChatSession (no main)
    │       └── chat_completer.py
    │
    ├── llm-workers-tools/                      # CLI tools package
    │   ├── pyproject.toml
    │   ├── README.md
    │   └── src/llm_workers_tools/
    │       ├── __init__.py
    │       ├── cli_main.py                     # main() for llm-workers-cli
    │       └── chat_main.py                    # main() for llm-workers-chat
    │
    └── llm-workers-evaluation/                 # Evaluation framework package
        ├── pyproject.toml
        ├── README.md
        └── src/llm_workers_evaluation/
            ├── __init__.py
            ├── config.py                       # Pydantic models for evaluation YAML
            ├── evaluation_lib.py               # Core evaluation logic
            └── evaluation_main.py              # main() for llm-workers-evaluate
```

## Dependency Management

### For Development
Run at repository root only:
```bash
poetry lock
poetry install
```

This installs all three packages in development mode via path dependencies.

### For Publishing
No lock files needed in individual packages. Just `poetry build` in each package directory.

### Checking Dependencies
```bash
# Show dependency tree
poetry show --tree

# Check specific package
poetry show <package-name>

# Find what requires a package
poetry show --why <package-name>  # Poetry 1.8+
```

## Import Patterns

### Core Package Imports
```python
from llm_workers import Worker
from llm_workers.config import WorkersConfig
from llm_workers.api import UserContext, WorkersContext
from llm_workers.cli_lib import run_llm_script
```

### Console Package Imports
```python
from llm_workers_console import ChatSession, ConsoleController
from llm_workers_console.chat import chat_with_llm_script
```

### Cross-Package Imports
Console package imports from core:
```python
from llm_workers.worker import Worker
from llm_workers.config import DisplaySettings
```

Tools package imports from both:
```python
from llm_workers.cli_lib import run_llm_script
from llm_workers_console.chat import chat_with_llm_script
```

### Evaluation Package Imports
```python
from llm_workers_evaluation import run_evaluation, format_results
from llm_workers_evaluation.config import EvaluationSuiteFile
```

## Version Management

When releasing, update versions in all four `pyproject.toml` files:
- `packages/llm-workers/pyproject.toml`
- `packages/llm-workers-console/pyproject.toml`
- `packages/llm-workers-tools/pyproject.toml`
- `packages/llm-workers-evaluation/pyproject.toml`
- Root `pyproject.toml` (for consistency)

Also update dependency version constraints in console, tools, and evaluation packages to match.
