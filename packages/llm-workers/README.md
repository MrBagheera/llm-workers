# llm-workers

Core library for experimenting with Large Language Models (LLMs).

## Overview

`llm-workers` is the core package providing:

- **Worker System**: Central orchestrator for LLM interactions and tool execution
- **Configuration**: YAML-based configuration system for LLM scripts
- **Tools**: Built-in tools for file operations, web scraping, nested LLM calls, and custom tool building
- **Expression System**: Dynamic value evaluation with Starlark-like expressions
- **LLM Integrations**: Support for OpenAI, Anthropic, Google, and other providers via LangChain
- **MCP Server Support**: Integration with Model Context Protocol servers

## Installation

```bash
pip install llm-workers
```

For console/chat interface, also install:

```bash
pip install llm-workers-console llm-workers-tools
```

## Usage

### Programmatic Usage

```python
from llm_workers.user_context import StandardUserContext
from llm_workers.workers_context import StandardWorkersContext
from llm_workers.cli_lib import run_llm_script

# Run a script programmatically
user_context = StandardUserContext()
context = StandardWorkersContext.create("my-script.yaml", user_context=user_context)
```

### Command-Line Usage

For command-line tools, install `llm-workers-tools`:

```bash
llm-workers-cli my-script.yaml "prompt here"
llm-workers-chat my-script.yaml
```

## Documentation

Full documentation: https://mrbagheera.github.io/llm-workers/

## License

See main repository for license information.
