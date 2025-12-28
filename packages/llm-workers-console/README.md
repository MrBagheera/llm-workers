# llm-workers-console

Console UI components for LLM Workers - interactive chat interface and rich terminal output.

## Overview

`llm-workers-console` provides the user interface components for LLM Workers:

- **ChatSession**: Interactive chat interface with command system
- **ConsoleController**: Rich terminal output with markdown rendering, syntax highlighting, and streaming
- **ChatCompleter**: Auto-completion for slash commands and file paths

## Installation

```bash
pip install llm-workers-console
```

This package depends on `llm-workers` (core library).

## Features

- **Interactive Chat**: Multi-line input with history, command completion
- **Rich Output**: Markdown rendering, syntax highlighting, streaming responses
- **Chat Commands**: `/help`, `/rewind`, `/model`, `/display`, `/cost`, `/export`, `/save`, `/clear`, `/exit`
- **Session Management**: Save and resume chat sessions
- **Token Usage Display**: Real-time token tracking and cost estimation
- **File Monitoring**: Auto-open changed files during chat

## Usage

### Programmatic Usage

```python
from llm_workers_console import chat_with_llm_script

# Start an interactive chat session
chat_with_llm_script("my-script.yaml")
```

### With Command-Line Tool

For command-line usage, install `llm-workers-tools`:

```bash
pip install llm-workers-tools
llm-workers-chat my-script.yaml
```

## Documentation

Full documentation: https://mrbagheera.github.io/llm-workers/

## License

See main repository for license information.
