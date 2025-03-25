# Project Overview

Simple library and command-line tools for experimenting with LLMs.

## Goals

Provide developers with a simple way to experiment with LLMs and LangChain:
- Easy setup and configuration
- Basic chat / CLI tools
- Own tool integration (both in Python and via composition of other tools)
- Support for less-mainstream LLMs like AWS Bedrock

## What This Project Is *Not*

- **Not an end-user tool**: This project is geared toward developers and researchers with knowledge of Python, LLM capabilities, and programming fundamentals.
- **Not a complete automation system**: It relies on human oversight and guidance for optimal performance.

# Configuration

Configuration is done via YAML-based "LLM scripts". See [`examples`](examples/) directory. This is WIP and
subject to change without notice until version 0.1.0.

# Running 

Library comes with two command-line tools that can be used to run LLM scripts: `llm-workers-cli` and `llm-workers-chat`.

To run LLM script with default prompt:
```shell
llm-workers-cli [--verbose] [--debug] <script_file>
```

To run LLM script with prompt(s) as command-line arguments:
```shell
llm-workers-cli [--verbose] [--debug] <script_file> [<prompt1> ... <promptN>]
```

To run LLM script with prompt(s) read from `stdin`, each line as separate prompt:
```shell
llm-workers-cli [--verbose] [--debug] <script_file> --
```

Results of LLM script execution will be printed to the `stdout` without any
extra formatting. 

To chat with LLM script:
```shell
llm-workers-chat [--verbose] [--debug] <script_file>
```
The tool provides terminal chat interface where user can interact with LLM script.

Common flags:
- `--verbose` flag triggers some debug prints to stderr
- `--debug` - enables LangChain's debug mode, which prints additional information about script execution

# Releases

## Next

- [0.1.0-alpha5](https://github.com/MrBagheera/llm-workers/milestone/1)

## Version 0.1.0

- simplify result referencing in chains - `{last_result}` and `store_as`
- `prompts` section
- `for_each` statement
- `ReadFileTool`: support `first_lines` / `last_lines`
- `WriteFileTool`: support `append`
- add `ListFilesTool`
- add `ShellTool`
- add `Execute` tool
- support `ui_hint` tool flag
- support `confirmation_prompt`/`confirmation_args` tool flags
- support accessing nested JSON elements in templates
- review error handling
- improve documentation - installing, running, developing

## Further Ideas

- structured output
- async versions for all built-in tools
- proper error handling
- "safe" versions of "unsafe" tools
- write trail
- resume trail
- support acting as MCP server (expose `custom_tools`)
- support acting as MCP host (use tools from configured MCP servers)

# Devlopment

## Packaging for release

- Bump up version in `pyproject.toml`
- Run `poetry build`
- Run `poetry publish` to publish to PyPI