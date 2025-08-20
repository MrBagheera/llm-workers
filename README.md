Table of Contents
=================

<!--ts-->
* [Project Overview](#project-overview)
   * [Goals](#goals)
   * [What This Project Is <em>Not</em>](#what-this-project-is-not)
* [Configuration](#configuration)
* [Example scripts](#example-scripts)
* [Running](#running)
* [Releases](#releases)
   * [Next](#next)
   * [Version 0.1.0](#version-010)
   * [Version 0.1.1](#version-011)
   * [Further Ideas](#further-ideas)
* [Devlopment](#devlopment)
   * [Packaging for release](#packaging-for-release)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: dmikhaylov, at: Wed Aug 20 16:26:22 EEST 2025 -->

<!--te-->

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

LLM scripts are YAML configuration files that define how to interact with large language models (LLMs) and what
tools LLMs can use. You should treat them like a normal scripts. In particular - DO NOT run LLM scripts from
unknown / untrusted sources. Scripts can easily download and run malicious code on your machine, or submit your secrets
to some web site.

See [LLM Script.md](LLM%20Script.md) file for reference.

# Example scripts

The [`examples`](examples/) directory contains sample LLM scripts demonstrating various features:

- **[Metacritic-monkey.yaml](examples/Metacritic-monkey.yaml)** - Custom tools with statement composition, web fetching tools, inline tool definitions, match statements with stubbed data, LLM tool integration, template variables, UI hints
- **[explicit-approval-tools.yaml](examples/explicit-approval-tools.yaml)** - Explicit approval workflow with token-based confirmation system, custom tool composition with inline imports, approval tools (request/validate/consume), safe execution of potentially dangerous operations
- **[find-concurrency-bugs.yaml](examples/find-concurrency-bugs.yaml)** - CLI mode with statement composition, AWS Bedrock model configuration with thinking mode, file reading tools, structured LLM output with JSON schema validation, inline tool definitions
- **[navigation-planning.yaml](examples/navigation-planning.yaml)** - Multiple model configurations (OpenAI + Bedrock), web fetching tools with markdown conversion, nested custom tools, tool composition with return_direct flag, CLI mode with tool restrictions, chat mode configuration
- **[reformat-Scala.yaml](examples/reformat-Scala.yaml)** - AWS Bedrock model configuration, CLI mode with complex file processing pipeline, match statements with conditional file operations, file I/O tools, LLM tool integration for code transformation
- **[using-context-help-tool.yaml](examples/using-context-help-tool.yaml)** - Shared configuration section for reusable data, context help tools with dynamic key resolution, thinking mode configuration with rate limiting, chat mode with advanced model settings, literal type parameters

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
- `--verbose` - increases verbosity of stderr logging, can be used multiple times (info / debug)
- `--debug` - increases amount of debug logging to file/stderr, can be used multiple times (debug only main worker / 
debug whole `llm_workers` package / debug all)


# Releases

- [0.1.0-alpha5](https://github.com/MrBagheera/llm-workers/milestone/1)
- [0.1.0-rc1](https://github.com/MrBagheera/llm-workers/milestone/3)
- [0.1.0-rc2](https://github.com/MrBagheera/llm-workers/milestone/4)
- [0.1.0-rc3](https://github.com/MrBagheera/llm-workers/milestone/5)
- [0.1.0-rc4](https://github.com/MrBagheera/llm-workers/milestone/6)
- [0.1.0-rc5](https://github.com/MrBagheera/llm-workers/milestone/8)
- [0.1.0-rc6](https://github.com/MrBagheera/llm-workers/milestone/9)
- [0.1.0-rc7](https://github.com/MrBagheera/llm-workers/milestone/10)
- [0.1.0-rc8](https://github.com/MrBagheera/llm-workers/milestone/11)
- [0.1.0-rc9](https://github.com/MrBagheera/llm-workers/milestone/12)
- [0.1.0-rc10](https://github.com/MrBagheera/llm-workers/milestone/13)

## Next

- [0.1.0](https://github.com/MrBagheera/llm-workers/milestone/7)

## Version 0.1.0

- basic assistant functionality

## Version 0.1.1

- simplify result referencing in chains - `{last_result}` and `store_as`
- `prompts` section
- `for_each` statement
- support accessing nested JSON elements in templates

## Further Ideas

- structured output
- async versions for all built-in tools
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