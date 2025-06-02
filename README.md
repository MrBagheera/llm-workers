Table of Contents
=================

* [Project Overview](#project-overview)
    * [Goals](#goals)
    * [What This Project Is <em>Not</em>](#what-this-project-is-not)
* [Configuration](#configuration)
* [Running](#running)
* [Releases](#releases)
    * [Next](#next)
    * [Version 0.1.0](#version-010)
    * [Version 0.1.1](#version-011)
    * [Further Ideas](#further-ideas)
* [Devlopment](#devlopment)
    * [Packaging for release](#packaging-for-release)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

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

See [LLM Scipt.md](LLM%20Scipt.md) file for reference.


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

## Next

- [0.1.0-rc2](https://github.com/MrBagheera/llm-workers/milestone/4)

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