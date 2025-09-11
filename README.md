Table of Contents
=================

<!--ts-->
* [Project Overview](#project-overview)
   * [Goals](#goals)
   * [What This Project Is <em>Not</em>](#what-this-project-is-not)
* [Configuration](#configuration)
   * [User-specific Configuration](#user-specific-configuration)
   * [LLM Scripts](#llm-scripts)
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
<!-- Added by: dmikhaylov, at: Mon Sep  8 06:50:59 EEST 2025 -->

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

## User-specific Configuration

User-specific configuration is stored in `~/.config/llm-workers/config.yaml`.

```yaml
models:
  - name: <model_name>
    provider: <provider_name>
    model: <model_id>
    rate_limiter: # Optional
      requests_per_second: <float>
      check_every_n_seconds: <float> # Optional, defaults to 0.1
      max_bucket_size: <float>
    # Optional model parameters (any additional parameters are passed directly to the model)
    temperature: <float>
    max_tokens: <int>
    # [additional parameters...]

```

On first launch, `llm-workers` CLI will guide you through initial setup. You can choose from:
- **OpenAI presets**: Configure GPT-4o, GPT-4o-mini for the standard model slots
- **Anthropic presets**: Configure Claude models for the standard model slots
- **Manual configuration**: Set up custom model configurations

### Models Section

Defines the LLMs to use. Configuration must define at least those standard models:
- `fast`: Optimized for speed and simple tasks
- `default`: Balanced performance for most use cases
- `thinking`: Advanced reasoning with internal thought processes

There are two types of model configurations:

#### Standard Model Configuration
- `name`: Identifier for the model
- `provider`: Service provider (e.g., `bedrock`, `bedrock_converse`, `openai`)
- `model`: Model identifier
- `rate_limiter`: Optional rate limiting configuration
- `config`: Optional model-specific parameters (overrides main section parameters if used)

```yaml
models:
  - name: default
    provider: openai
    model: gpt-4o
    rate_limiter:
      requests_per_second: 1.0
      max_bucket_size: 10
    # model specific parameters defined inline
    temperature: 0.7
    max_tokens: 1500

  - name: thinking
    provider: bedrock_converse
    model: us.anthropic.claude-3-7-sonnet-20250219-v1:0
    config: # model specific parameters defined in separate section
      temperature: 1
      max_tokens: 32768
      additional_model_request_fields:
        thinking:
          type: enabled
          budget_tokens: 16000
```

#### Import Model Configuration
- `name`: Identifier for the model
- `import_from`: Fully-qualified Python class/function path for custom model implementation
- `rate_limiter`: Optional rate limiting configuration
- `config`: Optional parameters passed to the model constructor/factory (overrides main section parameters if used)

The imported symbol can be:
- A `BaseChatModel` instance (used directly)
- A class (instantiated with config parameters)
- A function/method (called with config parameters to create the model)

```yaml
models:
  - name: custom_model
    import_from: my_module.models.CustomChatModel
    rate_limiter:
      requests_per_second: 2.0
      max_bucket_size: 5
    config:
      base_url: "https://api.example.com"
      api_key: "your-api-key"
      model_type: "advanced"
      timeout: 30
```

#### Model Parameters

Any extra parameters not defined above will be passed to the model. If model requires
specific parameters that conflict with standard parameters, those specific parameters can be defined in the `config` section.
In this case no parameters from main section will be passed to the model, only those defined in `config`.


## LLM Scripts

LLM scripts are YAML configuration files that define how to interact with large language models (LLMs) and what
tools LLMs can use. You should treat them like a normal scripts. In particular - DO NOT run LLM scripts from
unknown / untrusted sources. Scripts can easily download and run malicious code on your machine, or submit your secrets
to some web site.

See [LLM Script.md](LLM%20Script.md) file for reference.

# Example scripts

The [`examples`](examples/) directory contains sample LLM scripts demonstrating various features:

- **[Metacritic-monkey.yaml](examples/Metacritic-monkey.yaml)** - Custom tools with statement composition, web fetching tools, inline tool definitions, match statements with stubbed data, LLM tool integration, template variables, UI hints
- **[explicit-approval-tools.yaml](examples/explicit-approval-tools.yaml)** - Explicit approval workflow with token-based confirmation system, custom tool composition with inline imports, approval tools (request/validate/consume), safe execution of potentially dangerous operations
- **[find-concurrency-bugs.yaml](examples/find-concurrency-bugs.yaml)** - CLI mode with statement composition, file reading tool, thinking model via model_ref, structured JSON output (by instruction)
- **[navigation-planning.yaml](examples/navigation-planning.yaml)** - Web fetching tools with markdown conversion, nested custom tools, tool composition with return_direct flag, CLI mode with tool restrictions, chat mode configuration
- **[reformat-Scala.yaml](examples/reformat-Scala.yaml)** - CLI mode with complex file processing pipeline, match statements with conditional file operations, file I/O tools, LLM tool integration for code transformation

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
- [0.1.0-rc11](https://github.com/MrBagheera/llm-workers/milestone/14?closed=1)

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