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
* [Development](#development)
   * [Packaging for release](#packaging-for-release)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: dmikhaylov, at: Mon Sep 29 22:48:05 EEST 2025 -->

<!--te-->

# Project Overview

Simple library and command-line tools for experimenting with LLMs.

See [docs/README.md](docs/README.md) for more detailed documentation.

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
- [0.1.0-rc12](https://github.com/MrBagheera/llm-workers/milestone/15?closed=1)
- [0.1.0-rc13](https://github.com/MrBagheera/llm-workers/milestone/16?closed=1)
- [1.0.0-rc1](https://github.com/MrBagheera/llm-workers/milestone/17?closed=1)

## Version 1.0.0

- [0.1.0](https://github.com/MrBagheera/llm-workers/milestone/7)

## Further Ideas

https://github.com/MrBagheera/llm-workers/milestone/17

- basic assistant functionality
- simplify result referencing in chains - `{last_result}` and `store_as`
- `prompts` section
- `for_each` statement
- run as MCP client
- support accessing nested JSON elements in templates
- structured output
- async versions for all built-in tools
- "safe" versions of "unsafe" tools
- write trail
- resume trail
- support acting as MCP server (expose `custom_tools`)
- support acting as MCP host (use tools from configured MCP servers)


# Development

## Packaging for release

- Bump up version in `pyproject.toml`
- Run `poetry build`
- Run `poetry publish` to publish to PyPI