Table of Contents
=================

<!--ts-->
* [Project Overview](#project-overview)
* [Releases](#releases)
  * [Past Releases](#past-releases)
  * [Version 1.0.0](#version-100)
  * [Further Ideas](#further-ideas)
* [Development](#development)
  * [Packaging for release](#packaging-for-release)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

<!--te-->

# Project Overview

Simple library and command-line tools for experimenting with LLMs.

See [docs/index.md](docs/index.md) for more detailed documentation.

# Releases

## Past Releases

See [docs/release-notes.md](docs/release-notes.md) for detailed release notes.

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