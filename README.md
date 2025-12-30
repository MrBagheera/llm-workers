Table of Contents
=================

<!--ts-->
* [Project Overview](#project-overview)
* [Installation](#installation)
* [Releases](#releases)
  * [Past Releases](#past-releases)
  * [Version 1.0.0](#version-100)
  * [Further Ideas](#further-ideas)
* [Development](#development)
  * [Project Structure](#project-structure)
  * [Setup](#setup)
  * [Testing](#testing)
  * [Packaging for Release](#packaging-for-release)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

<!--te-->

# Project Overview

Simple library and command-line tools for experimenting with LLMs.

**Architecture:** Monorepo with three separate packages:
- **llm-workers** - Core library (worker, config, tools)
- **llm-workers-console** - Console UI components (chat session, rich terminal output)
- **llm-workers-tools** - CLI entry points (llm-workers-cli, llm-workers-chat)

See [docs/index.md](https://mrbagheera.github.io/llm-workers/) for more detailed documentation.

# Installation

## For End Users

Install the tools package (includes core and console as dependencies):

```bash
pip install llm-workers-tools
```

This provides:
- `llm-workers-cli` - Batch processing tool
- `llm-workers-chat` - Interactive chat interface

## For Library Usage

If you only need the core library without CLI tools:

```bash
# Core library only (no UI)
pip install llm-workers

# Core + console UI (programmatic use)
pip install llm-workers-console
```

## Optional Dependencies

For HTML parsing tools:

```bash
pip install llm-workers[html]
```

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

## Project Structure

This is a monorepo containing three packages:

```
llm-workers/                        # Repository root
├── packages/
│   ├── llm-workers/                # Core library
│   ├── llm-workers-console/        # Console UI
│   └── llm-workers-tools/          # CLI entry points
├── tests/                          # All tests
├── docs/                           # Documentation
└── pyproject.toml                  # Meta-project
```

See [AGENTS.md](AGENTS.md) for detailed architecture and file structure.

## Setup

```bash
# Clone the repository
git clone https://github.com/MrBagheera/llm-workers.git
cd llm-workers

# Install all packages in development mode
poetry install

# This installs all three packages via path dependencies
```

## Testing

```bash
# Run all tests
poetry run python -m unittest discover tests/

# Run with verbose output
poetry run python -m unittest discover tests/ -v

# Run specific test
poetry run python -m unittest tests.test_worker
```

## Packaging for Release

### Version Management

Before releasing, update versions in all three `pyproject.toml` files:
- `packages/llm-workers/pyproject.toml`
- `packages/llm-workers-console/pyproject.toml`
- `packages/llm-workers-tools/pyproject.toml`

Also update dependency version constraints in console and tools packages.

### Build Each Package

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

cd ../..
```

### Publish to PyPI

**Important:** Publish in dependency order:

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

cd ../..
```

### Test on TestPyPI First (Recommended)

```bash
# Configure TestPyPI (one time)
poetry config repositories.testpypi https://test.pypi.org/legacy/

# Publish to TestPyPI
cd packages/llm-workers
poetry publish -r testpypi

cd ../llm-workers-console
poetry publish -r testpypi

cd ../llm-workers-tools
poetry publish -r testpypi

cd ../..

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llm-workers-tools
```

## Running CLI Tools in Development

```bash
# CLI tool
poetry run llm-workers-cli [--verbose] [--debug] <script_file> [prompts...]

# Chat interface
poetry run llm-workers-chat [--verbose] [--debug] <script_file>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `poetry run python -m unittest discover tests/`
5. Submit a pull request

See [AGENTS.md](AGENTS.md) for detailed development guidelines.
