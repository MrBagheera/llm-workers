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

- [1.0.0](https://github.com/MrBagheera/llm-workers/milestone/7)

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

The project includes automated scripts in the `scripts/` directory to streamline the release process.

1. **Update Version**
   ```bash
   # Set version across all packages
   ./scripts/set-version.sh 1.0.0-rc9
   ```
   This updates version numbers in all `pyproject.toml` files (including root) and dependency constraints.

2. **Update Release Notes**

   Edit `docs/release-notes.md` and add release notes for the new version:
   ```markdown
   ## Version 1.0.0-rc9 (YYYY-MM-DD)

   ### Features
   - New feature description

   ### Bug Fixes
   - Bug fix description

   ### Breaking Changes
   - Breaking change description (if any)
   ```

3. **Review and Commit Changes**
   ```bash
   # Review version changes
   git diff

   # Commit version bump
   git add -A
   git commit -m "Bump version to 1.0.0-rc9"
   ```

4. **Build All Packages**
   ```bash
   ./scripts/build-all.sh
   ```
   This builds all three packages in the correct order.

5. **Test on TestPyPI (Recommended)**
   ```bash
   # Publish to TestPyPI
   ./scripts/publish-testpypi.sh

   # Test installation
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     llm-workers-tools
   ```

6. **Publish to PyPI**
   ```bash
   # Publish to production PyPI
   ./scripts/publish-pypi.sh
   ```
   **Important:** This publishes in dependency order (core → console → tools) with proper wait times.

7. **Create Git Tag and GitHub Release**
   ```bash
   # Get current version
   VERSION=$(cd packages/llm-workers && poetry version -s)

   # Create and push tag
   git tag "v$VERSION"
   git push origin "v$VERSION"
   ```

   Then create a GitHub release:
   - Go to https://github.com/MrBagheera/llm-workers/releases/new
   - Select the tag you just created
   - Copy release notes from `docs/release-notes.md`
   - Publish the release

### Available Scripts

- **`./scripts/set-version.sh <version>`** - Update version across all packages
- **`./scripts/build-all.sh`** - Build all packages
- **`./scripts/publish-testpypi.sh`** - Publish to TestPyPI for testing
- **`./scripts/publish-pypi.sh`** - Publish to production PyPI
- **`./scripts/clean-build.sh`** - Clean all build artifacts


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
