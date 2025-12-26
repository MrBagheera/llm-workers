---
layout: default
title: Examples
nav_order: 3
---

# Example scripts

This page indexes the example LLM scripts shipped with the project.

> Tip: these example files are YAML, intended to be run with `llm-workers-cli` or `llm-workers-chat`.

## Examples

- [Metacritic-monkey.yaml](examples/Metacritic-monkey.yaml)
- [explicit-approval-tools.yaml](examples/explicit-approval-tools.yaml)
- [find-concurrency-bugs.yaml](examples/find-concurrency-bugs.yaml)
- [mcp-example.yaml](examples/mcp-example.yaml)
- [navigation-planning.yaml](examples/navigation-planning.yaml)
- [reformat-Scala.yaml](examples/reformat-Scala.yaml)
- [starlark-demo.yaml](examples/starlark-demo.yaml)
- [toolkit-example.yaml](examples/toolkit-example.yaml)

## Running an example

```bash
poetry run llm-workers-cli docs/examples/<example>.yaml
# or
poetry run llm-workers-chat docs/examples/<example>.yaml
```

