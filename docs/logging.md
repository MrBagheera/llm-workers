---
layout: default
title: Logging
nav_order: 7
---

## Table of contents
{: .no_toc }

* TOC
{:toc}

# Logging

All `llm-workers` command-line tools write logs to a file and optionally to the console (stderr). The amount of
detail is controlled by command-line flags.

## Default Behavior

By default (no flags):

- **Log file**: written to `llm-workers.log` (or `llm-workers-evaluate.log` for `llm-workers-evaluate`) in the
  current directory, opened fresh on each run. Format: `<timestamp>: <logger> - <level> - <message>`.
- **Console (stderr)**: only `ERROR`-level messages are shown.
- **Log file level**: `INFO` for all loggers.

The `llm-workers-chat` tool always prints the path to the log file on startup:
```
Logging to /path/to/llm-workers.log
```

## Command-Line Arguments

All three tools (`llm-workers-cli`, `llm-workers-chat`, `llm-workers-evaluate`) accept the same logging flags:

| Flag | Repeatable | Effect |
|------|-----------|--------|
| `--verbose` | yes | Increases the **console (stderr)** log level |
| `--debug` | yes | Enables debug logging for specific loggers in the **log file** |
| `--level <module>=<level>` | yes | Sets the log level for a specific logger |

## `--verbose`: Console Output

Each additional `--verbose` flag lowers the console threshold:

| Flags | Console level |
|-------|--------------|
| _(none)_ | `ERROR` |
| `--verbose` | `INFO` |
| `--verbose --verbose` | `DEBUG` |
| `--verbose --verbose --verbose` | `ALL` (everything, including `TRACE`) |

Console messages use the format: `<logger-name>: <message>`.

## `--debug`: File Debug Logging

The `--debug` flag progressively enables debug logging for specific loggers in the log file:

| Flags | Loggers set to `DEBUG` in file |
|-------|-------------------------------|
| _(none)_ | _(none — root logger at `INFO`)_ |
| `--debug` | `llm-script`, `llm_workers.worker` |
| `--debug --debug` | `llm_workers` (entire core package) |
| `--debug --debug --debug` | root logger (everything) set to `DEBUG` |
| `--debug --debug --debug --debug` | root logger set to `ALL` (everything including `TRACE`) |

With `--debug`, the log file will include the full LLM input/output messages (formatted as YAML), tool call
arguments and results, and other internal worker details.

With `--debug --debug`, all internal `llm_workers` operations are logged at debug level — tool filtering,
MCP server connections, cache hits/misses, process execution, web fetch details, and more.

## `--level`: Per-Module Overrides

For fine-grained control, use `--level` to set any logger to any level:

```shell
llm-workers-cli --level llm_workers.worker=DEBUG --level llm_workers.tools.fetch=INFO script.yaml
```

Level names are case-insensitive. Aliases `WARN` and `FATAL` are also accepted. Numeric levels work too:
```shell
llm-workers-cli --level llm_workers.cache=5 script.yaml   # TRACE level
```

## Custom Logging from Scripts

LLM scripts can emit log messages from within expressions using the built-in `log()` function. Log messages
are emitted at `DEBUG` level and appear in the log file when `--debug` is active (or on the console with
`--verbose --verbose`).

**Signature**: `log(msg, *args)` — `msg` is a format string, `args` are substituted with `%`-style formatting.

**One-liner pattern** — log a value without affecting the pipeline (the expression still returns `_`):
```yaml
- eval: ${log("validation:\n%s", as_yaml(_)) or _}
```

The `as_yaml()` helper converts any value to a `LazyFormatter` object that renders as YAML only when the
log message is actually emitted — so there is no formatting cost when debug logging is disabled.
`log()` returns `None`, so `or _` passes the original value through unchanged.

### `as_yaml()` and the `trim` parameter

`as_yaml(value, trim=True)` controls how long strings inside the YAML output are trimmed:

| `trim` value | Behavior |
|---|---|
| `True` _(default)_ | Each string is trimmed to 1 visual line (80 characters) |
| `N` (integer) | Each string is trimmed to N visual lines (80 chars each); excess is shown as `[N lines trimmed]` |
| `False` | No trimming; multiline strings are rendered as YAML literal blocks (`\|`) |

### `%s` vs `%r` in format strings

The `LazyFormatter` returned by `as_yaml()` behaves differently depending on how Python formats it:

- **`%s`** calls `__str__()` — uses the `trim` value passed to `as_yaml()` (default: trim to 1 line).
- **`%r`** calls `__repr__()` — always renders the full value with **no trimming**, regardless of the `trim` parameter.

Use `%s` for brief annotations in normal debug output, and `%r` when you need to inspect the full value.

### Effect of logger level on trimming

When the script's logger is set to `ALL` level (numeric level 1, enabled by `--debug --debug --debug --debug`
or `--level llm-script=1`), `as_yaml()` automatically disables trimming — even for `%s` format specifiers.
This means you can turn up verbosity globally without modifying your script expressions.

### Examples

```yaml
# Log a simple message (no as_yaml needed for plain values)
- eval: ${log("processing item: %s", item_id) or _}

# Log trimmed YAML (default: 1 line per string field)
- eval: ${log("validation:\n%s", as_yaml(_)) or _}

# Log as YAML, trimming long strings to at most 3 lines each
- eval: ${log("details: %s", as_yaml(_, trim=3)) or _}

# Log full untrimmed YAML using %r
- eval: ${log("full state: %r", as_yaml(_)) or _}

# Log before and after a transformation
- eval: ${log("input: %s", as_yaml(_)) or _}
- call: ...
- eval: ${log("output: %s", as_yaml(_)) or _}
```

## Common Logger Names

These logger names appear in log output and can be targeted with `--level`:

| Logger | Description                                                                                    |
|--------|------------------------------------------------------------------------------------------------|
| `llm-script` | Default logger for script-level `log()` calls                                                  |
| `llm-script.<scope>` | Scoped script logger (e.g. `llm-script.cli`, `llm-script.evaluation`, `llm-script.<tool-name>`) |
| `llm_workers.worker` | LLM input/output, tool calls, main execution flow                                              |
| `llm_workers.workers_context` | Script loading, tool registration, MCP connections                                             |
| `llm_workers.user_context` | Model registration, user config loading                                                        |
| `llm_workers.cache` | Cache hits/misses, TTL management                                                              |
| `llm_workers.tools.custom_tool.<name>` | Per-tool logger for custom tools                                                               |
| `llm_workers.tools.fetch` | Web fetch tools                                                                                |
| `llm_workers.tools.fs` | Filesystem tools                                                                               |
| `llm_workers.tools.unsafe` | Shell/process execution tools                                                                  |
| `llm_workers.tools.llm_tool` | Nested LLM tool calls                                                                          |
| `llm_workers_evaluation.evaluation` | Evaluation framework, test results                                                             |
| `llm_workers_evaluation.evaluation.<test>` | Per-test logger                                                                                |
