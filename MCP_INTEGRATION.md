# MCP (Model Context Protocol) Integration

This document describes the MCP (Model Context Protocol) servers support in llm-workers.

## Overview

MCP integration allows you to connect to external MCP servers and use their tools alongside the built-in tools in llm-workers. Tools from MCP servers are automatically prefixed with the server name to avoid conflicts.

## Configuration

Add an `mcp:` section to your YAML configuration file:

```yaml
mcp:
  # Server name (used as tool prefix)
  server_name:
    transport: "stdio" | "streamable_http"

    # For stdio transport (local subprocess)
    command: "command_to_run"
    args: ["arg1", "arg2"]

    # For HTTP transport (remote server)
    url: "http://localhost:8000/mcp"

    # Tool filtering (optional, default: ["*"])
    tools:
      - "pattern*"      # Include tools matching pattern
      - "!exclude*"     # Exclude tools matching pattern

    # UI hints (optional, default: [])
    ui_hints_for:
      - "pattern*"      # Show UI hints for matching tools

    # Confirmation (optional, default: [])
    require_confirmation_for:
      - "pattern*"      # Require confirmation for matching tools
```

## Transport Types

### Stdio Transport

For local MCP servers running as subprocesses:

```yaml
mcp:
  math:
    transport: "stdio"
    command: "python"
    args: ["/path/to/math_server.py"]
```

### HTTP Transport

For remote MCP servers accessible via HTTP:

```yaml
mcp:
  weather:
    transport: "streamable_http"
    url: "http://localhost:8000/mcp"
```

## Tool Filtering

Use glob patterns to include or exclude specific tools:

```yaml
mcp:
  github:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    tools:
      - "gh*"           # Include all gh tools
      - "!gh_write*"    # Exclude write operations
      - "!gh_delete*"   # Exclude delete operations
```

Pattern matching uses Unix shell-style wildcards:
- `*` - matches everything
- `?` - matches any single character
- `[seq]` - matches any character in seq
- `[!seq]` - matches any character not in seq
- `!pattern` - negation (exclude matching tools)

## Environment Variables

Reference environment variables in args using the `env.` prefix:

```yaml
mcp:
  server:
    transport: "stdio"
    command: "python"
    args:
      - "env.HOME/mcp_servers/server.py"
      - "--token"
      - "env.MCP_TOKEN"
```

## Tool Naming

Tools from MCP servers are prefixed with the server name:

- Original tool: `add`
- Registered as: `math_add` (from server named "math")
- Original tool: `gh_read_file`
- Registered as: `github_gh_read_file` (from server named "github")

## Complete Example

```yaml
mcp:
  # Local math server
  math:
    transport: "stdio"
    command: "python"
    args: ["env.HOME/mcp_servers/math_server.py"]
    tools: ["*"]
    ui_hints_for: ["*"]

  # GitHub server with filtering
  github:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    tools:
      - "gh*"
      - "!gh_write*"
      - "!gh_delete*"
    ui_hints_for: ["gh*"]
    require_confirmation_for: ["gh_delete*"]

  # Remote weather API
  weather:
    transport: "streamable_http"
    url: "http://localhost:8000/mcp"
    tools:
      - "get_*"
      - "!get_internal_*"
    require_confirmation_for: ["*"]

# Regular tools work alongside MCP tools
tools:
  - name: fetch
    import_from: llm_workers.tools.fetch.FetchWebPage

chat:
  default_prompt: "You are a helpful assistant with access to MCP tools."
```

## Error Handling

- If an MCP server fails to connect, the error is logged and the system continues with other servers
- If a tool name conflicts with an existing tool, the MCP tool is skipped with a warning
- Environment variables that don't exist will raise an error during initialization

## Logging

Enable verbose logging to see MCP initialization details:

```bash
llm-workers-chat --verbose examples/mcp-example.yaml
```

Log messages include:
- MCP server configuration
- Connection attempts
- Tool loading progress
- Tool filtering decisions
- Registration confirmations

## Dependencies

MCP support requires the `langchain-mcp-adapters` package, which is automatically installed with llm-workers:

```bash
poetry install
```

## Testing

See `examples/mcp-example.yaml` for a complete working example.

To test the configuration without connecting to real servers:

```python
from llm_workers.config import load_config

config = load_config('your-config.yaml')
print(f"MCP servers: {list(config.mcp.keys())}")
```

## References

- [LangChain MCP Documentation](https://docs.langchain.com/oss/python/langchain/mcp)
- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- [MCP Server Examples](https://github.com/modelcontextprotocol/servers)
