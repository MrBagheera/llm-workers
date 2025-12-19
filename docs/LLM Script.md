LLM scripts are YAML configuration files that define how to interact with large language models (LLMs) and what 
tools LLMs can use. You should treat them like a normal scripts. In particular - DO NOT run LLM scripts from
unknown / untrusted sources. Scripts can easily download and run malicious code on your machine, or submit your secrets 
to some web site. 

For real examples, see [generic-assistant.yaml](src/llm_workers/generic-assistant.yaml)
and files in [`examples`](examples/) directory.


Table of Contents
=================
<!--ts-->
* [Basic Structure](#basic-structure)
   * [Environment Variables Section](#environment-variables-section)
   * [MCP Servers Section](#mcp-servers-section)
   * [Tools Section](#tools-section)
      * [Common Tool Parameters](#common-tool-parameters)
   * [Shared Section](#shared-section)
   * [Chat Section](#chat-section)
   * [CLI Section](#cli-section)
* [Using Tools](#using-tools)
   * [Importing Tools](#importing-tools)
   * [Built-in Tools](#built-in-tools)
      * [Web Fetching Tools](#web-fetching-tools)
         * [fetch_content](#fetch_content)
         * [fetch_page_markdown](#fetch_page_markdown)
         * [fetch_page_text](#fetch_page_text)
         * [fetch_page_links](#fetch_page_links)
      * [LLM Tool](#llm-tool)
         * [build_llm_tool](#build_llm_tool)
      * [File and System Tools](#file-and-system-tools)
         * [read_file](#read_file)
         * [write_file](#write_file)
         * [run_python_script](#run_python_script)
         * [show_file](#show_file)
         * [bash](#bash)
         * [list_files](#list_files)
         * [run_process](#run_process)
      * [Miscellaneous Tools](#miscellaneous-tools)
         * [user_input](#user_input)
         * [request_approval](#request_approval)
         * [validate_approval](#validate_approval)
         * [consume_approval](#consume_approval)
* [Defining Custom Tools](#defining-custom-tools)
   * [call Statement](#call-statement)
   * [eval Statement](#eval-statement)
   * [match Statement](#match-statement)
   * [Composing Statements](#composing-statements)
   * [Template Variables](#template-variables)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: dmikhaylov, at: Fri Sep 19 11:19:35 EEST 2025 -->

<!--te-->

# Basic Structure

```yaml
# Environment variables configuration (optional)
env:
  <var_name>:
    description: <description> # Optional
    persistent: <boolean> # Optional, default: false
    secret: <boolean> # Optional, default: false

# MCP Servers configuration (optional)
mcp:
  <server_name>:
    transport: "stdio" | "streamable_http"
    command: <command> # For stdio
    args: [<arg1>, <arg2>] # For stdio
    env: # Optional, environment variables for the server process
      <key>: <value>
    url: <url> # For streamable_http
    headers: # Optional, for streamable_http
      <key>: <value>
    auto_import_scope: none|shared|chat

# Shared configuration
shared:
  data: # Optional shared data accessible to all tools via ${shared['key']}
    <key>: <value>
    # Can contain any JSON-serializable data

  tools: # Optional shared tools that can be referenced from chat/cli
    # Single tool import
    - import_tool: <import_path>
      name: <tool_name> # Optional
      # ... other tool parameters

    # Mass import from toolkit or MCP server
    - import_tools: <toolkit_or_mcp_server>
      prefix: <prefix> # Mandatory
      filter: [<pattern>, ...] # Optional
      ui_hints_for: [<pattern>, ...] # Optional
      ui_hints_args: [<arg>, ...] # Optional
      require_confirmation_for: [<pattern>, ...] # Optional

    # Custom tool definition
    - name: <tool_name>
      description: <description> # Optional
      input: # Required for custom tools
        - name: <param_name>
          description: <description>
          type: <type>
          default: <default_value> # Optional
      confidential: <boolean> # Optional
      return_direct: <boolean> # Optional
      ui_hint: <template_string> # Optional
      do: # Required for custom tools
        <statement(s)> # List of statements for more complex flows

chat: # For interactive chat mode
  model_ref: <model_name> # Optional, references model by name (fast/default/thinking). If not defined, uses "default".
  user_banner: | # Optional, markdown-formatted text displayed at the beginning of chat
    <banner text>
  system_message: |
    <system prompt>
  default_prompt: | # Optional
    <default prompt>

cli: # For command-line interface
  <statement(s)> # List of statements for more complex flows
```

## Environment Variables Section

The `env` section allows you to declare environment variables required by your script. Variables can be marked as persistent (saved to `.env` file) or transient (prompted each time the script loads).

**Structure:**
```yaml
env:
  VAR_NAME:
    description: "Description of what this variable is used for"
    persistent: true  # or false (default)
    secret: true   # or false (default)
```

**Parameters:**
- `description`: (Optional) Human-readable description shown when prompting the user
- `persistent`: (Optional, default: `false`)
  - `true`: Value is saved to `.env` file and persists across sessions
  - `false`: Value is prompted each time the script is loaded (transient)
- `secret`: (Optional, default: `false`)
  - `true`: Input is hidden with asterisks when prompting (requires `prompt_toolkit` and TTY)
  - `false`: Input is shown normally
  - Use for passwords, API keys, tokens, and other sensitive values

**Behavior:**
- Environment variables are inherited from parent process and loaded from `.env` in current directory
(if exists) or `~/.config/llm-workers/.env` (default) 
- Variable statements are processed at startup
- If a variable is already set in the environment, no prompting occurs
- Persistent variables are saved back to `.env` file used at startup

**Example:**
```yaml
env:
  API_KEY:
    description: "API key for external service"
    persistent: true
    secret: true  # Input will be hidden

  SESSION_TOKEN:
    description: "Temporary session token for this run"
    persistent: false
    secret: true  # Hidden but not saved

  DATABASE_URL:
    description: "PostgreSQL connection string"
    persistent: true
    secret: false  # Not sensitive, can be shown
```

**When to use `secret`:**
- ✓ API keys, authentication tokens
- ✓ Passwords, private keys
- ✓ OAuth secrets, session tokens
- ✗ Non-sensitive config like URLs, paths, usernames

## MCP Servers Section

The `mcp` section allows you to connect to external MCP (Model Context Protocol) servers and use their tools alongside built-in tools. Tools from MCP servers are automatically prefixed with the server name to avoid conflicts.

**Structure:**
```yaml
mcp:
  server_name:  # Used as prefix for tools
    transport: "stdio" | "streamable_http"

    # For stdio transport (local subprocess)
    command: "command_to_run"
    args: ["arg1", "arg2"]
    env:  # Optional, environment variables for server process
      KEY: "${env.VAR_NAME}"  # Can reference env variables

    # For streamable_http transport (remote server)
    url: "http://localhost:8000/mcp"
    headers:  # Optional
      X-API-KEY: "${env.API_KEY}"

    # Auto-import scope (optional, default: "none")
    # Controls where tools from this server are automatically imported
    auto_import_scope: none | shared | chat
```

### Auto-Import Scope

The `auto_import_scope` field controls where tools from an MCP server are automatically imported:

- `none` (default): Tools are not automatically imported. You must explicitly import them using `import_tool` or `import_tools` statements.
- `shared`: Tools are automatically imported to the `shared.tools` section, making them accessible across all agents.
- `chat`: Tools are automatically imported to the `chat.tools` section, making them available only in chat mode.

**Example with auto-import:**
```yaml
mcp:
  echo:
    transport: "stdio"
    command: "uvx"
    args: ["echo-mcp-server-for-testing"]
    auto_import_scope: chat  # All tools automatically available in chat

chat:
  system_message: "You are a helpful assistant."
  # No need to explicitly import echo tools - they're already available
```

**Example without auto-import (manual import):**
```yaml
mcp:
  github:
    transport: "streamable_http"
    url: "https://api.githubcopilot.com/mcp/"
    auto_import_scope: none  # Must import manually

chat:
  system_message: "You are a helpful assistant."
  tools:
    - import_tools: mcp:github  # Explicitly import GitHub tools
      prefix: gh_
      filter: ["!*delete*"]
```

### Transport Types

**Stdio Transport** - For local MCP servers running as subprocesses:
```yaml
mcp:
  math:
    transport: "stdio"
    command: "uvx"
    args: ["mcp-server-math"]
    auto_import_scope: chat
```

**HTTP Transport** - For remote MCP servers accessible via HTTP:
```yaml
mcp:
  weather:
    transport: "streamable_http"
    url: "http://localhost:8000/mcp"
    headers:
      X-API-KEY: "${env.WEATHER_API_KEY}"
    auto_import_scope: none
```

### Environment Variable Substitution

MCP server configurations support environment variable substitution using the `${env.VAR_NAME}` syntax in both `args` and `env` fields:

```yaml
mcp:
  github:
    transport: "stdio"
    command: "npx"
    args:
      - "-y"
      - "@modelcontextprotocol/server-github"
      - "--config"
      - "${env.CONFIG_PATH}/github.json"  # In args
    env:
      GITHUB_TOKEN: "${env.GITHUB_TOKEN}"  # In env
      LOG_PATH: "/var/log/${env.USER}.log"  # Embedded substitution
```

**Key Features:**
- The `${env.VAR_NAME}` references are replaced with actual environment variable values at runtime
- Works in both `args` (list of arguments) and `env` (environment variables for the server process)
- Supports embedded substitutions: `"prefix_${env.VAR}_suffix"` → `"prefix_value_suffix"`
- Can use multiple variables in one string: `"${env.VAR1}_and_${env.VAR2}"`

### Tool Naming and Import Control

When importing tools from MCP servers (either via `auto_import_scope` or `import_tools`), you control the naming and behavior through the import statement:

**With auto-import:**
- Tools are imported with their original names from the MCP server
- The server name is used as a prefix: `server_name_tool_name`
- Example: `add` tool from `math` server → `math_add`

**With manual import (import_tools):**
- You have full control via the `prefix` parameter
- You can filter, add UI hints, and require confirmation
- See [Mass Import](#2-mass-import-import_tools) section for details

**Example with filtering and UI hints:**
```yaml
mcp:
  github:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    auto_import_scope: none  # Manual import required

chat:
  tools:
    - import_tools: mcp:github
      prefix: gh_
      filter:
        - "!*write*"    # Exclude write operations
        - "!*delete*"   # Exclude delete operations
      ui_hints_for: ["*"]
      ui_hints_args: ["owner", "repo"]
      require_confirmation_for: ["*push*"]
```

### Complete MCP Example

```yaml
env:
  GITHUB_TOKEN:
    description: "GitHub personal access token"
    persistent: true
  WEATHER_API_KEY:
    description: "API key for weather service"
    persistent: true

mcp:
  # Local math server with auto-import
  math:
    transport: "stdio"
    command: "uvx"
    args: ["mcp-server-math"]
    auto_import_scope: chat  # Tools automatically available in chat

  # GitHub server with manual import for fine-grained control
  github:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${env.GITHUB_TOKEN}"
    auto_import_scope: none  # Must import manually

  # Remote weather API with auto-import
  weather:
    transport: "streamable_http"
    url: "http://localhost:8000/mcp"
    headers:
      X-API-KEY: "${env.WEATHER_API_KEY}"
    auto_import_scope: chat

# Regular tools work alongside MCP tools
tools:
  - import_tool: llm_workers.tools.unsafe.ReadFileTool

chat:
  system_message: "You are a helpful assistant with access to MCP tools."
  tools:
    # GitHub tools imported manually with fine-grained control
    - import_tools: mcp:github
      prefix: gh_
      filter:
        - "!*write*"
        - "!*delete*"
      ui_hints_for: ["*"]
      ui_hints_args: ["owner", "repo"]
      require_confirmation_for: ["*push*"]

    # Regular tool reference
    - read_file

    # math and weather tools are auto-imported, no need to list them
```

### Error Handling

- If an MCP server fails to connect, the error is logged and the system continues with other servers
- If a tool name conflicts with an existing tool, the MCP tool is skipped with a warning
- Environment variables that don't exist will raise an error during initialization

## Shared Tools Section

The `shared.tools` section defines shared tools that can be used across agents and other tools. Tools can be:
- Imported from Python classes or functions (`import_tool`)
- Mass imported from toolkits or MCP servers (`import_tools`)
- Custom tools defined using statement composition (custom tool definition)

**Structure:**
```yaml
shared:
  tools:
    # Import single tool from Python class/function
    - import_tool: <import_path>
      name: <tool_name>  # Optional, can override the default name
      description: <description>  # Optional
      # ... other tool parameters

    # Import single tool from toolkit
    - import_tool: <toolkit_class>/<tool_name>
      # ... tool parameters

    # Import single tool from MCP server
    - import_tool: mcp:<server_name>/<tool_name>
      # ... tool parameters

    # Mass import from toolkit or MCP server
    - import_tools: <toolkit_class_or_mcp_server>
      prefix: <prefix>  # Mandatory, can be empty ""
      filter: [<pattern>, ...]  # Optional, default: ["*"]
      ui_hints_for: [<pattern>, ...]  # Optional, default: ["*"]
      ui_hints_args: [<arg>, ...]  # Optional, default: []
      require_confirmation_for: [<pattern>, ...]  # Optional, default: []

    # Custom tool definition
    - name: <tool_name>
      description: <description>
      input:
        - name: <param_name>
          description: <param_description>
          type: <type>
      do:
        # ... statements
```

**Example:**
```yaml
shared:
  tools:
    # Single tool import from Python class
    - import_tool: llm_workers.tools.fetch.FetchPageTextTool
      name: _fetch_page_text

    # Single tool import from Python function
    - import_tool: llm_workers.tools.llm_tool.build_llm_tool
      name: _LLM

    # Mass import from toolkit with filtering
    - import_tools: llm_workers.tools.fs.FilesystemToolkit
      prefix: fs_
      filter:
        - "read_*"  # Include read operations
        - "!write_*"  # Exclude write operations
      ui_hints_for: ["*"]
      ui_hints_args: ["path"]
      require_confirmation_for: []

    # Custom tool definition
    - name: metacritic_monkey
      description: >
        Finds the Metacritic score for a given movie title and year. Returns either a single number or "N/A" if the movie is not found.
      input:
        - name: movie_title
          description: Movie title
          type: str
        - name: movie_year
          description: Movie release year
          type: int
      ui_hint: Looking up Metacritic score for movie "{movie_title}" ({movie_year})
      do:
        - call: _fetch_page_text
          params:
            url: "https://www.metacritic.com/search/{movie_title}/?page=1&category=2"
            xpath: "//*[@class=\"c-pageSiteSearch-results\"]"
        - call: _LLM
          params:
            prompt: >
              Find Metacritic score for movie "{movie_title}" released in {movie_year}.
              To do so:
                - From the list of possible matches, chose the one matching movie title and year and return metacritic score as single number
                - If no matching movie is found, respond with just "N/A" (without quotes)
                - DO NOT provide any additional information in the response

              Possible matches:
              {_}
```

In addition to defining tools in the `shared.tools` section, you can define tools inline within `call` statements and within
the `tools` configuration of LLMs. This provides flexibility for single-use tools or when you need to customize
tool behavior for specific calls. See relevant sections below.

### Common Tool Parameters

- `name`: Unique identifier for the tool. This name is used to reference the tool in other parts of the script.
  Names should be unique within the script and should not contain spaces or special characters. Tools with names starting with `_`
  are considered to be "private". Those are not available for LLM use unless added to `tools` list explicitly.
- `description`: Brief description of the tool's purpose. For imported tools is optional, and is taken from Python code if omitted.
- `return_direct`: Optional, defaults to `false`. If `true`, the tool's result is returned directly to the user without
  further processing by the LLM. This is useful for tools that provide direct answers or results.
- `confidential`: Optional, defaults to `false`. If `true`, the tool's result is considered sensitive and will not be
  used in subsequent LLM calls. This is useful for tools that return sensitive data, such as passwords or personal information.
  Implies `return_direct: true`.
- `require_confirmation`: Optional, defaults to Python tool logic (if defined, can be on per-call basis) or `false`
    - If `true`, the tool requires user confirmation before executing. This is useful for tools that perform actions that may have significant consequences, such as deleting files or sending emails.
    - If `false`, the tool does not require user confirmation before executing, even if Python tool code does.
- `ui_hint`: Optional, defaults to Python tool logic (if defined), or empty. If defined, it is used to generate a UI
  hint for the tool. This can be a template string referencing tool's parameters.

See [Using Python Tools](#using-python-tools) and [Defining Custom Tools](#defining-custom-tools) sections below for
more details on how to define and use tools.

## Shared Data Section

The `shared.data` section provides a way to define reusable configuration data that can be accessed by all custom tools in the script. This is useful for avoiding duplication of common values like API endpoints, prompts, or configuration settings.

**Key Features:**
- Must be a dictionary (key-value pairs)
- Can contain any JSON-serializable data (strings, numbers, booleans, lists, nested objects)
- Accessible in custom tools via the `${shared['key']}` template syntax
- Supports nested access using bracket notation

**Example:**
```yaml
shared:
  data:
    prompts:
      test: Yada-yada-yada
  tools:
    - name: demo_shared_access
      input:
        - name: query
          description: "Search query"
          type: str
      do:
        eval: "Query ${query} returned ${shared['prompts']['test']}"
```

**Usage Notes:**
- The `shared.data` section is optional and defaults to an empty dictionary
- All tools automatically have access to shared data through the `shared` template variable
- Use bracket notation for accessing nested values: `${shared['category']['subcategory']}`
- Shared data is read-only during tool execution
- Changes to the shared section require reloading the script configuration

## Chat Section

Configuration for interactive chat mode:

- `model_ref`: References a model configured in `~/.config/llm-workers/config.yaml` (fast/default/thinking), defaults to "default"
- `system_message`: Instructions for the LLM's behavior
- `default_prompt`: Initial prompt when starting the chat, defaults to empty string
- `user_banner`: Optional markdown-formatted text displayed at the beginning of chat session, defaults to not shown
- `tools`: (Optional) List of tools to make available for this LLM. Defaults to all public tools (e.g. not starting with `_`).

**Tools can be specified in multiple ways:**

1. **By name** - Reference tools defined in the `tools` section:
   ```yaml
   tools:
     - read_file
     - write_file
   ```

2. **By pattern** - Use patterns to match multiple tools:
   ```yaml
   tools:
     - match: ["fs_*", "!fs_write*"]  # Include fs_* but exclude fs_write*
   ```

3. **Inline import_tool** - Import a single tool with full control:
   ```yaml
   tools:
     - import_tool: llm_workers.tools.unsafe.ReadFileTool
       name: read_file
       require_confirmation: false
   ```

4. **Inline import_tools** - Mass import from toolkit or MCP server:
   ```yaml
   tools:
     - import_tools: llm_workers.tools.fs.FilesystemToolkit
       prefix: fs_
       filter: ["read_*", "!write_*"]
   ```

5. **Inline custom tool** - Define a custom tool directly:
   ```yaml
   tools:
     - name: custom_processor
       description: "Process data"
       input:
         - name: data
           type: str
       do:
         - eval: "Processed: ${data}"
   ```

**Example:**
```yaml
chat:
  model_ref: thinking
  user_banner: |
    # Game Analytics Assistant

    Welcome to the mobile game analytics environment! I can help you investigate live game issues by:
    - Writing Python scripts to fetch and analyze data
    - Connecting data from various sources
    - Generating reports in JSON format

    Type your requests naturally and I'll get to work!
  system_message: |-
    You are AI assistant in a mobile game company.
    Your team is investigating issues in a live game. Your task is to help your team by writing
    and running Python scripts to fetch and connect data from various sources.

    If needed, preview text file content by reading first few lines from it.

    Unless explicitly asked by user, write script result to promptly named file in the current directory,
    output only progress information and file name. Prefer '.json' as output format.

    If unsure about the requirements, ask for clarification.
  default_prompt: |-
    Please run Python script to detect Python version.
  tools:
    # Reference by name
    - read_file

    # Pattern matching
    - match: ["fs_read*", "fs_list*"]

    # Inline tool import
    - import_tool: llm_workers.tools.unsafe.RunPythonScriptTool
      name: run_python
      require_confirmation: true

    # Mass import from toolkit
    - import_tools: llm_workers.tools.fs.FilesystemToolkit
      prefix: safe_
      filter: ["read_*", "list_*"]
```


## CLI Section

Configuration for command-line interface. Contains a list of statements that define the command-line flow.
Flow is run for each command line parameter. Strings can reference "${input}" variable for value of command-line parameter.

See "Custom Tools" section for details on statements.

Example:
```yaml
cli:
  - call: read_file
    params:
      filename: "${input}"
  - call: llm
    params:
      prompt: |-
        You are senior Scala developer. Your job is to reformat give file according to the rules below. 

        Rules:

        - Add types for local variables and method returns. Do not make assumptions about the type - if type cannot be 
        inferred from the provided code alone, omit type definition.

        - Break long statements up in several lines, use intermediate variables

        - When dealing with Java code, handle null values explicitly and as soon as possible
          - Don't add `null` checks unless they are really needed
          - Don't wrap `null`-s to `Option` unless you pass it further to Scala code
          - Handle the `null`-s immediately if possible, just wrapping them to `Option` pushes the responsibility to the receiver of the `Option`.

        - Get rid of Option-s as early as possible, prefer pattern matching over call chaining

        - Don't use infix or postfix notation, use dot notation with parenthesis everywhere: `obj.method(args)`

        - Chain method calls with dots on new line

        - Always use braces () in method definition and calls

        - Use curly braces {{}} in method definitions

        - Prefer code readability over complex pure functional code

        - Prefer for comprehension over chaining async method calls

        - Don't use curly braces for if-else with a single statement:
        ```scala
        if (playerContribution <= 2)
          1 + Math.floor(titanStars / 2.0).toInt
        else
          1
        ```

        - Don't use curly braces for if with return:
        ```scala
        if (playerContribution <= 2) return 1 + Math.floor(titanStars / 2.0).toInt
        ```

        After reformatting, output just the file content without any additional comments or formatting.      
        If no changes are needed, respond with string "NO CHANGES" (without quotes).

        Input file:
        ${_}
  - match: "${_}"
    matchers:
      - case: "NO CHANGES"
        then:
          eval: "${input}: NO CHANGES"
    default:
      - call: write_file
        params:
          path: "${input}"
          content: "{_}"
      - eval: "${input}: FIXED"
```

# Using Tools

## Referencing Tools

Tools can be referenced in different ways depending on the context:

**In `call` statements** (single tool reference):
- By name: `call: read_file`
- Inline import: `call: {import_tool: llm_workers.tools.unsafe.ReadFileTool, name: read_file}`
- Inline custom definition: `call: {name: my_tool, input: [...], do: [...]}`

**In `chat.tools` and `build_llm_tool` config** (multiple tools):
- By name: `- read_file`
- By pattern: `- match: ["fs_*", "!fs_write*"]`
- Inline import_tool: `- import_tool: llm_workers.tools.unsafe.ReadFileTool`
- Inline import_tools: `- import_tools: llm_workers.tools.fs.FilesystemToolkit`
- Inline custom definition: `- name: my_tool ...`

For most simple projects, tools should be defined in the global `tools` section and referenced by name in `chat.tools`.

## Importing Tools

LLM Workers provides three ways to import tools:

### 1. Import Single Tool (`import_tool`)

The `import_tool` statement imports a single tool with full control over its properties.

**Import from Python class or function:**
```yaml
tools:
  - import_tool: llm_workers.tools.unsafe.ReadFileTool
    name: read_file  # Optional, can override default name
    description: "Custom description"  # Optional
    require_confirmation: true  # Optional
```

**Import single tool from toolkit:**

You can import one specific tool from a toolkit using `<toolkit_class>/<tool_name>` syntax:

```yaml
tools:
  - import_tool: llm_workers.tools.fs.FilesystemToolkit/read_file
    name: my_read_file
    ui_hint: "Reading file: ${path}"
```

**Import single tool from MCP server:**

You can import one specific tool from an MCP server using `mcp:<server_name>/<tool_name>` syntax:

```yaml
tools:
  - import_tool: mcp:github/search_repositories
    name: gh_search
    require_confirmation: false
```

**Supported import sources:**
- Python class extending `langchain_core.tools.base.BaseTool` (instantiated with config parameters)
- Python factory function/method returning a `BaseTool` instance
- Single tool from a toolkit
- Single tool from an MCP server

Factory functions must conform to this signature:
```python
def build_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> BaseTool:
```

### 2. Mass Import (`import_tools`)

The `import_tools` statement imports multiple tools at once with basic control over their properties.

**Import from toolkit:**
```yaml
tools:
  - import_tools: llm_workers.tools.fs.FilesystemToolkit
    prefix: fs_  # Mandatory prefix (can be empty "")
    filter:  # Optional, default: ["*"]
      - "read_*"  # Include read operations
      - "list_*"  # Include list operations
      - "!write_*"  # Exclude write operations
    ui_hints_for: ["*"]  # Optional, patterns for UI hints
    ui_hints_args: ["path"]  # Optional, args to show in UI hints
    require_confirmation_for: ["write_*"]  # Optional, patterns requiring confirmation
```

**Import from MCP server:**
```yaml
tools:
  - import_tools: mcp:github
    prefix: gh_
    filter:
      - "!*delete*"  # Exclude delete operations
      - "!*force*"   # Exclude force operations
    ui_hints_for: ["*"]
    ui_hints_args: ["owner", "repo"]
    require_confirmation_for: ["*write*", "*push*"]
```

**Pattern Matching:**

The `filter`, `ui_hints_for`, and `require_confirmation_for` fields support Unix shell-style wildcards:
- `*` - matches everything
- `?` - matches any single character
- `[seq]` - matches any character in seq
- `[!seq]` - matches any character not in seq
- `!pattern` - negation (exclude matching tools)

Patterns are evaluated in order:
- Tools are included if they match any inclusion pattern
- Tools are excluded if they match any exclusion pattern (prefixed with `!`)

**Parameters:**
- `prefix`: (Mandatory) Prefix for all imported tool names. Can be empty `""` for no prefix
- `filter`: (Optional, default: `["*"]`) Patterns to include/exclude tools
- `ui_hints_for`: (Optional, default: `["*"]`) Patterns for which tools should show UI hints
- `ui_hints_args`: (Optional, default: `[]`) Tool arguments to include in UI hints (empty means show tool name only)
- `require_confirmation_for`: (Optional, default: `[]`) Patterns for tools requiring user confirmation

### 3. Custom Tool Definitions

Tools without `import_tool` or `import_tools` are custom tool definitions. See [Defining Custom Tools](#defining-custom-tools) section for details.

**Example:**
```yaml
tools:
  - name: metacritic_monkey
    description: "Finds Metacritic score for a movie"
    input:
      - name: movie_title
        description: Movie title
        type: str
    do:
      - call: fetch_page_text
        params:
          url: "https://www.metacritic.com/search/{movie_title}"
```

## Built-in Tools

In addition to the tools listed below, you can use (almost) any tool from [`langchain` library](https://docs.langchain.com/docs/integrations/tools/),
any imported library, or [define your own](https://python.langchain.com/docs/concepts/tools/) tools.


### Web Fetching Tools

#### fetch_content

```yaml
- import_tool: llm_workers.tools.fetch.FetchContentTool
  name: fetch_content
```

Fetches raw content from a given URL and returns it as a string.

**Parameters:**
- `url`: The URL to fetch from
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found ('raise_exception', 'return_error', 'return_empty')
- `on_error`: (Optional) What to do if an error occurs ('raise_exception', 'return_error', 'return_empty')

#### fetch_page_markdown

```yaml
- import_tool: llm_workers.tools.fetch.FetchPageMarkdownTool
  name: fetch_page_markdown
```

Fetches web page or page element and converts it to markdown.

**Parameters:**
- `url`: The URL of the page
- `xpath`: (Optional) The xpath to the element containing the text; if not specified the entire page will be returned
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found
- `on_error`: (Optional) What to do if an error occurs

#### fetch_page_text

```yaml
- import_tool: llm_workers.tools.fetch.FetchPageTextTool
  name: fetch_page_text
```

Fetches the text from web page or web page element.

**Parameters:**
- `url`: The URL of the page
- `xpath`: (Optional) The xpath to the element containing the text; if not specified the entire page will be returned
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found
- `on_error`: (Optional) What to do if an error occurs

#### fetch_page_links

```yaml
- import_tool: llm_workers.tools.fetch.FetchPageLinksTool
  name: fetch_page_links
```

Fetches the links from web page or web page element.

**Parameters:**
- `url`: The URL of the page
- `xpath`: (Optional) The xpath to the element containing the links; if not specified the entire page will be searched
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found
- `on_error`: (Optional) What to do if an error occurs

**Returns:** List of objects with `link` and optional `text` properties

Based on the `llm_tool.py` file, here is documentation for the LLM tool functionality:


### LLM Tool

#### build_llm_tool

```yaml
- import_tool: llm_workers.tools.llm_tool.build_llm_tool
  name: llm
```

Creates a tool that allows calling an LLM with a prompt and returning its response.

**Configuration Parameters**:
- `model_ref`: (Optional) Reference to a model configured in `~/.config/llm-workers/config.yaml` (fast/default/thinking), defaults to "default"
- `system_message`: (Optional) System message to use for this specific LLM tool
- `tools`: (Optional) List of tool names or inline tool definitions to make available for this specific LLM tool.
See `tools` definition in [Chat Section](#chat-section) for details.
- `remove_past_reasoning`: (Optional) Whether to hide past LLM reasoning, defaults to false
- `extract_json`: (Optional) Filters result to include only JSON blocks, defaults to "false".
Useful for models without Structured Output, like Claude. Fallbacks to entire message if no "```json" blocks are found. Possible values:
  - "first" - returns only the first JSON block found in the response
  - "last" - returns only the last JSON block found in the response
  - "all" - returns all JSON blocks found in the response as list
  - "false" - returns the full response without filtering

**Function**:
This factory method creates a `StructuredTool` that passes a prompt to an LLM and returns the model's response.

**Parameters**:
- `prompt`: Text prompt to send to the LLM
- `system_message`: (Optional) System message to prepend to the conversation at runtime

**Returns**:
- Text response from the LLM

**Behavior Notes**:
- When used in a workflow, this tool disables token streaming to prevent partial outputs
- For conversations with multiple messages, it returns only AI messages joined by newlines
- For a single message, it returns the raw message text
- Empty result lists return an empty string

**Example Usage**:
```yaml
- call: llm
  params:
    prompt: |
      Summarize the following text in three bullet points:
      {text_to_summarize}
```

**Example with System Message**:
```yaml
- call: llm
  params:
    system_message: |-
      You are a schema expert. Always respond with "SCHEMA_EXPERT:" prefix followed by your answer.
    prompt: |
      What is a database schema?
```

**Example with JSON only**:
```yaml
- call: llm
  json_only: true
  params:
    prompt: |
      Summarize the following text in three bullet points:
      {text_to_summarize}

      Return the result as a JSON array with object with following keys:
        - `bullet`: string - The bullet point text
        - `importance`: int - Importance of the bullet point (1-5)
        - `references: string[] - List of references relevant to the bullet point
```


The tool can be used to create custom LLM-powered tools within your workflows, enabling tasks like summarization,
analysis, formatting, or generating structured content based on input data.


Based on the `unsafe.py` file, here are the available tools that interact with the file system and execute code:

### File and System Tools

These tools provide access to the file system and allow code execution, which makes them potentially unsafe for use with untrusted inputs:

#### read_file

```yaml
- import_tool: llm_workers.tools.unsafe.ReadFileTool
  name: read_file
```

Reads a file and returns its content.

**Parameters:**
- `filename`: Path to the file to read
- `lines`: (Optional) Number of lines to read. If 0 (default), reads the entire file. If negative, reads from the end of file (tail)

#### write_file

```yaml
- import_tool: llm_workers.tools.unsafe.WriteFileTool
  name: write_file
```

Writes content to a file.

**Parameters:**
- `filename`: Path to the file to write
- `content`: Content to write
- `append`: (Optional) If true, append to the file instead of overwriting it (default: false)

#### run_python_script

```yaml
- import_tool: llm_workers.tools.unsafe.RunPythonScriptTool
  name: run_python_script
```

Runs a Python script and returns its output.

**Parameters:**
- `script`: Python script to run. Must be valid Python code

**Configuration Options:**
- `delete_after_run`: (Optional) Whether to delete the script file after running (default: false)
- `require_confirmation`: (Optional) Whether to require user confirmation before execution (default: true)

**Behavior:**
- Creates a temporary script file in `.cache/` directory with timestamp
- Executes it using the current Python interpreter (`sys.executable`)
- Returns stdout output
- Deletes the script file after execution (if `delete_after_run` is true)
- Requires user confirmation by default
- Raises `ToolException` if script returns non-zero exit code or encounters errors

#### show_file

```yaml
- import_tool: llm_workers.tools.unsafe.ShowFileTool
  name: show_file
```

Opens a file in the OS default application.

**Parameters:**
- `filename`: Path to the file to open

#### bash

```yaml
- import_tool: llm_workers.tools.unsafe.BashTool
  name: bash
```

Executes a bash script and returns its output.

**Parameters:**
- `script`: Bash script to execute
- `timeout`: (Optional) Timeout in seconds (default: 30)

**Behavior:**
- Creates a temporary executable script file
- Executes it using bash
- Returns stdout output
- Deletes the script file after execution
- Requires user confirmation by default

#### list_files

```yaml
- import_tool: llm_workers.tools.unsafe.ListFilesTool
  name: list_files
```

Lists files and directories with optional detailed information.

**Parameters:**
- `path`: Path to directory to list or file to show info
- `depth`: (Optional) Recursive directory depth (default: 0 - no recursion)
- `permissions`: (Optional) Whether to show permissions (default: false)
- `times`: (Optional) Whether to show creation and modification times (default: false)
- `sizes`: (Optional) Whether to show file sizes (default: false)

#### run_process

```yaml
- import_tool: llm_workers.tools.unsafe.RunProcessTool
  name: run_process
```

Runs a system process and returns its output.

**Parameters:**
- `command`: Command to run as a subprocess
- `args`: (Optional) List of arguments for the command
- `timeout`: (Optional) Timeout in seconds (default: 30)

**Behavior:**
- Executes the command with provided arguments
- Returns stdout output
- Requires user confirmation by default


### Miscellaneous Tools

#### user_input

```yaml
- import_tool: llm_workers.tools.misc.UserInputTool
  name: user_input
```

Prompts the user for input and returns their response.

**Parameters:**
- `prompt`: Text prompt to display to the user before requesting input

**Behavior:**
- Displays the prompt to the user
- Shows instruction about using empty line as end-of-input
- Reads lines from stdin until an empty line is encountered
- Returns all input as a single string with newlines preserved between lines
- Does not require user confirmation
- Handles EOF gracefully by returning collected input so far

**Primary Use Case:**
This tool is primarily intended for **prototyping new tools and prompt combinations**. It's designed to be used inside custom tools that mimic the intended tool's parameter schema and description. During development, you can create stub tools with `match` statements that return results for the most common input parameter combinations, while using `user_input` in the `default` block to handle unexpected inputs during testing.

**Example Usage for Prototyping:**
```yaml
- name: search_api_stub
  description: "Searches external API for data"
  input:
    - name: query
      description: "Search query"
      type: str
  do:
    - match: "${query}"
      matchers:
        - case: "common query 1"
          then:
            eval: "predefined result 1"
        - case: "common query 2"
          then:
            eval: "predefined result 2"
      default:
        - call: user_input
          params:
            prompt: "API returned: ${query}. Please provide mock response:"
```

This approach allows you to quickly prototype and test tool interactions before implementing the actual tool logic.



#### request_approval

The approval tools provide a way to require explicit user confirmation for potentially dangerous operations through a token-based system:

```yaml
- import_tool: llm_workers.tools.misc.RequestApprovalTool
  name: request_approval
```

Shows a prompt to the user for approval and returns an approval token that can be used to authorize subsequent operations.

**Parameters:**
- `prompt`: Text prompt to show to user for approval

**Returns:**
- JSON string containing `{"approval_token": "<token>"}` where token is a SHA256 hash

**Behavior:**
- Always requires user confirmation before proceeding
- Generates a unique token based on prompt and timestamp
- Stores token (and original prompt) in module-local memory for validation
- Not shown in UI by default

#### validate_approval

```yaml
- import_tool: llm_workers.tools.misc.ValidateApprovalTool
  name: validate_approval
```

Validates that an approval token exists and has not been consumed.

**Parameters:**
- `approval_token`: The approval token to validate

**Returns:**
- Return prompt originally passed to `request_approval` tool
- Throws `ToolException` if token is invalid or already consumed

**Behavior:**
- Does not require user confirmation
- Not shown in UI by default

#### consume_approval

```yaml
- import_tool: llm_workers.tools.misc.ConsumeApprovalTool
  name: consume_approval
```

Validates and marks an approval token as consumed, making it unusable for future operations.

**Parameters:**
- `approval_token`: The approval token to consume

**Returns:**
- Success message if token was valid and consumed
- Throws `ToolException` if token is invalid or already consumed

**Behavior:**
- Does not require user confirmation
- Not shown in UI by default
- Token becomes permanently unusable after consumption

**Example Usage:**

The approval tools are designed to work together in custom tool workflows to force LLM confirmation for dangerous operations:

```yaml
tools:
  - import_tool: llm_workers.tools.unsafe.RunPythonScriptTool
    name: _run_python_script_no_confirmation
    require_confirmation: false

  - import_tool: llm_workers.tools.misc.RequestApprovalTool
    name: show_plan_to_user
    description: |-
      Show plan to user and asks for explicit confirmation; upon confirmation return `approval_token` to be used in
      the following call to `run_script`.

  - import_tool: llm_workers.tools.misc.ValidateApprovalTool
    name: _validate_approval

  - import_tool: llm_workers.tools.misc.ConsumeApprovalTool
    name: _consume_approval

  - name: run_python_script
    description: Consume approval_token and run given Python script
    input:
      - name: approval_token
        description: `approval_token` from `show_plan_to_user` tool; upon successful tool completion is consumed and cannot be re-used
        type: str
      - name: script
        description: Python script to run
        type: str
    ui_hint: Running generated Python script...
    do:
      - call: _validate_approval
        params:
          approval_token: {approval_token}
      - call: _run_python_script_no_confirmation
        params:
          script: ${script}
      - call: _consume_approval
        params:
          approval_token: {approval_token}
```

This pattern allows LLMs to request approval for potentially dangerous operations, ensuring users explicitly consent before execution while preventing token reuse.


# Defining Custom Tools

To define custom tools, use the `input` and `body` sections of the tool definition:

```yaml
tools:
  - name: my_custom_tool
    description: "Description of what this tool does"
    input:
      - name: param1
        description: "Description of first parameter"
        type: "str"
        default: "default value"  # Optional
      - name: param2
        description: "Description of second parameter" 
        type: "int"
    return_direct: true  # Optional, returns result directly to user
    do:
      - call: some_tool
        params:
          tool_param: "${param1}"
      - match: "${_}"
        matchers:
          - case: "success"
            then:
              - eval: "Operation successful: ${_}"
        default:
          - eval: "Operation failed"
```
Input section defines the parameters that the tool accepts. These parameters can
later be referenced in the `body` section using the `${param_name}` syntax.

The `body` section contains one or more statements that can be composed in various ways:

## call Statement

Executes a specific tool with optional parameters. Tools can be referenced by name or defined inline for single-use scenarios.

**Call by name:**
```yaml
- call: tool_name
  params:
    param1: value1
    param2: value2
  catch: [error_type1, error_type2]  # Optional error handling
```

**Inline import_tool (recommended for single-use tool imports):**
```yaml
- call:
    import_tool: module.path.ToolClass
    name: tool_name  # Optional
    description: "Tool description"  # Optional
    config:  # Optional tool-specific configuration
      key: value
    return_direct: true  # Optional
    confidential: false  # Optional
    require_confirmation: true  # Optional
    ui_hint: "Processing ${param1}..."  # Optional
  params:
    param1: value1
    param2: value2
  catch: [error_type1, error_type2]  # Optional error handling
```

**Inline custom tool definition:**
```yaml
- call:
    name: custom_processor
    description: "Processes data with custom logic"
    input:
      - name: data
        description: "Data to process"
        type: str
    do:
      - call: some_other_tool
        params:
          input: "${data}"
      - eval: "Processed: ${_}"
  params:
    data: "input_value"
```

Inline tool definitions provide maximum flexibility by allowing you to:
- Define tools exactly where they're needed
- Avoid cluttering the global tools section with single-use tools  
- Customize tool behavior for specific contexts
- Create specialized tool configurations without affecting other usages

Like regular tool definitions, inline tools also support the `config` option for flexible parameter configuration, 
which is particularly useful when dealing with complex tool configurations or potential property conflicts.

## eval Statement

The `eval` statement evaluates an expression and returns the result. It supports all Simple Eval features including nested access, function calls, and conditional expressions.

### Basic Usage

```yaml
- eval: "This is a fixed response"
```

```yaml
- eval:
    status: success
    data:
      value: 42
```

### Dictionary Access

```yaml
# Static key
- eval: "${data['field_name']}"
- eval: "${data.field_name}"

# With default
- eval: "${data['field_name'] if 'field_name' in data else 'default_value'}"
# or
- eval: "${get(data, 'field_name', 'default_value')}"


# Dynamic key from variable
- eval: "${data[key_name]}"

# Nested
- eval: "${get(get(config, 'database', {}), 'host', 'localhost')}"
```

### List Access

```yaml
# Direct indexing
- eval: "${items[0]}"
- eval: "${items[-1]}"

# With bounds checking
- eval: "${items[index] if 0 <= index < len(items) else 'default'}"
# or
- eval: "${get(items, index, 'default')}"
```

### Conditional Expressions

```yaml
- eval: "${value if value is not None else 'N/A'}"
- eval: "${'valid' if score >= 80 else 'invalid'}"
- eval: "${'A' if score >= 90 else 'B' if score >= 80 else 'C'}"
```

### Expression Composition

```yaml
# String interpolation
- eval: "User ${user_name} has score ${score}"

# Arithmetic
- eval: "${total_price * 1.15}"  # Add 15% tax

# String operations
- eval: "${name.upper()}"

# List operations
- eval: "${len(items)}"
- eval: "${[x * 2 for x in numbers]}"  # List comprehension
```

## match Statement

Conditionally executes different actions based on matching patterns:

```yaml
- match: "${input}"
  trim: true  # Optional, removes whitespace before matching
  matchers:
    - case: "help"  # Exact match
      then:
        - eval: "Available commands: help, status, search"

    - pattern: "search (.+)"  # Regex pattern match
      then:
        - call: search_tool
          params:
            query: "${1}"  # Reference to captured group

  default:  # Executed if no matches found
    - eval: "Unknown command. Type 'help' for assistance."
```

## Composing Statements

Statements can be used as part of a tool's `body` section to create composite tools:

```yaml
tools:
  - name: advanced_search
    description: "Performs an enhanced search with preprocessing"
    input:
      - name: query
        type: str
    do:
      - call: normalize_input
        params:
          text: "{query}"
      - call: search_database
        params:
          query: "${_}"
      - match: "${_}"
        matchers:
          - case: ""
            then:
              - eval: "No results found"
        default:
          - eval: "${_}"
```

## Template Variables

Custom tools support template variables using the `${...}` expression syntax (powered by simpleeval):

- Direct references: `"${param_name}"` - preserves referenced parameter type (returns the actual value, not string)
- Complex templates: `"The value is ${param_name} and ${other_param}"` - returns string with interpolated values
- Nested element access:
  - Dictionary keys (bracket): `"${param_dict['key_name']}"` - accesses dictionary values by key using standard Python syntax
  - Dictionary keys (dot): `"${param_dict.key_name}"` - also works via simpleeval's "sweetener" feature
  - List indices: `"${param_list[0]}"` - accesses list elements by index
  - Nested structures: `"${param_dict['nested']['value']}"` or `"${param_dict.nested.value}"` - supports multiple levels of nesting
- Shared data access: `"${shared['key']}"` or `"${shared.key}"` - accesses values from the shared configuration section
- Tool input parameters: `"${param_name}"`
- (inside the list of statements) Previous statement results: `"${outputN}"` where N is the 0-based index of a previous statement
- (inside `match` statement) Regex capture groups: `"${matchN}"` when using regex patterns in match statements
- Python expressions: `"${a + b}"`, `"${len(items)}"`, `"${value if condition else default}"` - supports any safe Python expression via simpleeval

**Type Preservation:** When a string contains only a single expression (e.g., `"${param}"`), the original type is preserved. When text or multiple expressions are present, the result is converted to a string.

**Note:** Expressions are evaluated using simpleeval for safety, which supports standard Python operations but restricts potentially dangerous operations.

**Example with nested element and shared access:**
```yaml
shared:
  app:
    name: "MyApp"
    version: "1.0"
  templates:
    user_format: "Welcome to ${shared.app.name}!"

tools:
  - name: process_user_data
    description: "Processes user data with nested access"
    input:
      - name: user_profile
        description: "User profile object"
        type: object
      - name: settings
        description: "User settings array"
        type: array
    do:
      - eval: "User ${user_profile['name']} has email ${user_profile['contact']['email']} and first setting is ${settings[0]}. ${shared['templates']['user_format']}"
```

This would process input like:
```json
{
  "user_profile": {
    "name": "John",
    "contact": {"email": "john@example.com"}
  },
  "settings": ["dark_mode", "notifications"]
}
```

And return: `"User John has email john@example.com and first setting is dark_mode. Welcome to MyApp!"`

