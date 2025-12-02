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
   * [result Statement](#result-statement)
      * [Dynamic Key Resolution](#dynamic-key-resolution)
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
    tools: [<pattern>] # Optional, default: ["*"]
    ui_hints_for: [<pattern>] # Optional
    require_confirmation_for: [<pattern>] # Optional

# Built-in tools configuration
tools:
  - name: <tool_name>
    import_from: <import_path> # Required for importing tools
    description: <description> # Optional
    config: # Optional, tool-specific configuration
      <key>: <value>
    input: # Required for custom tools
      - name: <param_name>
        description: <description>
        type: <type>
        default: <default_value> # Optional
    confidential: <boolean> # Optional
    return_direct: <boolean> # Optional
    ui_hint: <template_string> # Optional
    body: # Required for custom tools (also can be specified in config section)
      <statement(s)> # List of statements for more complex flows

shared: # Optional shared configuration accessible to all tools
  <key>: <value>
  # Can contain any JSON-serializable data

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

### Transport Types

**Stdio Transport** - For local MCP servers running as subprocesses:
```yaml
mcp:
  math:
    transport: "stdio"
    command: "uvx"
    args: ["mcp-server-math"]
```

**HTTP Transport** - For remote MCP servers accessible via HTTP:
```yaml
mcp:
  weather:
    transport: "streamable_http"
    url: "http://localhost:8000/mcp"
    headers:
      X-API-KEY: "${env.WEATHER_API_KEY}"
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

### Tool Filtering

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

**Pattern matching** uses Unix shell-style wildcards:
- `*` - matches everything
- `?` - matches any single character
- `[seq]` - matches any character in seq
- `[!seq]` - matches any character not in seq
- `!pattern` - negation (exclude matching tools)

### Tool Naming

Tools from MCP servers are prefixed with the server name:
- Original tool: `add` → Registered as: `math_add` (from server named "math")
- Original tool: `gh_read_file` → Registered as: `github_gh_read_file` (from server named "github")

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
  # Local math server
  math:
    transport: "stdio"
    command: "uvx"
    args: ["mcp-server-math"]
    tools: ["*"]
    ui_hints_for: ["*"]

  # GitHub server with filtering
  github:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${env.GITHUB_TOKEN}"
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
    headers:
      X-API-KEY: "${env.WEATHER_API_KEY}"
    tools:
      - "get_*"
      - "!get_internal_*"
    require_confirmation_for: ["*"]

# Regular tools work alongside MCP tools
tools:
  - name: read_file
    import_from: llm_workers.tools.unsafe.ReadFileTool

chat:
  system_message: "You are a helpful assistant with access to MCP tools."
```

### Error Handling

- If an MCP server fails to connect, the error is logged and the system continues with other servers
- If a tool name conflicts with an existing tool, the MCP tool is skipped with a warning
- Environment variables that don't exist will raise an error during initialization

## Tools Section

Defines the tools available to the LLMs. Tools can be imported from Python libraries or combined from other tools
in LLM script.

Example:
```yaml
tools:
  - name: _fetch_page_text
    import_from: llm_workers.tools.fetch.FetchPageTextTool

  - name: _LLM
    import_from: llm_workers.tools.llm_tool.build_llm_tool

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
    body:
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
            {output0}
```

In addition to defining tools in the `tools` section, you can define tools inline within `call` statements and within 
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

## Shared Section

The `shared` section provides a way to define reusable configuration data that can be accessed by all custom tools in the script. This is useful for avoiding duplication of common values like API endpoints, prompts, or configuration settings.

**Key Features:**
- Must be a dictionary (key-value pairs)
- Can contain any JSON-serializable data (strings, numbers, booleans, lists, nested objects)
- Accessible in custom tools via the `{shared[key]}` template syntax
- Supports nested access using bracket notation

**Example:**
```yaml
shared:
  prompts:
    test: Yada-yada-yada

tools:
  - name: demo_shared_access
    input:
      - name: query
        description: "Search query"
        type: str
    body:
      result: "Query {query} returned {shared[prompts][test]}"
```

**Usage Notes:**
- The `shared` section is optional and defaults to an empty dictionary
- All tools automatically have access to shared data through the `shared` template variable
- Use bracket notation for accessing nested values: `{shared[category][subcategory]}`
- Shared data is read-only during tool execution
- Changes to the shared section require reloading the script configuration

## Chat Section

Configuration for interactive chat mode:

- `model_ref`: References a model configured in `~/.config/llm-workers/config.yaml` (fast/default/thinking), defaults to "default"
- `system_message`: Instructions for the LLM's behavior
- `default_prompt`: Initial prompt when starting the chat, defaults to empty string
- `user_banner`: Optional markdown-formatted text displayed at the beginning of chat session, defaults to not shown
- `tools`: (Optional) List of tool names or inline tool definitions to make available for this LLM.
  Defaults to all public tools (e.g. not starting with `_`). Supports:
  - Tool names (strings): References to tools defined in the global tools section
  - Inline tool definitions: Complete tool definitions with `name`, `import_from`/`input`, and other tool parameters
  - Mixed usage: Combination of tool names and inline definitions

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
    # reference to tool defined in the tools section
    - read_file
    # inline tool definition
    - name: _run_python_script
      import_from: llm_workers.tools.unsafe.RunPythonScriptTool
```


## CLI Section

Configuration for command-line interface. Contains a list of statements that define the command-line flow.
Flow is run for each command line parameter. Strings can reference "{input}" variable for value of command-line parameter.

See "Custom Tools" section for details on statements.

Example:
```yaml
cli:
  - call: read_file
    params:
      filename: "{input}"
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
        {output0}
  - match: "{output1}"
    matchers:
      - case: "NO CHANGES"
        then:
          result: "{input}: NO CHANGES"
    default:
      - call: write_file
        params:
          filename: "{input}"
          content: "{output1}"
      - result: "{input}: FIXED"
```

# Using Tools

## Importing Tools

To import tools from Python modules, use the `import_from` parameter with a fully-qualified Python path.

The imported symbol can be:
- A `BaseTool` instance (used directly)
- A class extending `langchain_core.tools.base.BaseTool` (instantiated with config parameters)
- A factory function/method returning a `BaseTool` instance

Factory functions must conform to this signature:
```python
def build_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> BaseTool:
```

Examples:

**Importing a tool class:**
```yaml
tools:
  - name: read_file
    import_from: llm_workers.tools.unsafe.ReadFileTool
```

**Importing a factory function:**
```yaml
tools:
  - name: _llm
    import_from: llm_workers.tools.llm_tool.build_llm_tool
```

**Importing with configuration:**
```yaml
tools:
  - name: custom_tool
    import_from: my_module.tools.CustomTool
    config:
      api_key: "your-key"
      timeout: 30
```

## Built-in Tools

In addition to the tools listed below, you can use (almost) any tool from [`langchain` library](https://docs.langchain.com/docs/integrations/tools/),
any imported library, or [define your own](https://python.langchain.com/docs/concepts/tools/) tools.


### Web Fetching Tools

#### fetch_content

```yaml
- name: fetch_content
  import_from: llm_workers.tools.fetch.FetchContentTool
```

Fetches raw content from a given URL and returns it as a string.

**Parameters:**
- `url`: The URL to fetch from
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found ('raise_exception', 'return_error', 'return_empty')
- `on_error`: (Optional) What to do if an error occurs ('raise_exception', 'return_error', 'return_empty')

#### fetch_page_markdown

```yaml
- name: fetch_page_markdown
  import_from: llm_workers.tools.fetch.FetchPageMarkdownTool
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
- name: fetch_page_text
  import_from: llm_workers.tools.fetch.FetchPageTextTool
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
- name: fetch_page_links
  import_from: llm_workers.tools.fetch.FetchPageLinksTool
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
- name: llm
  import_from: llm_workers.tools.llm_tool.build_llm_tool
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
- name: read_file
  import_from: llm_workers.tools.unsafe.ReadFileTool
```

Reads a file and returns its content.

**Parameters:**
- `filename`: Path to the file to read
- `lines`: (Optional) Number of lines to read. If 0 (default), reads the entire file. If negative, reads from the end of file (tail)

#### write_file

```yaml
- name: write_file
  import_from: llm_workers.tools.unsafe.WriteFileTool
```

Writes content to a file.

**Parameters:**
- `filename`: Path to the file to write
- `content`: Content to write
- `append`: (Optional) If true, append to the file instead of overwriting it (default: false)

#### run_python_script

```yaml
- name: run_python_script
  import_from: llm_workers.tools.unsafe.RunPythonScriptTool
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
- name: show_file
  import_from: llm_workers.tools.unsafe.ShowFileTool
```

Opens a file in the OS default application.

**Parameters:**
- `filename`: Path to the file to open

#### bash

```yaml
- name: bash
  import_from: llm_workers.tools.unsafe.BashTool
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
- name: list_files
  import_from: llm_workers.tools.unsafe.ListFilesTool
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
- name: run_process
  import_from: llm_workers.tools.unsafe.RunProcessTool
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
- name: user_input
  import_from: llm_workers.tools.misc.UserInputTool
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
  body:
    - match: "{query}"
      matchers:
        - case: "common query 1"
          then:
            result: "predefined result 1"
        - case: "common query 2"
          then:
            result: "predefined result 2"
      default:
        - call: user_input
          params:
            prompt: "API returned: {query}. Please provide mock response:"
```

This approach allows you to quickly prototype and test tool interactions before implementing the actual tool logic.



#### request_approval

The approval tools provide a way to require explicit user confirmation for potentially dangerous operations through a token-based system:

```yaml
- name: request_approval
  import_from: llm_workers.tools.misc.RequestApprovalTool
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
- name: validate_approval
  import_from: llm_workers.tools.misc.ValidateApprovalTool
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
- name: consume_approval
  import_from: llm_workers.tools.misc.ConsumeApprovalTool
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
  - name: _run_python_script_no_confirmation
    import_from: llm_workers.tools.unsafe.RunPythonScriptTool
    require_confirmation: false

  - name: show_plan_to_user
    import_from: llm_workers.tools.misc.RequestApprovalTool
    description: |-
      Show plan to user and asks for explicit confirmation; upon confirmation return `approval_token` to be used in
      the following call to `run_script`.

  - name: _validate_approval
    import_from: llm_workers.tools.misc.ValidateApprovalTool

  - name: _consume_approval
    import_from: llm_workers.tools.misc.ConsumeApprovalTool

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
    body:
      - call: _validate_approval
        params:
          approval_token: {approval_token}
      - call: _run_python_script_no_confirmation
        params:
          script: {script}
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
    body:
      - call: some_tool
        params:
          tool_param: "{param1}"
      - match: "{output0}"
        matchers:
          - case: "success"
            then:
              - result: "Operation successful: {param1}"
        default:
          - result: "Operation failed"
```
Input section defines the parameters that the tool accepts. These parameters can 
later be referenced in the `body` section using the `{param_name}` syntax.

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

**Inline tool definition (recommended for single-use tools):**
```yaml
- call:
    name: tool_name
    import_from: module.path.ToolClass
    description: "Tool description"  # Optional
    config:  # Optional tool-specific configuration
      key: value
    return_direct: true  # Optional
    confidential: false  # Optional
    require_confirmation: true  # Optional
    ui_hint: "Processing {param1}..."  # Optional
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
    body:
      - call: some_other_tool
        params:
          input: "{data}"
      - result: "Processed: {output0}"
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

## result Statement

Returns a specific value directly.

```yaml
- result: "This is a fixed response"
```

```yaml
- result: 
    status: success
    data:
      value: 42
```

### Dynamic Key Resolution

The `result` statement supports optional `key` and `default` parameters for dynamic value extraction from dictionaries and lists. 
**This feature is primarily intended for cases where the key itself is dynamic** (determined at runtime from template variables).
For static keys, standard templating with bracket notation (e.g., `"{data[static_key]}"`) works just fine.

```yaml
- result: "{shared.ask_schema_expert}"
  key: "{json_schema}"  # Dynamic key from template variable
  default: ""
```

**Parameters:**
- `key`: The key/index to extract from the result. For dictionaries, uses the key as a string. For lists, converts to integer index. **Most useful when the key value comes from template variables.**
- `default`: Optional fallback value if the key is not found or index is out of bounds.

**Examples:**

Primary use case - dynamic key from template variables:
```yaml
- result: "{api_response}"
  key: "{requested_field}"
  default: "N/A"
# Extracts api_response[requested_field] where requested_field is determined at runtime
```

For comparison, static keys should use standard templating:
```yaml
# Preferred for static keys:
- result: "{api_response[schema]}"

# Unnecessary for static keys:
- result: "{api_response}"
  key: "schema"
```

Additional examples:

Dynamic list index:
```yaml
- result: ["first", "second", "third"]
  key: "{index_param}"  # Dynamic index from parameter
# Returns element at runtime-determined index
```

With default value for missing keys:
```yaml
- result: "{user_data}"
  key: "{field_to_extract}"
  default: "field_not_found"
# Returns user_data[field_to_extract] or "field_not_found" if key doesn't exist
```

## match Statement

Conditionally executes different actions based on matching patterns:

```yaml
- match: "{input}"
  trim: true  # Optional, removes whitespace before matching
  matchers:
    - case: "help"  # Exact match
      then:
        - result: "Available commands: help, status, search"
    
    - pattern: "search (.+)"  # Regex pattern match
      then:
        - call: search_tool
          params:
            query: "{1}"  # Reference to captured group
  
  default:  # Executed if no matches found
    - result: "Unknown command. Type 'help' for assistance."
```

## Composing Statements

Statements can be used as part of a tool's `body` section to create composite tools:

```yaml
tools:
  - name: advanced_search
    description: "Performs an enhanced search with preprocessing"
    body:
      - call: normalize_input
        params:
          text: "{input}"
      - call: search_database
        params:
          query: "{output0}"
      - match: "{output1}"
        matchers:
          - case: ""
            then:
              - result: "No results found"
        default:
          - result: "{output1}"
```

## Template Variables

Custom tools support template variables in strings:

- Direct references: `"{param_name}"` - preserves referenced parameter type
- Complex templates: `"The value is {param_name} and {other_param}"`
- Nested element access: 
  - Dictionary keys: `"{param_dict[key_name]}"` - accesses dictionary values by key
  - List indices: `"{param_list[0]}"` - accesses list elements by index  
  - Nested structures: `"{param_dict[nested][value]}"` - supports multiple levels of nesting
- Shared data access: `"{shared[key]}"` - accesses values from the shared configuration section
- Tool input parameters: `"{<parm_name>}"`
- (inside the list of statements) Previous statement results: `"{outputN}"` where N is the 0-based index of a previous statement
- (inside `match` statement) Regex capture groups: `"{matchN}"` when using regex patterns in match statements

**Note:** Nested element access uses bracket notation (`[key]`) for both dictionary keys and list indices. This syntax works with complex templates but requires the entire template to be processed as a string (the nested values cannot preserve their original types).

**Example with nested element and shared access:**
```yaml
shared:
  app:
    name: "MyApp"
    version: "1.0"
  templates:
    user_format: "Welcome to {shared[app][name]}!"

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
    body:
      - result: "User {user_profile[name]} has email {user_profile[contact][email]} and first setting is {settings[0]}. {shared[templates][user_format]}"
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

