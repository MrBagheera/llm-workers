LLM scripts are YAML configuration files that define how to interact with large language models (LLMs) and what 
tools LLMs can use. You should treat them like a normal scripts. In particular - DO NOT run LLM scripts from
unknown / untrusted sources. Scripts can easily download and run malicious code on your machine, or submit your secrets 
to some web site. 

For real examples, see [generic-assistant.yaml](src/llm_workers/generic-assistant.yaml)
and files in [`examples`](examples/) directory.


Table of Contents
=================

* [Basic Structure](#basic-structure)
  * [Models Section](#models-section)
  * [Tools Section](#tools-section)
  * [Shared Section](#shared-section)
  * [Common Tool Parameters](#common-tool-parameters)
  * [Chat Section](#chat-section)
  * [CLI Section](#cli-section)
* [Using Python tools](#using-python-tools)
  * [Importing From Classes](#importing-from-classes)
  * [Importing From Factory Methods](#importing-from-factory-methods)
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
  * [Composing Statements](#composing-statements)
  * [Template Variables](#template-variables)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

# Basic Structure

```yaml
models:
  - name: <model_name>
    provider: <provider_name>
    model: <model_id>
    rate_limiter: # Optional
      requests_per_second: <float>
      max_bucket_size: <int>
    model_params: # Optional
      temperature: <float>
      max_tokens: <int>
      # [additional parameters...]

tools:
  - name: <tool_name>
    class: <tool_class> # Required for importing tools. Use class OR factory
    factory: <tool_factory> # Required for importing tools. Use class OR factory
    description: <description> # Optional
    input: # Required for custom tools
      - name: <param_name>
        description: <description>
        type: <type>
        default: <default_value> # Optional
    confidential: <boolean> # Optional
    return_direct: <boolean> # Optional
    ui_hint: <template_string> # Optional
    body: # Required for custom tools
      <statement(s)> # List of statements for more complex flows

shared: # Optional shared configuration accessible to all tools
  <key>: <value>
  # Can contain any JSON-serializable data

chat: # For interactive chat mode
  model_ref: <model_name> # Optional, references model by name. If not defined, uses model named "default".
  remove_past_reasoning: <boolean> # Optional
  show_reasoning: <boolean> # Optional
  auto_open_changed_files: <boolean> # Optional
  user_banner: | # Optional, markdown-formatted text displayed at the beginning of chat
    <banner text>
  system_message: |
    <system prompt>
  default_prompt: | # Optional
    <default prompt>

cli: # For command-line interface
  <statement(s)> # List of statements for more complex flows
```

## Models Section

Defines the LLMs to use:

- `name`: Identifier for the model
- `provider`: Service provider (e.g., `bedrock`, `bedrock_converse`, `openai`)
- `model`: Model identifier
- `rate_limiter`: Optional rate limiting configuration
- `model_params`: Model-specific parameters like temperature and token limits

Example:
```yaml
models:
  - name: default
    provider: openai
    model: gpt-4o
    rate_limiter:
      requests_per_second: 1.0
      max_bucket_size: 10
    model_params:
      temperature: 0.7
      max_tokens: 1500
  - name: thinking
    provider: bedrock_converse
    model: us.anthropic.claude-3-7-sonnet-20250219-v1:0
    model_params:
      temperature: 1
      max_tokens: 32768
      additional_model_request_fields:
        thinking:
          type: enabled
          budget_tokens: 16000
```

## Tools Section

Defines the tools available to the LLMs. Tools can be imported from Python libraries or combined from other tools
in LLM script.

Example:
```yaml
tools:
  - name: _fetch_page_text
    class: llm_workers.tools.fetch.FetchPageTextTool

  - name: _LLM
    factory: llm_workers.tools.llm_tool.build_llm_tool

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
        model: default
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

## Common Tool Parameters
- `name`: Unique identifier for the tool. This name is used to reference the tool in other parts of the script.
  Names should be unique within the script and should not contain spaces or special characters. Tools with names starting with `_`
  are considered to be "private". Those are not available for LLM use unless added to `tool_refs` list explicitly.
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


## Chat Section

Configuration for interactive chat mode:

- `model_ref`: References a model defined in the models section, defaults to "default"
- `system_message`: Instructions for the LLM's behavior
- `default_prompt`: Initial prompt when starting the chat, defaults to empty string
- `user_banner`: Optional markdown-formatted text displayed at the beginning of chat session, defaults to not shown
- `remove_past_reasoning`: Whether to hide past LLM reasoning from subsequent LLM calls, defaults to `false`
- `show_reasoning`: Whether to display LLM reasoning process to user, defaults to `false`
- `tool_refs`: Optional list of tool names to make available to the LLM, defaults to all public tools (e.g. not starting with `_`)
- `auto_open_changed_files`: Whether to automatically open files modified during LLM call, defaults to `false`
- `file_monitor_include`: List of glob patterns for files to monitor for changes, defaults to ['*']
- `file_monitor_exclude`: List of glob patterns for files to exclude from monitoring, defaults to ['.*', '*.log']

```yaml
chat:
  model_ref: thinking
  remove_past_reasoning: true
  show_reasoning: true
  auto_open_changed_files: true
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

# Using Python tools

## Importing

### From Classes

To import tools from Python class, use the `class` parameter. The class must be a fully-qualified Python class name.
Tool class must extend `langchain_core.tools.base.BaseTool` class.

Example:
```yaml
tools:
  - name: read_file
    class: llm_workers.tools.unsafe.ReadFileTool
```

### From Factory Methods

To import tools from factory methods, use the `factory` parameter. The factory method must be a fully-qualified Python method name.
The factory method must conform to following syntax:
```python
def build_llm_tool(context: WorkersContext, tool_config: Dict[str, Any]) -> BaseTool:
```
and must return an instance of `langchain_core.tools.base.BaseTool` class.

Example:
```yaml
tools:
  - name: _llm
    factory: llm_workers.tools.llm_tool.build_llm_tool
```

## Built-in Tools

In addition to the tools listed below, you can use (almost) any tool from [`langchain` library](https://docs.langchain.com/docs/integrations/tools/),
any imported library, or [define your own](https://python.langchain.com/docs/concepts/tools/) tools.


### Web Fetching Tools

#### `fetch_content`

```yaml
- name: fetch_content
  class: llm_workers.tools.fetch.FetchContentTool
```

Fetches raw content from a given URL and returns it as a string.

**Parameters:**
- `url`: The URL to fetch from
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found ('raise_exception', 'return_error', 'return_empty')
- `on_error`: (Optional) What to do if an error occurs ('raise_exception', 'return_error', 'return_empty')

#### `fetch_page_markdown`

```yaml
- name: fetch_page_markdown
  class: llm_workers.tools.fetch.FetchPageMarkdownTool
```

Fetches web page or page element and converts it to markdown.

**Parameters:**
- `url`: The URL of the page
- `xpath`: (Optional) The xpath to the element containing the text; if not specified the entire page will be returned
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found
- `on_error`: (Optional) What to do if an error occurs

#### `fetch_page_text`

```yaml
- name: fetch_page_text
  class: llm_workers.tools.fetch.FetchPageTextTool
```

Fetches the text from web page or web page element.

**Parameters:**
- `url`: The URL of the page
- `xpath`: (Optional) The xpath to the element containing the text; if not specified the entire page will be returned
- `headers`: (Optional) Extra headers to use for the request
- `on_no_content`: (Optional) What to do if no matching content is found
- `on_error`: (Optional) What to do if an error occurs

#### `fetch_page_links`

```yaml
- name: fetch_page_links
  class: llm_workers.tools.fetch.FetchPageLinksTool
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

#### `build_llm_tool`

```yaml
- name: llm
  factory: llm_workers.tools.llm_tool.build_llm_tool
```

Creates a tool that allows calling an LLM with a prompt and returning its response.

**Configuration Parameters**:
- `model_ref`: (Optional) Reference to a model defined in the models section, defaults to "default"
- `system_message`: (Optional) System message to use for this specific LLM tool
- `tool_refs`: (Optional) List of tool names to make available for this specific LLM tool, default to all public tools (e.g. not starting with `_`)
- `remove_past_reasoning`: (Optional) Whether to hide past LLM reasoning, defaults to false

**Function**:
This factory method creates a `StructuredTool` that passes a prompt to an LLM and returns the model's response.

**Parameters**:
- `prompt`: Text prompt to send to the LLM

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

The tool can be used to create custom LLM-powered tools within your workflows, enabling tasks like summarization,
analysis, formatting, or generating structured content based on input data.


Based on the `unsafe.py` file, here are the available tools that interact with the file system and execute code:

### File and System Tools

These tools provide access to the file system and allow code execution, which makes them potentially unsafe for use with untrusted inputs:

#### `read_file`

```yaml
- name: read_file
  class: llm_workers.tools.unsafe.ReadFileTool
```

Reads a file and returns its content.

**Parameters:**
- `filename`: Path to the file to read
- `lines`: (Optional) Number of lines to read. If 0 (default), reads the entire file. If negative, reads from the end of file (tail)

#### `write_file`

```yaml
- name: write_file
  class: llm_workers.tools.unsafe.WriteFileTool
```

Writes content to a file.

**Parameters:**
- `filename`: Path to the file to write
- `content`: Content to write
- `append`: (Optional) If true, append to the file instead of overwriting it (default: false)

#### `run_python_script`

```yaml
- name: run_python_script
  class: llm_workers.tools.unsafe.RunPythonScriptTool
```

Runs a Python script and returns its output.

**Parameters:**
- `script`: Python script to run. Must be valid Python code

**Behavior:**
- Creates a temporary file with the script
- Executes it using python3
- Returns stdout output
- Deletes the script file after execution
- Requires user confirmation by default

#### `show_file`

```yaml
- name: show_file
  class: llm_workers.tools.unsafe.ShowFileTool
```

Opens a file in the OS default application.

**Parameters:**
- `filename`: Path to the file to open

#### `bash`

```yaml
- name: bash
  class: llm_workers.tools.unsafe.BashTool
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

#### `list_files`

```yaml
- name: list_files
  class: llm_workers.tools.unsafe.ListFilesTool
```

Lists files and directories with optional detailed information.

**Parameters:**
- `path`: Path to directory to list or file to show info
- `depth`: (Optional) Recursive directory depth (default: 0 - no recursion)
- `permissions`: (Optional) Whether to show permissions (default: false)
- `times`: (Optional) Whether to show creation and modification times (default: false)
- `sizes`: (Optional) Whether to show file sizes (default: false)

#### `run_process`

```yaml
- name: run_process
  class: llm_workers.tools.unsafe.RunProcessTool
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

#### `user_input`

```yaml
- name: user_input
  class: llm_workers.tools.misc.UserInputTool
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



#### `request_approval`

The approval tools provide a way to require explicit user confirmation for potentially dangerous operations through a token-based system:

```yaml
- name: request_approval
  class: llm_workers.tools.misc.RequestApprovalTool
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

#### `validate_approval`

```yaml
- name: validate_approval
  class: llm_workers.tools.misc.ValidateApprovalTool
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

#### `consume_approval`

```yaml
- name: consume_approval
  class: llm_workers.tools.misc.ConsumeApprovalTool
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
    class: llm_workers.tools.unsafe.RunPythonScriptTool
    require_confirmation: false

  - name: show_plan_to_user
    class: llm_workers.tools.misc.RequestApprovalTool
    description: |-
      Show plan to user and asks for explicit confirmation; upon confirmation return `approval_token` to be used in
      the following call to `run_script`.

  - name: _validate_approval
    class: llm_workers.tools.misc.ValidateApprovalTool

  - name: _consume_approval
    class: llm_workers.tools.misc.ConsumeApprovalTool

  - name: run_python_script
    description: Consume approval_token and run given Python script
    params:
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

To define custom tools, use the `body` section of the tool definition:

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

The `body` section contains one or more statements that can be composed in various ways:

## `call` Statement

Executes a specific tool with optional parameters.

```yaml
- call: tool_name
  params:
    param1: value1
    param2: value2
  catch: [error_type1, error_type2]  # Optional error handling
```

## `result` Statement

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

## `match` Statement

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

