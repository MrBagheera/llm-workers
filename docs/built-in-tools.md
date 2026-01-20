---
layout: default
title: Built-in Tools
nav_order: 2.5
---

## Table of contents
{: .no_toc }

* TOC
{:toc}

# Built-in Tools

In addition to the tools listed below, you can use (almost) any tool from [`langchain` library](https://docs.langchain.com/docs/integrations/tools/),
any imported library, or [define your own](https://python.langchain.com/docs/concepts/tools/) tools.


## Web Fetching Tools

### fetch_content

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

### fetch_page_markdown

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

### fetch_page_text

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

### fetch_page_links

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


## LLM Tool

### build_llm_tool

```yaml
- import_tool: llm_workers.tools.llm_tool.build_llm_tool
  name: llm
```

Creates a tool that allows calling an LLM with a prompt and returning its response.

**Configuration Parameters**:
- `model_ref`: (Optional) Reference to a model configured in `~/.config/llm-workers/config.yaml` (fast/default/thinking), defaults to "default"
- `system_message`: (Optional) System message to use for this specific LLM tool
- `tools`: (Optional) List of tool names or inline tool definitions to make available for this specific LLM tool.
See `tools` definition in [Chat Section](llm-script.md#chat-section) for details.
- `remove_past_reasoning`: (Optional) Whether to hide past LLM reasoning, defaults to false
- `extract_json`: (Optional) Filters result to include only JSON blocks, defaults to "false".
Useful for models without Structured Output, like Claude. Fallbacks to entire message if no "```json" blocks are found. Possible values:
  - "first" - returns only the first JSON block found in the response
  - "last" - returns only the last JSON block found in the response
  - "all" - returns all JSON blocks found in the response as list
  - "false" - returns the full response without filtering
- `ui_hint`: (Optional) Template string to display in the UI when this tool is called. Supports variable interpolation.

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

**Example with Custom Configuration and UI Hint**:
```yaml
tools:
  - import_tool: llm_workers.tools.llm_tool.build_llm_tool
    name: llm_tool
    config:
      model_ref: fast
    ui_hint: Extracting Metacritic score from search results
```

The tool can be used to create custom LLM-powered tools within your workflows, enabling tasks like summarization,
analysis, formatting, or generating structured content based on input data.


## File and System Tools

These tools provide access to the file system and allow code execution, which makes them potentially unsafe for use with untrusted inputs:

### read_file

```yaml
- import_tool: llm_workers.tools.fs.ReadFileTool
  name: read_file
```

Reads a file and returns its content. Output is limited to `lines` parameter.

**Parameters:**
- `path`: Path to the file to read
- `lines`: Number of lines to read (required)
- `offset`: (Optional) Offset in lines. >=0 means from the start of the file, <0 means from the end of the file (default: 0)
- `show_line_numbers`: (Optional) If true, prefix each line with its line number (default: false)

### grep_files

```yaml
- import_tool: llm_workers.tools.fs.GrepFilesTool
  name: grep_files
```

Search for regex patterns within files. Returns matching lines with optional context.

**Parameters:**
- `pattern`: Regular expression pattern to search for
- `files_glob`: File path, directory, or glob pattern (e.g., `*.py`, `src/**/*.ts`). This parameter is required and determines where to search:
  - If a single file path: searches that file
  - If a directory path: recursively searches all files in that directory
  - If a glob pattern (containing `*`, `?`, or `[]`): searches files matching the pattern
- `lines_before`: (Optional) Number of lines to show before each match (default: 0)
- `lines_after`: (Optional) Number of lines to show after each match (default: 0)
- `case_insensitive`: (Optional) Ignore case when matching (default: false)
- `max_results`: (Optional) Maximum number of matches to return (default: 50)
- `output_mode`: (Optional) Output format (default: `"content"`):
  - `"content"`: Returns matching lines with file path and line number
  - `"files_only"`: Returns only file paths containing matches
  - `"count"`: Returns match counts

**Returns:**
A dictionary containing:
- `total_matches`: Total number of matches found
- `files_searched`: Number of files searched
- `matches` (when `output_mode="content"`): List of match objects with `file`, `line_number`, `content`, and optionally `context_before`/`context_after`
- `files` (when `output_mode="files_only"`): List of file paths
- `files_with_matches` (when `output_mode="count"`): Number of files with matches

**Example:**
```yaml
- call: grep_files
  params:
    pattern: "def.*test"
    files_glob: "**/*.py"
    lines_before: 2
    lines_after: 1
    case_insensitive: true
```

### write_file

```yaml
- import_tool: llm_workers.tools.unsafe.WriteFileTool
  name: write_file
```

Writes content to a file.

**Parameters:**
- `path`: Path to the file to write
- `content`: Content to write
- `append`: (Optional) If true, append to the file instead of overwriting it (default: false)
- `if_exists`: (Optional) What to do if file exists: `"overwrite"`, `"append"`, or `"error"` (default: `"overwrite"`)

### run_python_script

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

### show_file

```yaml
- import_tool: llm_workers.tools.unsafe.ShowFileTool
  name: show_file
```

Opens a file in the OS default application.

**Parameters:**
- `filename`: Path to the file to open

### bash

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

### list_files

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

### run_process

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


## Miscellaneous Tools

### user_input

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



### request_approval

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

### validate_approval

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

### consume_approval

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

This pattern allows LMs to request approval for potentially dangerous operations, ensuring users explicitly consent before execution while preventing token reuse.
