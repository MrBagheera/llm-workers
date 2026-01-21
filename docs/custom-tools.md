---
layout: default
title: Custom Tools
nav_order: 2.8
---

## Table of contents
{: .no_toc }

* TOC
{:toc}

# Defining Custom Tools

Custom tools allow you to compose existing tools and logic into reusable components. Use the `input`, `tools`, and `do` sections:

```yaml
tools:
  - name: my_custom_tool
    description: "Description of what this tool does"
    input:
    - name: name
      description: "Description of first parameter"
      type: "str"
      default: "default value"  # Optional
    - name: param2
      description: "Description of second parameter"
      type: "int"
    tools:  # Optional, local tools available only within this custom tool
    - import_tool: llm_workers.tools.fs.read_file
    return_direct: true  # Optional, returns result directly to user
    do:
      - call: read_file  # References local tool
        params:
          path: ${name}          
      - if: ${len(_).trim() == 0}
        then:
          eval: "File ${name} is empty"
        else:
          eval: ${_}
```

**Key Sections:**
- `input`: Defines the parameters that the tool accepts. These parameters can be referenced in the `do` section using the `${param_name}` syntax.
- `tools`: Optional list of tools available only within this custom tool. These are "local tools" that don't pollute the global namespace.
- `do`: Contains one or more statements that define the tool's behavior.

**Tool Scoping:**
Custom tools create their own tool scope. Tools defined in the `tools` field are only accessible within that custom tool's `do` section. Local tools can shadow global tools with the same name.

The `do` section contains one or more statements that can be composed in various ways:

## call Statement

Executes a specific tool with optional parameters. Tools must be defined in a `tools` section before they can be called.

**Syntax:**
```yaml
- call: tool_name
  params:
    param1: value1
    param2: value2
  catch: [error_type1, error_type2]  # Optional error handling
  store_as: result_var  # Optional, stores result in a named variable
  ui_hint: 'Processing ${param1}'  # Optional, overrides the tool's UI hint
```

**Parameters:**
- `call`: Name of the tool to execute
- `params`: Dictionary of parameters to pass to the tool
- `catch`: (Optional) List of error types to catch and handle gracefully
- `store_as`: (Optional) Variable name to store the result. Can be referenced later using `${variable_name}`
- `ui_hint`: (Optional) Override the tool's built-in UI hint with a custom message. Supports template variables from the current evaluation context (e.g., `${param_name}`, `${_}`)

**Example:**
```yaml
shared:
  tools:
    - import_tool: llm_workers.tools.fs.read_file
    - name: process_file
      input:
        - name: path
          type: str
      tools:
        - import_tool: llm_workers.tools.fs.read_file  # Local tool
      do:
        - call: read_file  # References local tool
          params:
            path: "${path}"
        - eval: "Processed: ${_}"
```

**Example with store_as:**
```yaml
tools:
  - name: multi_step_process
    input:
      - name: script
        type: str
      - name: approval_token
        type: str
    do:
      - call: validate_approval
        params:
          approval_token: "${approval_token}"
      - call: run_python_script
        params:
          script: "${script}"
        store_as: script_result  # Store result in named variable
      - call: consume_approval
        params:
          approval_token: "${approval_token}"
      - eval: "${script_result}"  # Reference stored result
```

**Example with ui_hint:**
```yaml
tools:
  - name: check_translations
    description: "Checks translations for a given key"
    input:
      - name: key
        type: str
      - name: locale
        type: str
    do:
      - call: fetch_translations
        params:
          key: "${key}"
          locale: "${locale}"
        ui_hint: 'Checking translations for key "${key}" (${locale})'
```

This will display `Checking translations for key "welcome_message" (en)` in the UI instead of the default tool hint when called with `key="welcome_message"` and `locale="en"`.

**Note:** Without `store_as`, the result of each statement is available as `${_}` in the next statement. With `store_as`, you can reference the result by name later in the workflow, which is useful when you need to perform multiple operations and reference earlier results.

**Important Changes:**
As of recent versions, inline tool definitions in `call` statements are no longer supported. Tools must be defined in a `tools` section (either in `shared.tools`, `chat.tools`, custom tool's `tools`, or CLI's `tools`) before they can be referenced by name in `call` statements.

**For custom tools:** Define local tools in the custom tool's `tools` field
**For global access:** Define tools in `shared.tools`
**For chat/CLI:** Define tools in `chat.tools` or `cli.tools`

This separation provides:
- Clearer distinction between tool definition and tool usage
- Better tool scoping and encapsulation
- Support for local tools that don't pollute the global namespace

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

## if Statement

Executes different actions based on a boolean condition:

```yaml
- if: "${condition}"
  then:
    <statement(s)>  # Executed if condition is truthy
  else:  # Optional
    <statement(s)>  # Executed if condition is falsy
  store_as: result_var  # Optional, stores the result of the executed branch
```

**Parameters:**
- `if`: Boolean condition expression to evaluate
- `then`: Statement(s) to execute if condition is truthy
- `else`: (Optional) Statement(s) to execute if condition is falsy
- `store_as`: (Optional) Variable name to store the result of whichever branch executes

### Basic Examples

**Simple boolean check:**
```yaml
- if: "${user_authenticated}"
  then:
    call: fetch_user_data
    params:
      user_id: "${user_id}"
  else:
    eval: "Please log in first"
```

**Boolean expression with comparison:**
```yaml
- if: "${score >= 80 and status == 'active'}"
  then:
    eval: "Eligible for promotion"
  else:
    eval: "Not eligible"
```

**Membership test:**
```yaml
- if: "${movie_title in stub_data}"
  then:
    eval: "${stub_data[movie_title]}"
  else:
    call: fetch_from_api
    params:
      query: "${movie_title}"
```

**Optional else clause:**
```yaml
- if: "${debug_mode}"
  then:
    call: log_debug_info
    params:
      message: "Debug enabled"
# If debug_mode is false, returns None and continues
```

**Multiple statements in branches:**
```yaml
- if: "${needs_preprocessing}"
  then:
    - call: normalize_data
      params:
        data: "${input}"
    - call: validate_data
      params:
        data: "${_}"
  else:
    eval: "${input}"
```

**Nested if statements:**
```yaml
- if: "${user_role == 'admin'}"
  then:
    - if: "${action == 'delete'}"
      then:
        call: delete_resource
        params:
          id: "${resource_id}"
      else:
        eval: "Action not allowed"
  else:
    eval: "Admin access required"
```

### Truthiness Rules

The condition uses Python truthiness rules:
- **Truthy values:** Non-empty strings, non-zero numbers, non-empty lists/dicts, `True`
- **Falsy values:** Empty string `""`, zero `0`, `False`, empty lists `[]`, empty dicts `{}`

**Note:** Direct evaluation of `None` variables is not supported due to expression system limitations. Use explicit comparisons like `${value is not None}` when checking for `None`.

## for_each Statement

Iterates over a collection and executes the body for each element:

```yaml
- for_each: ${collection}
  do:
    <statement(s)>  # Executed for each element
  store_as: result_var  # Optional, stores the final result
```

**Parameters:**
- `for_each`: Expression that evaluates to a list, dict, or scalar value
- `do`: Statement(s) to execute for each element
- `store_as`: (Optional) Variable name to store the result

**Behavior by Input Type:**

| Input Type | Output Type | Available Variables |
|------------|-------------|---------------------|
| Dict | Dict with same keys, mapped values | `_` = current value, `key` = current key |
| Iterable (list, tuple, set, etc.) | List of results | `_` = current element |
| Other (str, int, None, etc.) | Single result | `_` = the scalar value |

**Note:** Strings are treated as scalars, not as character iterables.

### Basic Examples

**Iterating over a list:**
```yaml
- for_each: ${names}
  do:
    eval: "Hello, ${_}!"
```

With input `["Alice", "Bob"]`, returns `["Hello, Alice!", "Hello, Bob!"]`.

**Iterating over a dict:**
```yaml
- for_each: ${users}
  do:
    eval: "User ${key} is ${_}"
```

With input `{"id1": "Alice", "id2": "Bob"}`, returns `{"id1": "User id1 is Alice", "id2": "User id2 is Bob"}`.

**Scalar passthrough:**
```yaml
- for_each: ${single_value}
  do:
    eval: "Value: ${_}"
```

With input `42`, returns `"Value: 42"`.

**Other iterables (tuple, set, etc.):**
```yaml
- for_each: ${items}
  do:
    eval: "${_ * 2}"
```

With input `(1, 2, 3)` (tuple) or `{1, 2, 3}` (set), returns `[2, 4, 6]` (always a list).

### Advanced Examples

**Multi-statement body:**
```yaml
- for_each: ${numbers}
  do:
    - eval: "${_ * 2}"
      store_as: doubled
    - eval: "${doubled + 1}"
```

With input `[1, 2, 3]`, returns `[3, 5, 7]`.

**Nested for_each:**
```yaml
- for_each: ${matrix}
  do:
    for_each: ${_}
    do:
      eval: "${_ * 10}"
```

With input `[[1, 2], [3, 4]]`, returns `[[10, 20], [30, 40]]`.

**With tool calls:**
```yaml
- for_each: ${file_paths}
  do:
    call: read_file
    params:
      path: "${_}"
      lines: 1000
```

**Accessing parent context:**
```yaml
- for_each: ${items}
  do:
    eval: "${prefix}: ${_}"
```

Variables from the parent context (like `${prefix}`) remain accessible inside the loop body.

**Storing results:**
```yaml
- for_each: ${items}
  do:
    eval: "${_ * 2}"
  store_as: doubled_items
- eval: "Processed ${len(doubled_items)} items"
```

### Edge Cases

- **Empty list:** Returns `[]`
- **Empty dict:** Returns `{}`
- **None input:** Applies body to `None`, returns single result

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
      - if: "${_ == ''}"
        then:
          eval: "No results found"
        else:
          eval: "${_}"
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
- Shared data variables: `"${key}"` - accesses variables defined in the `shared.data` section
- Tool input parameters: `"${param_name}"`
- (inside the list of statements) Previous statement results: `"${_}"` for the immediate previous result
- Python expressions: `"${a + b}"`, `"${len(items)}"`, `"${value if condition else default}"` - supports any safe Python expression via simpleeval

**Type Preservation:** When a string contains only a single expression (e.g., `"${param}"`), the original type is preserved. When text or multiple expressions are present, the result is converted to a string.

**Note:** Expressions are evaluated using simpleeval for safety, which supports standard Python operations but restricts potentially dangerous operations.

**Example with nested element and shared variables:**
```yaml
shared:
  data:
    app:
      name: "MyApp"
      version: "1.0"
    templates:
      user_format: "Welcome to ${app['name']}!"

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
        - eval: "User ${user_profile['name']} has email ${user_profile['contact']['email']} and first setting is ${settings[0]}. ${templates['user_format']}"
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

**Important:** Note that shared variables are accessed directly (e.g., `${app}`, `${templates}`) without a `shared.` prefix. This is different from earlier versions where you needed to use `${shared.app}` or `${shared['app']}`.
