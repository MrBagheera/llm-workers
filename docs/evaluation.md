---
layout: default
title: Evaluation Framework
nav_order: 4
---

## Table of contents
{: .no_toc }

* TOC
{:toc}

# Evaluation Framework

The evaluation framework allows you to run automated evaluation suites against LLM scripts and report scores. This is useful for:
- Testing LLM script behavior systematically
- Comparing performance across different model configurations
- Regression testing when modifying scripts
- Benchmarking prompt effectiveness

## Installation

The evaluation framework is provided by the `llm-workers-evaluation` package:

```bash
pip install llm-workers-evaluation
```

## Running Evaluations

Use the `llm-workers-evaluate` command to run evaluation suites:

```bash
# Basic usage
llm-workers-evaluate <script_file> <evaluation_suite>

# With custom iteration count
llm-workers-evaluate -n 5 <script_file> <evaluation_suite>

# With verbose output
llm-workers-evaluate --verbose <script_file> <evaluation_suite>

# With debug logging
llm-workers-evaluate --debug <script_file> <evaluation_suite>
```

**Arguments:**
- `script_file`: Path to the LLM script file or `module:resource.yaml` format
- `evaluation_suite`: Path to the evaluation suite YAML file

**Options:**
- `--iterations`, `-n`: Number of iterations per test (overrides suite default)
- `--verbose`: Enable verbose output (can be used multiple times)
- `--debug`: Enable debug mode (can be used multiple times)

## Evaluation Suite Format

Evaluation suites are YAML files defining tests that return scores between 0.0 and 1.0.

### Basic Structure

```yaml
# Shared configuration for all suites and tests
shared:
  data:
    <key>: <value>
  tools:
    - <tool_definition>

# Default number of iterations per test
iterations: 10

# Evaluation suites
suites:
  suite_name:
    data:
      <key>: <value>
    tools:
      - <tool_definition>
    tests:
      test_name:
        data:
          <key>: <value>
        tools:
          - <tool_definition>
        do:
          <statement(s)>  # Must return float score [0.0, 1.0]
```

### Sections

#### shared Section

The `shared` section defines data and tools available to all suites and tests:

```yaml
shared:
  data:
    expected_greeting: "Hello"
    api_base: "https://api.example.com"
  tools:
    - import_tool: llm_workers.tools.fetch.fetch_page_text
```

#### iterations

The `iterations` field specifies the default number of times each test is run:

```yaml
iterations: 10  # Each test runs 10 times by default
```

This can be overridden via the `-n` command-line option.

#### suites Section

Each suite groups related tests and can define its own data and tools:

```yaml
suites:
  response_quality:
    data:
      threshold: 0.8
    tests:
      format_check:
        do:
          eval: 1.0
      content_check:
        do:
          eval: 0.5
```

#### tests Section

Each test within a suite defines:
- `data`: Test-specific data (merges with shared + suite data)
- `tools`: Test-specific tools (concatenates with shared + suite tools)
- `do`: Statement(s) that must return a float score between 0.0 and 1.0

```yaml
tests:
  my_test:
    data:
      input_value: "test"
    do:
      - call: some_tool
        params:
          input: "${input_value}"
      - eval: "${1.0 if _ == expected_result else 0.0}"
```

## Data and Tool Merging

Data and tools are merged in order: `shared` → `suite` → `test`

**Data merging:** Later values override earlier ones with the same key.

```yaml
shared:
  data:
    value: "shared"
    common: "from_shared"

suites:
  my_suite:
    data:
      value: "suite"  # Overrides shared.value
    tests:
      my_test:
        data:
          value: "test"  # Overrides suite.value
        do:
          # value = "test", common = "from_shared"
          eval: "${value}"
```

**Tool merging:** Lists are concatenated.

```yaml
shared:
  tools:
    - import_tool: tool_a

suites:
  my_suite:
    tools:
      - import_tool: tool_b
    tests:
      my_test:
        tools:
          - import_tool: tool_c
        do:
          # Available tools: tool_a, tool_b, tool_c
          call: tool_c
```

## Writing Test Statements

Tests use the same statement syntax as custom tools (see [Custom Tools](custom-tools.md)). The final result must be a float between 0.0 and 1.0.

### Simple Score Returns

```yaml
tests:
  always_pass:
    do:
      eval: 1.0

  always_fail:
    do:
      eval: 0.0

  half_score:
    do:
      eval: 0.5
```

### Conditional Scoring

```yaml
tests:
  check_response:
    data:
      expected: "hello"
      actual: "hello"
    do:
      eval: "${1.0 if actual == expected else 0.0}"
```

### Multi-Step Tests

```yaml
tests:
  complex_validation:
    do:
      - call: fetch_data
        params:
          url: "${api_endpoint}"
      - eval: |
          ${
            1.0 if 'success' in _
            else 0.5 if 'partial' in _
            else 0.0
          }
```

### Using LLM for Evaluation

You can use LLM tools to evaluate responses:

```yaml
shared:
  tools:
    - import_tool: llm_workers.tools.llm_tool.build_llm_tool
      name: llm

suites:
  quality:
    tests:
      response_quality:
        data:
          test_prompt: "Explain what a variable is"
        do:
          - call: llm
            params:
              model_ref: default
              prompt: "${test_prompt}"
            store_as: response
          - call: llm
            params:
              model_ref: fast
              prompt: |
                Rate the following response on a scale of 0.0 to 1.0 for clarity and accuracy.
                Return ONLY a single decimal number, nothing else.

                Question: ${test_prompt}
                Response: ${response}
```

## Output Format

Evaluation results are output as YAML to stdout:

```yaml
final_score: 0.75
per_suite:
  suite_name:
    final_score: 0.80
    per_test:
      test_name_1: 0.85
      test_name_2: 0.75
```

Token usage statistics are printed to stderr (if any tokens were used):

```
Total Session Tokens: 1,234 total
Per-Model:
  default: 1,234 (500 in, 734 out)
```

## Score Handling

The evaluation framework handles various result scenarios:

| Scenario | Handling |
|----------|----------|
| Test returns valid score | Used as-is |
| Test returns `None` | Score = 0.0, warning logged |
| Test returns non-numeric | Score = 0.0, warning logged |
| Test returns < 0.0 | Clamped to 0.0, warning logged |
| Test returns > 1.0 | Clamped to 1.0, warning logged |
| Test raises exception | Score = 0.0 for that iteration, error logged |
| Suite has no tests | Score = 0.0, warning logged |

## Scoring Calculation

Scores are calculated hierarchically:

1. **Test score**: Average of all iteration scores for that test
2. **Suite score**: Average of all test scores in that suite
3. **Final score**: Average of all suite scores

Example with 2 iterations:
```
test_a: [1.0, 0.8] → average = 0.9
test_b: [0.5, 0.5] → average = 0.5
suite score = (0.9 + 0.5) / 2 = 0.7
```

## Complete Example

```yaml
# evaluation-suite.yaml
shared:
  data:
    greetings:
      - "Hello"
      - "Hi"
      - "Hey"
  tools:
    - import_tool: llm_workers.tools.llm_tool.build_llm_tool
      name: llm

iterations: 5

suites:
  greeting_tests:
    data:
      language: "English"
    tests:
      recognizes_hello:
        data:
          input: "Hello there!"
        do:
          - call: llm
            params:
              prompt: |
                Does this message contain a greeting? Answer only "yes" or "no".
                Message: ${input}
          - eval: "${1.0 if 'yes' in _.lower() else 0.0}"

      generates_greeting:
        do:
          - call: llm
            params:
              prompt: "Generate a simple ${language} greeting in one word."
          - eval: "${1.0 if any(g.lower() in _.lower() for g in greetings) else 0.0}"

  error_handling:
    tests:
      handles_empty_input:
        data:
          input: ""
        do:
          - call: llm
            params:
              prompt: "Respond to: ${input}"
            catch: ["*"]
          - eval: "${0.0 if _ is None else 1.0}"
```

Run with:
```bash
llm-workers-evaluate my-script.yaml evaluation-suite.yaml
```

## Best Practices

1. **Use meaningful test names** that describe what's being tested
2. **Set appropriate iteration counts** - more iterations give more stable scores but take longer
3. **Use data merging** to avoid repetition across tests
4. **Handle edge cases** with `catch` to prevent test failures from exceptions
5. **Use deterministic tests** where possible for consistent results
6. **Separate concerns** into different suites for better organization
7. **Start simple** with basic eval statements before adding LLM-based evaluation
