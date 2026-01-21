# llm-workers-evaluation

Evaluation framework for LLM Workers.

## Overview

`llm-workers-evaluation` provides tools for running evaluation suites against LLM scripts and reporting scores.

- **llm-workers-evaluate**: CLI tool for running evaluation suites

## Installation

```bash
pip install llm-workers-evaluation
```

This will install `llm-workers` (core) as a dependency.

## Usage

### Running Evaluations

```bash
# Basic usage
llm-workers-evaluate my-script.yaml my-suite.yaml

# With custom iteration count
llm-workers-evaluate -n 5 my-script.yaml my-suite.yaml

# With verbose output
llm-workers-evaluate --verbose my-script.yaml my-suite.yaml

# With debug mode
llm-workers-evaluate --debug my-script.yaml my-suite.yaml
```

### Evaluation Suite Format

Evaluation suites are YAML files defining tests that return scores between 0.0 and 1.0:

```yaml
shared:
  data:
    expected: "hello"
  tools: []

iterations: 10

suites:
  basic:
    data: {}
    tools: []
    tests:
      always-pass:
        do:
          eval: 1.0
      always-fail:
        do:
          eval: 0.0
      conditional:
        data:
          value: "hello"
        do:
          eval: "${1.0 if value == expected else 0.0}"
```

### Output Format

Results are output as YAML:

```yaml
final_score: 0.75
per_suite:
  basic:
    final_score: 0.75
    per_test:
      always-pass: 1.0
      always-fail: 0.0
      conditional: 1.0
```

### Score Handling

- Tests must return a float between 0.0 and 1.0
- `None` results are treated as 0.0
- Non-numeric results are treated as 0.0
- Scores below 0.0 are clamped to 0.0
- Scores above 1.0 are clamped to 1.0
- Exceptions during test execution result in 0.0 for that iteration

### Data and Tool Merging

Data and tools are merged in order: `shared` -> `suite` -> `test`

- For data: later values override earlier ones
- For tools: lists are concatenated

## Documentation

Full documentation: https://mrbagheera.github.io/llm-workers/

## License

See main repository for license information.
