# llm-workers-tools

Command-line tools for LLM Workers.

## Overview

`llm-workers-tools` provides command-line entry points for working with LLM scripts:

- **llm-workers-cli**: Batch processing tool for running LLM scripts with command-line or stdin input
- **llm-workers-chat**: Interactive chat interface for conversational LLM interactions

## Installation

```bash
pip install llm-workers-tools
```

This will install both `llm-workers` (core) and `llm-workers-console` (UI) as dependencies.

## Usage

### Batch CLI Tool

Process prompts in batch mode:

```bash
# Single prompt
llm-workers-cli my-script.yaml "What is the capital of France?"

# Multiple prompts
llm-workers-cli my-script.yaml "prompt 1" "prompt 2" "prompt 3"

# Read from stdin
echo "prompt from stdin" | llm-workers-cli my-script.yaml -

# Enable verbose output
llm-workers-cli --verbose my-script.yaml "prompt"

# Enable debug mode
llm-workers-cli --debug my-script.yaml "prompt"
```

### Interactive Chat

Start an interactive chat session:

```bash
# With specific script
llm-workers-chat my-script.yaml

# With default generic assistant
llm-workers-chat

# Resume last session
llm-workers-chat --resume

# With verbose logging
llm-workers-chat --verbose my-script.yaml
```

### Chat Commands

In chat mode, use these commands:

- `/help` - Show available commands
- `/model <name>` - Switch to different model
- `/rewind [N]` - Rewind to earlier message
- `/display [setting] [value]` - Show/modify display settings
- `/cost` - Show token usage and costs
- `/export <name>` - Export chat as markdown
- `/save <name>` - Save session to YAML
- `/clear` - Reset chat session
- `/exit` - Exit (auto-saves to `.last.chat.yaml`)

## Documentation

Full documentation: https://mrbagheera.github.io/llm-workers/

## License

See main repository for license information.
