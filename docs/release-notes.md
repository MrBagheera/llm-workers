---
layout: default
title: Release notes
nav_order: 90
---

# Release notes

## [1.0.0-rc4](https://github.com/MrBagheera/llm-workers/milestone/21?closed=1) (December 5, 2025)

UI polish and configuration improvements:
- Added `${env.}` substitution in environment descriptions
- Miscellaneous UI fixes

## [1.0.0-rc3](https://github.com/MrBagheera/llm-workers/milestone/20?closed=1) (December 5, 2025)

Tool functionality improvements and model updates:
- Fixed tool registration issues in `build_llm_tool`
- Fixed broken nested tool invocation UI hints
- Upgraded to Claude Sonnet 4.5 as default model

## [1.0.0-rc2](https://github.com/MrBagheera/llm-workers/milestone/19?closed=1) (December 3, 2025)

UI enhancements for streaming and tool parameters:
- Fixed markdown streaming functionality
- Added selective display of tool call arguments in MCP tool UI hints

## [1.0.0-rc1](https://github.com/MrBagheera/llm-workers/milestone/17?closed=1) (December 2, 2025)

Major infrastructure improvements and MCP integration:
- Replaced callbacks with notifications via astream
- Added basic MCP (Model Context Protocol) support
- Improved environment variable configuration
- Fixed LLM tool to return plain output
- Added `/new` command support

## [0.1.0-rc13](https://github.com/MrBagheera/llm-workers/milestone/16?closed=1) (October 4, 2025)

Comprehensive UX improvements:
- Added "Thinking..." prompt during LLM requests
- Improved commands support
- Enhanced token usage and cost reporting
- Moved UX-related switches to user config
- Changed confirmation to normal line input
- Improved handling of missing values from .env files
- Fixed reasoning display for OpenAI models
- Added @-triggered filename completion
- Passed token usage from LLM-backed tools
- Reduced colorfulness of chat interface

## [0.1.0-rc12](https://github.com/MrBagheera/llm-workers/milestone/15?closed=1) (September 20, 2025)

Chat interface and system enhancements:
- Added ability to switch models in chat
- Kept run Python scripts for audit purposes
- Improved caching
- Added `/export <name>` command

## [0.1.0-rc11](https://github.com/MrBagheera/llm-workers/milestone/14?closed=1) (September 11, 2025)

Model management improvements:
- Simplified model management functionality

## [0.1.0-rc10](https://github.com/MrBagheera/llm-workers/milestone/13?closed=1) (August 20, 2025)

Configuration flexibility enhancements:
- Added support for resolving using dynamic keys
- Implemented inline tool definition
- Added support for inline tool and model configuration

## [0.1.0-rc9](https://github.com/MrBagheera/llm-workers/milestone/12?closed=1) (July 1, 2025)

Tool result handling improvements:
- Added LLM tool support to filter results to JSON only
- Improved handling of tools returning non-string results

## [0.1.0-rc8](https://github.com/MrBagheera/llm-workers/milestone/11?closed=1) (July 1, 2025)

Chat output enhancements:
- Added experimental markdown output support to chat functionality

## [0.1.0-rc7](https://github.com/MrBagheera/llm-workers/milestone/10?closed=1) (June 19, 2025)

LLM tool customization:
- Added ability to specify system prompt as LLM tool parameter

## [0.1.0-rc6](https://github.com/MrBagheera/llm-workers/milestone/9?closed=1) (June 19, 2025)

Template and script organization:
- Added support for references to nested elements in templates
- Introduced "shared" section in LLM scripts

## [0.1.0-rc5](https://github.com/MrBagheera/llm-workers/milestone/8?closed=1) (June 11, 2025)

Bug fixes:
- Fixed RunPythonScriptTool to use configured Python executable instead of system default

## [0.1.0-rc4](https://github.com/MrBagheera/llm-workers/milestone/6?closed=1) (June 11, 2025)

Type system and UI enhancements:
- Added support for missing intermediate tool UI hints
- Added support for generic lists and dictionaries as parameter types

## [0.1.0-rc3](https://github.com/MrBagheera/llm-workers/milestone/5?closed=1) (June 11, 2025)

User interface and interaction improvements:
- Added support for hiding UI hints
- Implemented single-key confirmations
- Added support for showing nested tool executions
- Introduced approval tools

## [0.1.0-rc2](https://github.com/MrBagheera/llm-workers/milestone/4?closed=1) (June 10, 2025)

User experience enhancements:
- Added User Input tool
- Added optional welcome banner for users

## [0.1.0-rc1](https://github.com/MrBagheera/llm-workers/milestone/3?closed=1) (May 30, 2025)

Stability and robustness improvements:
- Improved error handling in main chat session loop
- Completed initial documentation
- Reconsidered and improved logging
- Enhanced tool error handling
- Fixed caching issues
- Added throttling configuration for AWS Bedrock models
- Added support for literal types in custom tools

## [0.1.0-alpha5](https://github.com/MrBagheera/llm-workers/milestone/1?closed=1) (March 26, 2025)

Initial alpha release with core features:
- Added Claude 3.7 thinking mode support via AWS Bedrock
- Implemented loading YAML files from resources
- Added environment-specific configuration
- Simplified tool run confirmations
- Improved handling of `return_direct` tools
- Added auto-open for new/updated files in current working directory
