# Project Overview

## Introduction

As of early 2025, even the most advanced publicly available LLM tools remain limited in their ability to conduct effective, autonomous internet research. Despite ambitious marketing claims, their functionality is often constrained to running basic web searches and synthesizing results, lacking deeper reasoning or operational autonomy.

This limitation stems from the absence of true **agentic capabilities**—the ability to recursively break down complex tasks into smaller, manageable subtasks and execute them in a structured sequence. While advancements in 2025-2026 may address this, either through emerging model capabilities or targeted training, the gap persists for now.

In the interim, research workflows rely on a "hybrid mode" where humans handle:
- Planning
- Task decomposition
- Tool configuration

The LLM, in turn, executes these instructions under human supervision. 

This project aims to streamline such workflows by simplifying the interaction between humans and LLMs for research tasks.


## Goals

This project is designed to:
- **Facilitate LLM-backed research** by providing a structured framework for human-guided assistance.
- **Enable debugging and flexibility** in the research process, including restarting workflows from specific checkpoints.
- **Maintain an audit trail** to document and evaluate the quality and reliability of research outputs.

Although this project may have a short lifespan as LLMs evolve toward true agentic capabilities, it seeks to provide valuable insights into designing systems that bridge current gaps.


## What This Project Is *Not*

- **Not an end-user tool**: This project is geared toward developers and researchers with knowledge of Python, LLM capabilities, and programming fundamentals.
- **Not a complete automation system**: It relies on human oversight and guidance for optimal performance.


# Running 

Envisioned workflow:
- Define what your research is, what information sources do you need
- Split this into the tasks LLM can do
- Define the tools needed for LLMs to do these tasks
- Debug the process using stub tool implementations
- Configure real tools
- Follow-up first few iterations of actual research execution to verify it is producing the expected results
- Run actual research until it finishes (which may take considerable amount of time)


# To Do

- [x] Running main prompt
- [x] General-purpose `debug` tool
- [x] Specifying LLM provider and model (ollama/open-ai/claude)
- [x] Specifying custom tools
- [x] Write introduction for README.md
- [ ] `LLM` tool
- [ ] `fetch_url_text` tool
- [ ] `search` tools
- [ ] Improve execution presentation
- [ ] Add support for interactive mode
- [ ] Add audit trail