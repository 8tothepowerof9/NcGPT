## app Directory Structure & Module Responsibilities

| Directory    | Responsibility                                                      |
| ------------ | ------------------------------------------------------------------- |
| `core/`      | Configuration and environment setup                                 |
| `models/`    | Provider/model definitions and configuration types                  |
| `llm/`       | DSPy LLM provider wiring and configuration                          |
| `graphs/`    | LangGraph graph orchestration (state, control flow, node functions) |
| `programs/`  | DSPy modules for prompt engineering and LLM calls                   |
| `services/`  | I/O adapters for DB, HTTP, file, vector stores                      |
| `tools/`     | Pure tool functions for agent/tool calling                          |
| `utils/`     | Utility functions for embedding, text, helpers                      |
| `pipelines/` | Data processing pipelines (chunking, indexing, etc.)                |
| `runtime/`   | Execution and runtime management                                    |

---

### Module Details

- **core/**  
  Configuration logic for providers, API keys, model names, and environment validation.

- **models/**  
  Enums and dataclasses for LLM and embedding providers, model configuration, and related types.

- **llm/**  
  Centralizes DSPy configuration for LLMs, allowing provider swapping without changing graph/program logic.

- **graphs/**  
  Implements state management, control flow, and node functions for LangGraph-based multi-agent interactions.

- **programs/**  
  Contains pure DSPy modules, each with its own signature, module logic, and optional optimizer configuration.

- **services/**  
  Handles external services such as database, HTTP, file, and vector store interactions. Keeps DSPy programs deterministic and separates side effects from graph logic.

- **tools/**  
  Contains pure, side-effect-free functions designed for use as callable tools in agent workflows.

- **utils/**  
  Provides reusable utilities for embedding, text processing, and helper functions.

- **pipelines/**  
  Implements chunking, indexing, and other document/data processing workflows.

- **runtime/**  
  Handles runtime orchestration and management for application processes.

---

This structure ensures each module has a clear, focused responsibility, supporting maintainability and extensibility for DSPy prompt engineering and LangGraph multi-agent workflows.
