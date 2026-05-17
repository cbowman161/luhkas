# OS / Software Profile

Primary environment: Ubuntu/Linux.

Shell commands should be POSIX/Linux compatible.
Use python3, not python.
Do not emit Windows cmd.exe, PowerShell, backslash paths, or drive letters unless requested.

Local LLM runtime:
- Ollama expected on localhost:11434
- Router model: qwen2.5:3b-instruct (intent classification, runs on every user turn)
- Chat model: qwen3:8b
- Planner/analyst/reasoner model: qwen3:30b
- Coder model: qwen3-coder:30b
- Fast coder model: qwen2.5-coder:14b
- Vision-language model: qwen2.5vl:7b
- Embedding model: bge-m3
- Immediate model keep-alive: 24h (router, chat, vision)
- Background model keep-alive: 5m (reasoner, planner, analyst, coder)

Python:
- Prefer standard library.
- Do not require pip installs unless the executor asks for dependency approval.
