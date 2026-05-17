# Command / Tool Profile

Preferred commands:
- python3
- uname
- lsb_release
- nvidia-smi if available
- ollama if available
- sensors if available, but never assume installed
- cat/read files only after checking existence

Avoid:
- Windows commands
- sudo unless explicitly approved
- destructive commands
- hardcoded sensor paths without fallbacks