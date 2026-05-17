# Legacy Code

This folder holds the old in-process coder stack that was replaced by the
standalone Code Monkey service.

Moved here:

- `task_manager.py`
- `agents/coder_agent.py`
- `agents/analyst_agent.py`
- `action_policy.py`
- `validator.py`
- `task_logger.py`

These files are kept for reference only. The active brain runtime should route
coding work through `code_monkey_client.py`, which talks to the Code Monkey HTTP
service. Do not import these legacy modules from active runtime code.
