# code_monkey async service

Run a bounded, persistent coder-agent service without changing the larger LUHKAS-BRAIN loop.

## Start manually

```bash
python3 -m code_monkey service --host 127.0.0.1 --port 8765 --workers 2
```

`--workers` is the hard limit for concurrently active coder tasks. Extra tasks remain queued in SQLite instead of all running at once.

## Install as a user systemd service

From the LUHKAS-BRAIN project root:

```bash
python3 -m code_monkey.install_service --workers 2
```

This writes `~/.config/systemd/user/code-monkey.service`, enables it, and starts it. The unit uses `Restart=always`, so it restarts after crashes and on boot/login. The installer also attempts `loginctl enable-linger $USER` so the user service can start at boot before login on systems that allow it.

Check it with:

```bash
systemctl --user status code-monkey.service
journalctl --user -u code-monkey.service -f
```

## Restart recovery

The service stores tasks in `code_monkey_data/code_monkey.sqlite3`. On startup it re-queues tasks interrupted in these states:

```text
claimed, planning, building, built, testing, repairing, running
```

Existing task workspaces are reused. If `work_order.json` already exists, the build resumes from that durable plan and proceeds through generation/verification again as needed.

## HTTP API

Submit a task:

```bash
curl -s -X POST http://127.0.0.1:8765/tasks \
  -H 'content-type: application/json' \
  -d '{"goal":"Build me a reminder system"}'
```

Poll status:

```bash
curl -s http://127.0.0.1:8765/tasks/TASK_ID
```

Read events:

```bash
curl -s http://127.0.0.1:8765/tasks/TASK_ID/events
```

Read session/blackboard:

```bash
curl -s http://127.0.0.1:8765/tasks/TASK_ID/session
```

Queue an existing non-final task:

```bash
curl -s -X POST http://127.0.0.1:8765/tasks/TASK_ID/enqueue
```

Health:

```bash
curl -s http://127.0.0.1:8765/health
```
