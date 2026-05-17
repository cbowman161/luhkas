# Brain Runtime Architecture

## Purpose

`vault_v2` is the orchestrator for the larger robot system. It owns the user-facing
runtime loop, session state, routing decisions, main event log, system capabilities,
and high-level coordination.

It is not the rover body controller and it is not Code Monkey.

## Runtime Ownership

### Main brain runtime

Owns:

- CLI interaction loop in `main.py`
- request interpretation and pending decisions
- main blackboard/session state
- main SQLite state in `vault_v2.db`
- system and learned capability registries
- local skill registry and skill execution
- high-level routing in `router.py`
- main-brain notifications and event log

### Code Monkey

Owns:

- its HTTP service
- its queue and worker pool
- its task database under `code_monkey_data/`
- its blackboard/session records
- build workspaces
- generated files, verification, repair loops, and lessons

The main brain talks to Code Monkey through `code_monkey_client.py` only. The main
brain must not import Code Monkey internals, write to Code Monkey storage, mark Code
Monkey notifications as read, or mutate Code Monkey blackboard/session state.
When submitting work, the main brain sends only a goal through the HTTP API.

The old in-process coder loop lives under `legacy/` for reference only. Active
runtime code should not import it.

### Rover/body layer

The rover is an external body/perception service. It owns camera capture, Hailo
inference, tracking memory, face-recognition cache, and motor/serial APIs. The
vault PC owns user chat, LLM inference, canonical person memory, persistent
unknown-face groups, face reference storage, vector memory, and GPU video
analysis.

The brain talks to the rover through narrow HTTP APIs:

- `GET rover:5000/meta`
- `GET rover:5000/snapshot`
- `POST rover:5000/clip`
- `POST rover:5000/learn_face?name=<identity>&face_id=<visible-face-id>`
- `POST rover:5000/tracking`
- `POST rover:5000/settings`
- `POST rover:5000/guard`
- `GET rover:5001/health`
- `POST rover:5001/pantilt`
- `POST rover:5001/move`

Do not add a second chat/LLM loop on the rover. CLI chat should call the brain
service, and the brain service should call rover APIs as needed.

The existing `ScoutVaultBridge` is the scout tool contract. It should use
`/meta.identity_prompt` or `/meta.identity_prompt_queue` to choose the current
face id for introductions, then call `learn_face` with that explicit `face_id`.
It also owns brain-side orchestration for tracking/follow/search-camera/guard
toggles, camera light settings, current-state explanations, snapshot capture,
clip capture, vision analysis, and person memory. Avoid creating another rover
chat bridge. `GET /scout/tools` reports the current bridge contract and rover
reachability.

## Request Flow

1. `main.py` reads user input.
2. `InteractionInterpreter` resolves pending choices when needed.
3. `Planner` classifies new requests using available capabilities and skills.
4. `Router` dispatches to one of:
   - main system capabilities
   - chat
   - skill confirmation/execution
   - capability proposals
   - Code Monkey via HTTP
5. Main-brain updates are stored in the main event log.
6. Code Monkey updates are queried from Code Monkey's HTTP API and displayed without
   mutating Code Monkey state.

## Service Boundary

Code Monkey is expected to run as a standalone local service:

```bash
python3 -m code_monkey service --host 127.0.0.1 --port 8765 --workers 2
```

The brain runtime uses `CODE_MONKEY_URL` from `config.py`, defaulting to:

```text
http://127.0.0.1:8765
```

Keep Code Monkey bound to localhost unless authentication and network policy are added.

The brain runtime also has a non-interactive HTTP service wrapper:

```bash
python3 vault_service.py --host 0.0.0.0 --port 7000
```

Install it as a user systemd service:

```bash
python3 install_vault_service.py
```

Useful service checks:

```bash
systemctl --user status code-monkey.service
systemctl --user status vault-runtime.service
curl -s http://127.0.0.1:7000/health
curl -s -X POST http://127.0.0.1:7000/presence/message \
  -H 'content-type: application/json' \
  -d '{"message":"Hi, I am Chris."}'
```

`POST /presence/message` is the unified Luhkas presence endpoint for every edge surface.
Rover CLI, wall-mounted camera/microphone/speaker nodes, brain-local terminals,
and future clients should all send user text or transcripts there. Edges may
include a `source` label for diagnostics, but they must not run separate chat,
personality, memory, or routing loops.

The original general runtime planner is still available for internal/dev use at
`POST /runtime/message`.

Rover chat endpoints exposed by the brain service:

- `GET /capabilities`
- `GET /session`
- `GET /presence/session`
- `GET /scout/state`
- `GET /scout/tools`
- `GET /whoami`
- `GET /identity`
- `POST /identity`
- `GET /debug/identity`
- `POST /presence/message`

The brain's self-identity is stored in:

```text
data/identity/profile.json
```

That profile defines the assistant name, creator, role, body, personality, and
behavioral boundaries. Prompts should read from this profile instead of
hardcoding self-identity.

## Model Roles

Brain model selection is centralized in `config.py` and `models.py`. Runtime code
should request a role with `get_model("<role>")` instead of hardcoding model
names.

Default role mapping for the RTX 3090 vault PC:

- `router`: `qwen2.5:3b-instruct`
- `chat`: `qwen3:8b`
- `reasoner`: `qwen3:30b`
- `planner`: `qwen3:30b`
- `analyst`: `qwen3:30b`
- `coder`: `qwen3-coder:30b`
- `fast_coder`: `qwen2.5-coder:14b`
- `vision`: `qwen2.5vl:7b`
- `embed`: `bge-m3`

Override with environment variables:

```bash
export VAULT_ROUTER_MODEL=qwen2.5:3b-instruct
export VAULT_CHAT_MODEL=qwen3:8b
export VAULT_REASONER_MODEL=qwen3:30b
export VAULT_CODER_MODEL=qwen3-coder:30b
export VAULT_FAST_CODER_MODEL=qwen2.5-coder:14b
export VAULT_VISION_MODEL=qwen2.5vl:7b
export VAULT_EMBED_MODEL=bge-m3
export VAULT_IMMEDIATE_KEEP_ALIVE=24h
export VAULT_BACKGROUND_KEEP_ALIVE=5m
export VAULT_WARM_MODEL_ROLES=router,chat,vision
```

Immediate models use Ollama `keep_alive` so chat-facing models stay loaded after
startup. Background Code Monkey models use the shorter background keep-alive
because they run asynchronously and should not evict interactive models unless
they are actively working.

Brain memory endpoints used by rover sync:

- `GET /faces/sync`
- `GET /faces/unknown`
- `POST /faces/unknown`
- `POST /faces/unknown/promote`
- `POST /people/<identity>/faces`
- `GET /people/<identity>/summary`
- `GET /people/<identity>/memory`
- `POST /people/<identity>/remember`
- `POST /people/<identity>/preference`

Camera nodes should send unknown face observations to `POST /faces/unknown`
instead of owning long-lived unknown groups locally. The vault groups samples by
face fingerprint across all camera nodes, using node/source track only as a
continuity hint when visual matching is unavailable. Groups are stored under
`data/unknown_face_groups/<group-id>/`. When a node provides a name, it calls
`POST /faces/unknown/promote`; the vault copies all samples for that group into
the named identity's face references so every camera node can receive them on the
next `GET /faces/sync`.

Brain GPU vision endpoint:

- `POST /vision/analyze`

## Project Direction

The long-term system is a distributed robotics brain:

- GPU vault PC: planning, memory, reasoning, coding, reflection, orchestration
- Code Monkey: standalone software/capability builder service
- Robot edge clients: perception, safety, local control, telemetry
- Main brain runtime: the coordinator that decides which service or capability should
  handle a request

Each boundary should stay explicit. New robot capabilities should be added as service
clients with safety gates, not as direct motor/sensor code inside the router.
