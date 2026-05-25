# Luhkas Project — Code Stacks

Luhkas is a distributed robotics brain. The **vault** owns reasoning, memory,
identity, and orchestration; **nodes** (edge devices like Scout) own sensing
and actuation. Everything runs locally — no cloud dependency.

```text
luhkas/
  vault/    Brain runtime for luhkas-vault PC (chat, routing, memory, capabilities, code monkey)
  node/     Deployable node runtime, copied to luhkas-scout and future nodes
  scripts/  Bootstrap and admin scripts
  tests/    Repo-level test harnesses
```

---

## Where to read next

| Topic | Doc |
|---|---|
| Full vault feature reference (routing, memory, identity, capabilities, scout integration, code monkey, translate, etc.) | [`vault/VAULT_RUNTIME.md`](vault/VAULT_RUNTIME.md) |
| Node-side service map (vision, robot API, presence proxy) | [`node/DOCUMENTATION.md`](node/DOCUMENTATION.md) |
| Camera node specifics | [`node/camera_node/README.md`](node/camera_node/README.md) |
| Node config layout | [`node/config/README.md`](node/config/README.md) |
| Legacy stack notes | [`vault/legacy/README.md`](vault/legacy/README.md) |

---

## Service ownership

| Owner | Hardware | What it runs |
|---|---|---|
| Vault PC (`luhkas-vault`) | RTX 3090, 96GB DDR5 | `vault-runtime.service` (port 7000), `code-monkey.service` (port 8765 localhost), `vault-autosync.timer`, Ollama, LanceDB |
| Scout (`luhkas-scout`) | Raspberry Pi 5, 16GB RAM, Hailo HAT+ | `scout-vision.service` (port 5000), `scout-robot-api.service` (port 5001), `scout-presence.service` (port 5002) |

Edge devices send user input to vault's `POST /presence/message` (or `/ui` for
the UI client). Vault decides everything: route, memory writes, identity
adoption, capability dispatch, recipe generation, response composition.
Nodes are interaction surfaces, not separate personalities.

---

## Node modules

The `node/` runtime is modularized into reusable Python packages, any subset
of which can run on a given node:

| Module | Purpose |
|---|---|
| `camera_node` | Camera capture + Hailo inference + tracking memory |
| `pantilt_node` | Pan-tilt servo control |
| `rover_node` | Drive/steering/wheel control via serial |
| `light_node` | Camera ring-light control |
| `luhkas_node` | Presence-message proxy that forwards user input to vault |

Each module ships its own systemd unit templates, command definitions, and
self-test. The presence service collects per-module self-tests at startup and
sends them to vault on `POST /node/register` + `POST /node/selftest`.

---

## Models (Ollama, vault-side)

| Role | Default | Hot? |
|---|---|---|
| `router` | `qwen2.5:3b-instruct` | yes (keep-alive 24h) |
| `chat` | `qwen3:8b` | yes (keep-alive 24h) |
| `vision` | `qwen2.5vl:7b` | yes (keep-alive 24h) |
| `coder` | `qwen3-coder:30b` | no (background) |
| `fast_coder` | `qwen2.5-coder:14b` | no (background) |
| `planner` / `reasoner` / `analyst` | `qwen3:30b` | no (background) |
| `embed` | `bge-m3` | always-loaded |

Configure via env vars (`VAULT_ROUTER_MODEL`, `VAULT_CHAT_MODEL`, etc.) or via
the `MODEL_ROLES` map in `vault/models.py`. Runtime code requests by role
(`get_model("chat")`) — never hardcoded model names.

---

## Deployment + sync

Bootstrap a new node:

```bash
NODE_ID=scout bash scripts/bootstrap_node.sh
```

The script clones the repo on the node, writes `node/.env`, renders systemd
unit templates, and enables user services.

Vault pushes node code via `vault/sync_manager.py`:

```bash
python3 vault/sync_manager.py        # push to all profiles
python3 vault/sync_manager.py scout  # push to one
```

It reads `node/profiles/<id>.json` (host, user, services to restart, rsync
excludes), rsyncs `node/`, and restarts services on the node only when files
changed. Vault has key-based SSH to nodes via `~/.ssh/id_ed25519_nodes`.

`vault-autosync.timer` runs the push periodically; `vault-runtime.service`'s
`ExecStartPost` also triggers a one-shot push so a vault restart propagates
fresh code to nodes.

---

## Dependency install

```bash
python3 -m pip install -r vault/requirements.txt
python3 -m pip install -r node/requirements.txt
```

Scout additionally needs the Hailo environment at `~/hailo-apps`; the node
service units activate that environment before launching the vision service.

---

## Legacy stack

The pre-repo runtime layout may still exist on mounted machines:

```text
/Volumes/luhkas-vault/vault_v2
/Volumes/luhkas-scout/scout_runtime
```

Treat as reference only. New development lands in this repo. The `vault_v2.db`
SQLite file under `vault/` is current state (events, notifications, jobs) — the
filename is legacy but the data is live.
