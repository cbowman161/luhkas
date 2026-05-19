# Luhkas Code Stacks

## Current Stack

The active stack lives in this repository.

```text
luhkas/
  vault/   Brain/runtime code for luhkas-vault
  node/    Deployable node runtime copied to luhkas-scout and future nodes
```

`vault/` owns chat, model routing, memory, node registration, guard alert
routing, Code Monkey orchestration, and the public brain HTTP API on port 7000.

`node/` owns the Scout-side services:

- `services/vision_service.py` on port 5000
- `services/robot_api.py` on port 5001
- `services/presence_client_service.py` on port 5002

Node code is modularized into reusable packages:

- `camera_node`
- `pantilt_node`
- `rover_node`
- `light_node`
- `luhkas_node`

The vault copy of `node/` is the source of truth. Scout runs a synchronized copy
at `~/luhkas/node`.

## Deployment And Sync

New nodes are bootstrapped with:

```bash
NODE_ID=scout bash scripts/bootstrap_node.sh
```

The bootstrap script clones this repo, writes `node/.env`, renders the systemd
unit templates, and enables user services.

The vault can push node code through `vault/sync_manager.py`. It reads node
profiles from `node/profiles/*.json`, rsyncs `node/` to the target host, and
restarts configured services only when files changed. The Scout profile is:

```text
node/profiles/scout.json
```

The node presence proxy registers itself with the vault at startup by calling
`POST /node/register`. Registration records the node IP, service ports,
capabilities, and module self-test data.

## Legacy Stack

The older runtime layout may still exist on mounted machines:

```text
/Volumes/luhkas-vault/vault_v2
/Volumes/luhkas-scout/scout_runtime
```

Treat those as legacy/reference copies unless a service is explicitly still
pointing at them. New development should land in this repo under `vault/` and
`node/`.

When moving behavior from the legacy stack, preserve the same service boundary:
the vault owns reasoning and memory; nodes own real-time sensing and physical
control.

## Dependency Files

Install vault dependencies with:

```bash
python3 -m pip install -r vault/requirements.txt
```

Install node dependencies with:

```bash
python3 -m pip install -r node/requirements.txt
```

Scout still requires the separate Hailo environment at `~/hailo-apps`; the node
start scripts activate that environment before launching vision and robot API
services.
