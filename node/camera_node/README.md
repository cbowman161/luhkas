# Camera Node Commands

Reusable camera package for any LUHKAS edge node with a camera service.

This folder is intentionally independent from Scout movement, wheels, pan/tilt,
and light behavior. Copy `camera_node/` into a new camera node and wire
`camera_node.commands.handle()` into that node's local command router.

`camera_node` owns camera capture/media, video feed, vision inference/detections,
face detection/recognition/learning, unknown-face vault observations, and guard
behavior. The current Scout vision HTTP service still hosts those live endpoints
while the reusable command and deterministic routing layer is split into this
package.

## Required Camera API

The local camera service should expose:

- `POST /snapshot` -> JSON with `{"ok": true, "path": "..."}`
- `POST /clip` with `{"seconds": 8.0}` -> JSON with `{"ok": true, "path": "..."}`
- `POST /guard` with `{"enabled": true|false}` -> JSON with `{"ok": true}`

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `CAMERA_SERVICE_URL` | `http://127.0.0.1:5000` | Local camera HTTP service |
| `CAMERA_NODE_ID` | `camera` | Capability owner/target node id |
| `CAMERA_COMMAND_SCOPE` | `node_local` | Capability scope label |
| `CAMERA_COMMAND_DISPATCH_TYPE` | `local_media` | Deterministic router type |
| `CAMERA_CLIP_SECONDS` | `8.0` | Default clip duration |
| `CAMERA_SNAPSHOT_TIMEOUT` | `5.0` | Snapshot request timeout |
| `CAMERA_CLIP_TIMEOUT` | `12.0` | Clip request timeout |

## Usage

```python
from camera_node.commands import capabilities, handle

response = handle("take a picture")
if response is not None:
    print(response["message"])

for command in capabilities():
    print(command["action"], command["triggers"])
```

## Deterministic Router Mappings

Camera media and guard command phrases are defined in
`camera_node/deterministic_mappings.json`.

Any LUHKAS runtime can compile package-level pre-vault routes by discovering:

```text
*_node/deterministic_mappings.json
```

Each file is a JSON object where keys are user phrases and values contain at
least a dispatch `type`:

```json
{
  "take a picture": {"type": "local_media"},
  "record a video": {"type": "local_media"}
}
```

The LUHKAS deterministic router loads every matching package mapping under the
runtime root, then overlays its writable learned cache at
`luhkas_node/data/deterministic_commands.json` or the path in
`LUHKAS_DETERMINISTIC_CACHE`.
