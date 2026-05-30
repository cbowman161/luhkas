# Luhkas Node Runtime Documentation

> **Where the code lives now.** The canonical source is `node/` in the Luhkas
> monorepo. On a node host (e.g. `luhkas-scout`), the deployed copy is rsynced
> from vault to `~/luhkas/node/` by `vault/sync_manager.py` (triggered on
> vault restart and on the `vault-autosync.timer` cadence). The older
> `~/scout_runtime` directory is legacy/reference only.

This document covers the node-side runtime: camera capture, Hailo inference,
tracking, behavior, manual control, robot serial control, telemetry, local
memory caches, and node-side HTTP APIs. The vendor Hailo install
(`~/hailo-apps`) stays clean; the node runtime activates it at service start.

For the **vault side** (chat, routing, identity, memory, learned capabilities,
Code Monkey, translation, etc.), see [`vault/VAULT_RUNTIME.md`](../vault/VAULT_RUNTIME.md).

For the project-wide stack overview, see [`STACKS.md`](../STACKS.md).

## Hardware And Network

| Component | Detail |
|-----------|--------|
| Scout node | Raspberry Pi 5 |
| AI accelerator | Hailo-10H / Hailo-8L compatible runtime |
| Camera | USB/V4L2 or configured camera index |
| Chassis comms | UART serial JSON on `/dev/ttyAMA0` at 115200 baud |
| Vault / brain | `luhkas-vault.local` / `10.10.1.1`, port 7000 |

Network topology:

```text
Internet
    |
[ Vault PC ] -- BowmanFamily LAN
           \-- luhkas-link AP (10.10.1.1/24)
                         |
                      [ Rover ]
```

The vault PC owns chat, LLM inference, vector/person memory, canonical face
references, and richer GPU vision analysis. The rover owns real-time sensing,
tracking, face recognition cache, manual control, motion APIs, and safety
behavior.

## Services

| Service | Port | Source | Role |
|---------|------|--------|------|
| `scout-robot-api` | 5001 | `services/robot_api.py` | HTTP to serial bridge, telemetry, OLED, heartbeat watchdog |
| `scout-vision` | 5000 | `services/vision_service.py` | Camera, inference, tracking, behavior, manual controller |
| `scout-presence` | 5002 | `services/presence_client_service.py` | Edge proxy to the vault brain |
| `scout-battery` | 5003 | `battery_node/service.py` | Canonical battery state (UART proxy backend on scout; INA219 on UPS-HAT nodes) |
| `scout-controller` | none | `tools/controller_drive.py` | Legacy/manual-start gamepad client; normal controller support now lives in `scout-vision` |

All services run from `~/scout_runtime` through scripts in `scripts/`, which
activate the Hailo environment from `~/hailo-apps/setup_env.sh`.

Useful service commands:

```bash
cd ~/scout_runtime
./scripts/install_user_services.sh
systemctl --user start scout-robot-api.service
systemctl --user start scout-vision.service
systemctl --user start scout-presence.service
systemctl --user restart scout-robot-api.service
systemctl --user restart scout-vision.service
journalctl --user -u scout-robot-api.service -f
journalctl --user -u scout-vision.service -f
```

Manual development startup:

```bash
cd ~/scout_runtime
source ~/hailo-apps/setup_env.sh
python3 services/robot_api.py
python3 services/vision_service.py --hef-path /path/to/model.hef --labels /path/to/labels.txt
```

The checked-in and installed `scout-vision.service` currently overrides several
runtime defaults for the live rover: tracking enabled, target label `person`,
score threshold `0.45`, autonomous wheel behavior disabled, pose joint threshold
`0.3`, slower search-camera timing, tuned absolute pan/tilt gains,
`SCOUT_TARGET_LOST_GRACE_SECONDS=1.5`, and
`SCOUT_BYTETRACKER_MATCH_THRESH=0.75`.

## Robot API

`robot_api` is the HTTP to UART bridge. It translates rover HTTP commands into
serial JSON packets, reads chassis telemetry, writes OLED messages, logs
telemetry to SQLite, and stops the wheels when the brain heartbeat is lost.

Threads:

| Thread | Role |
|--------|------|
| `serial_reader` | Reads UART, parses `T:1001`, publishes battery voltage to `/run/luhkas/battery_raw.json` for `battery_node`, updates motor/IMU telemetry, logs telemetry |
| `serial_worker` | Flushes pan-tilt, light, and move state every 30 ms |
| `oled_updater` | Alternates IP / `battery_node` percentage and accepts `/oled` overrides |
| `heartbeat_watchdog` | Polls brain `/health`; stops wheels after 5 seconds unreachable |
| HTTP server | Serves port 5001 |

Outbound serial commands:

| T code | Command |
|--------|---------|
| `T:1` | Direct wheels: `{"T":1,"L":<left>,"R":<right>}` |
| `T:133` | Pan-tilt absolute: `{"T":133,"X":<pan>,"Y":<tilt>,"SPD":0,"ACC":0}` |
| `T:134` | Pan-tilt relative: `{"T":134,"X":<pan>,"Y":<tilt>,"SX":<speed-x>,"SY":<speed-y>}` |
| `T:135` | Pan-tilt stop |
| `T:132` | Lights: `{"T":132,"IO4":<brightness>,"IO5":<brightness>}` |
| `T:3` | OLED text: `{"T":3,"lineNum":0|1,"Text":"..."}` |

Inbound serial telemetry:

| Field | Meaning |
|-------|---------|
| `T:1001` | Chassis telemetry packet |
| `v` | Battery voltage x100 |
| `L`, `R` | Left/right motor output telemetry |
| `odl`, `odr` | Left/right wheel encoders |
| `ax`, `ay`, `az` | Accelerometer |
| `gx`, `gy`, `gz` | Gyroscope |
| `mx`, `my`, `mz` | Magnetometer |

Robot API endpoints:

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | API/serial status |
| `GET /telemetry` | Latest parsed `T:1001` telemetry |
| `GET /telemetry/history?seconds=60` | Recent SQLite telemetry rows, max 3600 seconds |
| `GET /heartbeat` | Last heartbeat timestamp and timeout |
| `POST /heartbeat` | Updates heartbeat timestamp |
| `POST /pantilt` | Queues pan/tilt/light command |
| `POST /move` | Queues rover movement command |
| `POST /oled` | Queues two OLED text lines |
| `POST /send` | Sends raw serial JSON directly |

`POST /move` keeps the rover-facing API shape:

```json
{"x": 250, "z": 0}
```

`x` is forward/back and `z` is turn. `robot_api` converts that to direct wheel
command `T:1` with differential mixing:

```text
x_norm = clamp(x / 1000.0)
z_norm = clamp(z / 1000.0)
left = clamp(x_norm + z_norm)
right = clamp(x_norm - z_norm)
```

Generated wheel output is capped by `MAX_WHEEL_SPEED=0.25` for smoother manual
control. Positive `z` turns right; negative `z` turns left.

## Vision Service

`vision_service` owns camera capture, object detection, face recognition,
tracking, behavior, the browser UI, manual camera movement, USB gamepad control,
camera-light automation, snapshots/clips, and calls into `robot_api`.

Threads:

| Thread | Role |
|--------|------|
| `run_vision` | Main camera/inference/tracking loop |
| HTTP server | Serves port 5000 |
| `_gamepad_loop` | Reads `/dev/input/js*` directly for USB controller control |
| `_telemetry_poll_loop` | Optional gyro/telemetry polling for ego-motion compensation |
| Guard alert threads | Background alert POSTs |

Per-frame pipeline:

```text
Camera frame
  -> letterbox
  -> Hailo object detection
  -> parse detections
  -> optional OpenCV face detection when a person is present
  -> discard faces whose center is not inside a person bbox
  -> face recognition and auto reference capture
  -> copy face identity onto enclosing person
  -> SimpleTracker object matching and memory association
  -> target selection and prediction
  -> optional pose estimation and pose aim
  -> guard/collision/search/behavior decisions
  -> pan-tilt and optional wheel commands
  -> draw overlay, encode JPEG, update `/meta`
```

Tracking defaults to target label `person` and score threshold `0.45`.
`SCOUT_TARGET_LABEL` can be changed for object-class testing. If a selected
target briefly leaves the frame, the tracker exposes a predicted target for
continuity; predicted targets are not written back as real detections.

Face detections must be inside a `person` detection to be valid. Invalid faces
are filtered before recognition, overlay rendering, and `/meta` output.
When tracking is on and valid unknown faces are visible, the rover builds a
left-to-right introduction queue. The active unknown face becomes the pan/tilt
target, `/meta.identity_prompt` asks "Who are you?", and
`/meta.identity_prompt_queue` reports visible/unknown counts plus the active
face index. After that face is learned with `POST /learn_face`, the queue
advances to the next visible unknown face. The rover can keep a short local
unknown-face cache for continuity, but persistent unknown-face groups are owned
by the vault. Camera nodes upload unknown face observations to
`POST /faces/unknown`; when a person introduces themselves, the vault promotes
that unknown group into the named identity's face references.

CPU-side OpenCV Haar face detection is optional enrichment. It can be disabled
for a run with `--no-face-detection` and pointed at a specific cascade with
`SCOUT_FACE_CASCADE_PATH`. Common cascade locations are:

```text
/usr/share/opencv4/haarcascades/
/usr/share/opencv/haarcascades/
/usr/local/share/opencv4/haarcascades/
```

Behavior states:

| State | Meaning |
|-------|---------|
| `IDLE` | Tracking off and guard off |
| `FOLLOWING` | Tracking on with a visible or predicted target |
| `SEARCHING` | Tracking on without target; optional search camera movement |
| `GUARDING` | Guard on, tracking off |
| `AVOIDING` | Collision block while following; backs up briefly |
| `MANUAL` | USB/manual control owns camera and wheels |

Aim priority is face center, pose head/shoulders, upper body, then bbox center.
Pan/tilt inversion settings have been removed; the service uses one fixed
command convention. Absolute pan/tilt mode is the default and keeps an estimated
camera position so manual and tracking commands share the same frame of
reference.

### Browser UI

Open `http://<rover>:5000/`. The UI has one sidebar with:

- Behavior state, guard mode, guard alert count
- Manual camera controls and pan/tilt step sliders
- USB gamepad connection/manual status
- Tracking, follow, search camera, wheel drive, target identity, score sliders
- Follow tuning
- Camera light and pan-tilt tuning
- Collision avoidance
- Face and pose/vision settings
- Face introduction queue status
- Live detection list, target state, chat, and video feed

Scout-local chat commands are composed by `luhkas_node/local_commands.py` from
installed reusable `*_node` packages. `scout_node` is not part of the production
runtime. Scout currently composes:

- `camera_node`: camera capture/media, vision service ownership, face/guard
  behavior, and camera media commands.
- `pantilt_node`: manual pan/tilt, tracking toggles, target centering, search
  sweep, and edge reacquire behavior.
- `rover_node`: wheels, follow wheel movement, collision blocking for wheels,
  and gamepad/manual physical control ownership.
- `light_node`: generic light on/off, brightness, auto-light, and anti-blink
  probing behavior.

`GET /capabilities` marks every command with `owner_node: scout`,
`target_node: scout`, and `scope: scout_only`; these commands are not global
vault capabilities.

Each reusable package owns its deterministic phrases in
`<package>_node/deterministic_mappings.json`. The pre-vault deterministic router
discovers every `*_node/deterministic_mappings.json` package under the runtime
root and overlays the writable learned cache from
`luhkas_node/data/deterministic_commands.json` or `LUHKAS_DETERMINISTIC_CACHE`.

Manual camera controls call `POST /pantilt` repeatedly while held. Before
manual camera movement, the UI disables tracking. Centering the camera also
uses the manual pan/tilt path.

### Search Camera

When tracking is enabled and the selected target is missing, the tracker first
allows the predicted target state. After prediction expires, `SearchController`
can move the camera through a slower sweep phase and then a sinusoidal scan. The
UI `Search camera` toggle controls camera movement during `SEARCHING`; disabling
it keeps the state machine from driving pan/tilt while searching.

### USB Controller

`vision_service` reads the first `/dev/input/js*` device directly. Moving either
stick enables manual controller mode, disables tracking, disables search camera
movement, stops the wheels on entry/exit, and sets target state to `manual`.
Re-enabling tracking turns manual controller mode off.

| Input | Action |
|-------|--------|
| Left stick | Direct wheel movement through `robot.move(x,z)` |
| Right stick | Manual camera pan/tilt |
| A / button 0 | Center camera |
| B / button 1 | Toggle light and disable auto light |
| X / button 2 | Save snapshot under `captures/` |
| Y / button 3 | Save short MP4 clip under `captures/` |
| LB/RB / buttons 4/5 | Dim/brighten light and disable auto light |

The `Wheel drive` toggle applies to autonomous follow/collision behavior; manual
controller wheel commands are sent directly while manual mode is enabled.

### Camera Light

The camera light can be manual or automatic. Auto low-light mode estimates
ambient brightness from the mean grayscale video frame and smooths it over time.
If ambient light remains below `camera_light_low_threshold` for about two
seconds, the light turns on. Auto brightness scales with darkness up to
`camera_light_auto_brightness`.

The brightness slider has two meanings:

- Auto mode on: sets the maximum auto brightness.
- Auto mode off: sets manual light brightness immediately.

Scout-local chat commands such as `light on`, `light off`, and `light 50%`
are physical hardware commands owned by `light_node`.

To avoid blinking, the light does not simply turn off as soon as the scene looks
bright while illuminated. When the light is on, the service probes ambient light
about once per minute by halving current brightness for two seconds. If raw
ambient light at the reduced level is at least `low_threshold + 25`, it leaves
the light at that lower level until the next probe. If the scene is still too
dark, it restores the previous brightness.

### Vision API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health plus latest meta |
| `GET /meta` | Full per-frame state snapshot |
| `GET /chat_log?limit=200` | Session chat/prompt input-output log |
| `GET /video_feed` | MJPEG stream |
| `GET /snapshot` | Single JPEG frame |
| `GET /reference_poses` | Face reference pose coverage |
| `GET /people/<identity>/memory` | Local/vault memory profile |
| `POST /snapshot` | Save the latest JPEG frame under `captures/` |
| `POST /clip` | Save recent buffered frames as MP4 clip |
| `POST /learn_face?name=<name>&face_id=<id>` | Capture and train a targeted face reference |
| `POST /tracking` | Toggle tracking/follow and target identity |
| `POST /guard` | Toggle guard mode |
| `POST /pantilt` | Manual camera movement |
| `POST /move` | Manual wheel movement proxy |
| `POST /collision` | Collision settings |
| `POST /settings` | Live tuning values |
| `POST /people/<identity>/remember` | Store fact/preference |
| `POST /people/<identity>/preference` | Preference shorthand |

Important `/meta` fields include detections, object memory, tracker stats,
poses, target/target state, search phase, behavior state, guard status,
collision status, light state, ambient light, gamepad state, follow settings,
pan/tilt settings, face settings, vault memory status, and reference pose
coverage.

Node web chat is owned by `luhkas_node` on `/ui` and `/chat`; physical display
rendering is owned by `display_node`.

`POST /settings` can live-edit:

- Vision: `pose_enabled`, `pose_interval_frames`, `pose_score_threshold`,
  `pose_filter_persons`, `jpeg_quality`
- Tracking/control: `score_threshold`, `person_score_threshold`,
  `target_label`, `search_movement_enabled`, `manual_controller_enabled`
- Face: `face_interval_frames`, `face_detection_enabled`,
  `face_recognition_enabled`, `auto_reference_capture_enabled`,
  `auto_reference_min_confidence`
- Light: `camera_light_auto_enabled`, `camera_light_enabled`,
  `camera_light_brightness`, `camera_light_trigger_threshold`
- Motion: `follow_forward_speed`, `follow_steer_gain`,
  `follow_target_bbox_ratio`, `follow_deadzone_ratio`,
  `close_target_bbox_ratio`, `edge_reacquire_enabled`, `wheel_enabled`,
  `max_command`, `min_command`, `max_command_step`,
  `command_interval_seconds`, `settle_enter_degrees`,
  `settle_exit_degrees`, `estimated_pan_min`, `estimated_pan_max`,
  `estimated_tilt_min`, `estimated_tilt_max`, `pan_estimate_scale`,
  `tilt_estimate_scale`, `pan_limit_margin`, `absolute_max_step`,
  `absolute_min_step`, `absolute_pan_gain`, `absolute_tilt_gain`,
  `absolute_distance_gain`, `absolute_distance_max_multiplier`

## Face Recognition And Person Memory

Known face references live under `config/faces/<identity>/`. Use at least two
clear, consented images per person:

```text
config/faces/alex/01.jpg
config/faces/alex/02.jpg
```

Live enrollment:

```bash
curl -X POST "http://127.0.0.1:5000/learn_face?name=chris&face_id=12"
```

`learn_face` must only be called after a caller has selected a specific face
detection and knows the identity/name to attach to it. It requires a valid
`face_id` from the current `/meta` detections and will not choose a face
implicitly or fall back to a person bbox. Stand far enough back that both body
and face are visible, because live face detections are only valid inside person
boxes. Capture several angles. The service saves the targeted face crop,
retrains local recognition, advances the visible unknown-face introduction
queue, and reports how many samples are still needed.

Recognition can be disabled for a run with:

```bash
python3 services/vision_service.py --no-face-recognition
```

After a person is known, the recognizer maintains a reference-pose coverage map
for `frontal`, `left`, `right`, `up`, `down`, `close`, and `far`. Confident
live recognitions in under-covered poses are saved under
`config/faces/<identity>/_auto/<pose>/` and can be uploaded to the vault.

Local person facts and preferences live under `config/people/<identity>/`:

```text
config/people/chris/profile.json
config/people/chris/memories.jsonl
```

Examples:

```bash
curl -X POST "http://127.0.0.1:5000/people/chris/preference" \
  -H "Content-Type: application/json" \
  -d '{"key":"follow_distance","value":"far"}'

curl -X POST "http://127.0.0.1:5000/people/chris/remember" \
  -H "Content-Type: application/json" \
  -d '{"type":"fact","key":"display_name","value":"Chris"}'

curl "http://127.0.0.1:5000/people/chris/memory"
```

When `SCOUT_VAULT_MEMORY_ENABLED=1` and `SCOUT_VAULT_MEMORY_URL` are configured,
the vault brain is canonical for face references and person summaries. The rover
syncs face samples from `GET /faces/sync` into `config/vault_faces` by default
and uploads newly learned or auto-captured references back to the brain. Unknown
face groups are also vaulted: Scout uploads observations with `POST
/faces/unknown`, receives a vault group id, and asks the vault to compile that
group into a known identity with `POST /faces/unknown/promote`. The vault groups
unknown samples by face fingerprint across all camera nodes; Scout's local
unknown id is only a continuity hint.

Expected vault memory endpoints:

- `GET /faces/sync`
- `GET /faces/unknown`
- `POST /faces/unknown`
- `POST /faces/unknown/promote`
- `POST /people/<identity>/faces`
- `GET /people/<identity>/summary`
- `GET /people/<identity>/memory`
- `POST /people/<identity>/remember`
- `POST /people/<identity>/preference`

`GET /faces/sync` may return base64 blobs or URLs, grouped by identity:

```json
{
  "people": [
    {
      "identity": "chris",
      "samples": [
        {"path": "frontal/01.jpg", "image_b64": "..."}
      ]
    }
  ]
}
```

## Brain And Presence Contract

The vault PC owns the single Luhkas presence. Edge surfaces should use the same
brain chat endpoints and should not run independent personality/memory/router
loops.

Brain model roles currently documented for the vault:

| Role | Model |
|------|-------|
| Router | `qwen2.5:3b-instruct` |
| Chat | `qwen2.5:7b` |
| Reasoner/planner/analyst | `qwen3:30b` |
| Coder | `qwen3-coder:30b` |
| Fast coder | `qwen2.5-coder:14b` |
| Vision-language | `qwen2.5vl:7b` |
| Embeddings | `bge-m3` |

Brain chat endpoints:

- `GET /health`
- `GET /capabilities`
- `GET /session`
- `GET /scout/state`
- `GET /whoami`
- `GET /identity`
- `POST /identity`
- `GET /debug/identity`
- `POST /presence/message`

For first-time introductions, the brain should inspect:

- `GET http://<rover>:5000/meta`
- `GET http://<rover>:5000/snapshot`

If `/meta` shows a visible valid face and the user says "I am Chris" or similar,
the brain should call:

```text
POST http://<rover>:5000/learn_face?name=Chris&face_id=<visible-face-id>
```

The brain should only call this after the user has introduced themselves or
otherwise explicitly identified the selected face. It should then store
canonical person memory and face references. Current
`/meta` includes live target, object memory, behavior state, guard status,
collision status, gamepad/manual state, ambient light/light automation, search
phase, identity prompt/queue state, and tuning values. Face detections in
`/meta` are already filtered to faces inside person boxes.

For questions like "what do you see?", the brain should analyze the current
rover snapshot with its GPU vision model. Suggested internal endpoint:

```text
POST /vision/analyze
```

Suggested request shape:

```json
{
  "question": "What do you see?",
  "image_b64": "...",
  "tracking_memory": [],
  "active_identity": "chris"
}
```

The rover CLI talks to the brain, not directly to rover chat:

```bash
python3 tools/scout_chat.py --url http://brain.local:7000
```

Useful CLI commands: `/status`, `/who`, `/caps`, `/quit`.

`scout-presence` registers the rover node with the brain, proxies
`/presence/message`, exposes `/health`, `/session`, and
`/alerts/pending`, and buffers pending brain alerts.

## Configuration

Common environment variables:

| Variable | Default | Effect |
|----------|---------|--------|
| `ROBOT_API_PORT` | `5001` | Robot API listen port |
| `ROBOT_SERIAL_PORT` | `/dev/ttyAMA0` | UART device |
| `ROBOT_BAUD_RATE` | `115200` | Serial baud |
| `ROBOT_VAULT_URL` | `http://10.10.1.1:7000` | Brain URL for watchdog |
| `SCOUT_TELEMETRY_LOG_ENABLED` | `1` | Enable telemetry SQLite logging |
| `SCOUT_TELEMETRY_DB_PATH` | `config/telemetry.db` | Telemetry DB path |
| `SCOUT_VISION_PORT` | `5000` | Vision service listen port |
| `SCOUT_CAMERA_INDEX` | `0` | Camera device index |
| `ROBOT_API_URL` | `http://127.0.0.1:5001` | Robot API URL for vision service |
| `SCOUT_TARGET_LABEL` | `person` | Object label selected for tracking |
| `SCOUT_SCORE_THRESHOLD` | `0.45` | Generic detection/target threshold |
| `SCOUT_PERSON_SCORE_THRESHOLD` | `0.70` | Person-specific score threshold |
| `SCOUT_TRACKING_ENABLED` | `1` | Pan/tilt tracking on/off |
| `SCOUT_WHEEL_ENABLED` | `1` | Autonomous wheel behavior on/off |
| `SCOUT_FOLLOW_ENABLED` | `0` | Follow-person wheel mode |
| `SCOUT_COMMAND_INTERVAL` | `0.12` | Tracking command interval |
| `SCOUT_SEARCH_ENABLED` | `1` | Allow search camera movement |
| `SCOUT_SEARCH_SWEEP_DURATION` | `3.2` | Search sweep duration |
| `SCOUT_SEARCH_SWEEP_PAN_AMOUNT` | `60` | Search sweep pan amount |
| `SCOUT_SEARCH_SCAN_PAN_PERIOD` | `11.0` | Search scan pan period |
| `SCOUT_SEARCH_SCAN_TILT_PERIOD` | `18.0` | Search scan tilt period |
| `SCOUT_CAMERA_LIGHT_AUTO_ENABLED` | `1` | Enable auto low-light |
| `SCOUT_CAMERA_LIGHT_LOW_THRESHOLD` | `55` | Auto light trigger threshold |
| `SCOUT_CAMERA_LIGHT_AUTO_BRIGHTNESS` | `255` | Auto light maximum brightness |
| `SCOUT_FACE_DETECTION_ENABLED` | `1` | Enable face detection enrichment |
| `SCOUT_FACE_MIN_NEIGHBORS` | `3` | Haar cascade strictness for face candidates |
| `SCOUT_FACE_PERSON_UPPER_RATIO` | `0.55` | Reserved person/head association tuning |
| `SCOUT_FACE_MIN_PERSON_HEIGHT_RATIO` | `0.08` | Minimum face/person height ratio |
| `SCOUT_FACE_MAX_PERSON_HEIGHT_RATIO` | `0.50` | Maximum face/person height ratio |
| `SCOUT_FACE_INTRO_MIN_SEEN_FRAMES` | `2` | Grouped face observations before introduction prompts |
| `SCOUT_UNKNOWN_FACE_DIR` | `config/unknown_faces` | Short-lived local unknown face cache; vault owns persistent unknown groups |
| `SCOUT_FACE_UNKNOWN_MATCH_THRESHOLD` | `0.32` | Histogram/IoU matching threshold for unknown groups |
| `SCOUT_FACE_UNKNOWN_SAMPLE_INTERVAL` | `2.0` | Minimum seconds between saved unknown samples |
| `SCOUT_FACE_UNKNOWN_MAX_SAMPLES` | `24` | Max samples per unknown group |
| `SCOUT_FACE_UNKNOWN_PERSIST_SECONDS` | `8.0` | Keep unsampled unknown groups briefly after disappearance |
| `SCOUT_FACE_RECOGNITION_ENABLED` | `1` | Enable known-person recognition |
| `SCOUT_KNOWN_FACES_DIR` | `config/faces` | Local face reference directory |
| `SCOUT_FACE_RECOGNITION_INTERVAL_FRAMES` | `2` | Face recognition interval |
| `SCOUT_FACE_LBPH_THRESHOLD` | `72` | LBPH recognition threshold |
| `SCOUT_FACE_REFERENCE_POSES` | `frontal,left,right,up,down,close,far` | Reference pose buckets |
| `SCOUT_FACE_REFERENCE_SAMPLES_PER_POSE` | `3` | Target samples per reference pose |
| `SCOUT_FACE_AUTO_REFERENCE_CAPTURE` | `1` | Auto-capture missing pose refs |
| `SCOUT_FACE_AUTO_REFERENCE_MIN_CONFIDENCE` | `0.35` | Auto-capture confidence threshold |
| `SCOUT_FACE_AUTO_REFERENCE_COOLDOWN` | `20` | Auto-capture cooldown seconds |
| `SCOUT_IDENTITY_PROMPT_TEXT` | `Who are you?` | Unknown-face introduction prompt |
| `SCOUT_IDENTITY_PROMPT_REPEAT_SECONDS` | `45` | Repeat interval for the active unknown face prompt |
| `SCOUT_IDENTITY_PROMPT_COMPLETE_GRACE_SECONDS` | `20` | Time to suppress a just-learned face id while the queue advances |
| `SCOUT_PERSON_MEMORY_ENABLED` | `1` | Enable local person memory |
| `SCOUT_PEOPLE_DIR` | `config/people` | Local person memory directory |
| `SCOUT_VAULT_MEMORY_ENABLED` | `0` | Enable vault face/person sync |
| `SCOUT_VAULT_MEMORY_URL` | empty | Vault memory base URL |
| `SCOUT_VAULT_FACE_CACHE_DIR` | `config/vault_faces` | Brain-synced face cache |
| `SCOUT_VAULT_FACE_SYNC_INTERVAL` | `300` | Face sync interval seconds |
| `SCOUT_GUARD_ALERT_URL` | `http://luhkas-vault.local:7000/alerts` | Guard alert target |
| `SCOUT_GUARD_SNAPSHOT` | `1` | Include JPEG in guard alerts |
| `SCOUT_EGO_MOTION_ENABLED` | `0` | Enable gyro ego-motion feedback |
| `SCOUT_GYRO_PAN_SCALE` | `0.0` | Gyro z to pan delta |
| `SCOUT_GYRO_TILT_SCALE` | `0.0` | Gyro x to tilt delta |
| `VAULT_CHAT_URL` | `http://luhkas-vault.local:7000` | Presence brain URL |
| `VAULT_CHAT_SOURCE` | `scout_presence` | Presence source label |
| `SCOUT_PRESENCE_PORT` | `5002` | Presence listen port |
| `LUHKAS_TAILSCALE` | `1` | Install/start Tailscale during node bootstrap and service install |
| `TAILSCALE_AUTHKEY` | empty | Optional Tailscale auth key for unattended node enrollment |
| `TAILSCALE_HOSTNAME` | `luhkas-$LUHKAS_NODE_ID` | Tailnet hostname assigned during setup |
| `LUHKAS_PREFER_TAILSCALE` | `1` | Register the Tailscale IP as the preferred node service address |

Remote nodes are enrolled into the private tailnet by `scripts/bootstrap_node.sh`
and `node/scripts/install_user_services.sh` through
`node/scripts/setup_tailscale.sh`. For fully unattended provisioning, create a
reusable or ephemeral Tailscale auth key and store it on the vault at
`vault/secrets/tailscale.authkey` with mode `600`. Fresh nodes can start without
the key locally: they register with the vault over LAN, the vault SSHes back to
the node, writes `~/.config/luhkas/tailscale.authkey`, and reruns
`setup_tailscale.sh`. Nodes still register both `lan_ip` and `tailscale_ip` with
the vault; the vault prefers `tailscale_ip` when building service URLs.

## Local Data

| Path | Contents |
|------|----------|
| `captures/` | Snapshots and controller-triggered MP4 clips |
| `config/faces/<identity>/` | Local face reference images |
| `config/faces/<identity>/_auto/<pose>/` | Auto-captured pose references |
| `config/vault_faces/<identity>/` | Brain-synced face reference cache |
| `config/brain_faces/` | Older brain face cache path kept for migration |
| `config/unknown_faces/<unknown-id>/` | Local continuity cache for unknown faces before/while vaulting |
| `config/people/<identity>/profile.json` | Compact local facts/preferences |
| `config/people/<identity>/memories.jsonl` | Append-only local memory events |
| `config/telemetry.db` | SQLite telemetry history |

## Python Modules

| Module | Role |
|--------|------|
| `scout/config.py` | Dataclass configuration loaded from environment |
| `scout/types.py` | Detection dataclass and JSON conversion |
| `scout/vision.py` | Letterbox, Hailo output parsing, drawing, bbox helpers |
| `scout/tracking.py` | Multi-object tracking, target selection, prediction, object memory |
| `scout/motion.py` | Pan/tilt command generation, edge reacquire, wheel follow |
| `scout/search.py` | Optional sweep/scan camera search |
| `scout/behavior.py` | IDLE/FOLLOWING/SEARCHING/GUARDING/AVOIDING/MANUAL FSM |
| `scout/collision.py` | Frame-space collision blocking |
| `scout/pose.py` | Pose model integration and pose-based aim |
| `scout/face_detection.py` | OpenCV Haar face detection |
| `luhkas_node/` | Generic node registration, pre-vault router, package command composition |
| `camera_node/` | Reusable camera, media, vision, face, and guard ownership package |
| `pantilt_node/` | Reusable pan/tilt, tracking, search sweep, and target-centering package |
| `rover_node/` | Reusable wheel/follow/gamepad rover package; assumes `camera_node` |
| `light_node/` | Reusable generic light command package |
| `scout/face_recognition.py` | LBPH/histogram recognition and reference capture |
| `scout/vault_memory.py` | Brain face/person memory client |
| `scout/person_memory.py` | Local JSON/JSONL person memory store |
| `scout/robot_client.py` | HTTP client for `robot_api` |
| `scout/telemetry_logger.py` | SQLite WAL telemetry logger |

## Runtime Boundary

- `hailo-apps`: vendor examples/helpers/inference dependency.
- `scout_runtime`: camera loop, tracking, target selection, motion policy,
  controller input, robot API, serial protocol, telemetry, memory cache, UI, and
  video/status APIs.
