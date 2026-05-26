# display_node

Owns the on-screen UI for nodes that have a display. The kiosk launcher
runs Chromium in `--kiosk` mode pointed at `http://127.0.0.1:5005/ui`.

## Service

Default port `5005`.

| Endpoint | Purpose |
|---|---|
| `GET /` / `GET /ui` | Index HTML (vanilla, no build step) |
| `GET /ui/assets/*` | Static CSS/JS |
| `GET /ui/state` | Polled by the SPA every 500 ms |
| `POST /ui/event` | Other services push events here |
| `POST /ui/mute` | Proxies to `audio_node`'s `/listen` |
| `GET /health` | Service status |

## Event types

Other node services POST `application/json` to `/ui/event`:

| `type` | Fields | Source |
|---|---|---|
| `user_message` | `text`, optional `confidence` | `audio_node` after STT |
| `assistant_message` | `text` | `audio_node` after receiving vault reply |
| `status` | any of `battery`, `audio`, `camera`, `muted` | any service |
| `alert` | `level`, `message` | any service |

The service keeps a rolling history of 50 events; the SPA reads from
`/ui/state` and renders the latest user/assistant pair plus a scroll
history.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DISPLAY_HOST` | `0.0.0.0` | bind host |
| `DISPLAY_PORT` | `5005` | bind port |
| `AUDIO_SERVICE_URL` | `http://127.0.0.1:5004` | where `/ui/mute` proxies to |

## Kiosk browser

`scripts/start_kiosk_browser.sh` launches Chromium full-screen at
`/ui`. The kiosk profile (`profiles/kiosk.json`) enables a
`kiosk-browser.service` that runs it under the user's X session. The
service depends on `kiosk-display.service` being healthy.
