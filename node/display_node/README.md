# display_node

Owns the physical screen surface for a node. It does not own the chat UI or
`/chat`; those belong to `luhkas_node`.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Presence face HTML |
| `GET /presence/face` | Presence face HTML |
| `GET /presence/face/state` | State consumed by the face |
| `POST /ui/event` | Local event sink for user/assistant/status events |
| `GET /health` | Service status |

`POST /ui/event` is retained as the local display-event sink for compatibility
with `audio_node` and `luhkas_node`; it is not the owner of the web chat UI.
