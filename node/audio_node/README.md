# audio_node

Owns the microphone-to-speaker loop on a node. STT and TTS run **on the
node** — vault only receives text. The service:

1. Captures mic audio via `arecord` (ALSA, raw 16 kHz mono S16_LE).
2. Streams each chunk into the configured STT engine.
3. When the engine emits a final utterance, POSTs the text to the local
   presence service (`/presence/message`).
4. Receives vault's response from the presence proxy and speaks the
   `tts` (or `message`) field via the configured TTS engine.

## Service

Default port `5004`. Endpoints:

- `GET /health` — engine + capture status, last error, last transcript
- `POST /tts` — `{"text": "..."}`; synthesize and play locally
- `POST /listen` — `{"muted": bool}`; pause/resume mic capture
- `GET /transcripts` — last 20 recognized utterances (debug)

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `AUDIO_HOST` | `0.0.0.0` | bind host |
| `AUDIO_PORT` | `5004` | bind port |
| `AUDIO_STT_ENGINE` | `vosk` | `vosk`, or `none` to disable |
| `AUDIO_TTS_ENGINE` | `espeak` | `espeak`, `piper`, or `none` |
| `AUDIO_INPUT_DEVICE` | `default` | ALSA capture device (e.g. `plughw:1,0` for the RaspAudio HAT) |
| `AUDIO_OUTPUT_DEVICE` | `default` | ALSA playback device |
| `AUDIO_PRESENCE_URL` | `http://127.0.0.1:5002/presence/message` | where transcripts are posted |
| `AUDIO_SOURCE` | `audio_node` | tag attached to outbound transcripts |

### Vosk STT

| Env var | Default | Meaning |
|---|---|---|
| `AUDIO_VOSK_MODEL` | auto-detects `~/.local/share/luhkas/audio_node/vosk-model-en-us-0.22-lgraph`, else `~/.vosk-model` | path to extracted Vosk model |
| `AUDIO_STT_RATE` | `16000` | capture sample rate |

Install:

```bash
pip install vosk
# Better real-time English model (~128 MB):
curl -L https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip -o /tmp/m.zip
unzip -q /tmp/m.zip -d ~/.local/share/luhkas/audio_node
```

### espeak-ng TTS

| Env var | Default | Meaning |
|---|---|---|
| `AUDIO_TTS_VOICE` | `en-us` | espeak voice id |
| `AUDIO_TTS_RATE` | `175` | words per minute |

```bash
sudo apt install -y espeak-ng alsa-utils
```

### Piper TTS (opt-in)

```bash
pip install piper-tts
# Download a voice (.onnx + .onnx.json) and point AUDIO_PIPER_MODEL at it.
```

## Pipeline

```
arecord ──► chunks (3200 B = 100 ms)
                │
                ▼
          stt.accept(pcm)
                │
        STTResult(final=True)
                │
                ▼
   POST /presence/message  ──►  vault
                                  │
                              response
                                  │
                                  ▼
                          tts.speak(text)
                                  │
                                  ▼
                             aplay (ALSA)
```

The service holds a single TTS lock so concurrent `/tts` requests serialize
naturally; the capture loop pauses itself implicitly by waiting on TTS to
finish before yielding the next transcript (since `on_transcript` blocks).
