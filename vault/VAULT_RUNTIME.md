# Vault Runtime — Complete Reference

The vault is the brain of Luhkas. It owns chat, routing, LLM inference,
identity, canonical memory, face references, learned capabilities, and
orchestration. Edge nodes (Scout and future wall-mounted nodes) are
modular hardware surfaces; they witness and forward context but do not
own long-term memory or independent personality.

This document covers the full surface area as of the current `main` branch.

---

## 1. Service Boundary

| Owner | What it owns | Where |
|---|---|---|
| **Vault PC** | chat, routing, LLM inference, identity, canonical memory, face references, learned capabilities, Code Monkey orchestration, public API | `vault/`, port 7000 |
| **Code Monkey** | recipe generation, smoke-testing, job queue, worker pool, lessons | `vault/code_monkey/`, port 8765 (localhost-only) |
| **Edge nodes (Scout)** | camera capture, Hailo inference, tracking, motor/serial APIs, presence-message forwarding | `node/`, ports 5000-5002 |

**Hard rules:**
- Vault must not run a second chat/personality loop on a node.
- Nodes must not own canonical long-term memory.
- The brain talks to Code Monkey through `code_monkey_client.py` HTTP only — never imports internals.
- The brain talks to Scout through narrow HTTP APIs (`/meta`, `/snapshot`, `/learn_face`, `/pantilt`, `/move`, etc.).

---

## 2. HTTP API (port 7000)

### Presence & chat

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/presence/message` | Unified presence endpoint — every edge surface (CLI, UI, voice nodes) sends user text here |
| `POST` | `/ui` | UI-friendly wrapper around the presence flow; same dispatch, simpler payload |
| `POST` | `/runtime/message` | Internal/dev-only general planner entry point |

Payload: `{"message": "<user text>", "node_id": "<session id>"}`. Response includes the assistant reply, route, identity, actions taken, provenance, and optional `notification_alert` if there are background notifications.

### Session & identity

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/session` | Global bridge session (last 30 turns, identity, tracking memory) |
| `GET`  | `/presence/session` | Presence bridge state — same shape |
| `GET`  | `/whoami` | Quick assistant identity summary |
| `GET`  | `/identity` | Full identity profile |
| `POST` | `/identity` | Update identity profile |
| `GET`  | `/debug/identity` | Identity debug snapshot (visible subjects, face references, recognition state) |

### Scout state

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/scout/state` | Live tracking/detection snapshot from Scout |
| `GET`  | `/scout/tools` | Current Scout tool contract + reachability |
| `POST` | `/vision/analyze` | Run vault GPU vision over a Scout snapshot |

### Capabilities, jobs, updates

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/capabilities` | List of capabilities the assistant exposes |
| `GET`  | `/jobs` | Code Monkey job list |
| `GET`  | `/code-monkey` | Code Monkey service health |
| `GET`  | `/updates` | Recent system events |
| `GET`  | `/alerts/pending` | Pending guard alerts |

### Face memory (vault is canonical)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/faces/sync` | Snapshot of all known face references for nodes to mirror |
| `GET`  | `/faces/unknown` | Currently grouped unknown faces |
| `POST` | `/faces/unknown` | Submit an unknown-face observation (camera nodes only) |
| `POST` | `/faces/unknown/promote` | Promote an unknown group to a named identity |
| `POST` | `/people/<id>/faces` | Upload face references for an identity |
| `GET`  | `/people/<id>/summary` | Person summary |
| `GET`  | `/people/<id>/memory` | Person memory log |
| `POST` | `/people/<id>/remember` | Append a memory fact for an identity |
| `POST` | `/people/<id>/preference` | Set a preference for an identity |

### Node registry & admin

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/node/register` | Node self-registers at startup |
| `POST` | `/node/selftest` | Node reports module health |
| `GET`  | `/node/status` | Vault's view of node health |
| `POST` | `/admin/sync` | Push vault code to a node via rsync |
| `GET`  | `/admin/pubkey` | Vault's public SSH key for node access |

---

## 3. Request Flow

Every incoming `/ui` or `/presence/message` POST traverses this pipeline:

```
HTTP request
   ↓
vault_runtime.handle_presence(message, node_id)
   ↓
_handle_deterministic_presence_command  ← exact-match cmds + pending handlers
   ↓ (if no deterministic match)
scout.handle_message(...)               ← per-session bridge
   ↓
   _adopt_identity_from_state           ← read face detections from scout state
   ↓
   route_message(message, state)        ← LLM router → general_question / self_question /
                                          analyze_vision / greeting / direction
   ↓
   _pronoun_route_guard                 ← post-router override (e.g. "my X" → general_q)
   ↓
   _dispatch_route                      ← topic-specific dispatch
       ├─ translate? → maybe_handle_translation
       ├─ forget?    → maybe_handle_forget
       ├─ recall?    → maybe_handle_recall (deterministic)
       ├─ persist    → persist_user_facts (extract → classify → store)
       ├─ self?      → fast_self_answer or LLM-composed identity
       └─ general?   → answer_with_context (LLM compose with recalled memory)
   ↓
vault_runtime reads stash markers (vision conflict, memory conflict, ...)
   ↓ installs pending state on blackboard if needed
   ↓
Response returned to caller
```

---

## 4. Routing

### Top-level routes

`scout_integration.ROUTE_OPTIONS = {"greeting", "general_question", "self_question", "direction", "analyze_vision"}`

The LLM router (`qwen2.5:3b-instruct`) classifies each message into one of these. A fast-path (`_fast_route_message`) handles obvious cases without the LLM (greetings, broad status, assistant name, etc.).

### Pronoun guard

After every router result, `_pronoun_route_guard` checks the message against `_SPEAKER_ATTR_PATTERNS`:

- `what's my X`, `where do I X`, `tell me my X`, `do you know/remember my X`, `how do I X`

If any matches AND the message isn't an identity-recognition carve-out (`who am I`, `do you know me`, `do you recognize me`, `have we met`, `what do you know about me`, `am I Chris`), the route is overridden to `general_question`.

This is needed because the small router model has been unreliable on the pronoun convention — even with the rule in the prompt, a trailing `?` would flip `"what's my name"` → general_question into `"what's my name?"` → self_question, then self-route classifier would pick `assistant_identity` and answer with the Luhkas intro.

### Self-route classifier

When a message lands on `self_question`, `classify_self_question` picks a sub-route:
`assistant_identity`, `user_identity`, `personality`, `hardware`, `software`, `status`, `capabilities`, `memory`, `sensors`, `goals`, `other`.

`user_identity` falls back to `MemoryStore` for a stored name when `active_identity` (face-recognized) isn't set yet.

### Deterministic short-circuits (in order, before LLM routing)

| Handler | Trigger |
|---|---|
| `maybe_handle_translation` | "translate X to Spanish", "how do you say X in French", etc. |
| `maybe_handle_forget` | "forget my X", "delete my X", "remove my X" |
| `maybe_handle_recall` | "do you know my X", "what's my X" (with topic substring guard) |
| Memory-conflict short-circuit | new fact contradicts stored — returns "Oh, I thought…" |
| Duplicate short-circuit | new fact already known — returns "I already have that…" |

### Low-confidence gate

When the router returns `confidence < 0.88` AND the route isn't `general_question` (which is the catch-all), the bridge asks the user to confirm before learning the route: "I think you mean X. Is that right?".

---

## 5. Identity-Scoped Memory

Backed by **LanceDB** at `/home/vault/vault_data/memory.lance`, with `bge-m3` embeddings (1024 dim).

### Buckets

- `assistant` — seeded on bridge init from `data/identity/profile.json` + `data/self/identity.json` (name, role, creator, personality, boundaries, body framed as connected edge modules, primary user, etc.). Used by `_assistant_identity_answer`.
- `<identity-lowercased>` — per-user namespace populated as the user speaks. Identity comes from `_adopt_identity_from_state` (face-recognized) or `"unknown"` fallback.
- `unknown` — fallback bucket when no face is recognized. Carries optional `unidentified_face_ref` so future face-binding can migrate facts.

### Storage operations (`storage/vector_store.py`)

| Method | Purpose |
|---|---|
| `add(content, identity, ...)` | Embed + insert with pre-write duplicate guard (default ≤0.05 cosine) |
| `search(query, identity, top_k, distance_max)` | Identity-scoped top-k nearest neighbors |
| `find_conflict_candidates(content, identity, distance_min, distance_max, top_k)` | Nearest-neighbor band search for conflict detection |
| `replace(old_id, new_content, ...)` | Atomic delete-by-id + insert |
| `delete_by_id(fact_id)` | Remove a single fact |
| `list_for_identity(identity, limit)` | Enumerate (for inspection/debug) |
| `count()` | Total row count |

### Extraction (`extract_user_facts`)

Per-turn pipeline:

1. **Deterministic name fast-path** — `_extract_introduction_name` catches `"I'm <Name>"`, `"my name is <Name>"`, `"I am <Name>"`, `"call me <Name>"`. Rejects articles, common adjectives ("tired", "happy"), common professions ("librarian", "engineer"), non-capitalized non-names, and any candidate that isn't `[A-Za-z][A-Za-z'\-]*$`.
2. **LLM extractor** — `qwen3:8b` with JSON-schema output (`{"facts": [...]}`). Returns 0+ facts in third-person canonical form (`"the user's pet is named Salem"`, `"the user lives in Austin"`, etc.). Switched from the smaller router model after it silently dropped common preferences ("favorite drink", "favorite season", "favorite movie") even with examples.

### Per-fact pipeline (parallel when multi-fact)

`persist_user_facts` → for each extracted fact, runs `_classify_and_route_fact`:

1. **`find_conflict_candidates`** — vector search in identity's namespace, distance band 0.0-0.7, top-1.
2. **`classify_fact_relation`** — `qwen3:8b` with JSON-schema enforcement, returns `duplicate / contradicts / extends / unrelated`.
3. **Outcome:**
   - `contradicts` → don't store; surface a conflict marker for the confirmation flow.
   - `duplicate` → don't store; return the existing record so the dispatcher can answer "I already have that".
   - `extends` or `unrelated` → call `memory_store.add` with a tight in-store duplicate guard (≤0.05) as final safety net.

When the extractor produces ≥2 facts, the per-fact pipelines run in a `ThreadPoolExecutor` (max 4 workers). The classifier is the dominant cost (~300ms/fact), so multi-fact turns parallelize roughly `(N-1) × 300ms` of latency.

### Recall (`recall_user_facts`)

Identity-scoped top-5 vector search. The recall results are rewritten to **second person** before being injected into the answer composer prompt — the chat model occasionally misread "the user's name is Chris" as being about a different person.

### Conflict resolution (`memory_update_confirmation`)

When `persist_user_facts` detects a contradiction:

1. Bridge stashes `_stash_memory_conflict_marker` and returns "Oh, I thought you live in Austin. Has that changed — should I update it to you live in Ocala?"
2. `vault_runtime` reads the stash marker and installs a `memory_update_confirmation` pending state (per-node, 5-minute TTL).
3. Next turn:
   - `_is_affirmative("yes")` → `store.replace(old_id, new_fact)` → "Got it — updated. you live in Ocala."
   - `_is_denial("no")` → discard new fact → "OK, I'll keep it as you live in Austin."
   - Anything else → leave pending alive (the user might say something fact-related between the prompt and their decision; TTL handles abandonment).

### Forget (`maybe_handle_forget`)

Detects `"forget/delete/remove [my] X"` via regex. Search the speaker's namespace for the topic, require all key topic tokens to appear as substrings in the candidate content (vector search alone ranks "favorite movie" too close to "favorite drink" because the template dominates the embedding), then `delete_by_id`. Returns "Forgotten — your X is Y is no longer stored." or "I don't have anything about your X stored." on a miss.

### Recall fast-path (`maybe_handle_recall`)

Detects `"do you know/remember my X"` / `"what's my X"` patterns. Substring topic-verification on the top-5 candidates. Two outcomes:

- **Hit** → returns the second-person rewrite directly ("Your favorite drink is coffee."). Populates `_current_memory_sources` for provenance.
- **Pattern matched but no fact** → returns "I don't have that stored." deterministically, preventing the chat LLM from confabulating from prior knowledge.

### Answer composer (`answer_with_context`)

For `general_question` routes that didn't short-circuit:

- Pulls recalled facts (second-person rewritten).
- Pulls `recent_chat` from session turns — **only when memory is empty** for the query (otherwise rejected-conflict statements leak into recall).
- Injects facts_just_stored (this turn).
- Prompt rules: answer only what was asked, never volunteer unrelated stored facts, prefer memory over chat-context, say "I don't have that stored" rather than guessing.

### Provenance

`build_answer_provenance` emits a `memory_store` source (with `identity` + `facts_consulted`) when recall consulted the store, a `session_chat` source when chat-context was used, or `model_prior_knowledge` as the fallback when neither was used.

---

## 6. Assistant Identity (`_assistant_identity_answer`)

The assistant has its own bucket in MemoryStore (`identity = "assistant"`), seeded on bridge init from:

- `data/identity/profile.json` (name, role, creator, body, primary user, personality traits, boundaries)
- `data/self/identity.json` (curated fact list)

`identity_profile.json` remains canonical for structured access; MemoryStore is the semantic-recall view.

### Name-only short-circuit

When `_asks_assistant_name` matches (`"what's your name"`, `"what is your name"`, `"your name"`), the answer is the terse `"I'm <name>."` — no LLM call. Sub-300ms.

### Composed identity answer

For `"who are you"`, `"what are you"`, `"tell me about yourself"`, `"introduce yourself"`:

- Pulls top-6 from the assistant bucket via `recall_assistant_facts`.
- Rewrites `"the assistant"` → `"I"` and `"the assistant's"` → `"my"` before prompt injection.
- Calls `chat_model.generate` directly (bypasses `response_composer` to keep the prompt's STRICT WORD RULES intact), with `think=False` and `num_predict=220` (qwen3:8b's thinking-mode tokens used to eat the entire budget and return empty).
- Prompt rules: positive framing only (no "but I'm not the rover" disclaimers), describe self as AI/AI assistant/AI presence, mention Scout only when the question is about hardware/body/sensors.
- Validator: `_assistant_identity_response_violation` rejects responses containing `"you are"`/`"you're"`/`"aren't"` (user/assistant confusion guard).

### Hardware questions

`"tell me about your hardware"` routes to self_question/hardware and returns a structured statement: `"Vault PC: RTX 3090, 96GB DDR5. Scout body: Raspberry Pi 5 with 16GB RAM, Hailo HAT+."`

---

## 7. Translation (`maybe_handle_translation`)

Detects 6 request patterns:

| Pattern | Example |
|---|---|
| `translate <src> to <lang>` | `translate 'hello' to Spanish` |
| `translate to <lang>: <src>` | `translate to Spanish: I would like a coffee` |
| `how do you say <src> in <lang>` | `how do you say good morning in Spanish` |
| `say <src> in <lang>` | `say thank you very much in Spanish` |
| `what's <src> in <lang>` | `what's the cat is on the table in Spanish` |
| `<lang> for <src>` | `Spanish for water` |

Strips surrounding quotes, looks up the language in `_TRANSLATE_LANGUAGES` (12 langs: Spanish, French, German, Italian, Portuguese, Japanese, Chinese, Korean, Russian, Dutch, Swedish), calls `chat_model.generate` directly with a translate-only prompt that emphasizes natural/idiomatic output. Wired BEFORE persist so the source text isn't extracted as a speaker-fact.

Accepts routes `general_question`, `direction`, AND `analyze_vision` (router occasionally misclassifies quoted-source translations as visual).

---

## 8. Learned Capabilities (`learned_capabilities.py`)

Vault learns new system-state queries on demand. A learned cap = `(topic, aspect)` tuple + a recipe (bash, python, or list-form command) that produces output.

### Lookup chain

When a user says something not matching an exact deterministic command:

1. **Exact match** — `aliases` map → run the recipe directly.
2. **Concept match** — `_infer_topic_and_aspect` (router LLM) extracts `(topic, aspect)`, then `lookup_by_concept` finds a matching cap.
3. **Propose** — if no match, `_propose_code_monkey_recipe` queues a `learned_capability_confirmation` pending. User confirms with "yes" → async learn job kicks off via Code Monkey.

### Topic classifier (`_CLASSIFIER_PROMPT`)

JSON-output prompt. Returns `(topic, aspect)` for system queries; explicitly returns `(none, none)` for identity questions (`"who am I"`, `"what is my name"`, etc.) so they don't get hijacked into bogus "Vault user login" capabilities.

### Async learning (`_spawn_async_learn`)

User confirms a novel cap → background thread calls Code Monkey's `/learned-command-recipe` endpoint. Code Monkey runs a planner + smoke-test + retry-with-feedback loop. On success: cap saved to `capabilities.json`. On failure: `learn_failed` notification surfaced via `any updates`.

If the recipe needs a missing binary (e.g., `iotop` not installed), the planner detects `missing_binary` from final retry error and surfaces a `learned_install_confirmation` pending. User confirms → vault runs `sudo apt-get install <pkg>` via passwordless sudo rule at `/etc/sudoers.d/vault-autonomous-install`, then retries the learn flow.

### Correction flow

After a learned cap fires, a `learned_execution_review` pending state is set for one turn. User says `"no, the configuration"` → LLM `classify_pending_intent` returns `intent: correct, topic: memory, aspect: configuration` → cap engine removes the wrong alias, proposes a new cap with the corrected (topic, aspect).

### Audit (`audit caps` admin command)

Walks the cap registry, finds duplicates by python source / topic-aspect, queues per-pair merge confirmations. User answers `yes / no / skip / cancel` for each pair. Per-node pending so it doesn't trap other sessions.

---

## 9. Code Monkey

Standalone HTTP service on `127.0.0.1:8765`. Brain talks to it via `code_monkey_client.py` HTTP only.

### Capabilities it owns

- Recipe generation via `recipe_generator.py` (qwen3-coder:30b or qwen2.5-coder:14b)
- Smoke testing: runs the recipe in subprocess, validates output, retries with feedback up to 3 times
- Safety policy: `_is_safe_sudo_install` allows only `apt-get install/update`; `BLOCKED_COMMAND_SUBSTRINGS` and `FORBIDDEN_ARGV_TOKENS` block destructive ops
- Job queue + 2 worker threads
- Notification surface: `events` + `notifications` tables in `vault_v2.db` (the name is legacy; the file lives at `vault/vault_v2.db` but holds current state)

### `install <pkg>` admin command

Recognized via `_maybe_handle_install_command`. Spawns an async worker (`_spawn_async_install`) that runs `sudo apt-get install <pkg>`. Returns immediately with "I'll install X and let you know when it's done." On success: `install_succeeded` notification.

### Background notifications

Long-running jobs (async learn, install) surface results via the `events` and `notifications` SQLite tables. The bridge's `_attach_notification_alert` populates a separate `notification_alert` field on responses so the UI can announce them without polluting the main message. `"any updates"` lists them in-flight + unread.

---

## 10. Per-Node State

### Bridge sessions

`ScoutVaultBridge._sessions: dict[str, _NodeSession]` — each node_id has its own:

- `active_identity` (face-adopted or name-stored)
- `turns` (last 30, RAM only — not persisted across restarts)

`_migrate_identity` brings turn history forward when a known identity appears on a new node.

### Pending decisions

Vault blackboard stores ONE pending decision (e.g., `learned_capability_confirmation`, `learned_install_confirmation`, `audit_merge_confirmation`, `vision_full_analysis_confirmation`, `memory_update_confirmation`, `learned_execution_review`).

Tagged with `_node_id` and `_expires_at` (5-minute TTL). `_get_pending(node_id)` filters by owner and auto-clears expired. This prevents a stale audit from one node trapping every other UI session — which used to happen before per-node scoping landed.

### Identity adoption (`_adopt_identity_from_state`)

Each turn, before dispatch, reads `state.detections` (and `object_memory` as fallback). For any person with `identity_confidence ≥ 0.6`, the highest-confidence match becomes `self.active_identity`. **Sticky:** when nobody is visible, we keep the previously adopted identity so brief step-outs don't reset the session. Voice cross-check is planned to arbitrate later.

---

## 11. Models (`models.py` + `config.py`)

Centralized via `get_model("<role>")`. **Never hardcode model names in runtime code.**

| Role | Default model | Used for |
|---|---|---|
| `router` | `qwen2.5:3b-instruct` | Top-level routing (general_question / self_question / ...) |
| `chat` | `qwen3:8b` | Composer + extractor + relation classifier + translate + identity composition |
| `reasoner` | `qwen3:30b` | Deep reasoning (Code Monkey planning) |
| `planner` | `qwen3:30b` | Capability planning |
| `coder` | `qwen3-coder:30b` | Recipe generation |
| `fast_coder` | `qwen2.5-coder:14b` | Fast recipe iteration |
| `vision` | `qwen2.5vl:7b` | GPU scene analysis on Scout snapshots |
| `embed` | `bge-m3` | Vector memory embeddings (1024 dim) |

`VAULT_WARM_MODEL_ROLES=router,chat,vision` keeps interactive models loaded with Ollama `keep_alive: 24h`. Background Code Monkey models use a shorter 5-minute keep-alive so they don't evict interactive models.

Override via env vars: `VAULT_ROUTER_MODEL`, `VAULT_CHAT_MODEL`, etc.

---

## 12. Response Composer + Validators

`response_composer.py` (`ResponseComposer.compose`) is the standard path for LLM-generated user-facing replies. It wraps the prompt in scaffolding ("Write the final user-facing answer..."), tracks `recent_responses` to discourage exact repeats, runs sanitizers, and applies a per-`response_type` validator.

`response_policy_violation` checks:

- emoji rejection
- excessive foreign-character rejection
- `"Luhkas will"` third-person reference rejection
- node-identity confusion (`_claims_assistant_is_node_identity`)
- per-type rules: `assistant_identity`, `mood_statement`, `greeting`, `status_report`

`assistant_identity` validator only rejects `"you are"`/`"you're"` patterns now — was over-strict before, blocking any response with "creator/body/scout/chris", which is the content we now surface from the assistant memory bucket.

The composer is **bypassed** for:

- `_assistant_identity_answer` — direct `chat_model.generate` with strict word rules (composer scaffolding diluted them, validator then rejected most output).
- `_llm_translate` — direct call with a translate-only prompt.

---

## 13. Response Lessons

`data/self/response_lessons.json` stores LLM-authored guidance gleaned from user corrections. The original capture path put the user's raw correction text into the `avoid` field — the small chat model later parroted it as a literal answer ("what's my name" → "I don't have a pet"). Fixed two ways:

1. **Recorder:** `_extract_direct_response_lesson` no longer assigns `avoid = text`. General case leaves `avoid = ""`; the hardcoded special branches still produce proper directive strings.
2. **Sanitizer:** `_format_response_lessons_for_prompt` injects only `prefer` directives (positively-framed LLM-authored guidance) as `response_guidance: ["[scope] <text>"]`. The raw `avoid` field never reaches the prompt.

---

## 14. Vision Short-Circuit

`analyze_vision` route checks router confidence (`VISION_ROUTE_CONFIDENCE_FLOOR = 0.85`). Below floor, route is downgraded to `general_question` to avoid spurious vision calls on ambiguous phrasings.

Above floor:

1. `_detection_summary(state)` builds a quick summary from `state.detections` (e.g., `"I see 2 chairs and 1 person, identified as Chris"`).
2. Bridge stashes `_stash_vision_short_circuit_marker` and returns the summary + "Would you like me to analyze the scene?"
3. `vault_runtime` reads the marker and installs `vision_full_analysis_confirmation` pending.
4. User says `"yes"` → re-route through scout with `presence_context.force_full_vision=True` → heavy `analyze_scene` LLM call (~7s on qwen2.5vl:7b).
5. User says `"no"` → "OK, I won't run the full vision analysis." Pending cleared.

---

## 15. Data Layout

```text
vault/data/
  identity/profile.json          ← canonical assistant identity (name, role, creator, …)
  self/                          ← curated self-knowledge: identity, capabilities, hardware, software,
                                   personality, mood, response_lessons, response_settings, …
  people/<identity>/             ← per-person profile.json + memories.jsonl (legacy, parallel to MemoryStore)
  face_references/               ← face embedding references
  unknown_face_groups/<group>/   ← unknown-face samples grouped by fingerprint
  learned_capabilities/
    capabilities.json            ← learned cap registry + aliases
    scripts/                     ← generated executable recipes
  nodes_registered.json          ← node registry (IP, services, capabilities, modules)
  deterministic_routes.json      ← cache of router-confirmed message → route mappings

/home/vault/vault_data/
  memory.lance/                  ← LanceDB semantic memory (identity-scoped facts)

vault/vault_v2.db                ← SQLite: events, notifications, jobs, tasks, task_history
```

---

## 16. Systemd

Three user-service units under `vault/systemd/`:

| Unit | Purpose |
|---|---|
| `vault-runtime.service` | Main brain runtime on port 7000 |
| `code-monkey.service` | Code Monkey service on 8765 |
| `vault-autosync.service` + `.timer` | Periodic rsync to nodes (triggered by `ExecStartPost` on vault-runtime restart, also cron) |

Install via `python3 install_systemd_units.py`. Manage with `systemctl --user status/restart/...`.

`vault-runtime.service` includes:

```ini
ExecStartPost=/usr/bin/systemctl --user start --no-block vault-autosync.service
Environment=VAULT_WARM_MODEL_ROLES=router,chat,vision
Environment=VAULT_IMMEDIATE_KEEP_ALIVE=24h
Environment=VAULT_BACKGROUND_KEEP_ALIVE=5m
Environment=VAULT_SCOUT_URL=http://luhkas-scout.local:5000
Environment=VAULT_SCOUT_ROBOT_URL=http://luhkas-scout.local:5001
```

---

## 17. Sync to Nodes (`sync_manager.py`)

`push_all(node_id=None)` rsyncs `node/` from vault to each node profile in `node/profiles/*.json`. The function:

1. Reads node profile (host, services to restart, exclude list).
2. Runs `rsync -a --delete --itemize-changes node/ <user>@<host>:~/luhkas/node/`.
3. Restarts configured services on the node via SSH only if files changed.
4. Returns `{ok, node_id, files_changed, restarted_services}` per node.

Vault has key-based SSH to nodes via `~/.ssh/id_ed25519_nodes`. The autosync timer triggers `push_all()` on a cadence; manual sync is `python3 vault/sync_manager.py`.

---

## 18. Common Operations

### Run interactively

```bash
cd vault && python3 main.py
```

### Run as service

```bash
sudo python3 vault/install_vault_service.py
systemctl --user enable --now vault-runtime.service
```

### Submit a UI message

```bash
curl -s -X POST http://luhkas-vault.local:7000/ui \
  -H 'content-type: application/json' \
  -d '{"message":"what time is it","node_id":"chris-laptop"}'
```

### Inspect memory store

```python
from storage.vector_store import MemoryStore
m = MemoryStore()
print(m.count())
for r in m.list_for_identity("chris"):
    print(r["content"])
```

### Wipe user memory (keep assistant facts)

```python
m._table.delete('identity != "assistant"')
```

### Wipe everything (assistant gets re-seeded on next restart)

```python
m._table.delete('id IS NOT NULL')
```

### Tail event log

```python
import sqlite3
c = sqlite3.connect("vault/vault_v2.db")
for row in c.execute("SELECT event_type, message, created_at FROM events ORDER BY created_at DESC LIMIT 20"):
    print(row)
```

---

## 19. Testing

Battery scripts live in `/tmp/ui_*.py` (ephemeral) but the pattern is:

```python
import json, time, urllib.request
req = urllib.request.Request(
    "http://luhkas-vault.local:7000/ui",
    data=json.dumps({"message": "...", "node_id": f"test_{int(time.time())}"}).encode(),
    headers={"content-type": "application/json"}, method="POST",
)
with urllib.request.urlopen(req, timeout=60) as r:
    body = json.loads(r.read())
print(body["response"]["message"])
print(body["response"]["route"])
print(body["response"]["answer_provenance"])
```

The most recent full battery covered 71 cases across 19 sections (memory storage/recall/duplicate/conflict/forget, per-node isolation, pronoun routing, assistant identity, identity-scoped writes, hardware listing, regression of existing capabilities, plus 10 newer tests for multi-fact extraction, forget edge cases, restart persistence, face-recognized identity, apostrophe names, vision short-circuit, etc.). 71/71 passed.

---

## 20. Anti-Patterns (don't do these)

- **Don't add a second chat/LLM loop on a node.** All chat goes to vault.
- **Don't write to Code Monkey storage from the brain.** Use the HTTP client.
- **Don't hardcode model names.** Use `get_model("role")`.
- **Don't bypass MemoryStore for canonical user facts.** People profile JSONL is legacy parallel storage; vector memory is the source for recall.
- **Don't put negative framing in seed facts or LLM prompts.** "the assistant is not the rover" leaks into answers as "I'm not the rover". Frame positively.
- **Don't expand the `assistant_identity_response_violation` regex without good cause.** Over-blocking forces the LLM into fallback ("I don't know that from the information I have").
- **Don't store user's raw correction text into `lesson.avoid`.** The chat model can parrot it.
- **Don't run the recall fast-path through `answer_with_context` for clear hits** — the LLM has been unreliable on yes/no recall framing ("do you know my name" → "I don't know your name" despite name in facts).
- **Don't clear `memory_update_confirmation` pending on neutral messages.** Let the TTL handle abandonment; the user might say something fact-related between the prompt and their decision.
