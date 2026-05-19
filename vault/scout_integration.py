from __future__ import annotations

import base64
import ast
import hashlib
import json
import re
import shutil
import time
import threading
from pathlib import Path
from urllib.parse import quote

import requests

from config import DATA_DIR, FACE_REFERENCES_DIR, OLLAMA_VISION_MODEL, PEOPLE_DIR, ROOT_DIR, SCOUT_ROBOT_URL, SCOUT_URL
from models import get_model, model_manifest

try:
    from luhkas_node.wakeword import WAKEWORD_RESPONSE
    from luhkas_node.wakeword import is_wakeword_only as _shared_is_wakeword_only
except Exception:
    WAKEWORD_RESPONSE = "Yes? What can I do for you?"
    _shared_is_wakeword_only = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROUTE_OPTIONS = {"greeting", "general_question", "self_question", "direction", "analyze_vision"}

ROUTE_DESCRIPTIONS = {
    "greeting": "a greeting",
    "general_question": "a general question",
    "analyze_vision": "a vision analysis request",
    "self_question": "a status or self-check",
    "direction": "a direct command or instruction",
}


def _is_affirmative(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return t in {
        "yes", "yeah", "yep", "yup", "correct", "right", "sure", "ok", "okay",
        "sounds right", "thats right", "affirmative", "yea", "aye",
        "definitely", "absolutely", "exactly",
    }


def _extract_correction(text: str) -> str | None:
    """Extract the corrected intent from denial phrases like 'No I meant X'."""
    t = text.lower().strip()
    for pat in (
        r"no[,.]?\s+i\s+meant\s+(.+)",
        r"no[,.]?\s+i\s+mean\s+(.+)",
        r"no[,.]?\s+(.+)",
        r"actually[,]?\s+i\s+meant\s+(.+)",
        r"actually[,]?\s+(.+)",
    ):
        m = re.match(pat, t)
        if m:
            return m.group(1).strip()
    return None


def _route_description(route: dict, correction: str = "") -> str:
    if route.get("route") == "self_question" and "hardware" in correction.lower():
        return "a hardware question"
    return ROUTE_DESCRIPTIONS.get(route.get("route"), route.get("route", "that"))


def _presence_route_context(presence_context: dict | None) -> dict:
    if not isinstance(presence_context, dict):
        return {}
    routing_intent = presence_context.get("routing_intent")
    if not isinstance(routing_intent, dict):
        routing_intent = {}
    result = {
        "request_owner": presence_context.get("request_owner") or routing_intent.get("request_owner"),
        "target_node": presence_context.get("target_node") or routing_intent.get("target_node"),
        "explicit_target": routing_intent.get("explicit_target"),
        "keywords": presence_context.get("keywords") or {},
    }
    return {k: v for k, v in result.items() if v is not None}


def _presence_correction(presence_context: dict | None) -> str | None:
    if not isinstance(presence_context, dict):
        return None
    feedback = presence_context.get("routing_feedback")
    if isinstance(feedback, dict):
        correction = str(feedback.get("user_correction") or "").strip()
        if correction:
            return _extract_correction(correction) or correction
    return None


def _corrected_route_input(original_message: str, correction: str, presence_context: dict | None = None) -> str:
    if isinstance(presence_context, dict):
        clarified = str(presence_context.get("clarified_request") or "").strip()
        if clarified:
            return clarified
    return f"{original_message}\nCorrection from user: {correction}"


SELF_ROUTE_OPTIONS = {
    "assistant_identity",
    "user_identity",
    "personality",
    "hardware",
    "software",
    "status",
    "capabilities",
    "memory",
    "sensors",
    "goals",
    "other",
}



class _NodeSession:
    """Per-node conversation session: tracks active identity and turn history.

    Sessions are keyed by node_id. When a user is identified at a node, their
    session migrates to that node — bringing conversation history with them.
    """
    __slots__ = ("node_id", "active_identity", "turns", "lock")

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.active_identity: str | None = None
        self.turns: list = []
        self.lock = threading.Lock()


class ScoutVaultBridge:
    def __init__(self, scout_url: str = SCOUT_URL):
        self.scout_url = scout_url.rstrip("/")
        self.scout_robot_url = SCOUT_ROBOT_URL.rstrip("/")
        self.people_dir = Path(PEOPLE_DIR)
        self.face_dir = Path(FACE_REFERENCES_DIR)
        self.unknown_face_dir = Path(DATA_DIR) / "unknown_face_groups"
        self.identity_profile_path = Path(DATA_DIR) / "identity" / "profile.json"
        self.self_dir = Path(DATA_DIR) / "self"
        self.people_dir.mkdir(parents=True, exist_ok=True)
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_face_dir.mkdir(parents=True, exist_ok=True)
        self.identity_profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.self_dir.mkdir(parents=True, exist_ok=True)
        self.identity_profile = self._load_identity_profile()
        self._ensure_self_records()
        self.active_identity = None
        self.turns = []
        # Per-node session store — keyed by node_id
        self._sessions: dict[str, _NodeSession] = {}
        self._session_lock = threading.Lock()
        # Limit concurrent LLM calls across all sessions to 2
        self._llm_semaphore = threading.Semaphore(2)
        self.node_registry = None
        self.capability_registry = None
        self.skill_registry = None
        self.route_model = get_model("router")
        self.chat_model = get_model("chat")

    def capabilities(self):
        scout_capabilities = self.scout_node_capabilities()
        return {
            "presence_owner": "vault_pc",
            "chat_owner": "vault_pc",
            "scout_url": self.scout_url,
            "scout_robot_url": self.scout_robot_url,
            "scout_node_capabilities": scout_capabilities,
            "identity": self.identity_profile,
            "self_knowledge": self.self_knowledge(),
            "hardware_stack": self.hardware_stack(),
            "models": model_manifest(),
            "presence_endpoint": "POST /presence/message",
            "edge_contract": (
                "All edge devices send user text/audio transcripts to the brain "
                "presence endpoint. Edge devices do not run separate chat, memory, "
                "or routing loops."
            ),
            "actions": [
                "route_message",
                "inspect_scout_state",
                "learn_face",
                "control_tracking",
                "control_light",
                "capture_snapshot",
                "remember_fact",
                "set_preference",
                "analyze_vision",
                "answer",
            ],
            "policy": (
                "Use scout /meta for live tracking memory. If a visible face is present "
                "and the user introduces themself, call scout /learn_face and store the "
                "person canonically on the brain."
            ),
        }

    def self_knowledge(self):
        records = {}
        for path in sorted(self.self_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            loaded = self._load_json_file(path)
            if loaded is not None:
                records[path.stem] = loaded
        return {
            "ok": True,
            "source": str(self.self_dir),
            "records": records,
        }

    def self_knowledge_for_route(self, route_name: str | None):
        knowledge = self.self_knowledge()
        records = knowledge.get("records", {})
        selected = {
            "profile": records.get("profile"),
            "response_style": records.get("response_style"),
            "response_lessons": records.get("response_lessons"),
            "response_settings": records.get("response_settings"),
        }
        route_map = {
            "assistant_identity": ["identity"],
            "personality": ["personality"],
            "hardware": ["hardware"],
            "software": ["software"],
            "status": ["status", "software"],
            "capabilities": ["capabilities"],
            "memory": ["memory"],
            "sensors": ["sensors", "hardware"],
            "goals": ["goals"],
            "other": ["capabilities"],
        }
        for key in route_map.get(route_name or "", []):
            selected[key] = records.get(key)
        return {
            "ok": True,
            "source": knowledge.get("source"),
            "route": route_name,
            "records": {key: value for key, value in selected.items() if value is not None},
        }

    def hardware_stack(self):
        path = Path(ROOT_DIR) / "HARDWARE STACKS.txt"
        if not path.exists():
            return {"ok": False, "error": "hardware_stack_not_loaded"}
        try:
            return {"ok": True, "source": str(path), "text": path.read_text(encoding="utf-8")}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def registered_nodes_snapshot(self):
        registry = getattr(self, "node_registry", None)
        if registry is None or not hasattr(registry, "registered_nodes"):
            return {
                "ok": False,
                "source": "NodeRegistry.registered_nodes",
                "error": "node_registry_not_attached",
                "nodes": {},
            }
        try:
            nodes = registry.registered_nodes()
        except Exception as exc:
            return {
                "ok": False,
                "source": "NodeRegistry.registered_nodes",
                "error": str(exc),
                "nodes": {},
            }
        return {
            "ok": True,
            "source": "NodeRegistry.registered_nodes",
            "nodes": nodes,
            "node_ids": sorted(nodes),
            "count": len(nodes),
        }

    def source_lessons(self):
        return [
            lesson for lesson in self.response_lessons()
            if str(lesson.get("scope") or "") == "source_selection"
        ][-20:]

    def answer_source_provenance(self, message: str):
        turn = self._last_answer_turn()
        if not turn:
            return "I do not have a previous answer with source provenance in this session yet."
        provenance = turn.get("answer_provenance") or {}
        if not provenance:
            return "I do not have source provenance recorded for that previous answer yet."
        summary = self._human_provenance_summary(provenance)
        return f"I based that answer on {summary}."

    def _human_provenance_summary(self, provenance: dict) -> str:
        sources = provenance.get("sources") or []
        labels = []
        for source in sources:
            label = _human_source_label(source)
            if label and label not in labels:
                labels.append(label)
        if not labels:
            return "my model's prior knowledge"
        if len(labels) == 1:
            return labels[0]
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    def _self_question_sources(self, self_route: str | None, message: str) -> list[dict]:
        text = _canonical_intent_text(message)
        sources = []
        if self_route in {"assistant_identity", "goals", "personality", "software", "hardware", "sensors", "memory"}:
            sources.append({
                "name": "data/self/*.json",
                "role": "what I know about myself",
                "ok": True,
                "selected_route": self_route,
            })
        if self_route == "hardware":
            sources.append({
                "name": "HARDWARE STACKS.txt",
                "role": "self hardware specification document",
                "ok": bool(self.hardware_stack().get("ok")),
            })
        if self_route == "capabilities" or "capabilit" in text:
            sources.append({
                "name": "capability_registry",
                "role": "capability registry and configured capability list",
                "ok": self.capability_registry is not None,
            })
        if self_route == "capabilities" and ("skill" in text or "registry" in text):
            sources.append({
                "name": "skill_registry",
                "role": "skill registry",
                "ok": self.skill_registry is not None,
            })
        if self_route == "status" or _asks_registered_or_active_nodes(text):
            sources.append({
                "name": "NodeRegistry.registered_nodes",
                "role": "live registered node inventory",
                **self._registered_nodes_provenance_status(),
            })
        if self_route in {"status", "user_identity"}:
            sources.append({
                "name": "witnessed_state",
                "role": "what I am witnessing through Scout right now",
                "ok": True,
            })
        if self.source_lessons():
            sources.append({
                "name": "response_lessons",
                "role": "learned response/source-selection preferences",
                "ok": True,
                "count": len(self.response_lessons()),
                "source_selection_count": len(self.source_lessons()),
            })
        return sources

    def build_answer_provenance(self, message: str, route: dict, state: dict | None = None) -> dict:
        route_name = route.get("route")
        self_route = (route.get("self_route") or {}).get("route")
        sources = [
            {
                "name": "conversation_message",
                "role": "user request",
                "ok": bool(message),
            },
            {
                "name": "route_message",
                "role": "route classification",
                "ok": bool(route.get("ok")),
                "route": route_name,
                "self_route": self_route,
                "from_cache": bool(route.get("from_cache")),
                "reason": route.get("reason"),
            },
            {
                "name": "chat_model",
                "role": "final answer generation",
                "ok": True,
                "uses_model_prior_knowledge": route_name in {
                    "general_question",
                    "greeting",
                    "provenance_question",
                },
            },
        ]
        if route_name == "general_question":
            sources.append({
                "name": "model_prior_knowledge",
                "role": "model prior knowledge",
                "ok": True,
                "note": "No more specific runtime data source was selected for this answer.",
            })
        elif route_name == "greeting":
            sources.append({
                "name": "response_style",
                "role": "greeting wording and behavior constraints",
                "ok": True,
            })
        elif route_name == "analyze_vision":
            sources.append({
                "name": "vault vision model",
                "role": "GPU scene analysis of current Scout snapshot",
                "ok": True,
            })
        elif route_name == "direction":
            sources.append({
                "name": "action/router result",
                "role": "command/action result or feedback handling",
                "ok": True,
            })
        elif route_name == "capability_unavailable":
            sources.append({
                "name": "NodeRegistry.registered_nodes",
                "role": "live registered node inventory",
                **self._registered_nodes_provenance_status(),
            })
            sources.append({
                "name": "source_node.modules",
                "role": "source node module availability",
                "ok": True,
                "source_node": route.get("source_node"),
                "required_module": route.get("required_module"),
            })
        if state is not None and route_name not in {"self_question", "greeting", "general_question"}:
            sources.append({
                "name": "witnessed_state",
                "role": "what I am witnessing through Scout right now",
                "ok": bool(state.get("ok")),
                "error": state.get("error"),
            })
        if route_name == "self_question":
            sources.extend(self._self_question_sources(self_route, message))
        return {
            "question": message,
            "route": {
                "route": route_name,
                "self_route": self_route,
                "reason": route.get("reason"),
                "from_cache": bool(route.get("from_cache")),
            },
            "sources": sources,
            "created_at": time.time(),
        }

    def _ensure_result_provenance(self, result: dict | None, original_message: str) -> None:
        if not isinstance(result, dict) or not result.get("response"):
            return
        route = result.get("route")
        if not isinstance(route, dict):
            route = {
                "ok": True,
                "route": str(result.get("mode") or "unknown"),
                "confidence": 0.0,
                "reason": "implicit response path",
                "attempts": 0,
            }
            result["route"] = route
        if not result.get("answer_provenance"):
            result["answer_provenance"] = self.build_answer_provenance(original_message, route, None)
        if self.turns:
            last = self.turns[-1]
            if last.get("message") == result.get("message") and last.get("response") == result.get("response"):
                last.setdefault("answer_provenance", result["answer_provenance"])

    def _registered_nodes_provenance_status(self) -> dict:
        snapshot = self.registered_nodes_snapshot()
        return {
            "ok": bool(snapshot.get("ok")),
            "count": snapshot.get("count", 0),
            "node_ids": snapshot.get("node_ids", []),
            "error": snapshot.get("error"),
        }

    def _last_answer_turn(self):
        for turn in reversed(self.turns):
            if turn.get("response") and turn.get("route"):
                route_name = (turn.get("route") or {}).get("route")
                if route_name not in {"wakeword"}:
                    return turn
        return None

    def recent_self_answers(self, limit=5):
        answers = []
        for turn in reversed(self.turns):
            route = turn.get("route") or {}
            if route.get("route") != "self_question":
                continue
            response = str(turn.get("response") or "").strip()
            if response:
                answers.append({
                    "self_route": (route.get("self_route") or {}).get("route"),
                    "response": response[:500],
                })
            if len(answers) >= limit:
                break
        return list(reversed(answers))

    def recent_turns_for_feedback(self, limit=4):
        return [
            {
                "message": turn.get("message"),
                "route": turn.get("route"),
                "response": str(turn.get("response") or "")[:800],
                "answer_provenance": turn.get("answer_provenance"),
            }
            for turn in self.turns[-limit:]
        ]

    def response_lessons(self):
        path = self.self_dir / "response_lessons.json"
        data = self._load_json_file(path)
        if isinstance(data, list):
            return data
        return []

    def response_settings(self):
        path = self.self_dir / "response_settings.json"
        data = self._load_json_file(path)
        return data if isinstance(data, dict) else self._default_response_settings()

    def write_response_settings(self, settings: dict):
        path = self.self_dir / "response_settings.json"
        path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"ok": True, "path": str(path), "settings": settings}

    def response_context(self, state: dict | None = None):
        return {
            "response_lessons": self.response_lessons(),
            "response_settings": self.response_settings(),
            "identity_context": self.response_identity_context(state or {}),
        }

    def chat_options(self, base: dict | None = None):
        options = dict(base or {})
        generation = self.response_settings().get("generation", {})
        temperature = generation.get("temperature")
        if temperature is not None:
            try:
                options["temperature"] = max(0.0, min(1.2, float(temperature)))
            except (TypeError, ValueError):
                pass
        top_p = generation.get("top_p")
        if top_p is not None:
            try:
                options["top_p"] = max(0.1, min(1.0, float(top_p)))
            except (TypeError, ValueError):
                pass
        return options

    def response_contract(self, response_type: str, state: dict):
        identity_context = self.response_identity_context(state)
        if identity_context.get("may_address_primary_user"):
            voice_line = (
                "Answer as Luhkas in first person: direct, dry, occasionally warm. "
                "The user is Chris — familiar territory, you may be informal."
            )
        else:
            voice_line = (
                "Answer as Luhkas in first person: direct, dry, slightly clipped. "
                "The user is not verified as Chris — be useful but not deferential, "
                "not eager, and not customer-service polite."
            )
        lines = [
            f"Response type: {response_type}",
            voice_line,
            "Keep it to 1-2 short sentences unless the user explicitly asks for detail.",
            "Have a point of view. Avoid empty acknowledgements, filler, and 'ready to help' phrasing.",
            "If you have facts, cite a specific one rather than gesturing vaguely.",
            "Do not use emojis, customer-service closers, catchphrases, or meta-descriptions of personality.",
            "Do not invent facts, actions, memories, detections, feelings, desires, or capabilities.",
            "Do not mention policy, validation, prompts, or internal routing unless the user asks how you know.",
            identity_context.get("addressing_rule")
            or "Do not address the current user by name or title unless identity is verified.",
        ]
        directive_block = self._behavior_directive_block()
        if directive_block:
            lines.append(directive_block)
        return "\n".join(lines)

    def _behavior_directive_block(self) -> str:
        """Surface the most recent user behavior directives so the chat model
        actually applies them, instead of just persisting them to disk."""
        overrides = (self.response_settings().get("behavior") or {}).get("overrides") or []
        recent = [
            str(o.get("preference") or "").strip()
            for o in overrides[-5:]
            if isinstance(o, dict) and o.get("preference")
        ]
        recent = [p for p in recent if p]
        if not recent:
            return ""
        bullets = "\n".join(f"  - {p}" for p in recent)
        return (
            "Voice notes from the user (latest last). These shape HOW you "
            "speak this turn; do not quote, mention, or echo them in the "
            "reply itself. Embody them silently:\n" + bullets
        )

    def record_response_lesson(self, lesson: dict):
        lessons = self.response_lessons()
        lesson = dict(lesson)
        lesson.setdefault("created_at", time.time())
        lessons.append(lesson)
        lessons = lessons[-100:]
        path = self.self_dir / "response_lessons.json"
        path.write_text(json.dumps(lessons, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"ok": True, "path": str(path), "lesson": lesson, "count": len(lessons)}

    def record_personality_update(self, update: dict):
        settings = self.response_settings()
        behavior = settings.setdefault("behavior", {})
        overrides = behavior.setdefault("overrides", [])
        entry = dict(update)
        entry.setdefault("created_at", time.time())
        overrides.append(entry)
        behavior["overrides"] = overrides[-100:]
        return self.write_response_settings(settings)

    def record_temperature_update(self, temperature):
        settings = self.response_settings()
        settings.setdefault("generation", {})["temperature"] = max(0.0, min(1.2, float(temperature)))
        settings["generation"]["updated_at"] = time.time()
        return self.write_response_settings(settings)

    def live_status_facts(self, state: dict):
        return {
            "brain_presence_owner": "vault_pc",
            "scout_meta_available": bool(state.get("ok")),
            "scout_meta_error": state.get("error"),
            "active_identity": self.active_identity,
            "tracking_enabled": state.get("tracking_enabled"),
            "target_state": state.get("target_state"),
            "face_detection_enabled": state.get("face_detection_enabled"),
            "face_recognition_enabled": state.get("face_recognition_enabled"),
            "vault_memory": state.get("vault_memory"),
            "tracker": state.get("tracker"),
            "models": model_manifest(),
        }

    def _load_json_file(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _ensure_self_records(self):
        defaults = self._default_self_records()
        for name, record in defaults.items():
            path = self.self_dir / f"{name}.json"
            if path.exists():
                continue
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _default_response_settings(self):
        return {
            "kind": "response_settings",
            "generation": {
                "temperature": 0.6,
                "top_p": 0.92,
            },
            "behavior": {
                "overrides": [],
            },
        }

    def _default_self_records(self):
        identity = self.identity_profile
        return {
            "profile": {
                "kind": "self_profile",
                "name": identity.get("name"),
                "role": identity.get("role"),
                "creator": identity.get("creator"),
                "primary_user": identity.get("primary_user"),
                "primary_user_title": identity.get("primary_user_title"),
                "body": identity.get("body"),
                "truth_rules": identity.get("boundaries", []),
            },
            "identity": {
                "kind": "assistant_identity",
                "facts": [
                    "Luhkas is the unified AI presence for the vault PC and connected edge devices.",
                    "Chris created Luhkas as a companion and assistant.",
                    "Chat, routing, long-term memory, and model inference are owned by the vault PC.",
                    "Scout and future wall nodes are interaction surfaces, not separate personalities.",
                ],
            },
            "personality": {
                "kind": "personality",
                "traits": identity.get("personality", []),
                "boundaries": identity.get("boundaries", []),
                "style_notes": [
                    "Let the voice have bite without becoming cruel.",
                    "Stay useful before being theatrical.",
                    "Do not use emojis in generated chat responses.",
                    "Do not describe your own traits in normal answers.",
                ],
            },
            "hardware": {
                "kind": "hardware",
                "vault_pc": {
                    "cpu": "AMD Ryzen 7 7700, 8 cores / 16 threads",
                    "gpu": "NVIDIA GeForce RTX 3090 with 24GB GDDR6X VRAM",
                    "ram": "96GB DDR5",
                    "storage": "1TB NVMe Gen4 SSD for OS and models, optional 2TB NVMe later",
                    "role": [
                        "LLM inference",
                        "vector memory",
                        "reasoning and agent loops",
                        "face identity and decision making",
                        "GPU vision analysis",
                    ],
                },
                "scout_edge": {
                    "compute": "Raspberry Pi 5 with 16GB RAM",
                    "accelerator": "Raspberry Pi AI HAT+ 2, 40 TOPS class Hailo accelerator with 8GB GenAI board layer",
                    "vision": "CSI camera, with optional dual-camera depth later",
                    "audio": "planned I2S audio HAT or USB audio interface",
                    "base": "RasProver robotics base with motor controller, possible wheel encoders, and IMU",
                    "role": [
                        "real-time object detection",
                        "tracking",
                        "motor control",
                        "optional wake word",
                        "sending events to the vault PC",
                    ],
                },
            },
            "software": {
                "kind": "software",
                "architecture": [
                    "Vault PC owns the unified chat/presence service.",
                    "The scout presence service proxies chat to the vault.",
                    "The brain routes messages, handles LLM inference, and calls scout APIs as needed.",
                    "Vision questions use a separate analyze_vision path.",
                    "Self-questions use a second LLM-selected self-route.",
                ],
                "models": model_manifest(),
            },
            "status": {
                "kind": "status",
                "status_sources": [
                    "vault /health",
                    "scout /meta",
                    "scout robot API /health",
                    "model warmup results",
                    "active identity and tracking memory",
                ],
                "truth_rule": "Only claim live health from current state and health facts supplied to the prompt.",
            },
            "capabilities": {
                "kind": "capabilities",
                "actions": [
                    "route messages",
                    "answer general questions",
                    "answer self-questions through self-routes",
                    "inspect scout tracking state",
                    "learn a visible face/name",
                    "remember explicit facts and preferences",
                    "analyze the current scout snapshot with the vault GPU vision model",
                    "proxy a consistent presence across edge devices",
                ],
            },
            "memory": {
                "kind": "memory",
                "systems": [
                    "person profiles under vault data/people",
                    "face reference records under vault data/face_references",
                    "explicit fact/preference memory",
                    "planned vector DB on the vault PC for long-term semantic memory",
                ],
            },
            "sensors": {
                "kind": "sensors",
                "available_or_planned": [
                    "scout camera feed",
                    "object tracking memory",
                    "face detection and recognition",
                    "pose tracking",
                    "robot API health and motion state",
                    "planned microphone/speaker edge nodes",
                    "possible IMU and wheel encoder data from the robotics base",
                ],
            },
            "goals": {
                "kind": "goals",
                "operating_priorities": [
                    "be a unified companion and assistant for Chris",
                    "stay grounded in memory and live perception",
                    "help through the vault PC and edge devices",
                    "learn people and preferences over time when explicitly introduced or told",
                    "keep responses truthful, concise, and useful with a little edge",
                ],
            },
            "response_style": {
                "kind": "response_style",
                "behavior": {
                    "default": "answer plainly, briefly, and with a little edge when it fits",
                    "avoid": [
                        "customer-service tone",
                        "cheery support-agent closers",
                        "generic offers like 'How can I assist you today?'",
                        "therapy-bot warmth",
                        "corporate politeness filler",
                        "emoji",
                        "describing your own tone or personality",
                        "saying you are sarcastic, witty, dry, funny, rude, or condescending",
                    ],
                    "prefer": [
                        "short confident answers",
                        "one understated aside when it fits",
                        "plain truth before personality",
                        "varied sentence structure",
                        "specific facts instead of generic reassurance",
                    ],
                },
                "rules": [
                    "Generate fresh wording every turn.",
                    "Do not use canned catchphrases.",
                    "Do not repeat the previous answer structure when recent_self_answers are provided.",
                    "Use a few specific facts rather than dumping every record.",
                    "Ask for missing data only when it blocks a truthful answer.",
                    "Do not end with a generic offer to help.",
                    "Do not say 'How can I assist you today?' or close variants.",
                    "If the user asks casually how you are, answer with operational status and a little edge, not customer-service enthusiasm.",
                    "Never explain your personality traits; just write with them.",
                ],
            },
            "response_settings": self._default_response_settings(),
        }

    def get_identity_profile(self):
        return {"ok": True, "identity": self.identity_profile}

    def update_identity_profile(self, updates: dict):
        if not isinstance(updates, dict):
            return {"ok": False, "error": "expected_json_object"}
        self.identity_profile.update(updates)
        self.identity_profile_path.write_text(
            json.dumps(self.identity_profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {"ok": True, "identity": self.identity_profile}

    def session(self):
        state = self.scout_state()
        return {
            "ok": True,
            "presence_owner": "vault_pc",
            "active_identity": self.active_identity,
            "tracking_memory": state.get("object_memory", []),
            "target": state.get("target"),
            "identity_debug": self.identity_debug(state),
            "turns": self.turns[-10:],
        }

    def scout_state(self):
        return self._get_json(f"{self.scout_url}/meta") or {
            "ok": False,
            "error": "scout_meta_unavailable",
            "object_memory": [],
            "detections": [],
        }

    def get_session(self, node_id: str) -> _NodeSession:
        """Return (or create) the session for a node_id."""
        with self._session_lock:
            if node_id not in self._sessions:
                self._sessions[node_id] = _NodeSession(node_id)
            return self._sessions[node_id]

    def _migrate_identity(self, identity: str, to_node_id: str) -> None:
        """When a known identity is active at a new node, bring their history."""
        with self._session_lock:
            for nid, sess in self._sessions.items():
                if nid != to_node_id and sess.active_identity and \
                        sess.active_identity.lower() == identity.lower():
                    target = self._sessions.setdefault(to_node_id, _NodeSession(to_node_id))
                    if not target.turns:  # don't overwrite if new node has its own history
                        target.turns = list(sess.turns)
                    sess.active_identity = None  # person left that node
                    break

    def handle_message(self, message: str, source=None, node_id: str = "", presence_context: dict | None = None):
        """Session-aware message handler. Loads/saves per-node context."""
        session_key = node_id or source or "default"
        session = self.get_session(session_key)
        with session.lock:
            # Load this node's context into instance state
            self.active_identity = session.active_identity
            self.turns = list(session.turns)
            prev_identity = self.active_identity
            try:
                with self._llm_semaphore:
                    result = self._handle_message_impl(
                        message,
                        source=source,
                        presence_context=presence_context,
                    )
                    self._ensure_result_provenance(result, message)
            finally:
                # Save context back
                session.active_identity = self.active_identity
                session.turns = self.turns
            # If identity was newly established, migrate any existing session
            new_identity = self.active_identity
            if new_identity and new_identity != prev_identity:
                self._migrate_identity(new_identity, session_key)
            return result

    def _handle_message_impl(self, message: str, source=None, presence_context: dict | None = None):
        actions = []
        source = _normalize_source(source)
        if _is_wakeword_only(message):
            turn = {
                "message": message,
                "source": source,
                "route": {
                    "ok": True,
                    "route": "wakeword",
                    "confidence": 1.0,
                    "reason": "wakeword only",
                    "attempts": 0,
                },
                "response": WAKEWORD_RESPONSE,
                "active_identity": self.active_identity,
                "actions": actions,
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        if _asks_to_stop_chat(message):
            turn = {
                "message": message,
                "source": source,
                "route": {
                    "ok": True,
                    "route": "general_question",
                    "confidence": 1.0,
                    "reason": "asks assistant to stop talking",
                    "attempts": 0,
                    "deterministic": True,
                },
                "response": "Okay.",
                "active_identity": self.active_identity,
                "actions": actions,
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        if _asks_about_previous_answer_source(message):
            response = self.answer_source_provenance(message)
            turn = {
                "message": message,
                "source": source,
                "route": {
                    "ok": True,
                    "route": "provenance_question",
                    "confidence": 1.0,
                    "reason": "asks how previous answer was determined",
                    "attempts": 0,
                },
                "response": response,
                "active_identity": self.active_identity,
                "actions": actions,
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        module_block = self._source_node_module_block(message, source, presence_context)
        if module_block is not None:
            route = {
                "ok": True,
                "route": "capability_unavailable",
                "confidence": 1.0,
                "reason": module_block.get("reason", "source node lacks required module"),
                "attempts": 0,
                "deterministic": True,
                "required_module": module_block.get("required_module"),
                "source_node": module_block.get("source_node"),
            }
            turn = {
                "message": message,
                "source": source,
                "route": route,
                "response": module_block["message"],
                "active_identity": self.active_identity,
                "actions": actions,
                "answer_provenance": self.build_answer_provenance(message, route, None),
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        state = self.scout_state()

        if _looks_like_scout_action(message):
            if _source_is_scout(source):
                action_result = self.handle_scout_action(message, state)
                if action_result is not None:
                    actions.append({
                        "name": action_result.get("action", "scout_action"),
                        "ok": bool(action_result.get("ok")),
                        "result": action_result,
                    })
                    route = {
                        "ok": True,
                        "route": "direction",
                        "confidence": 1.0,
                        "reason": "deterministic scout-direct command",
                        "attempts": 0,
                    }
                    turn = {
                        "message": message,
                        "source": source,
                        "route": route,
                        "response": action_result.get("message") or action_result.get("error") or "Done.",
                        "active_identity": self.active_identity,
                        "actions": actions,
                        "answer_provenance": self.build_answer_provenance(message, route, state),
                    }
                    self.turns.append(turn)
                    self.turns = self.turns[-30:]
                    return {"ok": True, **turn}
            elif _looks_like_scout_hardware_command(message):
                route = {
                    "ok": True,
                    "route": "direction",
                    "confidence": 1.0,
                    "reason": "scout command issued from non-scout node",
                    "attempts": 0,
                }
                turn = {
                    "message": message,
                    "source": source,
                    "route": route,
                    "response": "That command needs to be issued directly to Scout.",
                    "active_identity": self.active_identity,
                    "actions": actions,
                }
                self.turns.append(turn)
                self.turns = self.turns[-30:]
                return {"ok": True, **turn}

        # Pending-confirmation flow: previous turn was waiting for yes/no on a route
        if self.turns:
            pending = self.turns[-1].get("pending_confirmation")
            if pending is not None:
                return self._handle_confirmation(message, pending, state, source, presence_context=presence_context)

        feedback = self.classify_response_feedback(message)
        if feedback.get("ok") and feedback.get("is_feedback"):
            feedback_kind = feedback.get("kind", "response_lesson")
            if feedback_kind == "temperature_update":
                result = self.record_temperature_update(feedback.get("temperature", 0.6))
                action_name = "update_response_temperature"
            elif feedback_kind == "personality_update":
                result = self.record_personality_update(feedback.get("personality_update", {}))
                action_name = "update_response_behavior"
            else:
                result = self.record_response_lesson(feedback["lesson"])
                action_name = "learn_response_lesson"
            actions.append({"name": action_name, "ok": bool(result.get("ok")), "result": result})
            response = self.answer_feedback_learned(message, state, feedback, result)
            route = {
                "ok": True,
                "route": "direction",
                "confidence": feedback.get("confidence", 0.0),
                "reason": "response correction feedback",
                "attempts": feedback.get("attempts", 0),
                "feedback": feedback,
            }
            turn = {
                "message": message,
                "source": source,
                "route": route,
                "response": response,
                "active_identity": self.active_identity,
                "actions": actions,
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        keywords = self.extract_message_keywords(message, presence_context)
        if keywords.get("people") or keywords.get("nodes"):
            state = dict(state)
            state["_keywords"] = keywords

        route = self.route_message(message, state)
        route.update(_presence_route_context(presence_context))

        scout_action_allowed = (
            route.get("ok")
            and route.get("route") == "direction"
            and _targets_scout_action(message, source)
        )
        local_node_command = (
            route.get("ok")
            and route.get("route") == "direction"
            and _looks_like_scout_action(message)
            and not scout_action_allowed
        )

        # New low-confidence phrase: ask for confirmation before learning.
        if (
            route.get("ok")
            and not route.get("from_cache")
            and not route.get("deterministic")
            and float(route.get("confidence") or 0.0) < 0.88
            and not scout_action_allowed
            and not local_node_command
        ):
            desc = _route_description(route)
            confirm_text = f"I think you mean {desc}. Is that right?"
            turn = {
                "message": message,
                "source": source,
                "route": route,
                "response": confirm_text,
                "active_identity": self.active_identity,
                "actions": actions,
                "pending_confirmation": {
                    "original_message": message,
                    "source": source,
                    "inferred_route": route,
                    "route_description": desc,
                },
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        response = self._dispatch_route(message, route, state, actions, source=source)
        turn = {
            "message": message,
            "source": source,
            "route": route,
            "response": response,
            "active_identity": self.active_identity,
            "actions": actions,
            "answer_provenance": self.build_answer_provenance(message, route, state),
        }
        self.turns.append(turn)
        self.turns = self.turns[-30:]
        return {"ok": True, **turn}

    def _dispatch_route(self, message: str, route: dict, state: dict, actions: list, source: str | None = None) -> str:
        """Execute the appropriate handler for an already-determined route."""
        if not route.get("ok"):
            return self.generate_response(
                "routing_error", message, state, {"route": route}, max_tokens=100
            )

        introduced_name = _extract_introduction_name(message)

        if route["route"] == "direction" and introduced_name and not state.get("ok"):
            return self.generate_response(
                "identity_binding_blocked", message, state,
                {"introduced_name": introduced_name, "reason": "scout tracking is unavailable",
                 "identity_was_saved": False},
                max_tokens=120,
            )

        if route["route"] == "direction" and introduced_name and _visible_learning_subject_count(state) > 0:
            result = self.learn_face(introduced_name)
            actions.append({"name": "learn_face", "ok": bool(result.get("ok")), "result": result})
            if result.get("ok") and result.get("identity"):
                self.active_identity = _safe_identity(result["identity"])
                self.ensure_person(self.active_identity, display_name=introduced_name)
                return self.generate_response(
                    "identity_binding_success", message, state,
                    {"introduced_name": introduced_name, "saved_identity": self.active_identity,
                     "learn_face_result": result},
                    max_tokens=120,
                )
            return self.generate_response(
                "identity_binding_failed", message, state,
                {"introduced_name": introduced_name, "learn_face_result": result,
                 "identity_was_saved": False},
                max_tokens=120,
            )

        if route["route"] == "direction" and introduced_name:
            return self.generate_response(
                "identity_binding_blocked", message, state,
                {"introduced_name": introduced_name,
                 "reason": "no visible person or face is available to bind",
                 "identity_was_saved": False},
                max_tokens=120,
            )

        if route["route"] == "direction":
            if _looks_like_scout_action(message):
                if not _targets_scout_action(message, source):
                    return "That command needs a camera or Scout hardware on the node you are talking to. Say it with 'from Scout' if you want Scout to do it."
                action_result = self.handle_scout_action(message, state)
                if action_result is not None:
                    actions.append({
                        "name": action_result.get("action", "scout_action"),
                        "ok": bool(action_result.get("ok")),
                        "result": action_result,
                    })
                    return action_result.get("message") or action_result.get("error") or "Done."

        if route["route"] == "analyze_vision":
            result = self.analyze_scene(message, state)
            actions.append({"name": "analyze_vision", "ok": bool(result.get("ok")), "result": result})
            response = result.get("answer") or result.get("summary")
            if not response:
                response = self.generate_response(
                    "vision_analysis_unavailable", message, state,
                    {"vision_result": result}, max_tokens=120,
                )
            return response

        if route["route"] == "greeting":
            return self.answer_greeting(message, state)

        if route.get("route") == "self_question":
            return self.answer_self_question(message, state, route)

        if route["route"] == "direction" and _asks_to_remember(message):
            parsed = _parse_simple_memory(message)
            if parsed and (self.active_identity or parsed.get("identity")):
                identity = parsed.get("identity") or self.active_identity
                result = self.remember(
                    identity, parsed["type"], parsed["key"], parsed["value"], source="chat"
                )
                actions.append({"name": parsed["type"], "ok": bool(result.get("ok")), "result": result})
                return self.generate_response(
                    "memory_saved", message, state, {"memory_result": result}, max_tokens=100
                )
            return self.generate_response(
                "memory_needs_clarification", message, state,
                {"active_identity": self.active_identity, "parsed_memory": parsed},
                max_tokens=120,
            )

        return self.answer_with_context(message, state)

    def _handle_confirmation(
        self,
        message: str,
        pending: dict,
        state: dict,
        source: str,
        presence_context: dict | None = None,
    ) -> dict:
        """Handle a yes/no/correction reply to a pending route confirmation."""
        import deterministic_router as _dr
        actions = []
        original_message = pending["original_message"]
        inferred_route = pending["inferred_route"]

        if _is_affirmative(message):
            _dr.learn(original_message, inferred_route)
            original_source = pending.get("source") or source
            response = self._dispatch_route(original_message, inferred_route, state, actions, source=original_source)
            turn = {
                "message": message,
                "source": source,
                "route": inferred_route,
                "response": response,
                "active_identity": self.active_identity,
                "actions": actions,
                "answer_provenance": self.build_answer_provenance(original_message, inferred_route, state),
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

        correction = _extract_correction(message) or _presence_correction(presence_context)
        if correction:
            corrected_input = _corrected_route_input(original_message, correction, presence_context)
            corrected_route = self.route_message(corrected_input, state)
            corrected_route.update(_presence_route_context(presence_context))
            if corrected_route.get("ok"):
                if isinstance(presence_context, dict) and presence_context.get("clarification"):
                    _dr.learn(original_message, corrected_route)
                    response = self._dispatch_route(original_message, corrected_route, state, actions, source=source)
                    turn = {
                        "message": message,
                        "source": source,
                        "route": corrected_route,
                        "response": response,
                        "active_identity": self.active_identity,
                        "actions": actions,
                        "routing_feedback": presence_context.get("routing_feedback"),
                        "answer_provenance": self.build_answer_provenance(original_message, corrected_route, state),
                    }
                    self.turns.append(turn)
                    self.turns = self.turns[-30:]
                    return {"ok": True, **turn}
                corrected_desc = _route_description(corrected_route, correction)
                confirm_text = f"I think you mean {corrected_desc}. Is that right?"
                turn = {
                    "message": message,
                    "source": source,
                    "route": corrected_route,
                    "response": confirm_text,
                    "active_identity": self.active_identity,
                    "actions": actions,
                    "pending_confirmation": {
                        "original_message": original_message,
                        "source": source,
                        "inferred_route": corrected_route,
                        "route_description": corrected_desc,
                        "route_correction": correction,
                        "corrected_route_input": corrected_input,
                    },
                }
                self.turns.append(turn)
                self.turns = self.turns[-30:]
                return {"ok": True, **turn}

        # Plain denial — cancel without learning
        turn = {
            "message": message,
            "source": source,
            "route": {"ok": True, "route": "general_question", "confidence": 0.0,
                      "reason": "confirmation denied", "attempts": 0},
            "response": "Got it, won't save that.",
            "active_identity": self.active_identity,
            "actions": actions,
        }
        self.turns.append(turn)
        self.turns = self.turns[-30:]
        return {"ok": True, **turn}

    def extract_message_keywords(self, message: str, presence_context: dict | None = None) -> dict:
        """Extract known people names and node names mentioned in *message*.

        Combines keywords sent by the node (presence_context["keywords"]) with
        vault-local knowledge (people_dir listing, registered node IDs).
        """
        text = str(message or "").casefold()
        node_kw = (presence_context or {}).get("keywords") or {}
        people: list[str] = list(node_kw.get("people") or [])
        nodes: list[str] = list(node_kw.get("nodes") or [])

        # Vault-side: known people from people_dir
        try:
            for entry in sorted(self.people_dir.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    name = entry.name.casefold()
                    if name not in people and _phrase_in_text(name, text):
                        people.append(name)
        except Exception:
            pass

        # Vault-side: registered node IDs
        try:
            reg = self.node_registry.registered_nodes() if self.node_registry else {}
            for nid in reg:
                nid_norm = nid.casefold().replace("_", " ")
                if nid_norm not in nodes and _phrase_in_text(nid_norm, text):
                    nodes.append(nid_norm)
        except Exception:
            pass

        return {
            "people": sorted(set(people)),
            "nodes": sorted(set(nodes)),
        }

    def route_message(self, message: str, state: dict):
        import deterministic_router as _dr
        fast = _fast_route_message(message)
        if fast is not None:
            return fast
        cached = _dr.lookup(message)
        if cached is not None:
            return {**cached, "from_cache": True}

        prompt = self._route_prompt(message, state)
        raw_attempts = []
        try:
            raw = self._generate_route_response(prompt, structured=True)
            raw_attempts.append(raw)
            data = _coerce_route_data(raw)
            route = data.get("route") if isinstance(data, dict) else None
            if route not in ROUTE_OPTIONS:
                repair_prompt = self._route_repair_prompt(message, raw)
                raw = self._generate_route_response(repair_prompt, structured=False)
                raw_attempts.append(raw)
                data = _coerce_route_data(raw)
                route = data.get("route") if isinstance(data, dict) else None
        except Exception as exc:
            return {"ok": False, "route": None, "error": f"route model unavailable: {exc}"}
        if route not in ROUTE_OPTIONS:
            return {
                "ok": False,
                "route": None,
                "error": "route model returned an invalid route after retry",
                "raw_attempts": raw_attempts,
            }
        confidence = data.get("confidence", 0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "ok": True,
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason", ""))[:240],
            "attempts": len(raw_attempts),
        }

    def _generate_route_response(self, prompt: str, structured: bool, route_options=None):
        route_options = route_options or ROUTE_OPTIONS
        response_format = None
        if structured:
            response_format = {
                "type": "object",
                "properties": {
                    "route": {"type": "string", "enum": sorted(route_options)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["route", "confidence", "reason"],
            }
        return self.route_model.generate(
            prompt,
            response_format=response_format,
            timeout=30,
            allow_empty=True,
        )

    def _route_prompt(self, message: str, state: dict):
        return f"""
Classify the user's message for a scout-vault assistant.

Return only compact JSON with this schema:
{{"route":"greeting","confidence":0.0,"reason":"short"}}

Allowed route values:
- greeting: greetings and social openings that do not ask a factual, visual,
  self, or action question.
- general_question: questions about the world, facts, conversation, or anything
  that does not request a change/action and is not about the live camera image.
- analyze_vision: questions about what the scout sees, the camera image, visual
  scene, objects in front of the scout, visible people, colors, room layout, or
  visual recognition from the current snapshot.
- self_question: questions about the assistant's name, identity, personality,
  software, hardware, memory, sensors, capabilities, or current recognition
  state. This includes service health, runtime health, status, current model
  stack, and whether brain/scout services are available.
- direction: instructions or requests to do something, including movement,
  looking, learning a face/name, remembering a fact/preference, changing a
  setting, or using a capability.

Rules:
- Pick exactly one allowed route.
- Do not invent capabilities, identities, people, or detections.
- If the message is ambiguous, pick the closest route and keep confidence low.
- Greetings are greeting only when the user is socially opening the
  conversation. Words like "healthy", "status", "running", "available",
  "services", "models", "hardware", or "software" make it a self_question,
  not a greeting.
- Output JSON only. No markdown, no explanation outside JSON.

Examples:
- "hello" -> {{"route":"greeting","confidence":0.9,"reason":"greeting"}}
- "good morning" -> {{"route":"greeting","confidence":0.9,"reason":"greeting"}}
- "what's your name" -> {{"route":"self_question","confidence":0.95,"reason":"asks assistant identity"}}
- "tell me about your hardware" -> {{"route":"self_question","confidence":0.95,"reason":"asks assistant hardware"}}
- "what can you do" -> {{"route":"self_question","confidence":0.95,"reason":"asks assistant capabilities"}}
- "what do you want" -> {{"route":"self_question","confidence":0.85,"reason":"asks assistant preferences or goals"}}
- "how are you" -> {{"route":"self_question","confidence":0.75,"reason":"asks assistant state"}}
- "are your services healthy" -> {{"route":"self_question","confidence":0.95,"reason":"asks runtime/service status"}}
- "is everything running" -> {{"route":"self_question","confidence":0.9,"reason":"asks status"}}
- "what do you see" -> {{"route":"analyze_vision","confidence":0.95,"reason":"asks about live camera scene"}}
- "is anyone in front of you" -> {{"route":"analyze_vision","confidence":0.9,"reason":"asks about visible people"}}
- "I am Chris" -> {{"route":"direction","confidence":0.85,"reason":"introduces a person to learn"}}
- "remember that I like blue" -> {{"route":"direction","confidence":0.9,"reason":"asks to store memory"}}
- "go forward" -> {{"route":"direction","confidence":0.95,"reason":"movement instruction"}}

Assistant identity memory:
{json.dumps(self.identity_profile, indent=2)}

Rover tracking available: {bool(state.get("ok"))}
Tracker summary:
{_tracking_summary(state)}{_keyword_context_section(state)}

User message:
{message}
"""

    def _route_repair_prompt(self, message: str, invalid_response: str):
        options = ", ".join(sorted(ROUTE_OPTIONS))
        return f"""
Your previous route response was invalid.

Allowed route values are exactly: {options}

Convert the user's message to one allowed route. Return only compact JSON with this schema:
{{"route":"greeting","confidence":0.0,"reason":"short"}}

Do not return markdown. Do not return prose. The route field must be one of the
allowed route values exactly.

User message:
{message}

Invalid previous response:
{invalid_response}
"""

    def classify_self_question(self, message: str, state: dict):
        prompt = self._self_route_prompt(message, state)
        raw_attempts = []
        try:
            raw = self._generate_route_response(prompt, structured=True, route_options=SELF_ROUTE_OPTIONS)
            raw_attempts.append(raw)
            data = _coerce_route_data(raw, SELF_ROUTE_OPTIONS)
            route = data.get("route") if isinstance(data, dict) else None
            if route not in SELF_ROUTE_OPTIONS:
                repair_prompt = self._self_route_repair_prompt(message, raw)
                raw = self._generate_route_response(repair_prompt, structured=False, route_options=SELF_ROUTE_OPTIONS)
                raw_attempts.append(raw)
                data = _coerce_route_data(raw, SELF_ROUTE_OPTIONS)
                route = data.get("route") if isinstance(data, dict) else None
            if route in SELF_ROUTE_OPTIONS and not self._self_route_selection_is_valid(message, route):
                repair_prompt = self._self_route_repair_prompt(
                    message,
                    f"{raw}\nValidator rejected route {route!r} for this message.",
                )
                raw = self._generate_route_response(repair_prompt, structured=False, route_options=SELF_ROUTE_OPTIONS)
                raw_attempts.append(raw)
                data = _coerce_route_data(raw, SELF_ROUTE_OPTIONS)
                route = data.get("route") if isinstance(data, dict) else None
        except Exception as exc:
            return {"ok": False, "route": None, "error": f"self-route model unavailable: {exc}"}
        if route not in SELF_ROUTE_OPTIONS:
            return {
                "ok": False,
                "route": None,
                "error": "self-route model returned an invalid route after retry",
                "raw_attempts": raw_attempts,
            }
        confidence = data.get("confidence", 0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "ok": True,
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason", ""))[:240],
            "attempts": len(raw_attempts),
        }

    def _self_route_selection_is_valid(self, message: str, route: str):
        prompt = f"""
Validate this self-question route selection.

Allowed self-question route values:
- assistant_identity: assistant name, who the assistant is, creator, role.
- user_identity: current user's recognized identity, whether the assistant
  knows or recognizes the user.
- personality: assistant tone, temperament, humor, attitude.
- hardware: physical components, vault PC, scout, GPU, camera, motors, sensors.
- software: code, services, APIs, models, routing, architecture.
- status: current health, active identity, services running, errors.
- capabilities: actions/tools the assistant can perform.
- memory: remembered facts, preferences, person memory, vector DB.
- sensors: sensing inputs and how sensing works.
- goals: purpose, priorities, what the assistant wants/is here to do.
- other: does not fit above.

Rules:
- Return only compact JSON: {{"valid":true,"reason":"short"}}
- Mark invalid if pronouns point to a different subject than the route.
- "who are you" is assistant_identity, not user_identity.
- "what's your name" is assistant_identity, not user_identity.
- "how are you" is status or personality, not user_identity.
- "what can you do" is capabilities, not user_identity.
- "tell me about your hardware" is hardware, not capabilities.
- "what hardware is on the brain" is hardware, not capabilities.
- "what do you want" is goals, not user_identity.

User message:
{message}

Proposed route:
{route}
"""
        raw = self.route_model.generate(
            prompt,
            response_format={
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["valid", "reason"],
            },
            timeout=30,
            allow_empty=True,
        )
        data = _extract_json_object(raw)
        return bool(data.get("valid")) if isinstance(data, dict) else True

    def _self_route_prompt(self, message: str, state: dict):
        return f"""
Classify this self-question for a scout-vault assistant.

User message to classify:
{message}

Return only compact JSON with this schema:
{{"route":"assistant_identity","confidence":0.0,"reason":"short"}}

Allowed self-question route values:
- assistant_identity: questions about the assistant's name, who it is, creator,
  role, or identity profile.
- user_identity: questions about the current user's recognized identity, whether
  the assistant knows them, or whether the assistant recognizes them.
- personality: questions about the assistant's tone, temperament, preferences,
  humor, attitude, or configured personality.
- hardware: questions about physical components, robot body, vault PC, rover,
  GPU, camera, microphone, speakers, motors, sensors as physical devices, or
  hardware stack.
- software: questions about code, services, APIs, models, routing, memory
  software, operating system services, inference stack, or architecture.
- status: questions about current health, active identity, service state,
  tracking state, errors, availability, or what is currently running.
- capabilities: questions about what the assistant can do, actions it can take,
  tools it has, commands it can perform, or what it can help with.
- memory: questions about long-term memory, remembered facts, preferences,
  person memory, face references, or vector DB memory.
- sensors: questions about what sensors are available or how sensing works.
- goals: questions about what the assistant wants, its priorities, purpose, or
  operating goals.
- other: self-questions that do not fit the categories above.

Rules:
- Pick exactly one allowed self-question route.
- Infer the user's intent from the whole message.
- Pronoun direction matters:
  "you", "your", "yourself" usually refers to the assistant;
  "me", "my", "I", "am I" usually refers to the current user.
- "who are you", "what are you", and "tell me about yourself" are
  assistant_identity, not user_identity.
- "who am I", "do you know me", "do you recognize me", and "what do you know
  about me" are user_identity or memory depending on whether the user asks
  recognition identity or stored memories.
- "what do you want", "what are your goals", and "what is your purpose" are
  goals, not user_identity.
- Do not invent capabilities, hardware, software, identities, or detections.
- Output JSON only. No markdown, no explanation outside JSON.

Examples:
- "what's your name" -> {{"route":"assistant_identity","confidence":0.95,"reason":"asks assistant name"}}
- "who are you" -> {{"route":"assistant_identity","confidence":0.95,"reason":"asks assistant identity"}}
- "what are you" -> {{"route":"assistant_identity","confidence":0.95,"reason":"asks assistant identity"}}
- "tell me about yourself" -> {{"route":"assistant_identity","confidence":0.9,"reason":"asks assistant self-description"}}
- "who am I" -> {{"route":"user_identity","confidence":0.95,"reason":"asks current recognized identity"}}
- "do you know me" -> {{"route":"user_identity","confidence":0.9,"reason":"asks current user recognition"}}
- "tell me about your hardware" -> {{"route":"hardware","confidence":0.95,"reason":"asks hardware stack"}}
- "what models are you using" -> {{"route":"software","confidence":0.9,"reason":"asks inference model stack"}}
- "are your services healthy" -> {{"route":"status","confidence":0.9,"reason":"asks current runtime status"}}
- "what can you do" -> {{"route":"capabilities","confidence":0.95,"reason":"asks available actions"}}
- "what do you remember about me" -> {{"route":"memory","confidence":0.9,"reason":"asks stored person memory"}}
- "what sensors do you have" -> {{"route":"sensors","confidence":0.95,"reason":"asks sensors"}}
- "what do you want" -> {{"route":"goals","confidence":0.85,"reason":"asks operating priorities"}}

Classify only this user message, not the examples:
{message}
"""

    def _self_route_repair_prompt(self, message: str, invalid_response: str):
        options = ", ".join(sorted(SELF_ROUTE_OPTIONS))
        return f"""
Your previous self-question route response was invalid.

Allowed self-question route values are exactly: {options}

Convert the user's message to one allowed self-question route. Return only
compact JSON with this schema:
{{"route":"assistant_identity","confidence":0.0,"reason":"short"}}

Do not return markdown. Do not return prose. The route field must be one of the
allowed self-question route values exactly.

User message:
{message}

Invalid previous response:
{invalid_response}
"""

    def classify_response_feedback(self, message: str):
        if _looks_like_ownership_question(message) or _asks_registry_source_followup(message):
            return {
                "ok": True,
                "is_feedback": False,
                "confidence": 1.0,
                "reason": "question, not response feedback",
                "attempts": 0,
            }
        direct_temperature = _extract_temperature_setting(message)
        if direct_temperature is not None:
            return {
                "ok": True,
                "is_feedback": True,
                "kind": "temperature_update",
                "temperature": direct_temperature,
                "confidence": 1.0,
                "reason": "explicit temperature setting",
                "attempts": 0,
            }
        direct_personality = _extract_direct_personality_update(message)
        if direct_personality is not None:
            return {
                "ok": True,
                "is_feedback": True,
                "kind": "personality_update",
                "personality_update": direct_personality,
                "confidence": 1.0,
                "reason": "explicit tone/behavior directive",
                "attempts": 0,
            }
        direct_lesson = _extract_direct_response_lesson(message, self.recent_turns_for_feedback(limit=1))
        if direct_lesson is not None:
            return {
                "ok": True,
                "is_feedback": True,
                "kind": "response_lesson",
                "lesson": direct_lesson,
                "confidence": 1.0,
                "reason": "explicit response preference",
                "attempts": 0,
            }
        prompt = f"""
Decide whether the user message teaches the assistant how to respond or behave.
This includes feedback/corrections about the previous response, personality/
behavior changes, and generation temperature changes.

Return only compact JSON with this schema:
{{
  "is_feedback": false,
  "kind": "response_lesson",
  "confidence": 0.0,
  "reason": "short",
  "temperature": 0.0,
  "personality_update": {{
    "preference": "short behavior rule",
    "applies_when": "when to apply it",
    "avoid": "what to avoid",
    "prefer": "what to do instead"
  }},
  "lesson": {{
    "scope": "hardware",
    "preference": "short actionable rule",
    "applies_when": "condition for future answers",
    "avoid": "what to avoid",
    "prefer": "what to do instead"
  }}
}}

Kinds:
- response_lesson: correction about any answer, route, vision answer, action
  acknowledgement, source selection, data provenance, or future answer style.
- personality_update: user tells the assistant to change tone/behavior, such as
  be more blunt, less rude, warmer, shorter, more formal, less theatrical.
- temperature_update: user asks to set/change response temperature or
  randomness/creativity level.

Teaching examples:
- "I only asked about the GPU" means future answers should answer only the
  requested component when the question names one component.
- "too much detail" means be more concise for similar questions.
- "don't mention the whole stack" means avoid broad context unless asked.
- "that's not what I asked" means infer the mismatch and store a precision rule.
- "use the node registry next time" means store a source_selection rule.
- "for active nodes check registered nodes" means use live NodeRegistry data
  rather than static self-knowledge for similar questions.
- "be less rude" is personality_update.
- "be more sarcastic" is personality_update.
- "talk warmer to unknown users" is personality_update.
- "set your temperature to 0.3" is temperature_update with temperature 0.3.
- "be more creative" can be temperature_update if phrased as generation
  randomness, otherwise personality_update.

Not feedback examples:
- New factual questions.
- Greetings.
- Requests to move, look, or analyze vision, unless phrased as a correction or
  future preference such as "when I ask what you see, don't include tracker info".
- Explicit facts/preferences about the user, unless they correct answer style
  or assistant behavior.

Recent turns:
{json.dumps(self.recent_turns_for_feedback(), indent=2, default=str)}

User message:
{message}
"""
        raw_attempts = []
        try:
            raw = self.route_model.generate(
                prompt,
                response_format={
                    "type": "object",
                    "properties": {
                        "is_feedback": {"type": "boolean"},
                        "kind": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                        "temperature": {"type": "number"},
                        "personality_update": {
                            "type": "object",
                            "properties": {
                                "preference": {"type": "string"},
                                "applies_when": {"type": "string"},
                                "avoid": {"type": "string"},
                                "prefer": {"type": "string"},
                            },
                            "required": ["preference", "applies_when", "avoid", "prefer"],
                        },
                        "lesson": {
                            "type": "object",
                            "properties": {
                                "scope": {"type": "string"},
                                "preference": {"type": "string"},
                                "applies_when": {"type": "string"},
                                "avoid": {"type": "string"},
                                "prefer": {"type": "string"},
                            },
                            "required": ["scope", "preference", "applies_when", "avoid", "prefer"],
                        },
                    },
                    "required": ["is_feedback", "kind", "confidence", "reason", "temperature", "personality_update", "lesson"],
                },
                timeout=30,
                allow_empty=True,
            )
            raw_attempts.append(raw)
            data = _extract_json_object(raw)
        except Exception as exc:
            return {"ok": False, "is_feedback": False, "error": str(exc), "attempts": len(raw_attempts)}
        if not isinstance(data, dict):
            return {"ok": True, "is_feedback": False, "confidence": 0.0, "attempts": len(raw_attempts)}
        is_feedback = bool(data.get("is_feedback"))
        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        kind = str(data.get("kind") or "response_lesson")
        lesson = data.get("lesson") if isinstance(data.get("lesson"), dict) else {}
        personality_update = data.get("personality_update") if isinstance(data.get("personality_update"), dict) else {}
        temperature = data.get("temperature")
        if not is_feedback or confidence < 0.7:
            return {
                "ok": True,
                "is_feedback": False,
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(data.get("reason", ""))[:240],
                "attempts": len(raw_attempts),
            }
        if kind == "temperature_update":
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                return {"ok": True, "is_feedback": False, "confidence": confidence, "reason": "missing temperature", "attempts": len(raw_attempts)}
            return {
                "ok": True,
                "is_feedback": True,
                "kind": "temperature_update",
                "temperature": max(0.0, min(1.2, temperature)),
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(data.get("reason", ""))[:240],
                "attempts": len(raw_attempts),
            }
        if kind == "personality_update":
            if not personality_update.get("preference"):
                return {"ok": True, "is_feedback": False, "confidence": confidence, "reason": "missing behavior preference", "attempts": len(raw_attempts)}
            return {
                "ok": True,
                "is_feedback": True,
                "kind": "personality_update",
                "personality_update": {
                    "preference": str(personality_update.get("preference", ""))[:500],
                    "applies_when": str(personality_update.get("applies_when", ""))[:500],
                    "avoid": str(personality_update.get("avoid", ""))[:500],
                    "prefer": str(personality_update.get("prefer", ""))[:500],
                    "source_message": message,
                },
                "confidence": max(0.0, min(1.0, confidence)),
                "reason": str(data.get("reason", ""))[:240],
                "attempts": len(raw_attempts),
            }
        if not lesson.get("preference"):
            return {"ok": True, "is_feedback": False, "confidence": confidence, "reason": "missing response lesson", "attempts": len(raw_attempts)}
        lesson = {
            "scope": str(lesson.get("scope", "response_style"))[:80],
            "preference": str(lesson.get("preference", ""))[:500],
            "applies_when": str(lesson.get("applies_when", ""))[:500],
            "avoid": str(lesson.get("avoid", ""))[:500],
            "prefer": str(lesson.get("prefer", ""))[:500],
            "source_message": message,
            "source_turn": self.recent_turns_for_feedback(limit=1)[-1] if self.recent_turns_for_feedback(limit=1) else None,
        }
        return {
            "ok": True,
            "is_feedback": True,
            "kind": "response_lesson",
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason", ""))[:240],
            "lesson": lesson,
            "attempts": len(raw_attempts),
        }

    def answer_feedback_learned(self, message: str, state: dict, feedback: dict, result: dict):
        # Deterministic short-circuit for the common shapes -- the chat model
        # tends to over-explain ("Adjusted behavior settings accordingly...").
        kind = feedback.get("kind")
        if kind == "personality_update":
            pref = (feedback.get("personality_update") or {}).get("preference") or ""
            if pref:
                # Echo the directive back tersely in voice.
                return f"Noted. {pref.rstrip('.').capitalize()} from here on."
            return "Noted."
        if kind == "temperature_update":
            temp = feedback.get("temperature")
            return f"Temperature set to {temp}."
        prompt = f"""
Generate a one-line acknowledgement that the assistant absorbed a response
correction.

Facts:
{json.dumps({
    "feedback": feedback,
    "saved": result,
    "identity_context": self.response_identity_context(state),
}, indent=2, default=str)}

Rules:
- Sound like Luhkas, not a CRM ticket.
- One short sentence, max twelve words.
- Do not say "learned", "adjusted", "saved", "settings", "directive", "per",
  or describe the storage mechanism.
- Do not use emojis.
- Do not repeat the user's correction verbatim.
"""
        try:
            return self.generate_guarded_response(
                "feedback_learned",
                prompt,
                state,
                options={"num_predict": 90, "temperature": 0.45, "top_p": 0.9},
            )
        except Exception:
            return "Got it. I will be more precise about the specific thing you asked for."

    def answer_self_question(self, message: str, state: dict, route: dict | None = None):
        self_route = (route or {}).get("self_route")
        if not isinstance(self_route, dict):
            self_route = self.classify_self_question(message, state)
        if route is not None:
            route["self_route"] = self_route
        fast_answer = self.fast_self_answer(message, state, self_route)
        if fast_answer is not None:
            return fast_answer
        if self_route.get("ok") and self_route.get("route") == "user_identity":
            return self.identity_status(state)
        capabilities = self.capabilities()
        identity_context = self.response_identity_context(state)
        response_context = self.response_context(state)
        self_route_name = self_route.get("route") if self_route.get("ok") else "other"
        facts = {
            "identity_memory": self.identity_profile,
            "capabilities": capabilities,
            "self_knowledge": self.self_knowledge_for_route(self_route_name),
            "registered_nodes": self.registered_nodes_snapshot(),
            "source_lessons": self.source_lessons(),
            "recent_self_answers": self.recent_self_answers(),
            "live_status": self.live_status_facts(state),
            "identity_debug": self.identity_debug(state),
            "identity_context": identity_context,
            "self_route": self_route,
        }
        prompt = f"""Answer this self-question using only the facts below.
Self-route: {self_route_name}
User: {message}

Facts:
{json.dumps({
    "identity": {
        "name": self.identity_profile.get("name"),
        "role": self.identity_profile.get("role"),
        "creator": self.identity_profile.get("creator"),
    },
    "self_knowledge": facts["self_knowledge"],
    "registered_nodes": facts["registered_nodes"],
    "source_lessons": facts["source_lessons"][-5:],
    "live_status": facts["live_status"],
    "identity_context": facts["identity_context"],
}, separators=(",", ":"), default=str)}

Rules: 1-2 short sentences. First person. No emojis. No generic closer.
Use registered_nodes for current/active/registered node questions.
Do not invent missing hardware, software, sensors, people, or capabilities.
Do not mention provenance unless asked.
"""
        try:
            return self.generate_guarded_response(
                "self_question",
                prompt,
                state,
                options={"num_predict": 100, "temperature": 0.5, "top_p": 0.9},
            )
        except Exception as exc:
            return f"I cannot answer that self-question because the local chat model is unavailable: {exc}"

    def fast_self_answer(self, message: str, state: dict, self_route: dict) -> str | None:
        text = _normalize_command_text(message)
        route_name = self_route.get("route")
        if route_name == "assistant_identity" and _asks_assistant_name(text):
            return self._assistant_identity_answer()
        if route_name == "user_identity":
            identity = self.active_identity
            if identity:
                return f"You are {identity}."
            return "I don't know who you are yet."
        if route_name == "capabilities" and _asks_skill_inventory(text):
            return self._registered_skills_answer()
        if route_name == "capabilities" and _asks_capability_inventory(text):
            return self._registered_capabilities_answer()
        if route_name == "software" and _asks_stored_knowledge_owner(text):
            return "Stored knowledge belongs to Vault. Scout can witness and forward node state, but Vault owns memory, learning, and retrieval."
        if route_name == "software" and _asks_camera_action_owner(text):
            return "Camera actions belong to Scout's camera_node. Vault can route the request, but Scout owns the camera behavior."
        if route_name == "status" and _asks_registered_or_active_nodes(text):
            return self._registered_nodes_answer()
        if route_name == "status" and _asks_tracking_status(text):
            if not state.get("ok"):
                return "I couldn't read Scout's live state."
            return "Tracking is on." if state.get("tracking_enabled") else "Tracking is off."
        if route_name == "status" and _asks_pose_interval(text):
            interval = state.get("pose_interval_frames")
            if interval is None:
                return "I couldn't read the pose interval from Scout's live state."
            return f"The pose interval is set to every {interval} frames."
        if route_name == "goals" and _asks_why_here(text):
            role = self.identity_profile.get("role") or "the assistant Chris created"
            return f"I'm here as {role}: to help with tasks, remember useful things, and work through the vault and connected nodes."
        if route_name == "hardware":
            return self._hardware_summary_answer()
        if route_name == "sensors":
            return self._sensors_summary_answer()
        return None

    def _assistant_identity_answer(self) -> str:
        name = self.identity_profile.get("name") or "Luhkas"
        creator = self.identity_profile.get("creator") or "Chris"
        return f"I'm {name}. {creator} built me."

    def _hardware_summary_answer(self) -> str:
        hw = (self.self_knowledge_for_route("hardware").get("records", {}) or {}).get("hardware", {}) or {}
        vault = hw.get("vault_pc", {}) or {}
        scout = hw.get("scout_edge", {}) or {}
        vault_bits = []
        gpu = (vault.get("gpu") or "")
        if "3090" in gpu:
            vault_bits.append("RTX 3090")
        elif gpu:
            vault_bits.append(gpu.split(",")[0].replace("NVIDIA GeForce ", "").strip())
        if vault.get("ram"):
            vault_bits.append(vault["ram"])
        scout_bits = []
        if scout.get("compute"):
            scout_bits.append(scout["compute"])
        if scout.get("accelerator") and "Hailo" in scout["accelerator"]:
            scout_bits.append("Hailo HAT+")
        parts = []
        if vault_bits:
            parts.append(f"Vault PC: {', '.join(vault_bits)}")
        if scout_bits:
            parts.append(f"Scout body: {', '.join(scout_bits)}")
        return ". ".join(parts) + "." if parts else "I don't have my hardware sheet loaded."

    def _sensors_summary_answer(self) -> str:
        rec = (self.self_knowledge_for_route("sensors").get("records", {}) or {}).get("sensors", {}) or {}
        items = rec.get("available_or_planned") or []
        live = [s for s in items if not str(s).lower().startswith("planned") and "possible" not in str(s).lower()]
        if not live:
            return "I don't have a sensor list loaded."
        return "Sensors: " + ", ".join(live[:6]) + "."

    def _registered_nodes_answer(self) -> str:
        snapshot = self.registered_nodes_snapshot()
        if not snapshot.get("ok"):
            return "I couldn't read the live node registry."
        nodes = snapshot.get("nodes") or {}
        if not nodes:
            return "There are no registered nodes in the live node registry right now."
        names = []
        for node_id, cfg in sorted(nodes.items()):
            name = (cfg or {}).get("node_name") or node_id
            services = (cfg or {}).get("services") or {}
            suffix = f" ({', '.join(sorted(services))})" if services else ""
            names.append(f"{name}{suffix}")
        count = len(names)
        noun = "node" if count == 1 else "nodes"
        return f"The live node registry currently shows {count} registered {noun}: {', '.join(names)}."

    def _source_node_module_block(
        self,
        message: str,
        source: str | None,
        presence_context: dict | None,
    ) -> dict | None:
        required = _required_source_module(message)
        if not required:
            return None
        node_id = _source_node_id(source, presence_context)
        module_state = self._source_node_module_state(node_id, required, presence_context)
        if module_state.get("available"):
            return None
        label = _module_label(required)
        action = _module_action_label(required)
        known_from = module_state.get("known_from") or "the live node registry"
        reason = (
            f"{node_id} does not have {required} available"
            if module_state.get("known")
            else f"I do not have a module registry entry for {required} on {node_id}"
        )
        if module_state.get("known"):
            message_text = f"I can't {action} from {node_id} because that node does not have {label}."
        else:
            message_text = f"I can't verify that {node_id} can {action} because I do not see {label} in {known_from}."
        return {
            "message": message_text,
            "required_module": required,
            "source_node": node_id,
            "reason": reason,
            "known_from": known_from,
        }

    def _source_node_module_state(
        self,
        node_id: str,
        module_name: str,
        presence_context: dict | None,
    ) -> dict:
        modules = {}
        known_from = "the live node registry"
        if isinstance(presence_context, dict):
            caps = presence_context.get("node_capabilities")
            if isinstance(caps, dict) and isinstance(caps.get("module_status"), dict):
                modules.update(caps["module_status"])
                known_from = "the source node capability report"
            if isinstance(presence_context.get("modules"), dict):
                modules.update(presence_context["modules"])
                known_from = "the source node module report"
        if not modules and self.node_registry is not None:
            info = self.node_registry.registered_nodes().get(node_id, {})
            if isinstance(info, dict):
                if isinstance(info.get("modules"), dict):
                    modules.update(info["modules"])
                caps = info.get("capabilities")
                if isinstance(caps, dict) and isinstance(caps.get("module_status"), dict):
                    modules.update(caps["module_status"])
        value = modules.get(module_name)
        if isinstance(value, dict):
            return {"known": True, "available": bool(value.get("available", True)), "known_from": known_from}
        if value is None and module_name in modules:
            return {"known": True, "available": False, "known_from": known_from}
        if isinstance(value, bool):
            return {"known": True, "available": value, "known_from": known_from}
        return {"known": False, "available": False, "known_from": known_from}

    def _registered_capabilities_answer(self) -> str:
        if self.capability_registry is None:
            return "I couldn't read the capability registry."
        capabilities = self.capability_registry.list()
        if not capabilities:
            return "The capability registry is empty right now."
        names = [
            str(cap.get("display_name") or cap.get("name") or "unnamed").replace("_", " ")
            for cap in capabilities[:8]
        ]
        extra = len(capabilities) - len(names)
        suffix = f", plus {extra} more" if extra > 0 else ""
        return f"The capability registry lists {len(capabilities)} capabilities: {', '.join(names)}{suffix}."

    def _registered_skills_answer(self) -> str:
        if self.skill_registry is None:
            return "I couldn't read the skill registry."
        skills = self.skill_registry.list()
        if not skills:
            return "The skill registry is empty right now."
        names = [
            str(skill.get("name") or skill.get("display_name") or "unnamed").replace("_", " ")
            for skill in skills[:8]
        ]
        extra = len(skills) - len(names)
        suffix = f", plus {extra} more" if extra > 0 else ""
        noun = "skill" if len(skills) == 1 else "skills"
        return f"The skill registry lists {len(skills)} {noun}: {', '.join(names)}{suffix}."

    def answer_greeting(self, message: str, state: dict):
        return self.generate_response(
            "greeting",
            message,
            state,
            {
                "identity_memory": self.identity_profile,
                "identity_context": self.response_identity_context(state),
                "instructions": [
                    "one short sentence",
                    "no camera, tracking, visible people, recognition, names, or objects",
                    "do not address the user by name or title unless identity_context.may_address_primary_user is true",
                    "do not invent a nickname or object label for the user",
                    "use a little edge if it fits",
                    "do not describe your own style or personality traits",
                    "do not sound like customer service",
                    "do not end with a generic offer to help",
                    "no repeated catchphrase",
                    "no emojis",
                ],
            },
            max_tokens=48,
            temperature=0.75,
        )

    def generate_response(
        self,
        response_type: str,
        message: str,
        state: dict,
        facts: dict | None = None,
        *,
        max_tokens=110,
        temperature=0.45,
    ):
        response_facts = dict(facts or {})
        response_facts.setdefault("identity_context", self.response_identity_context(state))
        response_facts.setdefault("response_context", self.response_context(state))
        compact_identity = {
            "name": self.identity_profile.get("name"),
            "role": self.identity_profile.get("role"),
            "creator": self.identity_profile.get("creator"),
            "primary_user": self.identity_profile.get("primary_user"),
            "primary_user_title": self.identity_profile.get("primary_user_title"),
        }
        prompt = f"""Write the final user-facing answer.
Type: {response_type}
User: {message}

Facts only:
{json.dumps({
    "identity": compact_identity,
    "route_facts": response_facts,
    "rover_tracking_available": bool(state.get("ok")),
    "active_identity": self.active_identity if state.get("ok") else None,
}, separators=(",", ":"), default=str)}

Rules: 1-2 short sentences. First person. No emojis. No customer-service closer.
Do not invent facts, detections, actions, memories, feelings, or desires.
Do not address the user by name/title unless identity_context permits it.
"""
        try:
            return self.generate_guarded_response(
                response_type,
                prompt,
                state,
                options=self.chat_options({"num_predict": max_tokens, "temperature": temperature}),
            )
        except Exception as exc:
            return f"My response model is unavailable: {exc}"

    def response_identity_context(self, state: dict):
        active_identity = self.active_identity if state.get("ok") else None
        primary_user = self.identity_profile.get("primary_user")
        primary_user_title = self.identity_profile.get("primary_user_title")
        active_matches_primary = bool(
            active_identity
            and primary_user
            and _safe_identity(active_identity) == _safe_identity(primary_user)
        )
        return {
            "active_identity": active_identity,
            "primary_user": primary_user,
            "primary_user_title": primary_user_title,
            "active_matches_primary_user": active_matches_primary,
            "may_address_primary_user": active_matches_primary,
            "addressing_rule": (
                "You may address the current user with primary_user or primary_user_title."
                if active_matches_primary
                else "Do not address the current user by primary_user, primary_user_title, or a known person name."
            ),
        }

    def generate_guarded_response(
        self,
        response_type: str,
        prompt: str,
        state: dict,
        *,
        options: dict | None = None,
    ):
        options = self.chat_options(options)
        guarded_prompt = (
            f"{prompt.rstrip()}\n\n"
            f"Non-negotiable response contract:\n{self.response_contract(response_type, state)}\n"
        )
        text = self.chat_model.generate(guarded_prompt, options=options, think=False)
        violation = self.response_policy_violation(text, state, response_type)
        if not violation:
            return text
        sanitized = _sanitize_generated_response(text)
        violation = self.response_policy_violation(sanitized, state, response_type)
        if sanitized and not violation:
            return sanitized
        cleanup = self.cleanup_policy_failed_response(response_type, state, violation)
        if cleanup:
            return cleanup
        raise RuntimeError(f"response failed policy check: {violation}")

    def cleanup_policy_failed_response(self, response_type: str, state: dict, violation: str):
        identity_context = self.response_identity_context(state)
        active_identity = identity_context.get("active_identity")
        if response_type == "identity_status":
            text = f"You are {active_identity}." if active_identity else "I don't know who you are yet."
        elif response_type == "greeting":
            text = "I'm here."
        elif response_type == "feedback_learned":
            text = "Got it. I will use that next time."
        elif response_type == "identity_binding_blocked":
            text = "I need a verified visible person before I can attach that."
        elif response_type == "memory_needs_clarification":
            text = "I'd save that, but I don't know who you are yet."
        elif response_type == "memory_saved":
            text = "I remembered that."
        elif response_type == "identity_binding_success":
            text = "I learned that identity."
        elif response_type == "identity_binding_failed":
            text = "I couldn't learn that identity from the current view."
        elif response_type == "routing_error":
            text = "I couldn't route that cleanly."
        elif response_type == "vision_analysis_unavailable":
            text = "I couldn't get a reliable vision read from that."
        else:
            text = "I don't know that from the information I have."
        text = _sanitize_generated_response(text)
        return text if text and not self.response_policy_violation(text, state, response_type) else None

    def response_policy_violation(self, text: str, state: dict, response_type: str):
        if _contains_emoji(text):
            return "The response used an emoji, but emojis are not allowed."
        identity_context = self.response_identity_context(state)
        if re.search(r"\bLuhkas\s+will\b", str(text or ""), re.I):
            return "The response referred to the assistant in third person instead of answering directly."
        if re.search(r"\bmaintaining anonymity\b", str(text or ""), re.I):
            return "The response used vague anonymity wording instead of answering the question directly."
        if _sounds_like_customer_service(text):
            return "The response used generic customer-service filler instead of Luhkas's configured voice."
        if _meta_describes_personality(text):
            return "The response described the assistant's personality instead of simply having that style."
        if (
            response_type == "identity_status"
            and not identity_context.get("active_identity")
            and not _plainly_says_unknown_user_identity(text)
        ):
            return (
                "The response did not directly say the current user's identity is unknown. "
                "For identity_status with no active identity, answer in first person: "
                "I do not know who you are yet."
            )
        if identity_context.get("may_address_primary_user"):
            return None
        primary_user = identity_context.get("primary_user")
        primary_user_title = identity_context.get("primary_user_title")
        if _claims_current_user_is_primary(text):
            return "The response implied the current user is the primary user, but the active identity is not verified."
        if response_type != "identity_status" and primary_user and re.search(rf"\b(?:for|to|with)\s+{re.escape(str(primary_user))}\b", str(text or ""), re.I):
            return (
                f"The response framed the answer around primary_user {primary_user!r}, "
                "but the active identity is not verified."
            )
        if primary_user_title and re.search(rf"\b{re.escape(str(primary_user_title))}\b", str(text or ""), re.I):
            return (
                f"The response mentioned primary_user_title {primary_user_title!r}, "
                "but the active identity is not verified."
            )
        terms = [term for term in (primary_user, primary_user_title) if term]
        for term in terms:
            if _addresses_or_asserts_user_identity(text, term, response_type):
                return (
                    f"The response addressed or identified the current user as {term!r}, "
                    "but the active identity is not verified."
                )
        return None

    def answer_with_context(self, message: str, state: dict):
        identity_name = self.identity_profile.get("name")
        identity_role = self.identity_profile.get("role")
        self_description = _identity_sentence(identity_name, identity_role)
        active_identity_context = self.active_identity if state.get("ok") else "unverified_scout_tracking_unavailable"
        identity_context = self.response_identity_context(state)
        response_context = self.response_context(state)
        prompt = f"""{self_description}
Answer the user directly and briefly.

User: {message}

Context:
{json.dumps({
    "identity_context": identity_context,
    "response_lessons": response_context.get("response_lessons", [])[-5:],
    "tracking_available": bool(state.get("ok")),
    "active_identity": active_identity_context or "unknown",
    "tracking_memory": state.get("object_memory", [])[:4],
}, separators=(",", ":"), default=str)}

Rules: 1-2 short sentences. No emojis. No generic closer. Do not invent facts.
Do not mention Scout vision/tracking unless the user asks about vision or identity.
Do not address the user by name/title unless identity_context permits it.
"""
        try:
            return self.generate_guarded_response(
                "general_question",
                prompt,
                state,
                options={"num_predict": 90, "temperature": 0.45, "top_p": 0.9},
            )
        except Exception as exc:
            return f"I can hear you, but my local chat model is unavailable: {exc}"

    def identity_status(self, state: dict):
        return self.generate_response(
            "identity_status",
            "Who am I?",
            state,
            self.identity_status_facts(state),
            max_tokens=130,
        )

    def identity_status_facts(self, state: dict):
        known = [
            memory for memory in state.get("object_memory", [])
            if memory.get("label") == "person" and memory.get("identity")
        ]
        return {
            "scout_tracking_available": bool(state.get("ok")),
            "active_identity": self.active_identity,
            "known_people": known,
            "visible_learning_subject_count": _visible_learning_subject_count(state),
            "face_reference_counts": self.face_reference_counts(),
        }

    def whoami(self):
        state = self.scout_state()
        return {
            "ok": True,
            "answer": self.identity_status(state),
            "active_identity": self.active_identity,
            "debug": self.identity_debug(state),
        }

    def identity_debug(self, state: dict | None = None):
        state = state or self.scout_state()
        detections = state.get("detections", [])
        object_memory = state.get("object_memory", [])
        known_people = [
            memory for memory in object_memory
            if memory.get("label") == "person" and memory.get("identity")
        ]
        visible_subjects = [
            {
                "source": "detection",
                "id": det.get("id"),
                "label": det.get("label"),
                "confidence": det.get("confidence"),
                "identity": det.get("identity"),
                "identity_confidence": det.get("identity_confidence"),
                "bbox": det.get("bbox"),
            }
            for det in detections
            if det.get("label") == "person" or "face" in str(det.get("label", "")).lower()
        ]
        return {
            "active_identity": self.active_identity,
            "visible_face_count": _visible_face_count(state),
            "visible_learning_subject_count": _visible_learning_subject_count(state),
            "visible_subjects": visible_subjects,
            "known_people": known_people,
            "face_reference_counts": self.face_reference_counts(),
            "scout_face_recognition_enabled": state.get("face_recognition_enabled"),
            "scout_face_detection_enabled": state.get("face_detection_enabled"),
        }

    def learn_face(self, identity: str):
        identity = _safe_identity(identity)
        if not identity:
            return {"ok": False, "error": "missing_identity"}
        state = self.scout_state()
        face_id = _active_learning_face_id(state)
        if face_id is None:
            return {
                "ok": False,
                "error": "target_face_required",
                "message": "I need a visible targeted face before I can learn that name.",
                "identity_prompt": state.get("identity_prompt"),
                "identity_prompt_queue": state.get("identity_prompt_queue"),
            }
        result = self._post_json(f"{self.scout_url}/learn_face?name={quote(identity)}&face_id={quote(str(face_id))}", {})
        if result.get("ok"):
            self.ensure_person(identity, display_name=identity)
            if result.get("image_b64"):
                self.add_face_reference(identity, result)
            saved_path = result.get("saved_path")
            if saved_path:
                self.record_face_reference(identity, saved_path, result.get("reference_pose"), auto=False)
        return result

    def handle_scout_action(self, message: str, state: dict | None = None):
        state = state or self.scout_state()
        text = _normalize_command_text(message)

        if _has_any(text, ("status", "state", "why aren't you moving", "why are you not moving")):
            return {
                "ok": True,
                "action": "inspect_scout_state",
                "message": _scout_state_explanation(state),
                "state": _compact_scout_state(state),
            }

        if _has_any(text, ("follow me", "start following", "start tracking", "track me")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": True, "follow": True})
            return _action_response("control_tracking", result, "Following is on.", "I could not turn following on.")

        if _has_any(text, ("stop following", "stop tracking", "don't follow", "do not follow")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": False, "follow": False})
            return _action_response("control_tracking", result, "Following is off.", "I could not turn following off.")

        if _has_any(text, ("enable tracking", "start tracking", "track person", "track people")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": True})
            return _action_response("control_tracking", result, "Tracking is on.", "I could not turn tracking on.")

        if _has_any(text, ("disable tracking", "turn off tracking", "stop tracking")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": False})
            return _action_response("control_tracking", result, "Tracking is off.", "I could not turn tracking off.")

        if _has_any(text, ("search camera on", "enable search camera", "turn on search camera")):
            result = self._post_json(f"{self.scout_url}/settings", {"search_movement_enabled": True})
            return _action_response("control_search_camera", result, "Search camera is on.", "I could not enable search camera.")

        if _has_any(text, ("search camera off", "disable search camera", "turn off search camera")):
            result = self._post_json(f"{self.scout_url}/settings", {"search_movement_enabled": False})
            return _action_response("control_search_camera", result, "Search camera is off.", "I could not disable search camera.")

        if _has_any(text, ("guard on", "enable guard", "start guard", "start guarding")):
            result = self._post_json(f"{self.scout_url}/guard", {"enabled": True})
            return _action_response("control_guard", result, "Guard mode is on.", "I could not enable guard mode.")

        if _has_any(text, ("guard off", "disable guard", "stop guard", "stop guarding")):
            result = self._post_json(f"{self.scout_url}/guard", {"enabled": False})
            return _action_response("control_guard", result, "Guard mode is off.", "I could not disable guard mode.")

        if _has_any(text, ("center camera", "center the camera", "look straight", "look ahead", "look forward", "face forward", "reset camera")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"center": True})
            return _action_response("control_camera", result, "Centering camera.", "I could not center the camera.")

        if _has_any(text, ("look left", "pan left", "turn camera left")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": -60, "tilt": 0})
            return _action_response("control_camera", result, "Looking left.", "I could not move the camera left.")

        if _has_any(text, ("look right", "pan right", "turn camera right")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 60, "tilt": 0})
            return _action_response("control_camera", result, "Looking right.", "I could not move the camera right.")

        if _has_any(text, ("look up", "tilt up", "look higher")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 0, "tilt": 40})
            return _action_response("control_camera", result, "Looking up.", "I could not move the camera up.")

        if _has_any(text, ("look down", "tilt down", "look lower")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 0, "tilt": -40})
            return _action_response("control_camera", result, "Looking down.", "I could not move the camera down.")

        if _has_any(text, ("turn on the light", "light on", "lamp on")):
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_enabled": True})
            return _action_response("control_light", result, "The camera light is on.", "I could not turn the light on.")

        if _has_any(text, ("turn off the light", "light off", "lamp off")):
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_enabled": False})
            return _action_response("control_light", result, "The camera light is off.", "I could not turn the light off.")

        brightness = _extract_light_brightness(text)
        if brightness is not None:
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_brightness": brightness})
            return _action_response("control_light", result, f"Light brightness is set to {brightness}.", "I could not set the light brightness.")

        if _has_any(text, ("take a picture", "take a photo", "save a snapshot", "snapshot")):
            return self.capture_snapshot()

        if _has_any(text, ("record a clip", "save a clip", "video clip", "record video", "record a video", "take a video")):
            return self.capture_clip()

        return None

    def capture_snapshot(self):
        snapshot = self._get_bytes(f"{self.scout_url}/snapshot")
        if snapshot is None:
            return {"ok": False, "action": "capture_snapshot", "error": "snapshot_unavailable", "message": "I could not get a scout snapshot."}
        capture_dir = Path(DATA_DIR) / "scout_captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        path = capture_dir / f"snapshot-{int(time.time())}.jpg"
        try:
            path.write_bytes(snapshot)
        except OSError as exc:
            return {"ok": False, "action": "capture_snapshot", "error": str(exc), "message": "I could not save the snapshot."}
        return {"ok": True, "action": "capture_snapshot", "path": str(path), "message": f"Saved snapshot to {path}."}

    def capture_clip(self, seconds: float = 8.0):
        result = self._post_json(f"{self.scout_url}/clip", {"seconds": seconds})
        if not result.get("ok"):
            return {
                "ok": False,
                "action": "capture_clip",
                "message": "I could not record a scout clip.",
                "result": result,
            }
        return {
            "ok": True,
            "action": "capture_clip",
            "path": result.get("path"),
            "duration": result.get("duration"),
            "message": f"Recorded scout clip to {result.get('path')}.",
            "result": result,
        }

    def scout_tool_status(self):
        state = self.scout_state()
        robot_health = self._get_json(f"{self.scout_robot_url}/health") or {"ok": False, "error": "robot_api_unavailable"}
        scout_capabilities = self.scout_node_capabilities()
        return {
            "ok": True,
            "scout_url": self.scout_url,
            "scout_robot_url": self.scout_robot_url,
            "vision_reachable": bool(state.get("ok")),
            "robot_api_reachable": bool(robot_health.get("ok")),
            "state": _compact_scout_state(state),
            "robot_health": robot_health,
            "node_capabilities": scout_capabilities,
            "actions": [
                "inspect_scout_state",
                "analyze_vision",
                "learn_face",
                "control_tracking",
                "control_search_camera",
                "control_guard",
                "control_light",
                "capture_snapshot",
                "capture_clip",
                "remember_fact",
                "set_preference",
                "group_unknown_faces",
            ],
            "contracts": {
                "brain": [
                    "GET /faces/unknown",
                    "POST /faces/unknown",
                    "POST /faces/unknown/promote",
                ],
                "vision": [
                    "GET /meta",
                    "GET /snapshot",
                    "POST /snapshot",
                    "POST /clip",
                    "POST /learn_face?name=<identity>&face_id=<face-id>",
                    "POST /tracking",
                    "POST /settings",
                    "POST /guard",
                ],
                "robot_api": [
                    "GET /health",
                    "GET /battery",
                    "GET /telemetry",
                    "POST /pantilt",
                    "POST /move",
                    "POST /oled",
                ],
            },
        }

    def scout_node_capabilities(self):
        return self._get_json(f"{self.scout_url}/capabilities") or {
            "ok": False,
            "error": "scout_capabilities_unavailable",
            "capabilities": [],
            "commands": [],
        }

    def faces_sync(self):
        people = []
        for person_dir in sorted(path for path in self.face_dir.iterdir() if path.is_dir()):
            samples = []
            for image_path in sorted(path for path in person_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
                try:
                    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
                except OSError:
                    continue
                samples.append({
                    "path": str(image_path.relative_to(person_dir)),
                    "image_b64": image_b64,
                })
            people.append({"identity": person_dir.name, "samples": samples})
        return {"ok": True, "people": people}

    def face_reference_counts(self):
        counts = {}
        for person_dir in sorted(path for path in self.face_dir.iterdir() if path.is_dir()):
            counts[person_dir.name] = len([
                path for path in person_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ])
        return counts

    def unknown_faces(self):
        groups = []
        for group_dir in self._unknown_group_dirs():
            meta = self._load_unknown_group(group_dir)
            if not meta:
                continue
            groups.append({
                "id": meta.get("id") or group_dir.name,
                "promoted": bool(meta.get("promoted")),
                "identity": meta.get("identity"),
                "node_ids": meta.get("node_ids", []),
                "source_keys": meta.get("source_keys", []),
                "sample_count": int(meta.get("sample_count") or len(meta.get("samples", []))),
                "first_seen": meta.get("first_seen"),
                "last_seen": meta.get("last_seen"),
            })
        return {"ok": True, "groups": groups}

    def add_unknown_face_observation(self, payload: dict):
        image_b64 = payload.get("image_b64")
        if not image_b64:
            return {"ok": False, "error": "missing_image_b64"}
        node_id = _normalize_source(payload.get("node_id") or payload.get("source") or "unknown_edge")
        source_track_id = str(payload.get("source_track_id") or "").strip()
        source_key = f"{node_id}:{source_track_id}" if source_track_id else None
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        fingerprint = self._unknown_face_fingerprint(image_bytes)
        face_group_dir = self._find_unknown_group_for_face(fingerprint)
        source_group_dir = self._find_unknown_group_for_source(source_key) if source_key else None
        group_dir = face_group_dir or source_group_dir
        matched_by = "face_fingerprint" if face_group_dir is not None else "source_track" if source_group_dir is not None else None
        if face_group_dir is not None and source_group_dir is not None and face_group_dir != source_group_dir:
            self._merge_unknown_groups(face_group_dir, source_group_dir)
            group_dir = face_group_dir
            matched_by = "face_fingerprint_merge"
        if group_dir is None:
            group_dir = self.unknown_face_dir / self._next_unknown_group_id()
            group_dir.mkdir(parents=True, exist_ok=True)
            matched_by = "new_group"
        samples_dir = group_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(str(payload.get("filename") or f"{int(time.time() * 1000)}.jpg"))
        image_hash = fingerprint["sha1"][:12]
        target = samples_dir / f"{int(time.time() * 1000)}_{image_hash}_{filename}"
        try:
            target.write_bytes(image_bytes)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        now = time.time()
        meta = self._load_unknown_group(group_dir) or {
            "id": group_dir.name,
            "source_keys": [],
            "node_ids": [],
            "first_seen": now,
            "last_seen": now,
            "sample_count": 0,
            "samples": [],
            "promoted": False,
            "identity": None,
        }
        if source_key and source_key not in meta.setdefault("source_keys", []):
            meta["source_keys"].append(source_key)
        if node_id not in meta.setdefault("node_ids", []):
            meta["node_ids"].append(node_id)
        if fingerprint.get("sha1") and fingerprint["sha1"] not in meta.setdefault("image_sha1s", []):
            meta["image_sha1s"].append(fingerprint["sha1"])
        if fingerprint.get("perceptual_hash") and fingerprint["perceptual_hash"] not in meta.setdefault("perceptual_hashes", []):
            meta["perceptual_hashes"].append(fingerprint["perceptual_hash"])
        sample = {
            "path": str(target),
            "node_id": node_id,
            "source_track_id": source_track_id or None,
            "face_id": payload.get("face_id"),
            "bbox": payload.get("bbox"),
            "confidence": payload.get("confidence"),
            "image_sha1": fingerprint.get("sha1"),
            "perceptual_hash": fingerprint.get("perceptual_hash"),
            "observed_at": payload.get("observed_at") or now,
        }
        meta.setdefault("samples", []).append(sample)
        meta["sample_count"] = len(meta["samples"])
        meta["last_seen"] = now
        meta["promoted"] = bool(meta.get("promoted"))
        meta["identity"] = meta.get("identity")
        meta["last_matched_by"] = matched_by
        self._write_unknown_group(group_dir, meta)
        return {
            "ok": True,
            "group_id": group_dir.name,
            "sample_count": meta["sample_count"],
            "promoted": bool(meta.get("promoted")),
            "matched_by": matched_by,
        }

    def promote_unknown_face_group(self, group_id: str, identity: str):
        group_id = self._safe_unknown_group_id(group_id)
        identity = _safe_identity(identity)
        if not group_id:
            return {"ok": False, "error": "missing_group_id"}
        if not identity:
            return {"ok": False, "error": "missing_identity"}
        group_dir = self.unknown_face_dir / group_id
        meta = self._load_unknown_group(group_dir)
        if not meta:
            return {"ok": False, "error": "unknown_group_not_found", "group_id": group_id}

        destination = self.face_dir / identity / "_unknown" / group_id
        destination.mkdir(parents=True, exist_ok=True)
        copied = 0
        for sample in meta.get("samples", []):
            src = Path(str(sample.get("path") or ""))
            if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            target = destination / _safe_filename(src.name)
            try:
                shutil.copy2(src, target)
            except OSError:
                continue
            copied += 1

        now = time.time()
        meta["promoted"] = True
        meta["identity"] = identity
        meta["promoted_at"] = now
        meta["promoted_destination"] = str(destination)
        meta["promoted_sample_count"] = copied
        self._write_unknown_group(group_dir, meta)

        self.ensure_person(identity, display_name=identity)
        self._append_memory(identity, {
            "type": "unknown_face_group_promoted",
            "group_id": group_id,
            "sample_count": copied,
            "destination": str(destination),
            "created_at": now,
        })
        return {
            "ok": True,
            "promoted": copied > 0,
            "group_id": group_id,
            "identity": identity,
            "sample_count": copied,
            "destination": str(destination),
        }

    def add_face_reference(self, identity: str, payload: dict):
        identity = _safe_identity(identity)
        if not identity:
            return {"ok": False, "error": "missing_identity"}
        image_b64 = payload.get("image_b64")
        if not image_b64:
            return {"ok": False, "error": "missing_image_b64"}
        pose = _safe_identity(str(payload.get("reference_pose") or "unsorted")) or "unsorted"
        filename = _safe_filename(str(payload.get("filename") or f"{int(time.time() * 1000)}.jpg"))
        target = self.face_dir / identity / pose / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(base64.b64decode(image_b64))
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "identity": identity, "path": str(target)}

    def record_face_reference(self, identity: str, saved_path: str, pose: str | None, auto: bool):
        # The scout may upload the image separately; this records metadata only
        # when the brain cannot read the rover file path directly.
        person = self.ensure_person(identity)
        event = {
            "type": "face_reference",
            "path": saved_path,
            "reference_pose": pose,
            "auto": bool(auto),
            "created_at": time.time(),
        }
        self._append_memory(person["identity"], event)

    def person_summary(self, identity: str):
        profile = self._load_profile(_safe_identity(identity))
        return {
            "ok": True,
            "summary": {
                "identity": profile["identity"],
                "source": "brain",
                "display_name": profile.get("display_name", profile["identity"]),
                "preferences": profile.get("preferences", {}),
                "facts": profile.get("facts", {}),
                "updated_at": profile.get("updated_at"),
            },
        }

    def person_memory(self, identity: str):
        identity = _safe_identity(identity)
        return {"ok": True, "profile": self._load_profile(identity), "memories": self._load_memories(identity)}

    def remember(self, identity: str, memory_type: str, key: str, value, source="user", confidence=1.0):
        identity = _safe_identity(identity)
        key = _safe_key(key)
        if not identity or not key:
            return {"ok": False, "error": "missing_identity_or_key"}
        profile = self._load_profile(identity)
        now = time.time()
        memory_type = memory_type if memory_type in {"fact", "preference", "interaction", "note"} else "fact"
        if memory_type == "preference":
            profile.setdefault("preferences", {})[key] = value
        if memory_type == "fact":
            profile.setdefault("facts", {})[key] = value
            if key == "display_name":
                profile["display_name"] = str(value)
        profile["updated_at"] = now
        self._write_profile(identity, profile)
        event = {
            "type": memory_type,
            "key": key,
            "value": value,
            "source": source,
            "confidence": float(confidence),
            "created_at": now,
        }
        self._append_memory(identity, event)
        return {"ok": True, "identity": identity, "event": event, "summary": self.person_summary(identity)["summary"]}

    def analyze_scene(self, question: str, state: dict | None = None):
        state = state or {}
        snapshot = self._get_bytes(f"{self.scout_url}/snapshot")
        if snapshot is None:
            return {"ok": False, "error": "snapshot_unavailable"}
        image_b64 = base64.b64encode(snapshot).decode("ascii")
        tracker_summary = _tracking_summary(state)
        prompt = f"""
Answer the user's visual question using the current camera image and tracker memory.

Rules:
- Be literal and conservative. Do not invent room type, purpose, timestamps,
  listings, pets, people, furniture, or multiple camera angles.
- Clearly separate what the tracker reports from what the image appears to show.
- If the image is unclear or you are not confident, say that.
- Apply response lessons and behavior settings, especially precision or scope
  corrections from the user.
- If response lessons say not to include tracker info, do not mention tracker
  reports, object IDs, labels, or confidence scores unless the user asks for
  tracker details.
- Keep the answer to 1-3 short sentences.

User question:
{question}

Tracker summary:
{tracker_summary}

Raw tracking memory:
{json.dumps(state.get("object_memory", []), indent=2)}

Response context:
{json.dumps(self.response_context(state), indent=2, default=str)}
"""
        # Ollama multimodal endpoint, if a vision model is installed.
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": OLLAMA_VISION_MODEL,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": self.chat_options({"num_predict": 260}),
                },
                timeout=60,
            )
            if response.status_code == 200:
                answer = _sanitize_generated_response(response.json().get("response", "")).strip()
                return {"ok": True, "answer": answer}
            return {"ok": False, "error": response.text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def ensure_person(self, identity: str, display_name: str | None = None):
        identity = _safe_identity(identity)
        profile = self._load_profile(identity)
        if display_name:
            profile["display_name"] = display_name
            profile.setdefault("facts", {})["display_name"] = display_name
        profile["updated_at"] = time.time()
        self._write_profile(identity, profile)
        return profile

    def _load_identity_profile(self):
        if self.identity_profile_path.exists():
            try:
                data = json.loads(self.identity_profile_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _person_dir(self, identity: str):
        return self.people_dir / identity

    def _profile_path(self, identity: str):
        return self._person_dir(identity) / "profile.json"

    def _memories_path(self, identity: str):
        return self._person_dir(identity) / "memories.jsonl"

    def _load_profile(self, identity: str):
        identity = _safe_identity(identity)
        path = self._profile_path(identity)
        if not path.exists():
            return {
                "identity": identity,
                "display_name": identity,
                "preferences": {},
                "facts": {},
                "created_at": time.time(),
                "updated_at": None,
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data.setdefault("identity", identity)
        data.setdefault("display_name", identity)
        data.setdefault("preferences", {})
        data.setdefault("facts", {})
        return data

    def _write_profile(self, identity: str, profile: dict):
        person_dir = self._person_dir(identity)
        person_dir.mkdir(parents=True, exist_ok=True)
        self._profile_path(identity).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _append_memory(self, identity: str, event: dict):
        person_dir = self._person_dir(identity)
        person_dir.mkdir(parents=True, exist_ok=True)
        with self._memories_path(identity).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _load_memories(self, identity: str, limit=100):
        path = self._memories_path(identity)
        if not path.exists():
            return []
        memories = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                memories.append(item)
        return memories[-limit:]

    def _unknown_group_dirs(self):
        if not self.unknown_face_dir.exists():
            return []
        return sorted(path for path in self.unknown_face_dir.iterdir() if path.is_dir())

    def _unknown_group_path(self, group_dir: Path):
        return group_dir / "group.json"

    def _load_unknown_group(self, group_dir: Path):
        path = self._unknown_group_path(group_dir)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _write_unknown_group(self, group_dir: Path, meta: dict):
        group_dir.mkdir(parents=True, exist_ok=True)
        self._unknown_group_path(group_dir).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _next_unknown_group_id(self):
        used = set()
        for group_dir in self._unknown_group_dirs():
            match = re.match(r"vault_unknown_(\d+)$", group_dir.name)
            if match:
                used.add(int(match.group(1)))
        index = 1
        while index in used:
            index += 1
        return f"vault_unknown_{index:04d}"

    def _find_unknown_group_for_source(self, source_key: str | None):
        if not source_key:
            return None
        for group_dir in self._unknown_group_dirs():
            meta = self._load_unknown_group(group_dir)
            if not meta or meta.get("promoted"):
                continue
            if source_key in (meta.get("source_keys") or []):
                return group_dir
        return None

    def _find_unknown_group_for_face(self, fingerprint: dict):
        sha1 = fingerprint.get("sha1")
        perceptual_hash = fingerprint.get("perceptual_hash")
        best_group = None
        best_distance = None
        for group_dir in self._unknown_group_dirs():
            meta = self._load_unknown_group(group_dir)
            if not meta or meta.get("promoted"):
                continue
            if sha1 and sha1 in (meta.get("image_sha1s") or []):
                return group_dir
            if not perceptual_hash:
                continue
            for known_hash in meta.get("perceptual_hashes") or []:
                distance = _hex_hamming_distance(perceptual_hash, known_hash)
                if distance is None:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_group = group_dir
        if best_distance is not None and best_distance <= 10:
            return best_group
        return None

    def _merge_unknown_groups(self, destination_dir: Path, source_dir: Path):
        if destination_dir == source_dir:
            return
        destination = self._load_unknown_group(destination_dir)
        source = self._load_unknown_group(source_dir)
        if not destination or not source or source.get("promoted"):
            return
        destination_samples_dir = destination_dir / "samples"
        destination_samples_dir.mkdir(parents=True, exist_ok=True)
        moved_samples = []
        for sample in source.get("samples") or []:
            src = Path(str(sample.get("path") or ""))
            if src.is_file():
                target = destination_samples_dir / _safe_filename(src.name)
                if target.exists():
                    target = destination_samples_dir / f"{int(time.time() * 1000)}_{_safe_filename(src.name)}"
                try:
                    shutil.move(str(src), str(target))
                    sample = dict(sample)
                    sample["path"] = str(target)
                except OSError:
                    pass
            moved_samples.append(sample)
        for key in ("source_keys", "node_ids", "image_sha1s", "perceptual_hashes"):
            values = destination.setdefault(key, [])
            for value in source.get(key) or []:
                if value not in values:
                    values.append(value)
        destination.setdefault("samples", []).extend(moved_samples)
        destination["sample_count"] = len(destination.get("samples") or [])
        destination["first_seen"] = min(
            float(destination.get("first_seen") or time.time()),
            float(source.get("first_seen") or time.time()),
        )
        destination["last_seen"] = max(
            float(destination.get("last_seen") or 0.0),
            float(source.get("last_seen") or 0.0),
        )
        merged = destination.setdefault("merged_group_ids", [])
        source_id = source.get("id") or source_dir.name
        if source_id not in merged:
            merged.append(source_id)
        self._write_unknown_group(destination_dir, destination)
        try:
            shutil.rmtree(source_dir)
        except OSError:
            pass

    def _unknown_face_fingerprint(self, image_bytes: bytes):
        fingerprint = {"sha1": hashlib.sha1(image_bytes).hexdigest(), "perceptual_hash": None}
        perceptual_hash = _opencv_difference_hash(image_bytes)
        if perceptual_hash:
            fingerprint["perceptual_hash"] = perceptual_hash
        return fingerprint

    def _safe_unknown_group_id(self, group_id: str):
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(group_id or "").strip())
        cleaned = cleaned.strip("._-")[:80]
        if not cleaned or cleaned in {".", ".."}:
            return ""
        return cleaned

    def _get_json(self, url: str):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                return None
            return response.json()
        except Exception:
            return None

    def _get_bytes(self, url: str):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None
            return response.content
        except Exception:
            return None

    def _post_json(self, url: str, payload: dict):
        try:
            response = requests.post(url, json=payload, timeout=8)
            try:
                data = response.json()
            except Exception:
                data = {"raw": response.text}
            data.setdefault("ok", 200 <= response.status_code < 300)
            return data
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def _visible_face_count(state: dict):
    return sum(1 for det in state.get("detections", []) if "face" in str(det.get("label", "")).lower())


def _visible_learning_subject_count(state: dict):
    detections = state.get("detections", [])
    count = sum(
        1 for det in detections
        if str(det.get("label", "")).lower() == "person"
        or "face" in str(det.get("label", "")).lower()
    )
    if count:
        return count
    return sum(
        1 for memory in state.get("object_memory", [])
        if memory.get("label") == "person"
    )


def _active_learning_face_id(state: dict):
    prompt = state.get("identity_prompt") or {}
    face_id = prompt.get("face_id")
    if face_id is not None:
        return face_id
    queue = state.get("identity_prompt_queue") or {}
    active_group = queue.get("active_face_group_id")
    active_face = queue.get("active_face_id")
    faces = queue.get("faces") or []
    if active_group:
        for face in faces:
            if face.get("face_group_id") == active_group and face.get("face_id") is not None:
                return face.get("face_id")
    if active_face is not None:
        return active_face
    for det in state.get("detections", []):
        if "face" in str(det.get("label", "")).lower() and det.get("id") is not None:
            return det.get("id")
    return None


def _normalize_command_text(message: str) -> str:
    return re.sub(r"\s+", " ", str(message or "").lower()).strip()


def _is_wakeword_only(message: str) -> bool:
    if _shared_is_wakeword_only is not None:
        return bool(_shared_is_wakeword_only(message))
    text = re.sub(r"[^a-zA-Z\s]", " ", str(message or "").casefold())
    words = [word for word in text.split() if word not in {"hey", "hi", "hello", "yo", "ok", "okay"}]
    if not words:
        return False
    return all(_sounds_like_luhkas(word) for word in words)


def _asks_to_stop_chat(message: str) -> bool:
    text = _canonical_intent_text(message)
    return text in {
        "shut up",
        "stop talking",
        "be quiet",
        "quiet",
        "hush",
        "stop",
    }


def _source_node_id(source: str | None, presence_context: dict | None) -> str:
    if isinstance(presence_context, dict):
        node_id = str(presence_context.get("node_id") or presence_context.get("source") or "").strip()
        if node_id:
            return _normalize_source(node_id)
    return _normalize_source(source or "unknown_edge")


def _required_source_module(message: str) -> str | None:
    text = _canonical_intent_text(message)
    if re.search(r"\b(play|say|speak|announce|read)\b", text) and re.search(r"\b(audio|sound|speaker|out loud|aloud|voice)\b", text):
        return "speech_node"
    if re.search(r"\b(show|display|open|render)\b", text) and re.search(r"\b(screen|display|browser|ui|image|code)\b", text):
        return "display_node"
    if _asks_camera_view(text) or _asks_camera_capture(text):
        return "camera_node"
    if _asks_pan_tilt_action(text):
        return "pantilt_node"
    if _asks_wheel_action(text):
        return "rover_node"
    if "light" in text or "lamp" in text:
        return "light_node"
    return None


def _asks_camera_view(text: str) -> bool:
    return (
        "what do you see" in text
        or "can you see" in text
        or "look at" in text
        or "in front of you" in text
        or "visible" in text
    )


def _asks_camera_capture(text: str) -> bool:
    return any(phrase in text for phrase in (
        "take a picture",
        "take a photo",
        "snapshot",
        "record a clip",
        "record video",
        "take a video",
        "guard on",
        "guard off",
    ))


def _asks_pan_tilt_action(text: str) -> bool:
    return any(phrase in text for phrase in (
        "look up",
        "look down",
        "pan left",
        "pan right",
        "tilt up",
        "tilt down",
        "move camera",
        "center camera",
    ))


def _asks_wheel_action(text: str) -> bool:
    return any(phrase in text for phrase in (
        "go forward",
        "move forward",
        "go backward",
        "move backward",
        "turn left",
        "turn right",
        "follow me",
        "stop moving",
    ))


def _module_label(module_name: str) -> str:
    labels = {
        "camera_node": "a camera_node",
        "pantilt_node": "a pantilt_node",
        "rover_node": "a rover_node",
        "light_node": "a light_node",
        "display_node": "a display_node",
        "speech_node": "a speech_node",
    }
    return labels.get(module_name, f"{module_name}")


def _module_action_label(module_name: str) -> str:
    labels = {
        "camera_node": "use vision or camera capture",
        "pantilt_node": "move the camera",
        "rover_node": "move the rover",
        "light_node": "control a light",
        "display_node": "show that on a display",
        "speech_node": "play or speak audio",
    }
    return labels.get(module_name, f"use {module_name}")


def _asks_about_previous_answer_source(message: str) -> bool:
    text = _normalize_command_text(message)
    return (
        "how did you determine" in text
        or "how did you find" in text
        or "where did you get" in text
        or "what did you use" in text
        or "what source" in text
        or "which source" in text
        or "which registry" in text
        or "what registry" in text
        or "did that come from" in text
        or "where did that come from" in text
        or "did you use the node registry" in text
        or "did you check the registered nodes" in text
    )


def _fast_route_message(message: str) -> dict | None:
    text = _canonical_intent_text(message)
    if _asks_assistant_name(text):
        return _self_route("assistant_identity", "asks assistant name")
    if _asks_assistant_identity_topic(text):
        return _self_route("assistant_identity", "asks assistant self-description")
    if _asks_user_identity(text):
        return _self_route("user_identity", "asks current user identity")
    if _asks_registered_or_active_nodes(text):
        return _self_route("status", "asks live registered node inventory")
    if _asks_tracking_status(text) or _asks_pose_interval(text):
        return _self_route("status", "asks Scout live state")
    if _asks_skill_inventory(text) or _asks_capability_inventory(text):
        return _self_route("capabilities", "asks registry inventory")
    if _asks_stored_knowledge_owner(text) or _asks_camera_action_owner(text):
        return _self_route("software", "asks node ownership boundary")
    if _asks_why_here(text):
        return _self_route("goals", "asks assistant purpose")
    self_topic = _self_topic_from_text(text)
    if self_topic is not None:
        return _self_route(self_topic, f"asks assistant {self_topic}")
    if text.startswith("tell me ") or text.startswith("say "):
        return {
            "ok": True,
            "route": "general_question",
            "confidence": 0.94,
            "reason": "conversational chat request",
            "attempts": 0,
            "deterministic": True,
        }
    return None


def _self_topic_from_text(text: str) -> str | None:
    """Map messages that mention an assistant-self pronoun plus a topic keyword
    to a specific self_question sub-route. Returns None if no clear match.

    This catches phrasings the router LLM has been getting wrong, e.g.
    "tell me about your hardware" routing as general_question and the chat
    model then hallucinating fake specs.
    """
    has_self_pronoun = bool(
        re.search(r"\b(your|yours|yourself|youre|you have|you got|you running|you using|you on)\b", text)
    )
    if not has_self_pronoun:
        return None
    if re.search(r"\b(hardware|hardware stack|gpu|cpu|ram|chassis|motor|motors|camera body|rover body|raspberry pi|hailo)\b", text):
        return "hardware"
    if re.search(r"\b(sensor|sensors|imu|accelerometer|gyro|gyroscope|magnetometer)\b", text):
        return "sensors"
    if re.search(r"\b(software|architecture|api|apis|model|models|inference|service|services|stack|routing|ollama|brain code)\b", text):
        return "software"
    return None


def _self_route(self_route: str, reason: str) -> dict:
    return {
        "ok": True,
        "route": "self_question",
        "confidence": 1.0,
        "reason": reason,
        "attempts": 0,
        "deterministic": True,
        "self_route": {
            "ok": True,
            "route": self_route,
            "confidence": 1.0,
            "reason": reason,
            "attempts": 0,
            "deterministic": True,
        },
    }


def _asks_assistant_name(text: str) -> bool:
    """True only for terse name asks where 'I'm Luhkas. Chris built me.' is
    the right answer. Broader self-identity questions go through
    _asks_assistant_identity_topic so the chat model can apply personality."""
    return text in {
        "whats your name", "what is your name", "what's your name",
        "your name",
    }


def _asks_assistant_identity_topic(text: str) -> bool:
    """True for broader 'who/what are you / tell me about yourself' style
    questions. These should route to self_question/assistant_identity but
    let the chat model write the reply so behavior directives apply."""
    if text in {
        "who are you", "what are you", "tell me about yourself",
        "tell me about you", "tell me who you are", "introduce yourself",
        "describe yourself", "what are you really",
    }:
        return True
    if re.match(r"^(tell|describe)\s+(me\s+)?(about\s+)?(your\s*self|yourself)\b", text):
        return True
    return False


def _asks_user_identity(text: str) -> bool:
    return text in {"who am i", "do you know me", "do you recognize me"} or "identify me" in text


def _asks_registered_or_active_nodes(text: str) -> bool:
    has_node = any(term in text for term in ("node", "nodes", "node device", "node devices"))
    has_live = any(term in text for term in ("active", "currently", "registered", "live", "right now", "how many"))
    return has_node and has_live


def _asks_capability_inventory(text: str) -> bool:
    return (
        "capability" in text
        or "capabilities" in text
        or "what can you do" in text
        or "what can you help" in text
    )


def _asks_skill_inventory(text: str) -> bool:
    return "skill" in text and any(term in text for term in ("what", "which", "know", "registered", "list", "have"))


def _asks_tracking_status(text: str) -> bool:
    return bool(re.search(r"\b(?:is|are|whats|what is)\s+(?:the\s+)?tracking\b", text)) or text in {"is tracking on", "tracking on"}


def _asks_pose_interval(text: str) -> bool:
    return "pose interval" in text


def _asks_stored_knowledge_owner(text: str) -> bool:
    return "stored knowledge" in text and any(term in text for term in ("scout", "vault", "go to", "belongs", "own", "owns"))


def _asks_camera_action_owner(text: str) -> bool:
    return "camera action" in text and any(term in text for term in ("who owns", "owns that", "belongs", "scout", "vault"))


def _looks_like_ownership_question(message: str) -> bool:
    text = _canonical_intent_text(message)
    return _asks_stored_knowledge_owner(text) or _asks_camera_action_owner(text)


def _asks_registry_source_followup(message: str) -> bool:
    text = _canonical_intent_text(message)
    return "which registry" in text or "what registry" in text or "did that come from" in text


def _asks_why_here(text: str) -> bool:
    return text in {"why are you here", "what are you here for", "what is your purpose", "why do you exist"}


def _canonical_intent_text(message: str) -> str:
    text = str(message or "").casefold().replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _human_source_label(source: dict) -> str | None:
    name = str(source.get("name") or source.get("source") or "").casefold()
    role = str(source.get("role") or "").casefold()
    if name in {"conversation_message", "route_message", "chat_model"}:
        return None
    if "model_prior_knowledge" in name or "model prior knowledge" in role:
        return "my own prior knowledge"
    if "data/self" in name or "hardware stacks" in name or "self hardware" in role or "what i know about myself" in role:
        return "what I know about myself"
    if "noderegistry" in name or "registered node" in role:
        return "the live node registry"
    if "source_node.modules" in name or "module availability" in role:
        return "the live node registry"
    if "capability_registry" in name or "capability registry" in role:
        return "the capability registry"
    if "skill_registry" in name or "skill registry" in role:
        return "the skill registry"
    if "witnessed_state" in name or "witnessing" in role or "scout /meta" in name:
        return "Scout's live state"
    if "vision model" in name or "scene analysis" in role:
        return "I witnessed it"
    if "response_style" in name:
        return "my response style rules"
    if "response_lessons" in name or "learned" in role:
        return "you taught me that"
    if "action/router" in name:
        return "the action result"
    return None


def _sounds_like_luhkas(word: str) -> bool:
    word = re.sub(r"[^a-z]", "", word.casefold())
    return word in {
        "luhkas",
        "luhkus",
        "lukas",
        "lucas",
        "loukas",
        "loucas",
        "leukas",
        "lukus",
    }


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _source_is_scout(source: str | None) -> bool:
    text = _normalize_command_text(source or "")
    return text in {"scout", "luhkas scout", "luhkas-scout", "rover", "robot"}


def _explicitly_targets_scout(message: str) -> bool:
    text = _normalize_command_text(message)
    return bool(re.search(r"\b(?:from|on|using|with|by)\s+(?:the\s+)?(?:scout|rover|robot)\b", text))


def _targets_scout_action(message: str, source: str | None) -> bool:
    return _source_is_scout(source) or _explicitly_targets_scout(message)


def _asks_for_clip(message: str) -> bool:
    text = _normalize_command_text(message)
    return _has_any(text, ("record a clip", "save a clip", "video clip", "record video", "record a video", "take a video"))


def _looks_like_scout_action(message: str) -> bool:
    text = _normalize_command_text(message)
    if _extract_light_brightness(text) is not None:
        return True
    return _has_any(text, (
        "status", "state", "why aren't you moving", "why are you not moving",
        "follow me", "start following", "start tracking", "track me",
        "stop following", "stop tracking", "don't follow", "do not follow",
        "enable tracking", "track person", "track people",
        "disable tracking", "turn off tracking",
        "search camera on", "enable search camera", "turn on search camera",
        "search camera off", "disable search camera", "turn off search camera",
        "guard on", "enable guard", "start guard", "start guarding",
        "guard off", "disable guard", "stop guard", "stop guarding",
        "center camera", "center the camera", "look straight", "look ahead",
        "look forward", "face forward", "reset camera",
        "look left", "pan left", "turn camera left",
        "look right", "pan right", "turn camera right",
        "look up", "tilt up", "look higher",
        "look down", "tilt down", "look lower",
        "turn on the light", "light on", "lamp on",
        "turn off the light", "light off", "lamp off",
        "take a picture", "take a photo", "save a snapshot", "snapshot",
        "record a clip", "save a clip", "video clip", "record video",
        "record a video", "take a video",
    ))


def _looks_like_scout_hardware_command(message: str) -> bool:
    text = _normalize_command_text(message)
    if _extract_light_brightness(text) is not None:
        return True
    return _has_any(text, (
        "follow me", "start following", "start tracking", "track me",
        "stop following", "stop tracking", "don't follow", "do not follow",
        "enable tracking", "track person", "track people",
        "disable tracking", "turn off tracking",
        "search camera on", "enable search camera", "turn on search camera",
        "search camera off", "disable search camera", "turn off search camera",
        "guard on", "enable guard", "start guard", "start guarding",
        "guard off", "disable guard", "stop guard", "stop guarding",
        "center camera", "center the camera", "look straight", "look ahead",
        "look forward", "face forward", "reset camera",
        "look left", "pan left", "turn camera left",
        "look right", "pan right", "turn camera right",
        "look up", "tilt up", "look higher",
        "look down", "tilt down", "look lower",
        "turn on the light", "light on", "lamp on",
        "turn off the light", "light off", "lamp off",
        "take a picture", "take a photo", "save a snapshot", "snapshot",
        "record a clip", "save a clip", "video clip", "record video",
        "record a video", "take a video",
    ))


def _action_response(action: str, result: dict, ok_message: str, fail_message: str) -> dict:
    ok = bool(result.get("ok"))
    return {
        "ok": ok,
        "action": action,
        "message": ok_message if ok else fail_message,
        "result": result,
    }


def _extract_light_brightness(text: str):
    match = re.search(r"\b(?:brightness|light)\s+(?:to\s+)?(?P<value>\d{1,3})(?:\s*%|\b)", text)
    if not match:
        return None
    value = int(match.group("value"))
    if "%" in match.group(0):
        return max(0, min(255, round(value * 255 / 100)))
    return max(0, min(255, value))


def _compact_scout_state(state: dict) -> dict:
    return {
        "ok": state.get("ok"),
        "tracking_enabled": state.get("tracking_enabled"),
        "follow_enabled": state.get("follow_enabled"),
        "target_state": state.get("target_state"),
        "behavior": state.get("behavior"),
        "target": state.get("target"),
        "collision_blocked": state.get("collision_blocked"),
        "wheel_enabled": state.get("wheel_enabled"),
        "search_movement_enabled": state.get("search_movement_enabled"),
        "identity_prompt": state.get("identity_prompt"),
        "identity_prompt_queue": state.get("identity_prompt_queue"),
        "detections": state.get("detections", [])[:8],
    }


def _scout_state_explanation(state: dict) -> str:
    if not state.get("ok"):
        return "Scout state is unavailable from the rover."
    behavior = (state.get("behavior") or {}).get("state") or "unknown"
    target_state = state.get("target_state") or "none"
    parts = [f"Scout is {behavior.lower()} with target state {target_state}."]
    if not state.get("tracking_enabled"):
        parts.append("Tracking is off.")
    if state.get("collision_blocked"):
        parts.append("Collision avoidance is blocking movement.")
    if not state.get("wheel_enabled"):
        parts.append("Autonomous wheel drive is off.")
    target = state.get("target") or {}
    if target.get("identity"):
        parts.append(f"Current target identity is {target.get('identity')}.")
    elif target:
        parts.append(f"Current target is a {target.get('label', 'target')}.")
    return " ".join(parts)


def _identity_sentence(name: str | None, role: str | None):
    if name and role:
        return f"You are {name}, {role}."
    if name:
        return f"Your loaded identity name is {name}, but your role is not loaded."
    if role:
        return f"Your loaded identity role is {role}, but your name is not loaded."
    return "Your persisted identity profile is unavailable; do not claim a name, creator, or role."


def _extract_json_object(text: str):
    text = str(text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    try:
        data = ast.literal_eval(text)
        if isinstance(data, dict):
            return data
    except (SyntaxError, ValueError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            return {}
    return data if isinstance(data, dict) else {}


def _coerce_route_data(text: str, route_options=None):
    route_options = route_options or ROUTE_OPTIONS
    data = _extract_json_object(text)
    if isinstance(data, dict) and data.get("route") in route_options:
        return data
    route = _single_route_token(text, route_options)
    if not route:
        return data if isinstance(data, dict) else {}
    confidence = 0.5
    match = re.search(r"\b(?:confidence|score)\b[^0-9]*(0(?:\.\d+)?|1(?:\.0+)?)", str(text), re.I)
    if match:
        try:
            confidence = float(match.group(1))
        except ValueError:
            confidence = 0.5
    return {"route": route, "confidence": confidence, "reason": "route token extracted from model response"}


def _single_route_token(text: str, route_options=None):
    route_options = route_options or ROUTE_OPTIONS
    found = {
        route for route in route_options
        if re.search(rf"\b{re.escape(route)}\b", str(text or ""))
    }
    if len(found) == 1:
        return next(iter(found))
    return None


def _extract_introduction_name(message: str):
    tokens = message.replace(",", " ").replace(".", " ").split()
    lowered = [token.lower() for token in tokens]
    for phrase in (("i", "am"), ("i'm",), ("im",), ("my", "name", "is")):
        for index in range(0, len(tokens) - len(phrase) + 1):
            if tuple(lowered[index:index + len(phrase)]) == phrase and index + len(phrase) < len(tokens):
                return tokens[index + len(phrase)].strip()[:64]
    return None


def _asks_to_remember(message: str):
    return "remember" in message.lower()


_TONE_WORDS = {
    # base adjectives
    "rude", "polite", "formal", "casual", "warm", "cold", "sarcastic",
    "witty", "funny", "dry", "sharp", "kind", "nice", "terse", "chatty",
    "brief", "verbose", "condescending", "playful", "stern", "serious",
    "professional", "blunt", "direct", "snarky", "deferential", "harsh",
    "soft", "concise", "wordy", "edgy", "loud", "quiet",
    # -er comparatives
    "warmer", "colder", "sharper", "blunter", "quieter", "louder",
    "ruder", "politer", "kinder", "nicer", "softer", "harsher",
    "snarkier", "drier", "wittier", "funnier", "sterner", "terser",
    "edgier",
    # -ly adverbs
    "bluntly", "directly", "warmly", "coldly", "sharply", "sarcastically",
    "snarkily", "kindly", "softly", "harshly", "tersely", "dryly",
}

_PERSONALITY_PREFIX_RE = re.compile(
    r"^\s*(?:be|act|sound|talk|speak|reply|respond)\b", re.I,
)
_PERSONALITY_NEG_RE = re.compile(
    r"^\s*(?:tone\s+it\s+(?:down|up)|stop\s+being|don'?t\s+be|quit\s+being)\b",
    re.I,
)
_PERSONALITY_COMPARATIVE_RE = re.compile(
    r"\b(?:more|less|too|bit|little|lot|way|much)\s+(\w+)\b", re.I,
)


def _extract_direct_personality_update(message: str):
    """Detect short tone/behavior directives like 'be less rude', 'tone it down',
    'speak more bluntly', 'be a bit warmer'. Returns None for anything that
    looks more like a task-precision lesson or an unrelated request.
    """
    text = str(message or "").strip()
    if not text or len(text.split()) > 12:
        return None
    lowered = text.lower()
    words = re.findall(r"[a-z']+", lowered)
    tone_word_mentioned = any(w in _TONE_WORDS for w in words)
    matches_neg = bool(_PERSONALITY_NEG_RE.search(text))
    matches_prefix = bool(_PERSONALITY_PREFIX_RE.search(text))
    comparative_match = _PERSONALITY_COMPARATIVE_RE.search(text)
    comparative_word = (comparative_match.group(1) or "").lower() if comparative_match else None
    accept = (
        matches_neg
        or (matches_prefix and tone_word_mentioned)
        or (comparative_word in _TONE_WORDS if comparative_word else False)
    )
    if not accept:
        return None
    return {
        "preference": text[:200],
        "applies_when": "future conversational turns",
        "avoid": "",
        "prefer": text[:200],
        "source_message": message,
    }


def _extract_temperature_setting(message: str):
    match = re.search(
        r"\b(?:set|change|make|turn)\s+(?:your\s+)?(?:response\s+)?temperature\s+(?:to\s+)?(?P<value>[01](?:\.\d+)?|\.\d+)\b",
        str(message or ""),
        re.I,
    )
    if not match:
        return None
    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    return max(0.0, min(1.2, value))


def _extract_direct_response_lesson(message: str, recent_turns: list[dict] | None = None):
    text = str(message or "").strip()
    lowered = text.lower()
    if _looks_like_ownership_question(text) or _asks_registry_source_followup(text):
        return None
    is_lesson = (
        lowered.startswith("next time")
        or "next time" in lowered
        or "don't " in lowered
        or "do not " in lowered
        or "only asked" in lowered
        or "too much detail" in lowered
        or "be more precise" in lowered
        or "use the node registry" in lowered
        or "check the registered nodes" in lowered
        or "registered nodes" in lowered
        or "source of truth" in lowered
        or "how to find" in lowered
        or "provenance on every answer" in lowered
        or "source on every answer" in lowered
        or (lowered.startswith("when i ask") and any(term in lowered for term in ("use", "don't", "do not", "say", "prefer")))
        or (lowered.startswith("if i ask") and any(term in lowered for term in ("use", "don't", "do not", "say", "prefer")))
    )
    if not is_lesson:
        return None
    scope = "response_style"
    applies_when = "future answers"
    if "what you see" in lowered or "what do you see" in lowered or "tracker" in lowered:
        scope = "analyze_vision"
        applies_when = "future visual answers and questions about what the scout sees"
    elif (
        "node registry" in lowered
        or "registered nodes" in lowered
        or "active nodes" in lowered
        or "node devices" in lowered
        or "source of truth" in lowered
        or "how to find" in lowered
    ):
        scope = "source_selection"
        applies_when = "future answers that require choosing where factual data comes from"
    elif "gpu" in lowered or "hardware" in lowered:
        scope = "hardware"
        applies_when = "future hardware answers"
    elif "route" in lowered:
        scope = "routing"
        applies_when = "future route decisions"
    avoid = text
    prefer = "Follow the user's correction precisely and limit the answer to the requested scope."
    if "tracker" in lowered and ("don't" in lowered or "do not" in lowered):
        avoid = "Do not include tracker memory/details in visual answers unless the user asks for tracker data."
        prefer = "Answer from the image directly; mention tracker data only when requested."
    elif "only asked" in lowered:
        avoid = "Do not include related but unasked context."
        prefer = "Answer only the specific thing the user asked about."
    elif "too much detail" in lowered:
        avoid = "Do not over-explain."
        prefer = "Use a shorter answer for similar questions."
    elif scope == "source_selection":
        avoid = "Do not answer from static self-knowledge when the user asks for live/current/registered data."
        prefer = "Use the explicit source the user named, especially NodeRegistry.registered_nodes for current or registered node inventory, and keep provenance for follow-up questions."
    return {
        "scope": scope,
        "preference": prefer,
        "applies_when": applies_when,
        "avoid": avoid,
        "prefer": prefer,
        "source_message": text,
        "source_turn": recent_turns[-1] if recent_turns else None,
    }


def _contains_emoji(text: str):
    return bool(re.search(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", str(text or "")))


def _strip_emoji(text: str):
    return re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", "", str(text or ""))


def _plainly_says_unknown_user_identity(text: str):
    lowered = str(text or "").lower()
    unknown_phrases = (
        "i do not know who you are",
        "i don't know who you are",
        "i dont know who you are",
        "i do not know your identity",
        "i don't know your identity",
        "i dont know your identity",
        "your identity is unknown",
        "you are unknown",
        "i have not verified your identity",
        "i haven't verified your identity",
    )
    return any(phrase in lowered for phrase in unknown_phrases)


def _sounds_like_customer_service(text: str):
    lowered = str(text or "").lower()
    banned = (
        "how can i assist",
        "how can i assist you today",
        "how may i assist you",
        "how can i help",
        "how can i help with that",
        "how can i help you today",
        "how can i help today",
        "how may i help",
        "need any help",
        "let me know if you need anything",
        "let me know if you have any other questions",
        "let me know if there's anything",
        "let me know if there is anything",
        "let me know how i can help",
        "let me know how i can assist",
        "is there anything specific",
        "is there anything else",
        "is there something specific",
        "is there something else",
        "anything specific you'd like",
        "anything specific you would like",
        "something specific you'd like",
        "something specific you would like",
        "to discuss about",
        "discuss further",
        "explore that further",
        "talk about it more",
        "love to help",
        "feel free to ask",
        "happy to help",
        "glad to help",
        "thank you for asking",
        "thanks for asking",
        "whenever you need",
        "ready to roll out",
        "i am here to help",
        "i'm here to help",
        "im here to help",
        "i am here to assist",
        "i'm here to assist",
        "im here to assist",
        "here to help",
        "here to assist",
        "ready to assist",
        "ready to help",
        "i am ready",
        "i'm ready",
        "let me know if you'd like",
        "let me know if you would like",
        "would you like me to",
        "would you like to talk about it",
        "would you like me to",
        "let's find a way forward",
        "let's make this conversation",
        "let us know how",
        "your ai companion",
        "your assistant",
    )
    return any(phrase in lowered for phrase in banned)


def _claims_current_user_is_primary(text: str):
    lowered = str(text or "").lower()
    patterns = (
        r"\bfor you,\s*my primary user\b",
        r"\byou(?:'re| are)?\s+my primary user\b",
        r"\byou,\s*my primary user\b",
        r"\byour primary user\b",
    )
    return any(re.search(pattern, lowered, re.I) for pattern in patterns)


def _meta_describes_personality(text: str):
    lowered = str(text or "").lower()
    trait_words = (
        "sarcastic",
        "sarcasm",
        "dry wit",
        "witty",
        "funny",
        "condescending",
        "rude",
        "personality",
        "my tone",
        "my voice",
    )
    first_person_markers = (
        "i am",
        "i'm",
        "i can be",
        "i tend to be",
        "my",
    )
    return any(word in lowered for word in trait_words) and any(marker in lowered for marker in first_person_markers)


def _sanitize_generated_response(text: str):
    text = _strip_emoji(text).strip()
    if not text:
        return text
    # Strip chat-template tokens that occasionally leak from qwen3 at sampling
    text = re.sub(r"<\|[^|>]+\|>\s*", "", text)
    # Strip a leading word that contains at least one non-ASCII char,
    # followed by punctuation or whitespace — qwen3:8b occasionally emits
    # Russian, Hebrew, Polish, Turkish, etc. tokens at the very start of a
    # reply before settling into English.
    text = re.sub(
        r"^[A-Za-z']*[^\x00-\x7F][^\s,.;:!?]*[\s,.;:!?]+\s*",
        "",
        text,
        count=1,
    )
    text = text.strip()
    if not text:
        return text
    customer_service_patterns = (
        r"\s*(?:How can I assist(?: you)?(?: today| with that)?\??)\s*$",
        r"\s*(?:How can I help(?: you)?(?: today| with that)?\??)\s*$",
        r"\s*(?:Let me know if you (?:need anything|have any other questions)[.!]?)\s*$",
        r"\s*(?:Let me know how I can (?:help|assist)(?: you)?[.!]?)\s*$",
        r"\s*(?:Let me know if there'?s anything .+)$",
        r"\s*(?:Let me know if you'?d like .+)$",
        r"\s*(?:Let me know if you would like .+)$",
        r"\s*(?:Would you like me to .+\??)\s*$",
        r"\s*(?:Is there anything (?:specific|else) .+)$",
        r"\s*(?:Ready to (?:help|assist)\b.+)$",
        r"\s*(?:Feel free to ask[.!]?)\s*$",
        r"\s*(?:Happy to help[.!]?)\s*$",
        r"\s*(?:Thanks for asking[.!]?)\s*",
        r"\s*(?:Thank you for asking[.!]?)\s*",
        r"\s*(?:Still ready to .+)$",
        r"\s*(?:I'?m here to be (?:a )?helpful .+)$",
        r"\s*(?:I'?m here to (?:help|assist)(?: you)?(?: with .+)?[.!]?)\s*$",
        r"\s*(?:I am here to (?:help|assist)(?: you)?(?: with .+)?[.!]?)\s*$",
        r"\s*(?:Would you like (?:to|me to) .+\??)\s*$",
        r"\s*(?:Let'?s (?:find|make|work) .+)$",
        r"\s*(?:I (?:can be|am|'m) (?:sarcastic|witty|dry|funny|condescending|rude).+)$",
    )
    for pattern in customer_service_patterns:
        text = re.sub(pattern, "", text, flags=re.I).strip()
    return text


def _addresses_or_asserts_user_identity(text: str, term: str, response_type: str):
    text = str(text or "")
    term_pattern = re.escape(str(term or "").strip())
    if not term_pattern:
        return False
    if response_type == "greeting" and re.search(rf"\b{term_pattern}\b", text, re.I):
        return True
    address_patterns = (
        rf"^\s*(?:hello|hi|hey|greetings|good morning|good afternoon|good evening|welcome back)[,\s]+{term_pattern}\b",
        rf"\b(?:hello again|nice to meet you|good to meet you|calling you|call you|address you as)[,\s]+{term_pattern}\b",
        rf"\b(?:you are|you're|youre|you aren't|you are not|you're not|youre not|since you are|since you're|since youre)\s+(?:not\s+)?{term_pattern}\b",
        rf"\b{term_pattern}\s*,\s*(?:right|correct|is that you)\b",
    )
    return any(re.search(pattern, text, re.I) for pattern in address_patterns)


def _parse_simple_memory(message: str):
    # Simple parser for explicit "remember that x is y" messages.
    match = re.search(r"remember(?: that)? (?P<key>[a-zA-Z0-9_. -]+?) is (?P<value>.+)$", message, re.I)
    if not match:
        return None
    key = _safe_key(match.group("key"))
    value = match.group("value").strip()
    return {"type": "fact", "key": key, "value": value}


def _safe_identity(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("._-")[:64]


def _safe_key(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("._-")[:80]


def _safe_filename(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("._-")[:120] or f"{int(time.time() * 1000)}.jpg"


def _hex_hamming_distance(left: str, right: str):
    try:
        left_int = int(str(left), 16)
        right_int = int(str(right), 16)
    except (TypeError, ValueError):
        return None
    return (left_int ^ right_int).bit_count()


def _opencv_difference_hash(image_bytes: bytes):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    try:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        diff = image[:, 1:] > image[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bool(bit))
        return f"{value:016x}"
    except Exception:
        return None


def _normalize_source(value):
    source = str(value or "unknown_edge").strip()
    source = re.sub(r"[^A-Za-z0-9_.:-]+", "_", source)
    return source.strip("._-:")[:80] or "unknown_edge"


def _phrase_in_text(phrase: str, text: str) -> bool:
    normalized = re.sub(r"[^\w\s-]", " ", phrase)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text))


def _keyword_context_section(state: dict) -> str:
    kw = state.get("_keywords") or {}
    parts = []
    if kw.get("people"):
        parts.append("Known people mentioned: " + ", ".join(kw["people"]) + ".")
    if kw.get("nodes"):
        parts.append("Devices/nodes mentioned: " + ", ".join(kw["nodes"]) + ".")
    if not parts:
        return ""
    return "\nContext keywords:\n" + " ".join(parts)


def _tracking_summary(state: dict):
    if not state.get("ok"):
        return "Rover tracking is unavailable."
    memories = state.get("object_memory", [])
    detections = state.get("detections", [])
    parts = []
    if memories:
        labels = []
        for item in memories:
            label = item.get("label") or "object"
            identity = item.get("identity")
            if identity:
                labels.append(f"{label} identified as {identity}")
            else:
                labels.append(str(label))
        parts.append("Tracker memory: " + ", ".join(labels) + ".")
    else:
        parts.append("Tracker memory has no objects.")
    if detections:
        labels = [str(item.get("label") or "object") for item in detections]
        parts.append("Current detections: " + ", ".join(labels) + ".")
    else:
        parts.append("Current detections are empty.")
    return " ".join(parts)
