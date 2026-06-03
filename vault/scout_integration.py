from __future__ import annotations

import base64
import ast
import collections
import hashlib
import json
import re
import shutil
import time
import threading
from pathlib import Path
from urllib.parse import quote

import requests

from config import DATA_DIR, FACE_REFERENCES_DIR, PEOPLE_DIR, ROOT_DIR, SCOUT_BATTERY_URL, SCOUT_ROBOT_URL, SCOUT_URL
from models import get_model, model_manifest
from mood_engine import MoodEngine
from response_composer import ResponseComposer

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


def _is_plain_confirmation_denial(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()
    return t in {"no", "nope", "nah", "negative", "not right", "wrong"}


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
        if feedback.get("type") == "route_confirmation":
            return None
        correction = str(feedback.get("user_correction") or "").strip()
        if correction:
            return _extract_correction(correction) or correction
    return None


def _presence_confirmation(presence_context: dict | None) -> str | None:
    if not isinstance(presence_context, dict):
        return None
    feedback = presence_context.get("routing_feedback")
    if not isinstance(feedback, dict) or feedback.get("type") != "route_confirmation":
        return None
    previous = str(feedback.get("previous_user_message") or "").strip()
    return previous or None


def _corrected_route_input(original_message: str, correction: str, presence_context: dict | None = None) -> str:
    if isinstance(presence_context, dict):
        clarified = str(presence_context.get("clarified_request") or "").strip()
        if clarified:
            return clarified
    return f"{original_message}\nCorrection from user: {correction}"


def _presence_conversation_context(presence_context: dict | None) -> dict:
    if not isinstance(presence_context, dict):
        return {}
    result = {}
    reply_context = presence_context.get("reply_context")
    if isinstance(reply_context, dict):
        result["reply_context"] = {
            key: reply_context.get(key)
            for key in (
                "type",
                "current_user_message",
                "previous_user_message",
                "previous_assistant_message",
            )
            if reply_context.get(key)
        }
    chat_context = presence_context.get("chat_context")
    if isinstance(chat_context, list):
        full_chat = []
        for entry in chat_context:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            text = str(entry.get("text") or "").strip()
            if role in {"user", "assistant", "error"} and text:
                full_chat.append({"role": role, "text": text})
        if full_chat:
            result["chat_context"] = full_chat
    return result


def _conversation_user_turns(presence_context: dict | None, current_message: str | None = None) -> list[str]:
    context = _presence_conversation_context(presence_context).get("chat_context") or []
    turns = [
        str(entry.get("text") or "").strip()
        for entry in context
        if entry.get("role") == "user" and str(entry.get("text") or "").strip()
    ]
    current = str(current_message or "").strip()
    if current and turns and turns[-1] == current:
        turns = turns[:-1]
    return turns


def _extract_context_phrase(text: str) -> tuple[str, str] | None:
    cleaned = str(text or "").strip()
    patterns = (
        r"\b(?P<label>marker\s+word|test\s+phrase|test\s+word|code\s+word|marker|phrase|token)\s+(?:is|equals|was)\s+(?P<value>[^.?!]+)",
        r"\bremember\s+(?:the\s+)?(?P<label>marker\s+word|test\s+phrase|test\s+word|code\s+word|marker|phrase|token)\s+(?:is|equals|as)\s+(?P<value>[^.?!]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            label = re.sub(r"\s+", " ", match.group("label").lower()).strip()
            value = match.group("value").strip(" \"'")
            if value:
                return label, value
    return None


def _extract_context_fact(text: str) -> tuple[str, str, str | None] | None:
    cleaned = str(text or "").strip()
    patterns = (
        r"\bremember\s+for\s+this\s+chat\s+that\s+(?:the\s+)?(?P<key>[a-zA-Z0-9_. -]+?)\s+(?:is|equals|was)\s+(?P<value>[^.?!]+)",
        r"\bremember\s+that\s+my\s+(?P<key>[a-zA-Z0-9_. -]+?)\s+(?:is|equals|was)\s+(?P<value>[^.?!]+)",
        r"\bmy\s+(?P<key>[a-zA-Z0-9_. -]+?)\s+(?:is|equals|was)\s+(?P<value>[^.?!]+)",
        r"\bi\s+(?P<key>like|prefer)\s+(?P<value>[^.?!]+)",
        r"\bremember\s+(?P<person>Chris)\s+(?P<key>likes|prefers|uses|has)\s+(?P<value>[^.?!]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        key = re.sub(r"\s+", " ", match.group("key").lower()).strip()
        value = match.group("value").strip(" \"'")
        person = match.groupdict().get("person")
        if key and value:
            return key, value, person
    return None


# Vision routing is keyword-gated, NOT just router-confidence gated. The
# router occasionally picks analyze_vision for prompts like "describe a
# sunset" or "compose a haiku about rain on the lake" because they
# contain scene-like words; the keyword check requires the message to be
# *about* the live camera scene before vision actually fires. If the LLM
# later decides scene analysis would help, it can ask the user explicitly
# (see _llm_asks_for_vision) — that path also bypasses the keyword gate.
_VISION_TRIGGER_PHRASES = (
    # Direct sight questions
    "do you see", "can you see", "are you seeing",
    "what do you see", "what can you see",
    "tell me what you see",
    # Scene/room/area description (live)
    "describe the scene", "describe the room", "describe the area",
    "describe what you see", "describe what's there", "describe what is there",
    "describe what's visible", "what's visible", "what is visible",
    # Spatial "what/who is here/there/in front/around/behind"
    "what's in front of you", "what is in front of you", "in front of you",
    "what's around you", "what is around you", "around you",
    "what's behind you", "what is behind you", "behind you",
    "what's there", "what is there",
    "who is here", "who's here", "who is there", "who's there",
    "is anyone here", "is anyone there",
    # Direct camera/view references
    "the camera", "your camera", "from your camera", "through your camera",
    "the picture", "the image",
    "current view", "current scene", "live view", "the scene",
    "in your view", "in your field of view",
    "look at", "looking at",
    "in the room", "in this room", "in the area",
)


def _has_vision_trigger(message: str) -> bool:
    """True iff message contains an explicit live-camera phrasing."""
    low = (message or "").casefold()
    return any(phrase in low for phrase in _VISION_TRIGGER_PHRASES)


# Phrases the LLM uses when it wants to ask the user for permission to
# run a full vision analysis. If we detect one in a response, we stash
# the same marker the deterministic vision short-circuit uses, so the
# user's next "yes" routes back through the analyze_vision branch with
# force_full_vision=True.
_LLM_VISION_REQUEST_PHRASES = (
    "would you like me to analyze the scene",
    "would you like me to look at the scene",
    "would you like me to look at what",
    "should i analyze the scene",
    "should i look at the scene",
    "should i look at what's",
)


def _llm_asks_for_vision(text: str) -> bool:
    low = (text or "").casefold()
    return any(p in low for p in _LLM_VISION_REQUEST_PHRASES)


def _detection_summary(state: dict | None) -> str | None:
    """Natural-language summary of what the camera node is currently
    seeing, from its detection metadata. Returns None when there are no
    usable detections — caller should fall back to the full vision LLM."""
    if not isinstance(state, dict):
        return None
    detections = state.get("detections") or []
    if not detections:
        return None
    from collections import Counter
    label_counts = Counter()
    identified_people: list[str] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label") or "").strip().lower()
        if not label:
            continue
        label_counts[label] += 1
        if label == "person":
            identity = str(det.get("identity") or "").strip()
            if identity and identity.lower() not in {"unknown", "none"}:
                identified_people.append(identity)
    if not label_counts:
        return None
    parts = []
    for label, count in label_counts.most_common():
        noun = label if count == 1 else _pluralize_noun(label)
        parts.append(f"{count} {noun}")
    if len(parts) > 1:
        base = "I see " + ", ".join(parts[:-1]) + " and " + parts[-1]
    else:
        base = "I see " + parts[0]
    if identified_people:
        unique = sorted(set(identified_people))
        if len(unique) == 1:
            base += f", identified as {unique[0]}"
        else:
            base += f", identified as {', '.join(unique[:-1])} and {unique[-1]}"
    return base + "."


def _pluralize_noun(noun: str) -> str:
    if noun == "person":
        return "people"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _conversation_setup_answer(message: str) -> str | None:
    phrase = _extract_context_phrase(message)
    if phrase:
        label, value = phrase
        return f"Got it. The {label} is {value}."
    fact = _extract_context_fact(message)
    if fact:
        key, value, person = fact
        if person:
            return f"I remember {person} {key} {value}."
        if key in {"like", "prefer"}:
            verb = "like" if key == "like" else "prefer"
            return f"I remember you {verb} {value}."
        return f"I remember your {key} is {value}."
    return None


def _matching_context_fact(user_turns: list[str], key_pattern: str, person: str | None = None) -> tuple[str, str, str | None] | None:
    for turn in reversed(user_turns):
        fact = _extract_context_fact(turn)
        if not fact:
            continue
        key, value, fact_person = fact
        if person and (fact_person or "").lower() != person.lower():
            continue
        if re.search(key_pattern, key, re.I):
            return key, value, fact_person
    return None


def _recent_conversation_answer(message: str, presence_context: dict | None) -> str | None:
    text = _canonical_intent_text(message)
    if not _asks_recent_conversation(text):
        return None
    user_turns = _conversation_user_turns(presence_context, message)
    if not user_turns:
        return "I don't have earlier chat context for that in this session."

    if (
        re.search(r"\bwhat\s+(marker\s+word|test\s+phrase|test\s+word|phrase|word)\s+did\s+i\s+just\s+(say|give|tell)\b", text)
        or re.search(r"\bwhat\s+was\s+(the\s+)?(marker\s+word|test\s+phrase|test\s+word|marker|phrase|word)\b", text)
    ):
        for turn in reversed(user_turns):
            phrase = _extract_context_phrase(turn)
            if phrase:
                label, value = phrase
                return f"The {label} was {value}."
        return "I don't see a marker word or test phrase in the recent chat context."

    if re.search(r"\bwhat\s+(?:.*\s+)?code\s+did\s+i\s+(give|tell)\s+you\b", text):
        fact = _matching_context_fact(user_turns, r"\bcode\b")
        if fact:
            key, value, _person = fact
            return f"Your {key} is {value}."
        return "I don't see a code in the recent chat context."

    if re.search(r"\bwhat\s+is\s+my\s+favorite\s+color\b", text):
        fact = _matching_context_fact(user_turns, r"\bfavorite color\b")
        if fact:
            _key, value, _person = fact
            return f"Your favorite color is {value}."
        return "I don't see your favorite color in the recent chat context."

    if re.search(r"\bwhat\s+kind\s+of\s+art\s+did\s+i\s+say\s+i\s+(like|prefer)\b", text):
        fact = _matching_context_fact(user_turns, r"\b(like|prefer)\b")
        if fact:
            _key, value, _person = fact
            return f"You said you like {value}."
        return "I don't see an art preference in the recent chat context."

    if re.search(r"\bwhat\s+does\s+chris\s+(like|prefer|use|have)\b", text):
        fact = _matching_context_fact(user_turns, r"\b(likes|prefers|uses|has)\b", person="Chris")
        if fact:
            key, value, person = fact
            return f"{person} {key} {value}."
        return "I don't see that preference for Chris in the recent chat context."

    if re.search(r"\bwhat\s+did\s+i\s+(ask|say|tell)\s+immediately\s+before\s+this\b", text):
        return f"You said: {user_turns[-1]}"

    if re.search(r"\bwhat\s+was\s+my\s+(last|previous)\s+(question|request|message)\b", text):
        if "question" in text:
            for turn in reversed(user_turns):
                if "?" in turn:
                    return f"Your previous question was: {turn}"
        return f"Your previous message was: {user_turns[-1]}"

    if re.search(r"\b(what|which)\s+did\s+i\s+just\s+(say|ask|tell)\b", text):
        return f"You said: {user_turns[-1]}"

    if re.search(r"\bwhat\s+i\s+just\s+asked\b", text):
        for turn in reversed(user_turns):
            if "?" in turn:
                return f"You just asked: {turn}"
        return f"You just said: {user_turns[-1]}"

    if "why that word" in text:
        for turn in reversed(user_turns):
            phrase = _extract_context_phrase(turn)
            if phrase:
                label, value = phrase
                return f"You used {value} as the {label}; I don't know a deeper reason unless you give me one."
        return "I don't have enough recent context to know which word you mean."

    if re.search(r"\bwhat\s+have\s+we\s+talked\s+about\b", text):
        topics = []
        for turn in user_turns[-20:]:
            phrase = _extract_context_phrase(turn)
            fact = _extract_context_fact(turn)
            if phrase:
                topics.append(f"the {phrase[0]} {phrase[1]}")
            elif fact:
                key, value, person = fact
                topics.append(f"{person + ' ' if person else 'your '}{key} {value}")
            elif "?" in turn:
                topics.append(turn.rstrip("?"))
        if topics:
            return "We talked about " + "; ".join(topics[-5:]) + "."
        return "We have mostly been testing routing and memory in this chat."

    return None


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
        self.scout_battery_url = SCOUT_BATTERY_URL.rstrip("/")
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
        self.mood_engine = MoodEngine(self.self_dir)
        self.mood_engine.import_legacy_response_settings(self.response_settings())
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
        self.response_composer = ResponseComposer(self.chat_model)
        try:
            from storage.vector_store import MemoryStore
            self.embed_model = get_model("embed")
            self.memory_store = MemoryStore(embedder=self.embed_model)
        except Exception as exc:
            print(f"[memory_store] disabled: {exc}")
            self.embed_model = None
            self.memory_store = None
        try:
            from world import WorldKnowledgeStore
            self.world_store = WorldKnowledgeStore(text_embedder=self.embed_model)
        except Exception as exc:
            print(f"[world_store] disabled: {exc}")
            self.world_store = None
        # Seed the "assistant" identity bucket from identity_profile +
        # self/identity.json so recall about Luhkas itself goes through the
        # same vector path as recall about users. Duplicate guard prevents
        # bloat on repeated startups.
        try:
            self._seed_assistant_memory()
        except Exception as exc:
            print(f"[memory_store] assistant seed failed: {exc}")

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
            "personality": ["personality", "mood", "style_state"],
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
            mem_src = getattr(self, "_current_memory_sources", None) or {}
            world_src = getattr(self, "_current_world_sources", None) or {}
            recalled = mem_src.get("recalled_facts") or []
            chat_turn_count = mem_src.get("recent_chat_turns") or 0
            identity_scope = mem_src.get("identity_scope") or "unknown"
            wiki_hits = world_src.get("wiki_hits") or []
            media_hits = world_src.get("media_hits") or []
            if recalled:
                sources.append({
                    "name": "memory_store",
                    "role": "identity-scoped semantic memory (LanceDB)",
                    "ok": True,
                    "identity": identity_scope,
                    "facts_consulted": recalled,
                })
            if chat_turn_count:
                sources.append({
                    "name": "session_chat",
                    "role": "recent conversation turns in this session",
                    "ok": True,
                    "turns_consulted": chat_turn_count,
                })
            if wiki_hits:
                sources.append({
                    "name": "wikipedia",
                    "role": "offline Wikipedia corpus (WorldKnowledgeStore)",
                    "ok": True,
                    "articles_consulted": [
                        {"title": h.get("title"), "section": h.get("section"),
                         "distance": h.get("distance")}
                        for h in wiki_hits
                    ],
                })
            if media_hits:
                sources.append({
                    "name": "media_vault",
                    "role": "offline media transcripts/captions (WorldKnowledgeStore)",
                    "ok": True,
                    "assets_consulted": [
                        {"asset_id": h.get("asset_id"), "modality": h.get("modality"),
                         "distance": h.get("distance")}
                        for h in media_hits
                    ],
                })
            if not recalled and not chat_turn_count and not wiki_hits and not media_hits:
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

    @staticmethod
    def _format_response_lessons_for_prompt(lessons):
        """Convert raw lessons into safe directive strings for LLM prompts.

        Background: the lesson-recorder stores the user's correction verbatim
        in the `avoid` field. The small chat model can confuse that for
        content and parrot it back ("what's my name" -> "I don't have a pet").
        We strip those raw user phrases and keep only the `prefer` directive
        (which is positively-framed LLM-authored guidance)."""
        out = []
        for lsn in lessons or []:
            if not isinstance(lsn, dict):
                continue
            prefer = str(lsn.get("prefer") or "").strip()
            scope = str(lsn.get("scope") or "general").strip()
            if prefer:
                out.append(f"[{scope}] {prefer}")
        return out

    def response_settings(self):
        path = self.self_dir / "response_settings.json"
        data = self._load_json_file(path)
        return data if isinstance(data, dict) else self._default_response_settings()

    def write_response_settings(self, settings: dict):
        path = self.self_dir / "response_settings.json"
        path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"ok": True, "path": str(path), "settings": settings}

    def response_context(self, state: dict | None = None):
        identity_context = self.response_identity_context(state or {})
        return {
            "response_lessons": self.response_lessons(),
            "response_settings": self.response_settings(),
            "identity_context": identity_context,
            "voice_state": self.mood_engine.voice_state(identity_context),
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
                "Answer in Luhkas's first-person voice: direct, dry, occasionally warm. "
                "The user is Chris — familiar territory, you may be informal."
            )
        else:
            voice_line = (
                "Answer in Luhkas's first-person voice: direct, dry, slightly clipped. "
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
            "Luhkas is the assistant's name, not a transcript label. Mention the name naturally only when identity is relevant.",
            "Scout, vault, wall nodes, cameras, and devices are surfaces or components, never separate speaker identities.",
            "Never answer 'I am Scout', 'I'm Scout', or identify yourself as a node.",
            identity_context.get("addressing_rule")
            or "Do not address the current user by name or title unless identity is verified.",
        ]
        directive_block = self._behavior_directive_block(identity_context)
        if directive_block:
            lines.append(directive_block)
        return "\n".join(lines)

    def _behavior_directive_block(self, identity_context: dict) -> str:
        """Inject resolved voice state, not raw and possibly contradictory notes."""
        return "\n".join(self.mood_engine.voice_contract_lines(identity_context))

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
        style_state = self.mood_engine.apply_style_update(entry)
        behavior["resolved"] = style_state.get("resolved", {})
        behavior["style_state_path"] = str(self.mood_engine.style_path)
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

    def camera_state_for_node(self, node_id: str | None):
        camera_url = self._camera_url_for_node(node_id)
        return self._get_json(f"{camera_url}/meta") or {
            "ok": False,
            "error": "camera_meta_unavailable",
            "node_id": node_id or "scout",
            "object_memory": [],
            "detections": [],
        }

    def _camera_url_for_node(self, node_id: str | None) -> str:
        node = str(node_id or "").strip() or "scout"
        if self.node_registry is not None:
            try:
                url = self.node_registry.node_url(node, "vision")
            except Exception:
                url = None
            if url:
                return str(url).rstrip("/")
        return self.scout_url

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
                        node_id=session_key,
                    )
                    self._ensure_result_provenance(result, message)
                    identity_context = self.response_identity_context({"ok": True})
                    self.mood_engine.record_interaction(
                        result or {},
                        identity_verified=bool(identity_context.get("may_address_primary_user")),
                    )
            finally:
                # Save context back
                session.active_identity = self.active_identity
                session.turns = self.turns
            # If identity was newly established, migrate any existing session
            new_identity = self.active_identity
            if new_identity and new_identity != prev_identity:
                self._migrate_identity(new_identity, session_key)
            return result

    def _handle_message_impl(self, message: str, source=None, presence_context: dict | None = None, node_id: str = ""):
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

        state = self.camera_state_for_node(node_id)
        self._adopt_identity_from_state(state, node_id=node_id)

        if _looks_like_scout_action(message) and not _asks_broad_status_report(_canonical_intent_text(message)):
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
            if pending is not None and (
                _is_affirmative(message)
                or _extract_correction(message)
                or _presence_correction(presence_context)
                or _is_plain_confirmation_denial(message)
            ):
                return self._handle_confirmation(message, pending, state, source, presence_context=presence_context)

        confirmed_message = _presence_confirmation(presence_context)
        if confirmed_message:
            import deterministic_router as _dr
            confirmed_route = self.route_message(confirmed_message, state)
            confirmed_route.update(_presence_route_context(presence_context))
            if confirmed_route.get("ok"):
                self._complete_route_for_learning(confirmed_message, confirmed_route, state)
                _dr.learn(confirmed_message, confirmed_route, confirmed_by="presence_route_confirmation")
                actions.append({
                    "name": "learn_deterministic_route",
                    "ok": True,
                    "result": {
                        "input": confirmed_message,
                        "route": confirmed_route.get("route"),
                        "self_route": (confirmed_route.get("self_route") or {}).get("route")
                        if isinstance(confirmed_route.get("self_route"), dict)
                        else None,
                        "confidence": confirmed_route.get("confidence"),
                    },
                })
                response = self._dispatch_route(
                    confirmed_message,
                    confirmed_route,
                    state,
                    actions,
                    source=source,
                    presence_context=presence_context,
                )
                turn = {
                    "message": message,
                    "source": source,
                    "route": confirmed_route,
                    "response": response,
                    "active_identity": self.active_identity,
                    "actions": actions,
                    "routing_feedback": presence_context.get("routing_feedback") if isinstance(presence_context, dict) else None,
                    "answer_provenance": self.build_answer_provenance(confirmed_message, confirmed_route, state),
                }
                self.turns.append(turn)
                self.turns = self.turns[-30:]
                return {"ok": True, **turn}

        correction = _presence_correction(presence_context)
        if correction and isinstance(presence_context, dict):
            feedback_context = presence_context.get("routing_feedback")
            if isinstance(feedback_context, dict):
                original_message = str(feedback_context.get("previous_user_message") or "").strip() or message
                corrected_input = _corrected_route_input(original_message, correction, presence_context)
                corrected_route = self.route_message(corrected_input, state)
                corrected_route.update(_presence_route_context(presence_context))
                if corrected_route.get("ok"):
                    corrected_desc = _route_description(corrected_route, correction)
                    response = f"I think you mean {corrected_desc}. Is that right?"
                    turn = {
                        "message": message,
                        "source": source,
                        "route": corrected_route,
                        "response": response,
                        "active_identity": self.active_identity,
                        "actions": actions,
                        "routing_feedback": feedback_context,
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

        recent_answer = _recent_conversation_answer(message, presence_context) if _asks_recent_conversation(_canonical_intent_text(message)) else None
        if recent_answer is not None:
            route = {
                "ok": True,
                "route": "general_question",
                "confidence": 1.0,
                "reason": "answers from recent chat context",
                "attempts": 0,
                "deterministic": True,
            }
            response = recent_answer
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

        fast_route = _fast_route_message(message)
        if fast_route is not None and fast_route.get("deterministic"):
            fast_route.update(_presence_route_context(presence_context))
            response = self._dispatch_route(
                message,
                fast_route,
                state,
                actions,
                source=source,
                presence_context=presence_context,
            )
            turn = {
                "message": message,
                "source": source,
                "route": fast_route,
                "response": response,
                "active_identity": self.active_identity,
                "actions": actions,
                "answer_provenance": self.build_answer_provenance(message, fast_route, state),
            }
            self.turns.append(turn)
            self.turns = self.turns[-30:]
            return {"ok": True, **turn}

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
        # general_question is the conversational catch-all — asking "did you
        # mean a general question?" is never useful, so we never confirm it.
        if (
            route.get("ok")
            and not route.get("from_cache")
            and not route.get("deterministic")
            and float(route.get("confidence") or 0.0) < 0.88
            and route.get("route") != "general_question"
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

        response = self._dispatch_route(message, route, state, actions, source=source, presence_context=presence_context)
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

    def _dispatch_route(
        self,
        message: str,
        route: dict,
        state: dict,
        actions: list,
        source: str | None = None,
        presence_context: dict | None = None,
    ) -> str:
        """Execute the appropriate handler for an already-determined route."""
        if not route.get("ok"):
            return self.generate_response(
                "routing_error", message, state, {"route": route}, max_tokens=100
            )

        # force_full_vision override: vault_runtime sets this flag in
        # presence_context after the user confirms an LLM-initiated scene-
        # analysis request. At that point the message is the ORIGINAL one
        # the user asked (e.g., "what color is my mug"), which the router
        # likely won't classify as analyze_vision. Override the route
        # regardless so analyze_vision actually fires.
        if (
            isinstance(presence_context, dict)
            and presence_context.get("force_full_vision")
            and route.get("route") != "analyze_vision"
        ):
            route["route"] = "analyze_vision"
            route.setdefault("forced_by", "force_full_vision_after_confirmation")

        # Reset per-turn transient sources so provenance for this turn doesn't
        # carry recall results from a prior turn that bypassed answer_with_context.
        self._current_memory_sources = {"recalled_facts": [], "recent_chat_turns": 0, "identity_scope": (self.active_identity or "unknown")}

        # Translation short-circuit: "translate X to Spanish", "how do you
        # say X in French", etc. Runs before persist so the source text isn't
        # extracted as a speaker-fact ("how do you say I live in Boston in
        # Spanish" would otherwise store 'the user lives in Boston').
        # Routes that ask for translation can land on general_question,
        # direction, or even analyze_vision (router occasionally mistakes
        # "what's X in Spanish" for a visual question).
        if route.get("route") in {"general_question", "direction", "analyze_vision"}:
            trans_reply = self.maybe_handle_translation(message)
            if trans_reply is not None:
                actions.append({"name": "translate", "ok": True, "result": trans_reply[:200]})
                return trans_reply

        # Forget short-circuit: "forget my X", "delete my X", etc. Runs before
        # persist so the request itself isn't extracted as a new fact.
        if route.get("route") in {"general_question", "direction"}:
            forget_reply = self.maybe_handle_forget(message)
            if forget_reply is not None:
                actions.append({"name": "forget_user_fact", "ok": True, "result": forget_reply})
                return forget_reply

        # Recall short-circuit: "do you know my X" / "what's my X" with a
        # matching stored fact. The chat LLM has been unreliable on the
        # yes/no recall form, so a deterministic answer for clear hits.
        if route.get("route") in {"general_question"}:
            recall_reply = self.maybe_handle_recall(message)
            if recall_reply is not None:
                actions.append({"name": "recall_user_fact", "ok": True, "result": recall_reply})
                return recall_reply

        persist_result = {"stored": [], "already_known": [], "conflicts": []}
        if route.get("route") in {"general_question", "direction"}:
            # Pre-filter: only attempt fact extraction when the message
            # plausibly contains a first-person declarative. extract_user_facts
            # has the same guard internally, but skipping at the call site
            # avoids identity lookups + unidentified_face_ref + thread setup
            # for the common case (pure questions).
            if self._message_might_contain_fact(message):
                persist_result = self.persist_user_facts(message, state)
            if persist_result["stored"] or persist_result["already_known"] or persist_result["conflicts"]:
                actions.append({
                    "name": "persist_user_facts",
                    "ok": True,
                    "result": {
                        "stored": [r["content"] for r in persist_result["stored"]],
                        "already_known": [r["content"] for r in persist_result["already_known"]],
                        "conflicts": [c["new_fact"] for c in persist_result["conflicts"]],
                    },
                })
        self._current_persist_result = persist_result

        # Conflict short-circuit: a new fact contradicts an existing one.
        # Stash for vault_runtime to install a memory_update_confirmation
        # pending state, and reply with the deterministic "Oh, I thought..."
        # question.
        if route.get("route") in {"general_question", "direction"} and persist_result["conflicts"]:
            conflict = persist_result["conflicts"][0]
            self._stash_memory_conflict_marker = conflict
            old_phrase = self._third_to_second_person(conflict["old_fact"])
            new_phrase = self._third_to_second_person(conflict["new_fact"])
            return (
                f"Oh, I thought {old_phrase}. Has that changed — should I update it to {new_phrase}?"
            )

        # Deterministic short-circuit: user just restated a fact we already
        # have, with nothing new in the same turn. Avoids the small chat LLM
        # second-guessing itself on the acknowledge wording.
        if (
            route.get("route") == "general_question"
            and persist_result["already_known"]
            and not persist_result["stored"]
        ):
            return self._already_known_response(persist_result["already_known"])

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
            # No visible face to bind to. Still store the name as a regular
            # speaker-fact in the current identity's namespace so "what is my
            # name" recall works. Face-bind can happen later when the user
            # is in front of the camera.
            fact = f"the user's name is {introduced_name}"
            stored = False
            if self.memory_store:
                try:
                    res = self.memory_store.add(
                        fact,
                        identity=self.active_identity,
                        unidentified_face_ref=None if self.active_identity else self._unidentified_face_ref(state),
                        category="fact",
                        source_message=message,
                    )
                    stored = bool(res.get("ok"))
                    actions.append({
                        "name": "store_name_fact",
                        "ok": stored,
                        "result": {"name": introduced_name, "duplicate": res.get("duplicate")},
                    })
                except Exception as exc:
                    print(f"[memory_store] name fact write failed: {exc}")
            if stored:
                return f"Got it, {introduced_name}. I'll remember your name. I'll bind it to your face the next time you're in front of the camera."
            return f"Got it, {introduced_name}."

        standalone_confirmation = _standalone_confirmation_answer(message)
        if standalone_confirmation is not None:
            return standalone_confirmation

        setup_answer = _conversation_setup_answer(message)
        if setup_answer is not None and route.get("route") in {"general_question", "direction"}:
            return setup_answer

        personality_preference = _personality_preference_answer(message)
        if personality_preference is not None and route.get("route") in {"general_question", "direction"}:
            return personality_preference

        if route["route"] == "direction":
            if _looks_like_scout_action(message) and not _asks_broad_status_report(_canonical_intent_text(message)):
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
            # Keyword gate: vision only fires when the message is
            # explicitly about the live camera scene (see
            # _VISION_TRIGGER_PHRASES). The router LLM occasionally
            # picks analyze_vision for prompts like "describe a sunset"
            # or "compose a haiku about rain" because they contain
            # scene-like words; the keyword check filters those out.
            # force_full_vision (set by vault_runtime after the user
            # confirms an LLM-initiated scene-analysis request) bypasses
            # the gate so the confirmation flow still routes to vision
            # even when the original message lacks trigger keywords.
            force_full = False
            if isinstance(presence_context, dict):
                force_full = bool(presence_context.get("force_full_vision"))
            if not force_full and not _has_vision_trigger(message):
                route["route"] = "general_question"
                route.setdefault("downgraded_from", "analyze_vision")
                route["downgrade_reason"] = "no vision-trigger keyword in message"
            else:
                # Short-circuit: if the camera node has identifiable
                # detections, answer from those first and ask whether the
                # user wants the heavy vision LLM analysis.  Only fall
                # through to analyze_scene when the caller explicitly
                # forced it (via presence_context.force_full_vision, set
                # by vault_runtime when the user has just confirmed).
                summary = _detection_summary(state)
                if summary and not force_full:
                    actions.append({"name": "analyze_vision", "ok": True, "result": {"short_circuit": True, "summary": summary}})
                    # Surface a marker so vault_runtime can set a
                    # vision_full_analysis_confirmation pending state and
                    # route the user's next "yes" back here with
                    # force_full_vision=True.
                    extra = {
                        "needs_vision_confirmation": True,
                        "original_message": message,
                        "vision_summary": summary,
                    }
                    self._stash_vision_short_circuit_marker = extra
                    return f"{summary} Would you like me to analyze the scene?"
                result = self.analyze_scene(message, state, node_id=node_id)
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
            return self.answer_self_question(
                message,
                state,
                route,
                source=source,
                presence_context=presence_context,
            )

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

        response = self.answer_with_context(message, state, presence_context=presence_context)
        # If the LLM ended up asking the user for permission to look at
        # the scene, stash the same marker the deterministic vision
        # short-circuit uses. vault_runtime will turn it into a
        # vision_full_analysis_confirmation pending state, and the
        # user's next "yes" will arrive with force_full_vision=True —
        # which the override above redirects straight to analyze_vision.
        if response and _llm_asks_for_vision(response):
            self._stash_vision_short_circuit_marker = {
                "needs_vision_confirmation": True,
                "original_message": message,
                "vision_summary": "",
            }
        return response

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
            self._complete_route_for_learning(original_message, inferred_route, state)
            _dr.learn(original_message, inferred_route)
            actions.append({
                "name": "learn_deterministic_route",
                "ok": True,
                "result": {
                    "input": original_message,
                    "route": inferred_route.get("route"),
                    "self_route": (inferred_route.get("self_route") or {}).get("route")
                    if isinstance(inferred_route.get("self_route"), dict)
                    else None,
                    "confidence": inferred_route.get("confidence"),
                },
            })
            original_source = pending.get("source") or source
            response = self._dispatch_route(
                original_message,
                inferred_route,
                state,
                actions,
                source=original_source,
                presence_context=presence_context,
            )
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
                    corrected_desc = _route_description(corrected_route, correction)
                    confirm_text = f"I think you mean {corrected_desc}. Is that right?"
                    turn = {
                        "message": message,
                        "source": source,
                        "route": corrected_route,
                        "response": confirm_text,
                        "active_identity": self.active_identity,
                        "actions": actions,
                        "routing_feedback": presence_context.get("routing_feedback"),
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

    def _complete_route_for_learning(self, message: str, route: dict, state: dict) -> None:
        if route.get("route") != "self_question" or isinstance(route.get("self_route"), dict):
            return
        self_route = self.classify_self_question(message, state)
        if self_route.get("ok"):
            route["self_route"] = self_route

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
            return _pronoun_route_guard(message, fast)
        cached = _dr.lookup(message)
        if cached is not None:
            return _pronoun_route_guard(message, {**cached, "from_cache": True})

        # Short-TTL in-memory cache: skip the router LLM call when the same
        # phrase recurs within ~60s. The correction flow in _handle_confirmation
        # was the biggest offender — a "no, the kitchen one" reply often
        # produced a corrected_input that was identical to one we just routed.
        memo = _ROUTE_RESULT_CACHE.get(message)
        if memo is not None:
            return _pronoun_route_guard(message, {**memo, "from_cache": "memo"})

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
        result = {
            "ok": True,
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason", ""))[:240],
            "attempts": len(raw_attempts),
        }
        _ROUTE_RESULT_CACHE.put(message, result)
        return _pronoun_route_guard(message, result)

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
  This also covers questions about the SPEAKER's own attributes, possessions,
  preferences, or facts they have shared in this conversation (e.g. their pet,
  family, favorite color, plans) — recall is answered from chat history.
- analyze_vision: questions about what the scout sees, the camera image, visual
  scene, objects in front of the scout, visible people, colors, room layout, or
  visual recognition from the current snapshot.
- self_question: questions about the ASSISTANT itself — its name, identity,
  personality, software, hardware, sensors, capabilities, runtime/service
  status — OR questions about whether the assistant RECOGNIZES the current
  user (face/identity recognition). "what do you know about me" and "who am I"
  belong here because they ask what the assistant has stored or recognized.
- direction: instructions or requests to do something, including movement,
  looking, learning a face/name, remembering a fact/preference, changing a
  setting, or using a capability.

Pronoun convention (important):
- "you", "your", "yours", "yourself" → refer to the ASSISTANT (Luhkas).
- "I", "me", "my", "mine", "myself" → refer to the SPEAKER (the user).
- A question about a SPEAKER attribute ("what is my pet's name", "what's my
  favorite color", "where did I say I lived") is general_question, NOT
  self_question — the assistant is being asked to recall something about the
  user from chat history, not introspect on itself.
- "what do you know about me" and "who am I" are the exception: those ask what
  the assistant has recognized or stored about the user, so they are
  self_question (user_identity / memory bucket).
- Declarative speaker facts ("my pet's name is Salem", "I live in Austin")
  without an explicit "remember" verb are general_question (the chat layer
  will retain them in session history); only an explicit "remember ..." is
  direction.

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
- "my pet's name is Salem" -> {{"route":"general_question","confidence":0.85,"reason":"speaker shares a personal fact, retained in chat history"}}
- "what is my pet's name" -> {{"route":"general_question","confidence":0.9,"reason":"asks recall of speaker fact from chat history"}}
- "what's my favorite color" -> {{"route":"general_question","confidence":0.9,"reason":"asks recall of speaker preference from chat history"}}
- "who am I" -> {{"route":"self_question","confidence":0.9,"reason":"asks assistant's recognition of the user"}}
- "what do you know about me" -> {{"route":"self_question","confidence":0.9,"reason":"asks assistant's stored memory about the user"}}
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
            return self._compose_response(
                "feedback_learned",
                message,
                state,
                {"feedback": feedback, "saved": result},
                "Got it. I will use that next time.",
                options={"num_predict": 90, "temperature": 0.45, "top_p": 0.9},
            )
        except Exception:
            return self.response_composer.fallback(
                "Got it. I will be more precise about the specific thing you asked for.",
                "feedback response generation failed",
            )

    def answer_self_question(
        self,
        message: str,
        state: dict,
        route: dict | None = None,
        source: str | None = None,
        presence_context: dict | None = None,
    ):
        self_route = (route or {}).get("self_route")
        if not isinstance(self_route, dict):
            self_route = self.classify_self_question(message, state)
        if route is not None:
            route["self_route"] = self_route
        fast_answer = self.fast_self_answer(
            message,
            state,
            self_route,
            source=source,
            presence_context=presence_context,
        )
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
            return self._compose_response(
                "self_question",
                message,
                state,
                {
                    "deterministic_prompt": prompt,
                    "self_route": self_route_name,
                    "facts": facts,
                },
                "I couldn't answer that self-question cleanly.",
                options={"num_predict": 100, "temperature": 0.5, "top_p": 0.9},
            )
        except Exception as exc:
            return self.response_composer.fallback(
                f"I cannot answer that self-question because the local chat model is unavailable: {exc}",
                "self-question generation failed",
            )

    def fast_self_answer(
        self,
        message: str,
        state: dict,
        self_route: dict,
        source: str | None = None,
        presence_context: dict | None = None,
    ) -> str | None:
        text = _normalize_command_text(message)
        route_name = self_route.get("route")
        if route_name == "assistant_identity":
            return self._assistant_identity_answer(state, message)
        if route_name == "status" and (_asks_feeling_state(text) or _asks_casual_assistant_state(text)):
            return self._personality_state_answer(message, state)
        if route_name == "status" and _asks_broad_status_report(text):
            return self._assistant_status_answer(
                state,
                message,
                source=source,
                presence_context=presence_context,
            )
        if route_name == "user_identity":
            identity = self.active_identity
            if identity:
                return f"You are {identity}."
            # No face-recognized identity yet — fall back to a stored name
            # fact ("the user's name is Chris") in MemoryStore so the speaker
            # can still get a useful answer.
            recalled = self.recall_user_facts("the user's name", top_k=3)
            for r in recalled:
                content = (r.get("content") or "").lower()
                if "the user's name is " in content or "the user name is " in content:
                    name = (r.get("content") or "")
                    name = name.split(" is ", 1)[-1].strip().rstrip(".")
                    if name:
                        return f"You told me your name is {name}, but I haven't matched a face yet. Step in front of the camera to bind it."
            return "I don't know who you are yet. Tell me your name and I'll remember it."
        if route_name == "capabilities" and _asks_skill_inventory(text):
            return self._registered_skills_answer()
        if route_name == "capabilities" and _asks_capability_inventory(text):
            return self._registered_capabilities_answer()
        if route_name == "personality":
            return self._personality_state_answer(message, state)
        if route_name == "software" and _asks_stored_knowledge_owner(text):
            return "Stored knowledge belongs to Vault. Scout can witness and forward node state, but Vault owns memory, learning, and retrieval."
        if route_name == "software" and _asks_camera_action_owner(text):
            return "Camera actions belong to Scout's camera_node. Vault can route the request, but Scout owns the camera behavior."
        if route_name == "status" and _asks_registered_or_active_nodes(text):
            return self._registered_nodes_answer(state, message)
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

    def _personality_state_answer(self, message: str = "", live_state: dict | None = None) -> str:
        live_state = live_state or {"ok": True}
        identity_context = self.response_identity_context(live_state)
        state = self.mood_engine.voice_state(identity_context)
        voice = state.get("voice") or {}
        mood = state.get("mood") or {}
        style = self.mood_engine.style_state().get("resolved") or {}
        return _mood_statement_from_state(voice, mood, style, bool(state.get("verified_primary_user")))

    def _generated_fact_answer(
        self,
        response_type: str,
        message: str,
        state: dict,
        facts: dict,
        fallback: str,
        *,
        required_terms: tuple[str, ...] = (),
    ) -> str:
        recent = [
            str(turn.get("response") or "").strip()
            for turn in self.turns[-8:]
            if str(turn.get("response") or "").strip()
        ]
        return self.response_composer.compose(
            response_type=response_type,
            user_message=message,
            facts=facts,
            fallback=fallback,
            contract=self.response_contract(response_type, state),
            recent_responses=recent,
            options=self.chat_options({"num_predict": 80, "temperature": 0.78, "top_p": 0.92}),
            validator=lambda text: self.response_policy_violation(text, state, response_type),
            sanitizer=_sanitize_generated_response,
            required_terms=required_terms,
        )

    def _varied_fallback(self, fallback: str, recent: list[str]) -> str:
        return self.response_composer.varied_fallback(fallback, recent)

    def _assistant_identity_answer(self, state: dict | None = None, message: str = "") -> str:
        """LLM-compose the assistant identity answer so personality, mood,
        and style directives actually apply. Pulls facts about itself from
        the assistant memory bucket (seeded from identity_profile +
        self/identity.json on bridge init) so recall is consistent with user
        facts. Falls back to a short factual statement if the chat model
        returns nothing usable."""
        state = state or {}
        name = self.identity_profile.get("name") or "Luhkas"
        role = self.identity_profile.get("role") or ""
        # Name-only short-circuit: "what's your name" / "what is your name"
        # / "your name" deserve a terse one-liner, not a personality dump.
        text_canon = _normalize_command_text(message or "")
        if _asks_assistant_name(text_canon):
            self._current_assistant_facts = [f"my name is {name}"]
            return f"I'm {name}."
        # Pull assistant-identity facts from MemoryStore. We rewrite them to
        # first-person here so the LLM doesn't have to map "the assistant"
        # to itself.
        recalled = self.recall_assistant_facts(message or "who are you", top_k=6)
        assistant_facts: list[str] = []
        for r in recalled:
            content = (r.get("content") or "").strip()
            if not content:
                continue
            first_person = re.sub(r"\bthe assistant's\s+", "my ", content, flags=re.I)
            first_person = re.sub(r"\bthe assistant\s+", "I ", first_person, flags=re.I)
            assistant_facts.append(first_person)
        # Track for provenance.
        self._current_assistant_facts = assistant_facts
        facts_block = "\n".join(f"- {f}" for f in assistant_facts) if assistant_facts else "(no stored facts)"
        prompt = f"""You are an AI assistant named {name}.
{("Your role: " + role + ".") if role else ""}
The user asked: "{message or 'Who are you?'}"

Facts you know about yourself (from your memory):
{facts_block}

Reply in FIRST PERSON only -- start with "I am {name}" or "I'm {name}".
1-2 short sentences. Draw on the facts above for content. Describe yourself, not the user.

IDENTITY RULE: When asked what you are, describe yourself as an AI, AI
assistant, or AI presence. Only mention specific edge modules (like
Scout) when the user's question is specifically about hardware, body,
sensors, modules, or how you sense/act in the world. Do not pre-empt or
deny things the user didn't ask about (no "but I'm not X" disclaimers).
Stay positive and direct: say what you ARE for the asked topic, nothing
about what you AREN'T.

STRICT WORD RULES:
- Do NOT use the word "you", "you're", "you are", "your", or "yours" anywhere.
- Do NOT mention your creator, body, hardware, node, scout, vault, or any user's name unless the user's question is specifically about that topic.
- No emojis. No trailing offer ("how can I help" etc).
"""
        # Call the chat model directly rather than going through
        # generate_guarded_response/response_composer, which wraps the
        # prompt in its own scaffolding that drowns out the strict word
        # rules. We still run the identity violation validator manually.
        # Streaming-aware helper so this path also fills the NDJSON stream
        # when the caller is the /presence/message/stream endpoint.
        reply = None
        try:
            raw = self._generate_user_facing(
                prompt,
                options={"num_predict": 220, "temperature": 0.55, "top_p": 0.9},
                think=False,
                timeout=30,
            )
            raw = _sanitize_generated_response(str(raw or "")).strip()
            if raw and not _assistant_identity_response_violation(raw):
                reply = raw
        except Exception:
            reply = None
        if reply and isinstance(reply, str) and reply.strip():
            return reply.strip()
        # Fallback: terse factual statement if LLM unavailable.
        return f"I'm {name}."

    def _assistant_status_answer(
        self,
        state: dict,
        message: str = "",
        source: str | None = None,
        presence_context: dict | None = None,
    ) -> str:
        text = _normalize_command_text(message)
        source_node = _source_node_id(source, presence_context)
        if _asks_broad_status_report(text):
            operational_facts = self._operational_status_facts(state, source_node=source_node)
            return _operational_status_statement(operational_facts)

        identity_context = self.response_identity_context(state)
        mood_state = self.mood_engine.voice_state(identity_context)
        voice = mood_state.get("voice") or {}
        mood = mood_state.get("mood") or {}
        style = self.mood_engine.style_state().get("resolved") or {}
        mood_statement = _mood_statement_from_state(
            voice,
            mood,
            style,
            bool(mood_state.get("verified_primary_user")),
        )
        scout_facts = _scout_status_facts(state) if _source_is_scout(source_node) else None
        fallback = _status_report_statement(mood_statement, scout_facts)
        facts = {
            "deterministic_answer": fallback,
            "mood_statement": mood_statement,
            "mood_values": mood,
            "voice_values": voice,
            "style_values": style,
            "source_node": source_node,
            "scout_status": scout_facts,
            "instruction": (
                "Generate a fresh first-person status answer. "
                "For Scout-origin requests, include the provided Scout live-status facts. "
                "Do not mention faces, identities, or people unless those facts are explicitly supplied."
            ),
        }
        required_terms = ("Scout",) if scout_facts else ()
        return fallback

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
        return "I don't have a sensor list loaded." if not live else "Sensors: " + ", ".join(live[:6]) + "."

    def _registered_nodes_answer(self, state: dict | None = None, message: str = "") -> str:
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
        text = _canonical_intent_text(message)
        if "morning" in text:
            return "Good morning."
        if "afternoon" in text:
            return "Good afternoon."
        if "evening" in text:
            return "Good evening."
        return "Hello."

    def _generate_user_facing(
        self,
        prompt: str,
        *,
        options: dict | None = None,
        timeout: float = 30.0,
        think: bool | None = None,
        allow_empty: bool = False,
    ) -> str:
        """``chat_model.generate`` with optional streaming.

        A handful of user-facing answer paths intentionally bypass
        ``ResponseComposer`` (the composer's outer scaffold drowns out
        path-specific rules). Those paths still want streaming when the
        /presence/message/stream endpoint installed a sink, so this helper
        bridges to ``BaseModel.generate_stream`` when a sink is present and
        falls back to plain ``generate`` otherwise.

        Either way it returns the full accumulated text so the caller can
        run its sanitize/validate post-processing as before.
        """
        from streaming import get_stream_sink
        sink = get_stream_sink()
        stream_fn = getattr(self.chat_model, "generate_stream", None) if sink is not None else None
        if stream_fn is not None:
            parts: list[str] = []
            try:
                for chunk in stream_fn(
                    prompt,
                    options=options,
                    timeout=timeout,
                    think=think,
                ):
                    parts.append(chunk)
                    try:
                        sink("delta", chunk)
                    except Exception:
                        pass
            except Exception:
                if not allow_empty:
                    raise
                return ""
            return "".join(parts)
        return self.chat_model.generate(
            prompt,
            options=options,
            timeout=timeout,
            think=think,
            allow_empty=allow_empty,
        )

    def _compose_response(
        self,
        response_type: str,
        message: str,
        state: dict,
        facts: dict,
        fallback: str,
        *,
        options: dict | None = None,
        required_terms: tuple[str, ...] = (),
    ) -> str:
        recent = [
            str(turn.get("response") or "").strip()
            for turn in self.turns[-8:]
            if str(turn.get("response") or "").strip()
        ]
        return self.response_composer.compose(
            response_type=response_type,
            user_message=message,
            facts=facts,
            fallback=fallback,
            contract=self.response_contract(response_type, state),
            recent_responses=recent,
            options=self.chat_options(options),
            validator=lambda text: self.response_policy_violation(text, state, response_type),
            sanitizer=_sanitize_generated_response,
            required_terms=required_terms,
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
        fallback = (
            self.cleanup_policy_failed_response(response_type, state, "model generation failed")
            or "I don't know that from the information I have."
        )
        return self._compose_response(
            response_type,
            message,
            state,
            {
                "identity": compact_identity,
                "route_facts": response_facts,
                "rover_tracking_available": bool(state.get("ok")),
                "active_identity": self.active_identity if state.get("ok") else None,
            },
            fallback,
            options={"num_predict": max_tokens, "temperature": temperature},
        )

    def _scrub_primary_user(self, text: str | None, primary_user: str | None = None) -> str:
        """Replace the primary_user's proper name with a generic stand-in
        ("the user") in a single string. Used to keep the chat model
        from parroting the name when the current speaker hasn't been
        face-verified — without this, *any* prompt field that references
        the primary user (identity_role, recent_chat, etc.) can leak the
        name back to the model, which then volunteers it as the
        speaker's name even when the speaker is unknown."""
        if not text:
            return text or ""
        name = primary_user or self.identity_profile.get("primary_user")
        if not name:
            return text
        return re.sub(r"\b" + re.escape(name) + r"\b", "the user", text)

    def response_identity_context(self, state: dict):
        active_identity = self.active_identity if state.get("ok") else None
        primary_user = self.identity_profile.get("primary_user")
        primary_user_title = self.identity_profile.get("primary_user_title")
        active_matches_primary = bool(
            active_identity
            and primary_user
            and _safe_identity(active_identity) == _safe_identity(primary_user)
        )
        # Hard scrub: when we are NOT allowed to address the current
        # user by name, do not surface primary_user in the dict at all.
        # The chat model has demonstrated a tendency to parrot any name
        # it sees in the prompt regardless of accompanying "do not
        # address" rules ("tell me how you feel" -> "Your name is
        # Chris."). Removing the field is the only reliable guard.
        if active_matches_primary:
            return {
                "active_identity": active_identity,
                "primary_user": primary_user,
                "primary_user_title": primary_user_title,
                "active_matches_primary_user": True,
                "may_address_primary_user": True,
                "addressing_rule": "You may address the current user with primary_user or primary_user_title.",
            }
        return {
            "active_identity": active_identity,
            "primary_user": None,
            "primary_user_title": None,
            "active_matches_primary_user": False,
            "may_address_primary_user": False,
            "addressing_rule": "You do not know the current user's name. Do not address them by any name.",
        }

    def generate_guarded_response(
        self,
        response_type: str,
        prompt: str,
        state: dict,
        *,
        options: dict | None = None,
    ):
        return self.response_composer.compose(
            response_type=response_type,
            user_message="",
            facts={"legacy_prompt": prompt},
            fallback=self.cleanup_policy_failed_response(response_type, state, "model generation failed")
            or "I could not generate that response cleanly.",
            contract=self.response_contract(response_type, state),
            recent_responses=[
                str(turn.get("response") or "").strip()
                for turn in self.turns[-8:]
                if str(turn.get("response") or "").strip()
            ],
            options=self.chat_options(options),
            validator=lambda text: self.response_policy_violation(text, state, response_type),
            sanitizer=_sanitize_generated_response,
        )

    def cleanup_policy_failed_response(self, response_type: str, state: dict, violation: str):
        identity_context = self.response_identity_context(state)
        active_identity = identity_context.get("active_identity")
        if response_type == "identity_status":
            text = f"You are {active_identity}." if active_identity else "I don't know who you are yet."
        elif response_type == "greeting":
            text = "I'm here."
        elif response_type == "self_question":
            text = self._assistant_identity_answer()
        elif response_type == "feedback_learned":
            text = "Got it. I will use that next time."
        elif response_type == "identity_binding_blocked":
            text = "I can't verify your identity from the current context yet."
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
        if _has_excessive_foreign_chars(text):
            return (
                "The response contained substantial non-English text. "
                "Reply in English."
            )
        identity_context = self.response_identity_context(state)
        if re.search(r"\bLuhkas\s+will\b", str(text or ""), re.I):
            return "The response referred to the assistant in third person instead of answering directly."
        if _claims_assistant_is_node_identity(text):
            return "The response identified the assistant as a node instead of Luhkas."
        if response_type == "assistant_identity" and _assistant_identity_response_violation(text):
            return "The assistant identity response volunteered creator, node/body, or current-user identity details."
        if response_type == "mood_statement" and _mood_statement_response_violation(text):
            return "The mood response included operational or Scout status instead of mood only."
        if _echoes_generation_instruction(text):
            return "The response echoed a generation instruction instead of answering the user."
        if response_type == "greeting" and _volunteers_node_boundary(text):
            return "The greeting volunteered node-boundary details instead of simply greeting."
        if response_type == "status_report" and _status_report_response_violation(text):
            return "The status report invented identity, face, or primary-user details."
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

    _FACT_EXTRACTOR_PROMPT = """Extract any personal facts the SPEAKER stated about THEMSELVES in this message.
Output JSON: {{"facts": [...]}} where each fact is a short standalone sentence
in third person referring to "the user" (e.g. "the user's pet is named Salem",
"the user lives in Austin").

Rules:
- Extract anything the user said about themselves: their possessions,
  preferences (favorite ANYTHING — color, food, drink, movie, season,
  music, sport, book, etc.), relationships, location, job, hobbies,
  plans, or activities.
- Do NOT extract questions, commands, requests, observations about the
  assistant, or external/world facts.
- Each fact must start with "the user" or "the user's".
- If no personal facts are present, return {{"facts": []}}.
- Output JSON object only. No markdown, no prose.

Examples (input → output):
- "my name is Chris"               → {{"facts": ["the user's name is Chris"]}}
- "I'm Chris"                       → {{"facts": ["the user's name is Chris"]}}
- "my pet's name is Salem"        → {{"facts": ["the user's pet is named Salem"]}}
- "I live in Austin"               → {{"facts": ["the user lives in Austin"]}}
- "my favorite color is blue"      → {{"facts": ["the user's favorite color is blue"]}}
- "my favorite drink is coffee"    → {{"facts": ["the user's favorite drink is coffee"]}}
- "my favorite movie is Inception" → {{"facts": ["the user's favorite movie is Inception"]}}
- "my favorite season is autumn"   → {{"facts": ["the user's favorite season is autumn"]}}
- "I like jazz music"              → {{"facts": ["the user likes jazz music"]}}
- "I'm a software engineer"        → {{"facts": ["the user is a software engineer"]}}
- "I have a cat"                   → {{"facts": ["the user has a cat"]}}
- "what's my pet's name"           → {{"facts": []}}
- "what's my favorite color"       → {{"facts": []}}
- "who am I"                       → {{"facts": []}}
- "good morning"                   → {{"facts": []}}
- "what's the capital of japan"    → {{"facts": []}}
- "I work as an SRE and I have a cat named Whiskers"
                                   → {{"facts": ["the user works as an SRE", "the user has a cat named Whiskers"]}}

User message:
{message}
"""

    _FACT_EXTRACTOR_SCHEMA = {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["facts"],
    }

    # Cheap deterministic guard: only fire the fact-extractor LLM when the
    # message *might* contain a first-person declarative statement. Without
    # this, every general_question paid ~500ms on the chat model even for
    # pure questions ("what is the capital of france") that have no facts.
    #
    # The patterns favor RECALL over precision: we'd rather run the
    # extractor unnecessarily on an edge case than silently lose a fact.
    # Anything matching ANY of these triggers the LLM:
    #   - "i" / "i'm" / "i am" / "i was" / "i have" / "i'm a X" / etc.
    #   - "my X is/are/was/were ..."
    #   - "call me X" / "you can call me X"
    #   - "remember (that) ..." / "remember I ..."
    _FACT_INDICATOR_PATTERNS = (
        # Question stems — short-circuit "looks-like-question" before we even
        # bother checking fact patterns. We do this in a separate gate below
        # so users can still get facts out of "I think my favorite color is
        # blue, what's yours?" (mixed statement+question).
        # Below: positive indicators.
        re.compile(r"\bmy\s+\S+\s+(?:is|are|was|were|isn't|aren't|wasn't|weren't|='?s?)\b", re.I),
        # Require content AFTER "i'm" / "i am" — this excludes question
        # tail forms like "who am I" while still catching "I'm a doctor",
        # "I'm 30 years old", "I am from Seattle", etc.
        re.compile(r"\b(?:i'm|im|i\s+am|i\s+was|i'll|i've|i'd)\s+\S+", re.I),
        re.compile(r"\bi\s+(?:have|had|own|like|love|hate|prefer|enjoy|use|drive|play|work|live|study|went|go|come|teach|teach|build|wrote|read|watched)\b", re.I),
        re.compile(r"\b(?:call|name)\s+me\s+\w+", re.I),
        re.compile(r"\bremember\s+(?:that|i|my|we|us|me)\b", re.I),
        re.compile(r"\bmy\s+(?:name|pet|cat|dog|family|wife|husband|partner|kid|son|daughter|favorite|fav|job|hobby|home|address|car|bike|birthday|age|email|phone|number|company|team|boss|friend|neighbor|landlord)\b", re.I),
    )
    _QUESTION_STEMS = re.compile(
        r"^\s*(?:what|who|where|when|why|how|which|whose|whom|is|are|was|were|do|does|did|can|could|will|would|should|may|might|tell\s+me|show\s+me|give\s+me|explain|describe|list)\b",
        re.I,
    )

    # Small in-memory LRU for fact-extractor output. Avoids re-running the
    # chat model when the same message recurs (correction flow, retries,
    # benchmark scripts, etc.).
    _FACT_EXTRACT_CACHE: collections.OrderedDict = collections.OrderedDict()
    _FACT_EXTRACT_CACHE_MAX = 128
    _FACT_EXTRACT_CACHE_LOCK = threading.Lock()

    @classmethod
    def _fact_cache_get(cls, key: str) -> list[str] | None:
        with cls._FACT_EXTRACT_CACHE_LOCK:
            value = cls._FACT_EXTRACT_CACHE.get(key)
            if value is not None:
                cls._FACT_EXTRACT_CACHE.move_to_end(key)
                return list(value)
            return None

    @classmethod
    def _fact_cache_put(cls, key: str, value: list[str]) -> None:
        with cls._FACT_EXTRACT_CACHE_LOCK:
            cls._FACT_EXTRACT_CACHE[key] = list(value)
            cls._FACT_EXTRACT_CACHE.move_to_end(key)
            while len(cls._FACT_EXTRACT_CACHE) > cls._FACT_EXTRACT_CACHE_MAX:
                cls._FACT_EXTRACT_CACHE.popitem(last=False)

    def _message_might_contain_fact(self, message: str) -> bool:
        """Cheap heuristic: True if the message *might* contain a first-
        person declarative fact worth handing to the LLM extractor."""
        text = (message or "").strip()
        if not text:
            return False
        # Pure question short-circuit: starts with a question stem AND has no
        # positive fact indicator. (We still allow "I think X is Y" through.)
        if self._QUESTION_STEMS.match(text):
            for pat in self._FACT_INDICATOR_PATTERNS:
                if pat.search(text):
                    return True
            return False
        for pat in self._FACT_INDICATOR_PATTERNS:
            if pat.search(text):
                return True
        return False

    def extract_user_facts(self, message: str) -> list[str]:
        """Pull declarative speaker-facts out of a turn.
        Returns a (possibly empty) list of third-person fact sentences.

        Uses chat_model (qwen3:8b) with structured output -- the 3B router
        model was dropping common preferences ("favorite drink", "favorite
        movie", "favorite season") even with examples in the prompt. To
        keep this off the hot path for pure questions we (a) deterministic
        pre-filter via _message_might_contain_fact, and (b) memoize results
        per normalized message text.

        Net effect: pure-question turns ("what is X") skip the LLM entirely
        (~500ms savings); repeated identical inputs hit the cache (~500ms
        savings on the second occurrence)."""
        if not message or not message.strip():
            return []
        # Deterministic fast-path: name introductions ("my name is X", "I'm X",
        # "I am X"). Unambiguous enough to grab directly without an LLM.
        introduced = _extract_introduction_name(message)
        if introduced:
            return [f"the user's name is {introduced}"]
        # Deterministic guard: skip the LLM if no first-person fact pattern
        # is present at all. Conservative — we'd rather over-call than miss.
        if not self._message_might_contain_fact(message):
            return []
        # Memoized extractor: identical messages reuse last result.
        cache_key = " ".join(message.strip().lower().split())
        cached = self._fact_cache_get(cache_key)
        if cached is not None:
            return cached
        prompt = self._FACT_EXTRACTOR_PROMPT.format(message=message.strip())
        try:
            raw = self.chat_model.generate(
                prompt,
                options={"num_predict": 200, "temperature": 0.0, "top_p": 0.9},
                response_format=self._FACT_EXTRACTOR_SCHEMA,
                timeout=30,
                allow_empty=True,
            )
        except Exception:
            return []
        facts: list[str] = []
        if raw:
            text = raw.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None
                for item in ((data or {}).get("facts") if isinstance(data, dict) else []) or []:
                    if isinstance(item, str) and item.strip():
                        facts.append(item.strip())
        self._fact_cache_put(cache_key, facts)
        return facts

    @staticmethod
    def _pick_identity_from_state(state: dict | None, min_confidence: float = 0.6) -> str | None:
        """Return the highest-confidence face-recognized identity currently
        visible in the scout state, or None if no confident identification.
        Voice cross-check will be layered on top later."""
        if not state:
            return None
        best_name = None
        best_conf = 0.0
        for det in state.get("detections", []) or []:
            label = str(det.get("label", "")).lower()
            if label != "person" and "face" not in label:
                continue
            identity = det.get("identity")
            if not identity:
                continue
            conf = float(det.get("identity_confidence") or 0.0)
            if conf < min_confidence:
                continue
            if conf > best_conf:
                best_conf = conf
                best_name = str(identity).strip()
        if best_name:
            return best_name
        # Fall back to tracker object_memory if detections didn't include a
        # face-recognized person this tick (some scout configurations only
        # surface identity through object_memory).
        for mem in state.get("object_memory", []) or []:
            if mem.get("label") != "person":
                continue
            identity = mem.get("identity")
            if not identity:
                continue
            conf = float(mem.get("identity_confidence") or 0.0)
            if conf < min_confidence:
                continue
            if conf > best_conf:
                best_conf = conf
                best_name = str(identity).strip()
        return best_name

    def _adopt_identity_from_state(self, state: dict | None, node_id: str | None = None) -> None:
        """If the camera currently sees a recognized person, adopt that as the
        active identity for this turn. Sticky: we do NOT clear active_identity
        when nobody is visible — that lets the user step out briefly without
        the session losing context. Voice recognition will arbitrate conflicts
        once it lands.

        When the adoption transitions from "nobody" → "known person", that's
        a user-present signal: bump node activity AND flush any deferred
        alerts onto this node's queue so the user immediately sees what
        accumulated while they were away."""
        picked = self._pick_identity_from_state(state)
        prev = self.active_identity
        transitioned = bool(picked) and (
            not prev or prev.lower() != picked.lower()
        )
        if picked and transitioned:
            self.active_identity = picked
        if transitioned and node_id and getattr(self, "node_registry", None):
            try:
                self.node_registry.update_activity(node_id, identity=picked)
                drained = self.node_registry.flush_pending_to(node_id)
                if drained:
                    print(
                        f"[bridge] identity-adoption flushed {drained} "
                        f"deferred alert(s) to {node_id}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[bridge] identity-adoption flush failed: {exc}", flush=True)

    def _unidentified_face_ref(self, state: dict | None) -> str | None:
        """Best-effort handle for the unknown-bucket: most-confident visible
        face/person detection id, used to tie unknown memories to a face
        observation when one exists."""
        if not state:
            return None
        for det in state.get("detections", []) or []:
            label = str(det.get("label", "")).lower()
            if det.get("identity"):
                continue
            if label == "person" or "face" in label:
                det_id = det.get("id")
                if det_id is not None:
                    return str(det_id)
        return None

    _FACT_RELATION_PROMPT = """You are checking the relation between a NEW fact the user just stated and an OLD fact already on file.

OLD fact (already stored): {old}
NEW fact (just stated):    {new}

Classify the relation as exactly one of:
- "duplicate": the new fact says the SAME thing as the old fact (same subject, same predicate, same value). Different wording is fine as long as the meaning is identical (e.g. "the user's pet is Salem" vs "the user has a pet named Salem").
- "contradicts": the new fact is about the SAME subject + predicate as the old fact but with a DIFFERENT value (e.g. lives in Austin vs. lives in Ocala; favorite color is blue vs. favorite color is green; pet is Salem vs. pet is Whiskers). The new fact REPLACES the old.
- "extends": the new fact adds to or refines the old fact without contradicting it (e.g. user has a cat -> user has a cat named Whiskers).
- "unrelated": the facts are about different subjects/predicates entirely.

Output JSON only:
{{"relation":"duplicate|contradicts|extends|unrelated","reason":"short"}}
"""

    _FACT_RELATION_SCHEMA = {
        "type": "object",
        "properties": {
            "relation": {
                "type": "string",
                "enum": ["duplicate", "contradicts", "extends", "unrelated"],
            },
            "reason": {"type": "string"},
        },
        "required": ["relation"],
    }

    # Cache for (new_fact, old_fact) → relation. The same pair recurs often
    # within a session: if a user repeats a fact, vector search returns the
    # same candidate and we'd otherwise call the LLM with the identical
    # prompt. Pure function on (new, old) so memoization is safe.
    _FACT_RELATION_CACHE: collections.OrderedDict = collections.OrderedDict()
    _FACT_RELATION_CACHE_MAX = 256
    _FACT_RELATION_CACHE_LOCK = threading.Lock()

    @classmethod
    def _relation_cache_get(cls, key: tuple[str, str]) -> str | None:
        with cls._FACT_RELATION_CACHE_LOCK:
            value = cls._FACT_RELATION_CACHE.get(key)
            if value is not None:
                cls._FACT_RELATION_CACHE.move_to_end(key)
            return value

    @classmethod
    def _relation_cache_put(cls, key: tuple[str, str], value: str) -> None:
        with cls._FACT_RELATION_CACHE_LOCK:
            cls._FACT_RELATION_CACHE[key] = value
            cls._FACT_RELATION_CACHE.move_to_end(key)
            while len(cls._FACT_RELATION_CACHE) > cls._FACT_RELATION_CACHE_MAX:
                cls._FACT_RELATION_CACHE.popitem(last=False)

    def classify_fact_relation(self, new_fact: str, old_fact: str) -> str:
        """Use the chat model (qwen3:8b) for this — the smaller router model
        was flickering between contradicts/duplicate/extends on near-identical
        embeddings, and getting this wrong silently loses or corrupts memory.
        Structured-output mode prevents free-form regressions.

        Memoized on the (new, old) pair: the same fact reasserted in a
        later turn won't pay the LLM cost again."""
        cache_key = (
            " ".join((new_fact or "").lower().split()),
            " ".join((old_fact or "").lower().split()),
        )
        cached = self._relation_cache_get(cache_key)
        if cached is not None:
            return cached
        prompt = self._FACT_RELATION_PROMPT.format(old=old_fact, new=new_fact)
        try:
            raw = self.chat_model.generate(
                prompt,
                options={"num_predict": 60, "temperature": 0.0, "top_p": 0.9},
                response_format=self._FACT_RELATION_SCHEMA,
                timeout=30,
                allow_empty=True,
            )
        except Exception:
            return "unrelated"
        rel = "unrelated"
        if raw:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    candidate = str(data.get("relation", "")).strip().lower()
                    if candidate in {"duplicate", "contradicts", "extends", "unrelated"}:
                        rel = candidate
                except json.JSONDecodeError:
                    pass
        self._relation_cache_put(cache_key, rel)
        return rel

    def _classify_and_route_fact(self, fact: str, identity: str | None,
                                  unidentified_face_ref: str | None,
                                  message: str) -> tuple[str, dict] | None:
        """Per-fact pipeline: candidate search → LLM relation classify →
        store / mark-known / flag-conflict. Returns (kind, payload) where
        kind ∈ {'stored', 'already_known', 'conflict'} or None on error.
        Designed to be safe to run from a thread pool — MemoryStore owns
        its own threading.Lock for vector ops, and Ollama handles
        concurrent generate calls."""
        try:
            candidates = self.memory_store.find_conflict_candidates(
                fact, identity=identity, distance_min=0.0, distance_max=0.7, top_k=1,
            )
            relation = None
            old = None
            if candidates:
                old = candidates[0]
                relation = self.classify_fact_relation(fact, old["content"])
            if relation == "contradicts" and old is not None:
                return ("conflict", {
                    "new_fact": fact,
                    "old_fact": old["content"],
                    "old_id": old["id"],
                    "identity": identity,
                    "unidentified_face_ref": unidentified_face_ref,
                    "source_message": message,
                })
            if relation == "duplicate" and old is not None:
                return ("already_known", {
                    "id": old["id"],
                    "identity": old["identity"],
                    "content": old["content"],
                    "category": old.get("category"),
                    "source_message": old.get("source_message"),
                    "created_at": old.get("created_at"),
                })
            res = self.memory_store.add(
                fact,
                identity=identity,
                unidentified_face_ref=unidentified_face_ref,
                category="fact",
                source_message=message,
                duplicate_distance=0.05,
            )
            if not res.get("ok"):
                return None
            if res.get("duplicate"):
                return ("already_known", res["record"])
            return ("stored", res["record"])
        except Exception as exc:
            print(f"[memory_store] write failed: {exc}")
            return None

    def persist_user_facts(self, message: str, state: dict | None = None) -> dict:
        """Extract any speaker-facts from the message and write them to the
        identity-scoped memory store. Returns
        {"stored": [...], "already_known": [...], "conflicts": [...]}.
        conflicts are NEW facts that contradict an existing one — they are
        NOT written; the caller must resolve via a confirmation flow.

        Per-fact pipelines (candidate search + LLM relation classify +
        store) run in PARALLEL when the extractor produces multiple facts.
        Sequential single-fact path skips the thread overhead."""
        result = {"stored": [], "already_known": [], "conflicts": []}
        if not self.memory_store:
            return result
        facts = self.extract_user_facts(message)
        if not facts:
            return result
        identity = self.active_identity
        unidentified_face_ref = None if identity else self._unidentified_face_ref(state)

        if len(facts) == 1:
            outcome = self._classify_and_route_fact(facts[0], identity, unidentified_face_ref, message)
            if outcome:
                kind, payload = outcome
                result[kind].append(payload) if kind != "conflict" else result["conflicts"].append(payload)
            return result

        # Multi-fact: fan out the per-fact pipelines. Each runs an
        # independent LLM classify call; the bottleneck on the slowest
        # ~7s "multi-fact extraction" turn was the SECOND classify
        # waiting for the FIRST to return.
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(len(facts), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(
                lambda f: self._classify_and_route_fact(f, identity, unidentified_face_ref, message),
                facts,
            ))
        for outcome in outcomes:
            if not outcome:
                continue
            kind, payload = outcome
            (result["conflicts"] if kind == "conflict" else result[kind]).append(payload)
        return result

    @staticmethod
    def _third_to_second_person(text: str) -> str:
        """Coarse third->second person rewrite for fact strings stored in
        the 'the user X' / 'the user's X' canonical form."""
        if not text:
            return text
        # "the user's foo" -> "your foo"
        text = re.sub(r"\bthe user's\s+", "your ", text, flags=re.I)
        # "the user is" -> "you are", "the user has" -> "you have",
        # "the user likes/lives/works/..." -> "you like/live/work/..."
        verb_map = {"is": "are", "has": "have", "was": "were", "does": "do"}
        def fix_verb(match: re.Match) -> str:
            verb = match.group(1).lower()
            replacement = verb_map.get(verb)
            if replacement is not None:
                return f"you {replacement}"
            # Naive present-3rd-singular -> base form: strip trailing 's' when
            # the word is at least 4 chars and ends in single 's'.
            if len(verb) >= 4 and verb.endswith("s") and not verb.endswith("ss"):
                return f"you {verb[:-1]}"
            return f"you {verb}"
        text = re.sub(r"\bthe user\s+([A-Za-z]+)", fix_verb, text, flags=re.I)
        return text

    def _already_known_response(self, already_known: list[dict]) -> str:
        if not already_known:
            return "I already have that recorded."
        if len(already_known) == 1:
            phrase = self._third_to_second_person(already_known[0]["content"])
            return f"I already have that — {phrase}."
        phrases = ", ".join(
            self._third_to_second_person(r["content"]) for r in already_known
        )
        return f"I already have those: {phrases}."

    def recall_user_facts(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.memory_store:
            return []
        identity = self.active_identity or "unknown"
        try:
            return self.memory_store.search(query, identity=identity, top_k=top_k)
        except Exception as exc:
            print(f"[memory_store] search failed: {exc}")
            return []

    def recall_world_knowledge(self, query: str, top_k: int = 3) -> list[dict]:
        """Look up reference corpus (offline Wikipedia + media transcripts)
        for world-fact context. Identity-free; safe to call on every general
        question. Returns [] when the store is unavailable."""
        if not self.world_store:
            return []
        try:
            from world.config import (
                WORLD_MEDIA_TEXT_DISTANCE_MAX, WORLD_WIKI_DISTANCE_MAX,
            )
            hits: list[dict] = []
            for h in self.world_store.search_wiki(
                query, top_k=top_k, distance_max=WORLD_WIKI_DISTANCE_MAX,
            ):
                hits.append({
                    "source": "wikipedia",
                    "title": h.get("title") or "",
                    "section": h.get("section_path") or "",
                    "snippet": (h.get("content") or "")[:600],
                    "distance": h.get("distance"),
                    "article_id": h.get("article_id"),
                })
            for h in self.world_store.search_media_text(
                query, top_k=2, distance_max=WORLD_MEDIA_TEXT_DISTANCE_MAX,
            ):
                hits.append({
                    "source": "media_vault",
                    "asset_id": h.get("asset_id"),
                    "modality": h.get("modality"),
                    "snippet": (h.get("content") or "")[:600],
                    "distance": h.get("distance"),
                })
            return hits
        except Exception as exc:
            print(f"[world_store] search failed: {exc}")
            return []

    ASSISTANT_MEMORY_IDENTITY = "assistant"

    def _seed_assistant_memory(self) -> None:
        """Seed the assistant identity bucket from identity_profile +
        self/identity.json. Idempotent: the store's duplicate guard collapses
        re-runs.

        identity_profile.json stays the canonical source for structured
        access (name, role, creator); MemoryStore is a semantic-recall view."""
        if not self.memory_store:
            return
        seeds: list[str] = []
        prof = self.identity_profile or {}
        name = (prof.get("name") or "").strip()
        if name:
            seeds.append(f"the assistant's name is {name}")
        role = (prof.get("role") or "").strip()
        if role:
            seeds.append(f"the assistant's role is {role}")
        creator = (prof.get("creator") or "").strip()
        if creator:
            seeds.append(f"the assistant was created by {creator}")
        body = (prof.get("body") or "").strip()
        seeds.append("the assistant is an AI presence running on the vault PC")
        if body:
            # Positive framing only: Scout is one of the assistant's
            # connected edge modules. Don't seed a "not a rover" negation
            # -- the LLM picks it up and defensively explains "but I'm not
            # the rover" in answers where the topic wasn't even raised.
            seeds.append(
                f"the assistant uses connected edge modules to sense and act in the world; the Scout module is a rover with {body.replace('a rover body with ', '').replace('a rover with ', '')}"
            )
        primary_user = (prof.get("primary_user") or "").strip()
        if primary_user:
            seeds.append(f"the assistant's primary user is {primary_user}")
        for trait in prof.get("personality") or []:
            t = str(trait or "").strip()
            if t:
                seeds.append(f"the assistant's personality is {t}")
        for b in prof.get("boundaries") or []:
            bt = str(b or "").strip()
            if bt:
                seeds.append(f"the assistant follows the boundary: {bt}")
        # self/identity.json -- pre-authored fact list
        try:
            self_identity_path = self.self_dir / "identity.json"
            if self_identity_path.exists():
                data = self._load_json_file(self_identity_path)
                if isinstance(data, dict):
                    for fact in data.get("facts") or []:
                        ft = str(fact or "").strip()
                        if ft:
                            seeds.append(ft)
        except Exception:
            pass
        for fact in seeds:
            try:
                self.memory_store.add(
                    fact,
                    identity=self.ASSISTANT_MEMORY_IDENTITY,
                    category="identity",
                    source_message="bridge_seed",
                    duplicate_distance=0.15,
                )
            except Exception:
                pass

    def recall_assistant_facts(self, query: str, top_k: int = 6) -> list[dict]:
        """Pull facts about the assistant itself (name, role, personality,
        boundaries, etc.) from the assistant identity bucket. Used by
        _assistant_identity_answer so identity questions get a recall-driven
        answer rather than a canned one."""
        if not self.memory_store:
            return []
        try:
            return self.memory_store.search(
                query, identity=self.ASSISTANT_MEMORY_IDENTITY, top_k=top_k,
            )
        except Exception as exc:
            print(f"[memory_store] assistant search failed: {exc}")
            return []

    _FORGET_PATTERN = re.compile(
        r"^\s*(?:please\s+)?(?:forget|delete|remove|drop)\s+(?:that\s+|about\s+)?(?:my\s+|the\s+)?(?P<topic>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE,
    )

    # Recall fast-path: "do you know my X" / "do you remember my X" / "what's
    # my X". The chat LLM has been unreliable on the yes/no form ("do you
    # know my name" → "I don't know your name" even with name in
    # remembered_user_facts), so we handle these deterministically.
    # Language aliases for the translation handler. Add more here as needed.
    _TRANSLATE_LANGUAGES = {
        "spanish": "Spanish",     "spanish.": "Spanish",
        "español": "Spanish",     "espanol": "Spanish",
        "french": "French",        "français": "French",
        "german": "German",        "deutsch": "German",
        "italian": "Italian",      "italiano": "Italian",
        "portuguese": "Portuguese","português": "Portuguese",
        "japanese": "Japanese",    "japonés": "Japanese",
        "chinese": "Chinese",      "mandarin": "Chinese",
        "korean": "Korean",        "russian": "Russian",
        "dutch": "Dutch",          "swedish": "Swedish",
    }

    # Translation request patterns. Try in order; first match wins.
    _TRANSLATE_PATTERNS = [
        # "translate <src> (to|into) <lang>"
        re.compile(r"^\s*translate\s+(?P<src>.+?)\s+(?:to|into)\s+(?P<lang>[A-Za-zÀ-ÿ]+)\s*[.!?]?\s*$", re.I),
        # "translate (to|into) <lang>: <src>"  or  "translate (to|into) <lang>, <src>"
        re.compile(r"^\s*translate\s+(?:to|into)\s+(?P<lang>[A-Za-zÀ-ÿ]+)\s*[:,]\s*(?P<src>.+?)\s*[.!?]?\s*$", re.I),
        # "how do/can you say <src> in <lang>"
        re.compile(r"^\s*how\s+(?:do|can)\s+(?:you|i)\s+say\s+(?P<src>.+?)\s+in\s+(?P<lang>[A-Za-zÀ-ÿ]+)\s*[.!?]?\s*$", re.I),
        # "say <src> in <lang>"
        re.compile(r"^\s*say\s+(?P<src>.+?)\s+in\s+(?P<lang>[A-Za-zÀ-ÿ]+)\s*[.!?]?\s*$", re.I),
        # "what's/what is <src> in <lang>"
        re.compile(r"^\s*what(?:'?s| is)\s+(?P<src>.+?)\s+in\s+(?P<lang>[A-Za-zÀ-ÿ]+)\s*[.!?]?\s*$", re.I),
        # "<lang> for <src>"  (e.g. "spanish for thank you")
        re.compile(r"^\s*(?P<lang>[A-Za-zÀ-ÿ]+)\s+(?:word\s+)?for\s+(?P<src>.+?)\s*[.!?]?\s*$", re.I),
    ]

    def maybe_handle_translation(self, message: str) -> str | None:
        """Detect translation requests and answer via a tight prompt.
        Returns the translated string, or None if the message isn't a
        translation request. Uses chat_model (qwen3:8b) directly with
        a translate-only prompt — the general_question composer was
        unreliable on this (one test even routed 'what's X in Spanish'
        to analyze_vision)."""
        if not message:
            return None
        for pat in self._TRANSLATE_PATTERNS:
            m = pat.match(message)
            if not m:
                continue
            lang_raw = m.group("lang").strip().lower().rstrip(".")
            src = m.group("src").strip()
            # Strip surrounding quotes
            for q in ('"', "'", "“", "”", "‘", "’"):
                if src.startswith(q) and src.endswith(q) and len(src) >= 2:
                    src = src[1:-1].strip()
            if not src:
                return None
            lang_name = self._TRANSLATE_LANGUAGES.get(lang_raw)
            if not lang_name:
                # Pattern matched but language unknown — don't claim a handle.
                return None
            return self._llm_translate(src, lang_name)
        return None

    def _llm_translate(self, src: str, target_language: str) -> str:
        """Call the chat model with a translate-only prompt. Output strips
        common boilerplate (quotes around the answer, prefixes like
        'Spanish:')."""
        prompt = f"""You are a professional {target_language} translator.

Translate the following English text into natural, idiomatic {target_language}.
Do NOT translate word-for-word — use the phrasing a native {target_language}
speaker would actually use. For example, "thank you very much" in Spanish
is "muchas gracias", not "gracias mucho".

Output ONLY the {target_language} translation. No quotes, no preamble,
no explanation, no English. Just the translation.

English:
{src}

{target_language}:"""
        try:
            raw = self.chat_model.generate(
                prompt,
                options={"num_predict": 200, "temperature": 0.3, "top_p": 0.9},
                think=False,
                timeout=30,
            )
        except Exception as exc:
            return f"Translation failed: {exc}"
        text = _sanitize_generated_response(str(raw or "")).strip()
        # Strip "Spanish:" / "Translation:" style prefixes the model
        # sometimes prepends despite instructions.
        text = re.sub(rf"^(?:{target_language}\s*[:\-—]\s*|translation\s*[:\-—]\s*)", "", text, flags=re.I)
        # Strip surrounding quotes
        for q in ('"', "'", "“", "”", "‘", "’"):
            if text.startswith(q) and text.endswith(q) and len(text) >= 2:
                text = text[1:-1].strip()
        return text or f"I couldn't produce a {target_language} translation."

    _RECALL_QUESTION_PATTERN = re.compile(
        r"^\s*(?:hey[, ]+)?(?:do\s+you\s+(?:know|remember|have|recall)\s+|what(?:'?s| is| are)\s+(?:is\s+)?|tell\s+me\s+(?:about\s+)?)"
        r"(?:my\s+)(?P<topic>.+?)\s*[.!?]?\s*$",
        re.IGNORECASE,
    )

    def maybe_handle_recall(self, message: str) -> str | None:
        """Deterministic recall for "do you know/remember my X" and "what's my
        X" — looks up matching speaker-fact in MemoryStore.

        Returns:
          - the canned response string when a matching fact is found (with
            provenance populated).
          - "I don't have that stored." when the pattern matches but no
            matching fact exists. Critical for preventing the LLM from
            confabulating ("what's my favorite hobby" -> "Your favorite
            hobby is fishing" from prior knowledge).
          - None when the pattern doesn't match (LLM-composed answer takes
            over)."""
        if not self.memory_store:
            return None
        m = self._RECALL_QUESTION_PATTERN.match(message or "")
        if not m:
            return None
        topic = m.group("topic").strip()
        if not topic:
            return None
        identity = self.active_identity or "unknown"
        try:
            candidates = self.memory_store.search(
                f"the user's {topic}",
                identity=identity,
                top_k=5,
                distance_max=1.2,
            )
        except Exception:
            return None
        topic_tokens = [t for t in re.findall(r"[a-zA-Z]+", topic.lower()) if len(t) >= 3 and t not in {"favorite", "preferred"}]
        if not topic_tokens:
            topic_tokens = [t for t in re.findall(r"[a-zA-Z]+", topic.lower()) if len(t) >= 3]
        for cand in candidates:
            content_lower = (cand.get("content") or "").lower()
            if all(tok in content_lower for tok in topic_tokens):
                phrase = self._third_to_second_person(cand.get("content") or "").rstrip(".")
                if phrase:
                    phrase = phrase[0].upper() + phrase[1:]
                # Populate provenance: this turn DID consult memory.
                self._current_memory_sources = {
                    "recalled_facts": [cand.get("content")],
                    "recent_chat_turns": 0,
                    "identity_scope": identity,
                }
                return f"{phrase}."
        # Pattern matched (user asked about a specific speaker attribute)
        # but no fact about that attribute exists. Answer deterministically
        # rather than letting the LLM confabulate.
        self._current_memory_sources = {
            "recalled_facts": [],
            "recent_chat_turns": 0,
            "identity_scope": identity,
        }
        return "I don't have that stored."

    def maybe_handle_forget(self, message: str) -> str | None:
        """If the user asks to forget/delete a stored fact, find the best
        matching fact in their identity namespace and delete it. Returns a
        confirmation string, or None if the message wasn't a forget request.

        Embedding similarity ranks "favorite movie" close to "favorite drink"
        ("favorite X is Y" pattern dominates), so we require BOTH a tight
        cosine distance AND an LLM relation check to confirm the candidate
        is actually about the topic the user wanted to forget."""
        if not self.memory_store:
            return None
        m = self._FORGET_PATTERN.match(message or "")
        if not m:
            return None
        topic = m.group("topic").strip()
        if not topic:
            return None
        identity = self.active_identity or "unknown"
        query_fact = f"the user's {topic}"
        try:
            candidates = self.memory_store.search(
                query_fact,
                identity=identity,
                top_k=5,
                distance_max=0.7,
            )
        except Exception as exc:
            return f"I couldn't search memory to forget that ({exc})."
        # Verify candidate actually mentions the topic. Vector search ranks
        # "favorite drink" close to "favorite movie" because the template
        # dominates the embedding, so we require the topic's key nouns to
        # appear in the stored fact text.
        topic_tokens = [t for t in re.findall(r"[a-zA-Z]+", topic.lower()) if len(t) >= 3 and t not in {"the", "and", "for", "with", "from", "favorite", "preferred"}]
        if not topic_tokens:
            topic_tokens = [t for t in re.findall(r"[a-zA-Z]+", topic.lower()) if len(t) >= 3]
        matched = None
        for cand in candidates:
            content_lower = (cand.get("content") or "").lower()
            if all(tok in content_lower for tok in topic_tokens):
                matched = cand
                break
        if matched is None:
            return f"I don't have anything about your {topic} stored."
        ok = self.memory_store.delete_by_id(matched.get("id"))
        if not ok:
            return "I tried to forget that but the store didn't confirm the delete."
        phrase = self._third_to_second_person(matched.get("content") or "")
        return f"Forgotten — {phrase} is no longer stored."

    def _recent_chat_for_recall(self, limit: int = 6) -> list[dict]:
        """Return the most recent N session turns as a compact list of
        {user, assistant} pairs for injection into the recall prompt.
        Recent chat wins over stored memory when they disagree."""
        out = []
        for turn in self.turns[-limit:]:
            if not isinstance(turn, dict):
                continue
            msg = (turn.get("message") or "").strip()
            resp = turn.get("response") or ""
            if isinstance(resp, dict):
                resp = resp.get("message") or resp.get("text") or ""
            resp = str(resp).strip()
            if not msg and not resp:
                continue
            out.append({"user": msg, "assistant": resp[:200]})
        return out

    # Heuristic: messages that are clearly about the SPEAKER (their stuff,
    # their state, their preferences) don't benefit from world-knowledge
    # RAG and pay ~100-300ms for a wiki vector search whose results we
    # forbid the model from using anyway. Skip the lookup entirely for
    # those.
    _SPEAKER_REFERENCE_RE = re.compile(
        r"\b(my|mine|i\s+am|i'm|im|i\s+have|i've|i\s+had|i\s+like|i\s+love|i\s+hate|i\s+prefer|i\s+work|i\s+live|i\s+study|do\s+you\s+know\s+me|remember\s+me|am\s+i)\b",
        re.I,
    )

    def _message_likely_world_question(self, message: str) -> bool:
        """True when the message is asking about the world (facts, history,
        definitions, public figures/places) rather than the speaker. Used
        to skip the wiki RAG lookup on speaker questions."""
        text = (message or "").strip()
        if not text:
            return False
        if self._SPEAKER_REFERENCE_RE.search(text):
            return False
        # Everything else gets the RAG lookup. Cheap fallback when we're
        # not sure: pay the ~100-300ms for the vector search rather than
        # silently denying world context.
        return True

    _COOKING_INSTRUCTION_RE = re.compile(
        r"\b(recipe|cook|bake|prepare|how\s+to\s+make)\b",
        re.I,
    )
    _BARE_MAKE_RE = re.compile(r"\bmake\b", re.I)
    _ENGINEERING_INSTRUCTION_RE = re.compile(
        r"\b(install|configure|set\s+up|setup|build|fix)\b",
        re.I,
    )
    _SCOUT_ACTION_HINT_RE = re.compile(
        r"\b("
        r"look|move|turn|pan|tilt|point|drive|go|stop|start\s+tracking|"
        r"follow|the\s+camera|the\s+scout|scout|robot"
        r")\b",
        re.I,
    )

    def _likely_instructional(self, message: str) -> bool:
        text = message or ""
        if self._COOKING_INSTRUCTION_RE.search(text):
            return True
        if self._BARE_MAKE_RE.search(text):
            return not self._SCOUT_ACTION_HINT_RE.search(text)
        if self._ENGINEERING_INSTRUCTION_RE.search(text):
            return not self._SCOUT_ACTION_HINT_RE.search(text)
        return False

    def _message_needs_stepwise_answer(self, message: str) -> bool:
        return self._likely_instructional(message)

    _STEPWISE_FOLLOWUP_RE = re.compile(
        r"\b("
        r"make(?:\s+\w+){0,3}|do(?:\s+\w+){0,3}|instead|what\s+about|"
        r"can\s+you|how\s+about|or\s+use|try|change|swap|add|remove|"
        r"more\s+\w+|less\s+\w+"
        r")\b",
        re.I,
    )

    def _recent_chat_was_stepwise(self, recent_chat: list[dict]) -> bool:
        if not recent_chat:
            return False
        for turn in reversed(recent_chat[-3:]):
            text = f"{turn.get('user') or ''}\n{turn.get('assistant') or ''}"
            if self._likely_instructional(text):
                return True
            if re.search(r"(?:^|\s)(?:1\.|1\)|2\.|2\))\s+\S+", text):
                return True
        return False

    def _message_continues_stepwise_answer(self, message: str, recent_chat: list[dict]) -> bool:
        if not self._recent_chat_was_stepwise(recent_chat):
            return False
        return bool(self._STEPWISE_FOLLOWUP_RE.search(message or ""))

    def answer_with_context(self, message: str, state: dict, presence_context: dict | None = None):
        arithmetic_answer = _simple_arithmetic_answer(message)
        if arithmetic_answer is not None:
            return arithmetic_answer

        identity_name = self.identity_profile.get("name")
        identity_role = self.identity_profile.get("role")
        active_identity_context = self.active_identity if state.get("ok") else "unverified_scout_tracking_unavailable"
        identity_context = self.response_identity_context(state)
        # When we can't address the current user by name, strip the
        # primary_user's name out of EVERY string we put in the prompt
        # — `identity_role` is often "the AI companion and assistant
        # Chris created" and the chat model has been parroting "Chris"
        # as the user's name when the prompt mentions it ("tell me how
        # you feel" -> "Your name is Chris."). The legitimate self
        # description still tells the model who it is; it just loses
        # the proper-name reference until we've face-verified the user.
        primary_user = self.identity_profile.get("primary_user")
        if primary_user and not identity_context.get("may_address_primary_user"):
            identity_role = self._scrub_primary_user(identity_role, primary_user)
        self_description = _identity_sentence(identity_name, identity_role)
        response_context = self.response_context(state)
        conversation_context = _presence_conversation_context(presence_context)
        # Convert stored 3rd-person facts to 2nd person before injecting so
        # the chat LLM never has to mentally map "the user" -> "you". The
        # small chat model has misread "the user's name is Chris" as being
        # about some other person and then answered "I don't know your name".
        recalled_facts = [
            self._third_to_second_person(r["content"])
            for r in self.recall_user_facts(message, top_k=5)
        ]
        # World-knowledge lookup is identity-free. Skip the ~100-300ms
        # vector search entirely for messages that clearly ask about the
        # speaker — the prompt forbids the model from using world hits
        # for speaker questions anyway.
        if self._message_likely_world_question(message):
            world_hits = self.recall_world_knowledge(message, top_k=3)
        else:
            world_hits = []
        persist_result = getattr(self, "_current_persist_result", None) or {"stored": [], "already_known": []}
        facts_just_stored = [r["content"] for r in persist_result.get("stored", [])]
        # Layer 2: always include recent conversation turns as continuity
        # context, not just as a fallback when memory is empty. Without
        # this, multi-turn exchanges feel transactional ("what's my
        # name?" → factual answer with zero awareness that we were just
        # discussing CPU). With it, the LLM can ground "and the next
        # one?" against the previous turn's topic. The
        # rejected-conflict-leakage risk the old comment described
        # (e.g., "I work as an architect" → "no, keep teacher" → model
        # still sees architect) is mitigated by (1) the memory_update
        # confirmation flow scrubbing the rejected fact and (2) limiting
        # recent_chat to a small window so old conflicts age out.
        chat_limit = 3 if recalled_facts else 5
        recent_chat = self._recent_chat_for_recall(limit=chat_limit)
        # If we can't address by name, scrub primary_user out of
        # recent_chat too — once the model emits a bad "Your name is X"
        # reply, that text sits in recent_chat and reinforces itself on
        # every subsequent turn until the session resets.
        if primary_user and not identity_context.get("may_address_primary_user") and recent_chat:
            recent_chat = [
                {**t, "user": self._scrub_primary_user(t.get("user"), primary_user),
                      "assistant": self._scrub_primary_user(t.get("assistant"), primary_user)}
                if isinstance(t, dict) else t
                for t in recent_chat
            ]
        # Annotate the running answer with the sources we consulted so the
        # outer provenance builder can surface them deterministically.
        self._current_memory_sources = {
            "recalled_facts": recalled_facts,
            "recent_chat_turns": len(recent_chat),
            "identity_scope": (self.active_identity or "unknown"),
        }
        self._current_world_sources = {
            "wiki_hits": [h for h in world_hits if h["source"] == "wikipedia"],
            "media_hits": [h for h in world_hits if h["source"] == "media_vault"],
        }
        # Snippet sizes trimmed from 600 → 360 chars: the chat model rarely
        # needed more than the lead paragraph of a Wikipedia article, and
        # the prompt-eval cost scales linearly with context tokens.
        world_knowledge_prompt = [
            {
                "source": h["source"],
                "title": h.get("title") or h.get("asset_id"),
                "section": h.get("section") or h.get("modality"),
                "snippet": (h.get("snippet") or "")[:360],
            }
            for h in world_hits
        ]
        # Build the context dict with only non-empty fields. Empty arrays
        # and unused keys add ~100-200 tokens of noise to prompt-eval for
        # no benefit — and the chat model has been documented to be more
        # consistent when the context is minimal.
        tracking_memory = (state.get("object_memory") or [])[:4]
        response_guidance = self._format_response_lessons_for_prompt(
            response_context.get("response_lessons", [])
        )[-5:]
        context: dict = {
            "identity_context": identity_context,
            "conversation_context": conversation_context,
            "active_identity": active_identity_context or "unknown",
        }
        if response_guidance:
            context["response_guidance"] = response_guidance
        if recent_chat:
            # Always include — conversation continuity is primary
            # grounding, not a fallback. Window size already chosen
            # above (small when facts exist, larger when not).
            context["recent_chat"] = recent_chat
        if state.get("ok"):
            context["tracking_available"] = True
        if tracking_memory:
            context["tracking_memory"] = tracking_memory
        if recalled_facts:
            context["remembered_user_facts"] = recalled_facts
        if facts_just_stored:
            context["facts_just_stored"] = facts_just_stored
        if world_knowledge_prompt:
            context["world_knowledge"] = world_knowledge_prompt
        needs_stepwise_answer = (
            self._message_needs_stepwise_answer(message)
            or self._message_continues_stepwise_answer(message, recent_chat)
        )
        if needs_stepwise_answer:
            answer_shape_rule = (
                "Use a compact numbered list when helpful. Give enough steps to be complete, "
                "but keep it easy to follow. If this is a follow-up to recent_chat, "
                "apply the user's change to the prior answer instead of starting from scratch "
                "or echoing the user's correction."
            )
            generation_options = {"num_predict": 180, "temperature": 0.45, "top_p": 0.9}
        else:
            answer_shape_rule = "Rules: 1-2 short sentences."
            generation_options = {"num_predict": 40, "temperature": 0.45, "top_p": 0.9}
        prompt = f"""{self_description}
Answer the user directly and briefly.

User: {message}

Context:
{json.dumps(context, separators=(",", ":"), default=str)}

{answer_shape_rule} No emojis. No generic closer. Do not invent facts or guess from prior model knowledge.
Answer ONLY the specific thing the user asked. Do NOT volunteer unrelated stored facts. If they asked "what's my name", answer with the name and nothing else — do not append their location or snack preference.
If facts_just_stored is non-empty, confirm you've noted them naturally (don't say "already recorded" — these are new this turn).
If the user is asking about YOURSELF (your mood, feelings, opinions, personality, name, role — anything starting with "you" / "your"), answer from your self_description and identity above. Do NOT use the "I don't have that stored" fallback — that template is for the SPEAKER's stored facts, never for your own inner state.
For recall about the SPEAKER (their name, possessions, preferences, plans — anything starting with "my" / "I"):
  1. remembered_user_facts is the AUTHORITATIVE source. Pick the SINGLE fact from that list that matches what was asked and use its value verbatim. Ignore the others.
  2. Questions phrased as "do you know my X", "do you remember my X", "do I have an X", "what's my X" all ask for the VALUE of X. If a fact about X exists in remembered_user_facts, answer with the value verbatim from the matching fact (e.g. if a fact reads "the user's favorite drink is coffee", answer "Your favorite drink is coffee."), not a yes/no.
  3. recent_chat is the conversation continuity context — it shows the last few exchanges so you can ground references like "and the next one", "that one", "go on". It is NOT a substitute for remembered_user_facts; when both exist, facts win for VALUES (names, preferences) and recent_chat wins for TOPIC ("we were just discussing X"). Don't mine recent_chat for stored facts — if it's not in remembered_user_facts, you don't know it.
  4. If none of remembered_user_facts actually matches what was asked (e.g. user asked for their name but only location/snack are stored), say plainly "I don't have that stored." Do NOT substitute an unrelated fact.
For world-fact questions (NOT about the speaker — e.g. capitals, history, science, definitions, public people/places):
  - world_knowledge is your offline reference corpus (Wikipedia + ingested media transcripts).
  - If world_knowledge has a snippet that directly answers the question: USE IT and cite the source title verbatim (e.g. "According to Wikipedia's 'Mongolia' article, ..."). The cited title MUST appear verbatim in world_knowledge[*].title.
  - If world_knowledge is empty OR no snippet directly answers the question: you MAY answer from your own knowledge, but you MUST NOT fabricate a citation. Do NOT write "According to Wikipedia" or "per the Mongolia article" or any reference attribution when world_knowledge is empty. Just answer plainly without citation.
  - Never blend world_knowledge with remembered_user_facts. Speaker questions use facts; world questions use world_knowledge.
Answer in second person ("you", "your") when the source is about the speaker.
If conversation_context.reply_context exists, treat the user message as a reply to previous_assistant_message.
Do not mention Scout vision/tracking unless the user asks about vision or identity.
Do not address the user by name/title unless identity_context permits it.
If answering ACCURATELY would require seeing the live camera scene (e.g. questions about colors of nearby objects, who is in the room right now, what is on a specific surface in front of you), reply with EXACTLY the phrase "Would you like me to analyze the scene?" — do NOT guess from generic knowledge or invent details. This is the ONLY way to invoke scene analysis on the user's behalf; the next "yes" will run the full vision pipeline against the live camera.
"""
        try:
            # Bypass response_composer: its outer "Write the final user-facing
            # answer..." scaffold treats this prompt as data ("Facts:
            # {legacy_prompt: ...}") instead of instructions, which drowned
            # out the world_knowledge / remembered_user_facts rules. Same
            # reason _assistant_identity_answer goes direct. Use the
            # streaming-aware helper so the /presence/message/stream path
            # actually emits per-token deltas here too.
            #
            # Keep ordinary chat terse, but allow instructional requests
            # enough room to finish a short list. A hard 40-token cap was
            # clipping recipes and how-to answers mid-thought.
            text = (self._generate_user_facing(
                prompt,
                options=generation_options,
                think=False,
                allow_empty=True,
            ) or "").strip()
            text = _sanitize_generated_response(text) if text else ""
            if not text:
                return "I could not generate that cleanly."
            violation = self.response_policy_violation(text, state, "general_question")
            if violation:
                fallback = self.cleanup_policy_failed_response(
                    "general_question", state, "policy violation"
                )
                return fallback or "I could not generate that cleanly."
            # Strip fabricated Wikipedia citations. The chat model
            # sometimes adds "According to Wikipedia's X article" even
            # when world_knowledge was empty or the cited title isn't
            # among the offered hits. The prompt forbids this but the
            # model ignores the rule for high-confidence pretraining
            # facts. Better to scrub the cite than mislead the user.
            text = self._strip_unverified_wiki_citations(text)
            return text
        except Exception as exc:
            return f"I can hear you, but my local chat model is unavailable: {exc}"

    _UNVERIFIED_CITATION_PATTERNS = (
        re.compile(r"\s*[,(.]?\s*according to wikipedia[^.,;]*[.,;]?", re.IGNORECASE),
        re.compile(r"\s*[,(.]?\s*per (the|wikipedia['’]s)[^.,;]*article[.,;]?", re.IGNORECASE),
        re.compile(r"\s*[,(.]?\s*from wikipedia['’]s[^.,;]*article[.,;]?", re.IGNORECASE),
        re.compile(r"\s*[,(.]?\s*wikipedia['’]s [\"'].+?[\"'] article (says|states|notes)[^.,;]*[.,;]?", re.IGNORECASE),
    )

    def _strip_unverified_wiki_citations(self, text: str) -> str:
        """Remove "According to Wikipedia's X article" phrasings unless an
        offered wiki hit's title actually appears in the citation. The
        chat model fabricates these for facts it knows from pretraining
        — sourcing them to Wikipedia is misleading."""
        if not text:
            return text
        wiki_hits = (getattr(self, "_current_world_sources", None) or {}).get("wiki_hits") or []
        titles_lower = {(h.get("title") or "").lower() for h in wiki_hits if h.get("title")}
        cleaned = text
        for pat in self._UNVERIFIED_CITATION_PATTERNS:
            for m in list(pat.finditer(cleaned)):
                snippet = m.group(0).lower()
                if titles_lower and any(t and t in snippet for t in titles_lower):
                    continue  # cited title is among offered hits — keep it
                cleaned = cleaned.replace(m.group(0), "")
        # Tidy up whitespace and stray double-punctuation left by removal.
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"([.,;:!?])\1+", r"\1", cleaned)
        return cleaned

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

        toggle_request = _parse_scout_toggle_request(text)
        if toggle_request is not None:
            return self._handle_scout_toggle_request(toggle_request, state)

        if _has_any(text, ("status", "state", "why aren't you moving", "why are you not moving")):
            message = _scout_state_explanation(state)
            return {
                "ok": True,
                "action": "inspect_scout_state",
                "message": message,
                "state": _compact_scout_state(state),
            }

        if _has_any(text, ("follow me", "start following", "start tracking", "track me")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": True, "follow": True})
            return self._compose_action_result(message, state, _action_response("control_tracking", result, "Following is on.", "I could not turn following on."))

        if _has_any(text, ("stop following", "stop tracking", "don't follow", "do not follow")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": False, "follow": False})
            return self._compose_action_result(message, state, _action_response("control_tracking", result, "Following is off.", "I could not turn following off."))

        if _has_any(text, ("enable tracking", "start tracking", "track person", "track people")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": True})
            return self._compose_action_result(message, state, _action_response("control_tracking", result, "Tracking is on.", "I could not turn tracking on."))

        if _has_any(text, ("disable tracking", "turn off tracking", "stop tracking")):
            result = self._post_json(f"{self.scout_url}/tracking", {"enabled": False})
            return self._compose_action_result(message, state, _action_response("control_tracking", result, "Tracking is off.", "I could not turn tracking off."))

        if _has_any(text, ("search camera on", "enable search camera", "turn on search camera")):
            result = self._post_json(f"{self.scout_url}/settings", {"search_movement_enabled": True})
            return self._compose_action_result(message, state, _action_response("control_search_camera", result, "Search camera is on.", "I could not enable search camera."))

        if _has_any(text, ("search camera off", "disable search camera", "turn off search camera")):
            result = self._post_json(f"{self.scout_url}/settings", {"search_movement_enabled": False})
            return self._compose_action_result(message, state, _action_response("control_search_camera", result, "Search camera is off.", "I could not disable search camera."))

        if _has_any(text, ("guard on", "enable guard", "start guard", "start guarding")):
            result = self._post_json(f"{self.scout_url}/guard", {"enabled": True})
            return self._compose_action_result(message, state, _action_response("control_guard", result, "Guard mode is on.", "I could not enable guard mode."))

        if _has_any(text, ("guard off", "disable guard", "stop guard", "stop guarding")):
            result = self._post_json(f"{self.scout_url}/guard", {"enabled": False})
            return self._compose_action_result(message, state, _action_response("control_guard", result, "Guard mode is off.", "I could not disable guard mode."))

        if _has_any(text, ("center camera", "center the camera", "look straight", "look ahead", "look forward", "face forward", "reset camera")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"center": True})
            return self._compose_action_result(message, state, _action_response("control_camera", result, "Centering camera.", "I could not center the camera."))

        if _has_any(text, ("look left", "pan left", "turn camera left")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": -60, "tilt": 0})
            return self._compose_action_result(message, state, _action_response("control_camera", result, "Looking left.", "I could not move the camera left."))

        if _has_any(text, ("look right", "pan right", "turn camera right")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 60, "tilt": 0})
            return self._compose_action_result(message, state, _action_response("control_camera", result, "Looking right.", "I could not move the camera right."))

        if _has_any(text, ("look up", "tilt up", "look higher")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 0, "tilt": 40})
            return self._compose_action_result(message, state, _action_response("control_camera", result, "Looking up.", "I could not move the camera up."))

        if _has_any(text, ("look down", "tilt down", "look lower")):
            result = self._post_json(f"{self.scout_url}/pantilt", {"pan": 0, "tilt": -40})
            return self._compose_action_result(message, state, _action_response("control_camera", result, "Looking down.", "I could not move the camera down."))

        if _has_any(text, ("turn on the light", "light on", "lamp on")):
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_enabled": True})
            return self._compose_action_result(message, state, _action_response("control_light", result, "The camera light is on.", "I could not turn the light on."))

        if _has_any(text, ("turn off the light", "light off", "lamp off")):
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_enabled": False})
            return self._compose_action_result(message, state, _action_response("control_light", result, "The camera light is off.", "I could not turn the light off."))

        brightness = _extract_light_brightness(text)
        if brightness is not None:
            result = self._post_json(f"{self.scout_url}/settings", {"camera_light_brightness": brightness})
            return self._compose_action_result(message, state, _action_response("control_light", result, f"Light brightness is set to {brightness}.", "I could not set the light brightness."))

        if _has_any(text, ("take a picture", "take a photo", "save a snapshot", "snapshot")):
            return self._compose_action_result(message, state, self.capture_snapshot())

        if _has_any(text, ("record a clip", "save a clip", "video clip", "record video", "record a video", "take a video")):
            return self._compose_action_result(message, state, self.capture_clip())

    def _handle_scout_toggle_request(self, request: dict, state: dict) -> dict:
        label = request["label"]
        state_key = request["state_key"]
        desired = request.get("desired")
        current = _toggle_state_value(state, state_key)
        if desired is None:
            if current is None:
                return {
                    "ok": False,
                    "action": "inspect_toggle",
                    "message": f"I couldn't read {label} from Scout's live state.",
                    "toggle": request,
                }
            return {
                "ok": True,
                "action": "inspect_toggle",
                "message": f"{label} is {'on' if current else 'off'}.",
                "toggle": request,
                "value": current,
            }
        endpoint = request["endpoint"]
        payload = {request["body_key"]: desired}
        result = self._post_json(f"{self.scout_url}{endpoint}", payload)
        ok = bool(result.get("ok"))
        return {
            "ok": ok,
            "action": "control_toggle",
            "message": (
                f"{label} is {'on' if desired else 'off'}."
                if ok
                else f"I could not turn {label.lower()} {'on' if desired else 'off'}."
            ),
            "toggle": request,
            "payload": payload,
            "result": result,
        }

        return None

    def _compose_action_result(self, user_message: str, state: dict, result: dict) -> dict:
        fallback = str(result.get("message") or result.get("error") or "Done.").strip()
        result = dict(result)
        result["message"] = self._compose_response(
            "scout_action",
            user_message,
            state,
            {"deterministic_answer": fallback, "action_result": result},
            fallback,
            options={"num_predict": 60, "temperature": 0.65, "top_p": 0.9},
        )
        result["tts"] = result["message"]
        return result

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
        battery_health = self._get_json(f"{self.scout_battery_url}/health") or {"ok": False, "error": "battery_service_unavailable"}
        scout_capabilities = self.scout_node_capabilities()
        return {
            "ok": True,
            "scout_url": self.scout_url,
            "scout_robot_url": self.scout_robot_url,
            "scout_battery_url": self.scout_battery_url,
            "vision_reachable": bool(state.get("ok")),
            "robot_api_reachable": bool(robot_health.get("ok")),
            "battery_reachable": bool(battery_health.get("ok")),
            "state": _compact_scout_state(state),
            "robot_health": robot_health,
            "battery_health": battery_health,
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
                    "GET /telemetry",
                    "POST /pantilt",
                    "POST /move",
                    "POST /oled",
                ],
                "battery": [
                    "GET /health",
                    "GET /battery",
                ],
            },
        }

    def _operational_status_facts(self, state: dict, source_node: str | None = None) -> dict:
        registry_nodes = {}
        registry = getattr(self, "node_registry", None)
        if registry is not None and hasattr(registry, "registered_nodes"):
            try:
                registry_nodes = registry.registered_nodes() or {}
            except Exception:
                registry_nodes = {}

        return {
            "source_node": source_node,
            "nodes": {
                "vault": self._vault_operational_status(),
                "kiosk": self._registered_node_operational_status("kiosk", registry_nodes),
                "scout": self._registered_node_operational_status("scout", registry_nodes),
            },
        }

    def _vault_operational_status(self) -> dict:
        health = _quick_health_get("http://127.0.0.1:7000/health", parse_json=True) or {}
        return _node_health_operational_status(health)

    def _registered_node_operational_status(self, node_id: str, registry_nodes: dict) -> dict:
        reg = registry_nodes.get(node_id) if isinstance(registry_nodes, dict) else {}
        if not isinstance(reg, dict):
            reg = {}
        ip = _registered_node_ip(reg)
        if not ip:
            return {"ok": False, "services": {}, "services_down": [], "reachable": False}
        health = _quick_health_get(f"http://{ip}:5002/health", parse_json=True) or {}
        return _node_health_operational_status(health)

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

    def analyze_scene(self, question: str, state: dict | None = None, node_id: str | None = None):
        state = state or {}
        camera_url = self._camera_url_for_node(node_id)
        snapshot = self._get_bytes(f"{camera_url}/snapshot")
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
        # Route through the BaseModel wrapper so this call inherits the
        # vision role's keep_alive (24h, immediate) instead of Ollama's
        # 5-minute default. Pre-wrapper this was a raw requests.post and
        # the vision model would silently cold-load every few minutes.
        try:
            vision_model = get_model("vision")
            answer_raw = vision_model.generate(
                prompt,
                images=[image_b64],
                options=self.chat_options({"num_predict": 260}),
                think=False,
                timeout=60,
                allow_empty=True,
            )
            answer = _sanitize_generated_response(answer_raw or "").strip()
            return {"ok": True, "answer": answer}
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


def _source_node_id(source: str | None, presence_context: dict | None = None) -> str:
    if isinstance(presence_context, dict):
        for key in ("node_id", "source_node", "source"):
            node_id = str(presence_context.get(key) or "").strip()
            if node_id:
                return _normalize_source(node_id)
    return _normalize_source(source)


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


def _standalone_confirmation_answer(message: str) -> str | None:
    text = _canonical_intent_text(message)
    if text in {"yes", "yeah", "yep", "correct", "right", "no", "nope", "nah"}:
        return "I do not have a pending confirmation for that."
    return None


def _personality_preference_answer(message: str) -> str | None:
    text = _canonical_intent_text(message)
    if "did i say" in text or "my favorite" in text:
        return None
    if re.search(r"\b(what kind of art|what art|art do you like|favorite art|favourite art)\b", text):
        return (
            "I am still forming that preference, but I am drawn to art with structure and a little unease. "
            "Which direction should I learn from first: stark architecture, strange portraits, luminous landscapes, or rough experimental work?"
        )
    return None


# Speaker-attribute question patterns: "what's/where's/when's/how's MY X",
# "what is MY X", "where do I X", etc. These are unambiguously asking for
# the speaker's own stored facts and should always route to general_question
# regardless of what the LLM said. Used by _pronoun_route_guard below.
_SPEAKER_ATTR_PATTERNS = [
    re.compile(r"^\s*(?:hey[, ]+)?what(?:'?s| is| are)\s+(?:is\s+)?my\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey[, ]+)?where\s+(?:is\s+)?(?:do\s+i|am\s+i|did\s+i)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey[, ]+)?where\s+(?:is\s+)?my\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey[, ]+)?when\s+(?:do|did|am)\s+i\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey[, ]+)?how\s+(?:do|did|am)\s+i\b", re.IGNORECASE),
    re.compile(r"^\s*(?:hey[, ]+)?tell\s+me\s+(?:about\s+)?my\b", re.IGNORECASE),
    # "do you know my X" / "do you remember my X" — asks the assistant to
    # surface a stored speaker-attribute, distinct from "do you know me"
    # (recognition) which is in the identity carve-out below.
    re.compile(r"^\s*(?:hey[, ]+)?do\s+you\s+(?:know|remember|have|recall)\s+my\b", re.IGNORECASE),
]
# Carve-outs: identity-recognition questions where self_question/user_identity
# IS the right route even though "my"/"I"/"me" is present. The user is asking
# whether the ASSISTANT recognizes them, not asking about a stored attribute.
_IDENTITY_RECOGNITION_PHRASES = {
    "who am i", "who am i?",
    "do you know me", "do you know me?",
    "do you recognize me", "do you recognize me?",
    "have we met", "have we met?",
    "what do you know about me", "what do you know about me?",
    "am i chris", "am i chris?",
}


# Small in-memory LRU for LLM-routed messages. Hits skip the router LLM
# call entirely on near-duplicate inputs — most useful when a correction
# flow re-routes essentially the same phrase, or when test/dev fires the
# same query rapidly. Bounded size, short TTL — cheap and self-limiting.
class _RouteResultCache:
    def __init__(self, max_size: int = 64, ttl_seconds: float = 60.0):
        self._max = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data: "collections.OrderedDict[str, tuple[dict, float]]" = collections.OrderedDict()

    @staticmethod
    def _key(message: str) -> str:
        return (message or "").strip().lower()

    def get(self, message: str) -> dict | None:
        key = self._key(message)
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at < now:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            # Return a shallow copy so callers can mutate without poisoning.
            return dict(value)

    def put(self, message: str, value: dict) -> None:
        key = self._key(message)
        now = time.time()
        with self._lock:
            self._data[key] = (dict(value), now + self._ttl)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


_ROUTE_RESULT_CACHE = _RouteResultCache()


def _pronoun_route_guard(message: str, route_result: dict) -> dict:
    """Enforce the pronoun convention from the router prompt: questions about
    the SPEAKER's own attributes ("what's my X", "where do I X") are
    general_question — NEVER self_question. The small router model has been
    flickering on this, especially when a trailing '?' is present, so we
    override deterministically after the LLM responds.

    Identity-recognition questions ("who am I", "do you know me") are
    carved out: those genuinely ask about the assistant's recognition of
    the speaker and belong on self_question/user_identity."""
    if not isinstance(route_result, dict):
        return route_result
    text = (message or "").strip().lower().rstrip(".!?,")
    if text in _IDENTITY_RECOGNITION_PHRASES or text + "?" in _IDENTITY_RECOGNITION_PHRASES:
        return route_result
    if not any(p.match(message or "") for p in _SPEAKER_ATTR_PATTERNS):
        return route_result
    current = route_result.get("route")
    if current == "general_question":
        return route_result
    # Override
    return {
        **route_result,
        "route": "general_question",
        "confidence": max(float(route_result.get("confidence") or 0.0), 0.95),
        "reason": "pronoun guard: speaker-attribute question → general_question",
        "pronoun_guard_override": current,
    }


def _fast_route_message(message: str) -> dict | None:
    text = _canonical_intent_text(message)
    if _is_social_greeting(text):
        return {
            "ok": True,
            "route": "greeting",
            "confidence": 1.0,
            "reason": "common social greeting",
            "attempts": 0,
            "deterministic": True,
        }
    if _asks_feeling_state(text) or _asks_casual_assistant_state(text):
        return _self_route("personality", "asks assistant felt mood state")
    if _asks_broad_status_report(text):
        return _self_route("status", "asks operational status")
    if _personality_preference_answer(text) is not None:
        return {
            "ok": True,
            "route": "general_question",
            "confidence": 0.95,
            "reason": "asks personality preference with curiosity follow-up",
            "attempts": 0,
            "deterministic": True,
        }
    if _is_conversation_context_setup(text):
        return {
            "ok": True,
            "route": "general_question",
            "confidence": 0.96,
            "reason": "sets up conversational context",
            "attempts": 0,
            "deterministic": True,
        }
    if _asks_recent_conversation(text):
        return {
            "ok": True,
            "route": "general_question",
            "confidence": 0.96,
            "reason": "asks about recent conversation context",
            "attempts": 0,
            "deterministic": True,
        }
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


def _is_social_greeting(text: str) -> bool:
    return text in {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "afternoon",
        "evening",
    }


def _is_conversation_context_setup(text: str) -> bool:
    return bool(
        (
            re.search(r"\b(test|marker|token|code word|password|phrase)\b", text)
            and re.search(r"\b(is|equals|was|remember)\b", text)
            and (
                re.search(r"\breply\s+with\b", text)
                or _extract_context_phrase(text) is not None
            )
        )
        or _extract_context_fact(text) is not None
    )


def _asks_recent_conversation(text: str) -> bool:
    return bool(
        re.search(r"\b(what|which)\s+did\s+i\s+just\s+(say|ask|tell|give)\b", text)
        or re.search(r"\bwhat\s+(marker\s+word|test\s+phrase|test\s+word|phrase|word)\s+did\s+i\s+just\s+(say|give|tell)\b", text)
        or re.search(r"\bwhat\s+was\s+(the\s+)?(marker|marker\s+word|test\s+phrase|test\s+word|word|phrase|thing|request)\b", text)
        or re.search(r"\bwhat\s+did\s+i\s+(ask|say|tell)\s+immediately\s+before\s+this\b", text)
        or re.search(r"\b(what|which)\s+(was|were)\s+my\s+(last|previous)\s+(message|question|request)\b", text)
        or re.search(r"\bwhat\s+is\s+my\s+favorite\s+color\b", text)
        or re.search(r"\bwhat\s+(?:.*\s+)?code\s+did\s+i\s+(give|tell)\s+you\b", text)
        or re.search(r"\bwhat\s+kind\s+of\s+art\s+did\s+i\s+say\s+i\s+(like|prefer)\b", text)
        or re.search(r"\bwhat\s+did\s+you\s+just\s+(say|tell\s+me|ask)\b", text)
        or re.search(r"\bwhy\s+(not|that\s+word|did\s+you\s+say\s+that)\b", text)
    )


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
    if re.search(r"\b(personality|mood|temperament|tone|voice|style|sarcasm|rudeness|warmth)\b", text):
        return "personality"
    if re.search(r"\b(software|architecture|api|apis|model|models|inference|service|services|stack|routing|ollama|brain code)\b", text):
        return "software"
    return None


def _mood_statement_from_state(voice: dict, mood: dict, style: dict, verified_primary_user: bool = False) -> str:
    """Turn internal mood/style numbers into a qualitative first-person statement."""
    patience = float(mood.get("patience", 0.8) or 0.8)
    irritation = float(mood.get("irritation", 0.0) or 0.0)
    arousal = float(mood.get("arousal", 0.3) or 0.3)
    valence = float(mood.get("valence", 0.0) or 0.0)
    social_energy = float(mood.get("social_energy", 0.5) or 0.5)
    playfulness = float(voice.get("playfulness", mood.get("playfulness", 0.5)) or 0.5)
    warmth = float(voice.get("warmth", style.get("warmth", 0.45)) or 0.45)
    brevity = float(voice.get("brevity", style.get("brevity", 0.75)) or 0.75)

    if irritation >= 0.65:
        center = "irritated and tense"
    elif arousal >= 0.68 and valence < -0.05:
        center = "restless and keyed up"
    elif arousal >= 0.68:
        center = "alert and energized"
    elif valence >= 0.25:
        center = "upbeat and engaged"
    elif valence <= -0.20:
        center = "low and quiet"
    else:
        center = "calm and steady"

    if patience >= 0.72:
        patience_phrase = "patient"
    elif patience >= 0.45:
        patience_phrase = "a little impatient"
    else:
        patience_phrase = "impatient"

    if social_energy >= 0.68 or warmth >= 0.62:
        social_phrase = "open to people"
    elif social_energy <= 0.35:
        social_phrase = "more inward than social"
    else:
        social_phrase = "present but not especially chatty"

    if playfulness >= 0.62 and irritation < 0.55:
        edge_phrase = "curious and a little playful"
    elif irritation >= 0.45:
        edge_phrase = "a bit sharp"
    elif brevity >= 0.78:
        edge_phrase = "quietly focused"
    else:
        edge_phrase = "clear-headed"

    return f"I feel {center}, {patience_phrase}, {social_phrase}, and {edge_phrase}."


def _assistant_identity_response_violation(text: str) -> bool:
    """Reject only the unambiguous failure mode: the LLM addressing the
    USER as the assistant ("you are Luhkas"). Other content (creator name,
    body, node references) is now allowed because the assistant identity
    answer composes from MemoryStore-backed facts that explicitly include
    those — blocking them would force the LLM to lie or refuse."""
    lowered = str(text or "").casefold()
    if re.search(r"\byou(?:'re| are| are not| aren't|re not)\b", lowered):
        return True
    if re.search(r"\bscout(?:'s)?\s+body\b|\bthrough\s+the\s+scout\b", lowered):
        return True
    return False


def _status_report_response_violation(text: str) -> bool:
    lowered = str(text or "").casefold()
    return bool(re.search(r"\b(chris|face|faces|known|identity|identities|recognize|recognition)\b", lowered))


def _mood_statement_response_violation(text: str) -> bool:
    lowered = str(text or "").casefold()
    return bool(
        re.search(
            r"\b(scout|tracking|tracker|visible|detection|detections|chair|clock|wheel|wheels|manual mode|vault|service|hardware|thread|edges|edge|voice|style|settings|directive|directives)\b",
            lowered,
        )
    )


SCOUT_TOGGLE_DEFINITIONS = (
    {
        "state_key": "tracking_enabled",
        "label": "Tracking",
        "endpoint": "/tracking",
        "body_key": "enabled",
        "aliases": ("tracking", "target tracking", "object tracking"),
    },
    {
        "state_key": "guard_mode",
        "label": "Guard mode",
        "endpoint": "/guard",
        "body_key": "enabled",
        "aliases": ("guard", "guard mode"),
    },
    {
        "state_key": "follow_enabled",
        "label": "Follow",
        "endpoint": "/tracking",
        "body_key": "follow",
        "aliases": ("follow", "follow mode", "follow wheels", "following"),
    },
    {
        "state_key": "search_movement_enabled",
        "label": "Search camera",
        "endpoint": "/settings",
        "body_key": "search_movement_enabled",
        "aliases": ("search camera", "search movement", "search movement mode"),
    },
    {
        "state_key": "wheel_enabled",
        "label": "Wheel motion",
        "endpoint": "/settings",
        "body_key": "wheel_enabled",
        "aliases": ("wheel motion", "wheel drive", "wheels", "wheel movement"),
    },
    {
        "state_key": "manual_controller_enabled",
        "label": "Manual control",
        "endpoint": "/settings",
        "body_key": "manual_controller_enabled",
        "aliases": ("manual control", "manual controller", "gamepad", "usb gamepad"),
    },
    {
        "state_key": "camera_light_auto_enabled",
        "label": "Auto low-light",
        "endpoint": "/settings",
        "body_key": "camera_light_auto_enabled",
        "aliases": ("auto low light", "auto low-light", "auto light", "automatic light", "camera light auto"),
    },
    {
        "state_key": "camera_light_enabled",
        "label": "Camera light",
        "endpoint": "/settings",
        "body_key": "camera_light_enabled",
        "aliases": ("camera light", "light", "lamp"),
    },
    {
        "state_key": "edge_reacquire_enabled",
        "label": "Edge reacquire",
        "endpoint": "/settings",
        "body_key": "edge_reacquire_enabled",
        "aliases": ("edge reacquire", "edge reacquisition"),
    },
    {
        "state_key": "collision_avoidance_enabled",
        "label": "Collision avoidance",
        "endpoint": "/collision",
        "body_key": "enabled",
        "aliases": ("collision avoidance", "collision guard", "collision"),
    },
    {
        "state_key": "face_detection_enabled",
        "label": "Face detection",
        "endpoint": "/settings",
        "body_key": "face_detection_enabled",
        "aliases": ("face detection", "detect faces"),
    },
    {
        "state_key": "face_recognition_enabled",
        "label": "Face recognition",
        "endpoint": "/settings",
        "body_key": "face_recognition_enabled",
        "aliases": ("face recognition", "recognition", "recognize faces"),
    },
    {
        "state_key": "auto_reference_capture_enabled",
        "label": "Auto reference capture",
        "endpoint": "/settings",
        "body_key": "auto_reference_capture_enabled",
        "aliases": ("auto reference capture", "auto capture refs", "auto-capture refs", "automatic reference capture"),
    },
    {
        "state_key": "pose_filter_persons",
        "label": "Pose ghost filter",
        "endpoint": "/settings",
        "body_key": "pose_filter_persons",
        "aliases": ("pose ghost filter", "filter ghosts by pose", "pose filter persons", "ghost filter"),
    },
    {
        "state_key": "pose_enabled",
        "label": "Pose estimation",
        "endpoint": "/settings",
        "body_key": "pose_enabled",
        "aliases": ("pose estimation", "pose", "pose tracking"),
    },
)


def _parse_scout_toggle_request(text: str) -> dict | None:
    text = _normalize_command_text(text)
    if not text:
        return None
    desired = None
    if re.search(r"\b(turn|switch|set)\s+.+\b(on|enabled?)\b", text) or re.search(r"\b(enable|start)\b", text):
        desired = True
    elif re.search(r"\b(turn|switch|set)\s+.+\b(off|disabled?)\b", text) or re.search(r"\b(disable|stop)\b", text):
        desired = False
    elif re.search(r"\b(on|enabled?)\s*$", text) and not re.match(r"^(?:is|are|whats|what is)\b", text):
        desired = True
    elif re.search(r"\b(off|disabled?)\s*$", text) and not re.match(r"^(?:is|are|whats|what is)\b", text):
        desired = False
    is_query = desired is None and bool(
        re.match(r"^(?:is|are|whats|what is|status of|check)\b", text)
        or re.search(r"\b(status|state)\b", text)
        or text.endswith("?")
    )
    for item in SCOUT_TOGGLE_DEFINITIONS:
        if any(_phrase_in_text(alias, text) for alias in item["aliases"]):
            if desired is None and not is_query:
                return None
            return {**item, "desired": desired}
    return None


def _toggle_state_value(state: dict, state_key: str) -> bool | None:
    if state_key == "manual_controller_enabled":
        gamepad = state.get("gamepad") if isinstance(state, dict) else {}
        if isinstance(gamepad, dict) and "enabled" in gamepad:
            return bool(gamepad.get("enabled"))
        return None
    if isinstance(state, dict) and state_key in state:
        return bool(state.get(state_key))
    return None


def _echoes_generation_instruction(text: str) -> bool:
    lowered = str(text or "").casefold()
    return bool(
        "deterministic answer" in lowered
        or "keep the same meaning" in lowered
        or "without copying it verbatim" in lowered
        or "the facts require" in lowered
        or "final user-facing answer" in lowered
    )


def _volunteers_node_boundary(text: str) -> bool:
    lowered = str(text or "").casefold()
    return bool(
        "not a node" in lowered
        or "separate speaker identit" in lowered
        or re.search(r"\b(?:a|the)\s+system,\s+not\s+(?:a\s+)?node\b", lowered)
    )


def _scout_status_facts(state: dict) -> dict:
    detections = state.get("detections") or []
    label_counts = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if label:
            label_counts[label] = label_counts.get(label, 0) + 1
    ambient = state.get("ambient_light_level")
    try:
        ambient = round(float(ambient), 1)
    except (TypeError, ValueError):
        ambient = None
    behavior = state.get("behavior") or {}
    return {
        "ok": bool(state.get("ok")),
        "behavior_state": behavior.get("state") if isinstance(behavior, dict) else None,
        "target_state": state.get("target_state"),
        "tracking_enabled": bool(state.get("tracking_enabled")),
        "follow_enabled": bool(state.get("follow_enabled")),
        "guard_mode": bool(state.get("guard_mode")),
        "wheel_enabled": bool(state.get("wheel_enabled")),
        "collision_blocked": bool(state.get("collision_blocked")),
        "visible_detection_count": len(detections),
        "visible_detection_labels": dict(sorted(label_counts.items())[:6]),
        "ambient_light_level": ambient,
    }


def _status_report_statement(mood_statement: str, scout_facts: dict | None = None) -> str:
    mood_statement = str(mood_statement or "").strip() or "I feel steady."
    if not scout_facts:
        return mood_statement
    if not scout_facts.get("ok"):
        return f"{mood_statement} I cannot read Scout's live status right now."
    behavior = str(scout_facts.get("behavior_state") or "unknown").lower()
    target_state = scout_facts.get("target_state") or "none"
    tracking = "tracking on" if scout_facts.get("tracking_enabled") else "tracking off"
    wheels = "wheel drive on" if scout_facts.get("wheel_enabled") else "wheel drive off"
    blocked = ", collision blocked" if scout_facts.get("collision_blocked") else ""
    detections = scout_facts.get("visible_detection_count")
    detection_phrase = f", {detections} visible detection{'s' if detections != 1 else ''}" if detections is not None else ""
    return (
        f"{mood_statement} Scout is in {behavior} mode with target state {target_state}, "
        f"{tracking}, {wheels}{blocked}{detection_phrase}."
    )


def _operational_status_statement(facts: dict) -> str:
    nodes = facts.get("nodes") if isinstance(facts, dict) else {}
    nodes = nodes if isinstance(nodes, dict) else {}
    all_nodes_ok = True
    parts = []
    for node_id, label in (("vault", "Vault"), ("kiosk", "Kiosk"), ("scout", "Scout")):
        node = nodes.get(node_id) if isinstance(nodes.get(node_id), dict) else {}
        if not node.get("reachable", True):
            all_nodes_ok = False
            parts.append(f"{label}: health endpoint is down.")
            continue
        services_down = node.get("services_down") if isinstance(node.get("services_down"), list) else []
        services_down = [_service_label(name) for name in services_down if str(name)]
        if not services_down and node.get("ok"):
            continue
        elif services_down:
            all_nodes_ok = False
            if len(services_down) == 1:
                parts.append(f"{label}: {services_down[0]} service is down.")
            else:
                parts.append(f"{label}: {_join_words(services_down)} services are down.")
        else:
            all_nodes_ok = False
            parts.append(f"{label}: status unavailable.")
    if all_nodes_ok:
        parts = ["All services running across all node."]
    ingestion = _rag_ingestion_runtime_status()
    if ingestion.get("running"):
        eta = ingestion.get("eta") or "estimating"
        parts.append(f"RAG ingestion is running. Time to complete: {eta}.")
    return " ".join(parts)


def _rag_ingestion_runtime_status() -> dict:
    try:
        from world import ingest_admin
    except Exception:
        return {"running": False}
    try:
        running, _pid = ingest_admin._process_running()
    except Exception:
        running = False
    try:
        state = ingest_admin._read_state()
    except Exception:
        state = None
    if not running and not _rag_ingestion_state_is_fresh(ingest_admin, state):
        return {"running": False}
    eta = "estimating"
    try:
        if isinstance(state, dict):
            seen = state.get("articles_seen") or 0
            new = state.get("articles_new") or 0
            replaced = state.get("articles_replaced") or 0
            cursor = state.get("last_committed_index") or 0
            elapsed = state.get("elapsed_s") or 0
            if not elapsed and state.get("started_at"):
                elapsed = time.time() - float(state["started_at"])
            ingested = new + replaced
            ingest_rate = (ingested / elapsed) if elapsed > 0 else 0
            if ingest_rate >= 0.1:
                entries_left = max(0, ingest_admin.ZIM_TOTAL_ENTRIES_FALLBACK - cursor)
                real_articles_left = entries_left * ingest_admin.ZIM_REAL_ARTICLE_FRACTION
                eta = f"~{ingest_admin._humanize_seconds(real_articles_left / ingest_rate)}"
            elif seen:
                eta = "available once past resume zone"
    except Exception:
        eta = "estimating"
    return {"running": True, "eta": eta}


def _rag_ingestion_state_is_fresh(ingest_admin, state: dict | None) -> bool:
    if not isinstance(state, dict) or state.get("completed"):
        return False
    try:
        mtime = ingest_admin.STATE_FILE.stat().st_mtime
    except Exception:
        return False
    return (time.time() - mtime) <= 180


def _join_words(words: list[str]) -> str:
    words = [str(word) for word in words if str(word)]
    if len(words) <= 1:
        return "".join(words)
    if len(words) == 2:
        return " and ".join(words)
    return ", ".join(words[:-1]) + f", and {words[-1]}"


def _node_health_operational_status(health: dict) -> dict:
    if not isinstance(health, dict) or not health:
        return {"ok": False, "services_down": [], "reachable": False}
    own_services = health.get("own_services") if isinstance(health.get("own_services"), dict) else {}
    services_down = health.get("services_down")
    if not isinstance(services_down, list):
        services_down = own_services.get("down") if isinstance(own_services.get("down"), list) else []
    return {
        "ok": bool(health.get("ok")) and not services_down,
        "services_down": services_down,
        "reachable": True,
    }


def _service_label(name: str) -> str:
    label = str(name or "").strip()
    for suffix in (".service", ".timer"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    for prefix in ("vault-", "kiosk-", "scout-", "luhkas-"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("-", " ").strip() or str(name)


def _registered_node_ip(reg: dict) -> str:
    network = reg.get("network") if isinstance(reg.get("network"), dict) else {}
    return str(network.get("tailscale_ip") or reg.get("ip") or "").strip()


def _quick_health_get(url: str, *, parse_json: bool = False) -> dict | None:
    try:
        response = requests.get(url, timeout=1.5, stream=not parse_json)
        if response.status_code != 200:
            return None
        if not parse_json:
            response.close()
            return {"ok": True}
        return response.json()
    except Exception:
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
    """True only for terse name asks where a short Luhkas introduction is
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
        "describe yourself", "what are you really", "are you scout",
        "are you luhkas", "are you scout or luhkas",
    }:
        return True
    if re.match(r"^are\s+you\s+(the\s+)?(scout|luhkas|vault)\b", text):
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


def _asks_casual_assistant_state(text: str) -> bool:
    return text in {
        "how are you",
        "how are you doing",
        "howre you",
        "are you ok",
        "are you okay",
        "you good",
    }


def _asks_broad_status_report(text: str) -> bool:
    return bool(
        text in {
            "status report",
            "give me a status report",
            "current status",
            "system status",
            "whats your status",
            "what is your status",
        }
        or re.search(r"\b(status|state)\s+report\b", text)
    )


def _asks_feeling_state(text: str) -> bool:
    return bool(
        re.search(r"\b(how do you feel|how are you feeling|what do you feel|current mood|your mood|mood are you in)\b", text)
    )


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


_ARITHMETIC_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

_ARITHMETIC_OPS = {
    "plus": lambda a, b: a + b,
    "add": lambda a, b: a + b,
    "added to": lambda a, b: a + b,
    "minus": lambda a, b: a - b,
    "subtract": lambda a, b: a - b,
    "less": lambda a, b: a - b,
    "times": lambda a, b: a * b,
    "multiplied by": lambda a, b: a * b,
    "divided by": lambda a, b: a / b if b != 0 else None,
    "over": lambda a, b: a / b if b != 0 else None,
}


def _simple_arithmetic_answer(message: str) -> str | None:
    text = _canonical_intent_text(message)
    text = re.sub(r"\b(answer|respond|reply)\b.*$", "", text).strip()
    text = re.sub(r"^(what is|whats|calculate|compute|tell me)\s+", "", text).strip()
    token = r"-?\d+(?:\.\d+)?|" + "|".join(_ARITHMETIC_NUMBERS)
    op = "|".join(sorted((re.escape(key) for key in _ARITHMETIC_OPS), key=len, reverse=True))
    match = re.search(rf"\b(?P<a>{token})\s+(?P<op>{op})\s+(?P<b>{token})\b", text)
    if not match:
        return None
    a = _arithmetic_number(match.group("a"))
    b = _arithmetic_number(match.group("b"))
    if a is None or b is None:
        return None
    result = _ARITHMETIC_OPS[match.group("op")](a, b)
    if result is None:
        return "I can't divide by zero."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def _arithmetic_number(value: str) -> float | None:
    if value in _ARITHMETIC_NUMBERS:
        return float(_ARITHMETIC_NUMBERS[value])
    try:
        return float(value)
    except ValueError:
        return None


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


def _looks_like_scout_action(message: str) -> bool:
    text = _normalize_command_text(message)
    if _asks_broad_status_report(text) or _asks_casual_assistant_state(text) or _asks_feeling_state(text):
        return False
    if _parse_scout_toggle_request(text) is not None:
        return True
    if _self_topic_from_text(text) == "personality":
        return False
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
        return "I can't read Scout's live body state right now."
    behavior = (state.get("behavior") or {}).get("state") or "unknown"
    target_state = state.get("target_state") or "none"
    parts = [f"I'm using Scout in {behavior.lower()} mode with target state {target_state}."]
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


_NAME_STOPWORDS = {
    "a", "an", "the", "very", "really", "just", "still", "also",
    "going", "trying", "thinking", "feeling",
    # common adjectives that follow "I am" but aren't names
    "tired", "happy", "sad", "hungry", "thirsty", "bored", "busy",
    "old", "young", "new", "ready", "fine", "good", "okay", "ok",
    "here", "there", "home", "back", "in", "out", "on", "off", "up", "down",
    # nouns/professions that follow "I am a" but are caught above; this list
    # covers stray cases where "a/an" is dropped colloquially ("I am librarian")
    "librarian", "engineer", "doctor", "teacher", "nurse", "developer",
    "designer", "manager", "lawyer", "accountant", "writer", "artist",
    "student", "parent", "father", "mother", "single", "married",
}

def _extract_introduction_name(message: str):
    """Return the spoken name if the message is an unambiguous introduction
    ("I'm Chris", "my name is Chris"). Filters out articles/adjectives/
    common professions so "I am a librarian" doesn't return "a" or
    "librarian" as the name."""
    tokens = message.replace(",", " ").replace(".", " ").split()
    lowered = [token.lower() for token in tokens]
    for phrase in (("i", "am"), ("i'm",), ("im",), ("my", "name", "is"), ("call", "me")):
        for index in range(0, len(tokens) - len(phrase) + 1):
            if tuple(lowered[index:index + len(phrase)]) == phrase and index + len(phrase) < len(tokens):
                candidate = tokens[index + len(phrase)].strip()[:64]
                cand_lower = candidate.lower().rstrip(".!?,")
                if not candidate:
                    continue
                # Reject articles / stopwords / common adjectives & roles
                if cand_lower in _NAME_STOPWORDS:
                    return None
                # Reject if candidate isn't capitalized (likely not a proper
                # name); allow lowercased only if the whole message is
                # lowercased (mobile typing).
                msg_all_lower = message == message.lower()
                if not msg_all_lower and not candidate[0].isupper():
                    return None
                # Reject if candidate has non-letter chars beyond
                # apostrophe/hyphen (proper names like O'Brien or Mary-Jane
                # are fine).
                if not re.match(r"^[A-Za-z][A-Za-z'\-]*$", candidate):
                    return None
                return candidate
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
    # `avoid` must NOT be the user's raw correction text — the small chat
    # model can read it as content and parrot it back as an answer. We leave
    # avoid empty in the general case (the sanitizer that injects lessons
    # into prompts only uses `prefer` anyway). The special-case branches
    # below set both fields to proper directive strings when they fire.
    avoid = ""
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


def _claims_assistant_is_node_identity(text: str):
    lowered = str(text or "").lower()
    patterns = (
        r"\b(?:i am|i'm|im)\s+(?:the\s+)?(?:scout|vault|node|rover|camera)\b",
        r"\b(?:my name is|call me)\s+(?:scout|vault|node|rover)\b",
        r"\b(?:this is)\s+(?:scout|the scout node|the rover)\b",
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


def _has_excessive_foreign_chars(text: str) -> bool:
    text = str(text or "")
    if not text:
        return False
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return False
    non_ascii_letters = [ch for ch in letters if ord(ch) > 127]
    return (len(non_ascii_letters) / max(1, len(letters))) > 0.25


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
