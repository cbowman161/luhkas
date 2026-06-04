"""vault/identity_actions.py — privileged identity-control verbal intents.

Two actions, both gated on visually verifying the speaker is the
current primary user:

  - **unset primary user** — clears the primary_user designation.
    The runtime returns to ``awaiting_primary_user`` mode and the
    background ticker greets the next visible face.

  - **transfer primary user to <name>** — flips the primary_user
    pointer to ``<name>`` if that identity is already enrolled. If
    ``<name>`` isn't enrolled yet, refuses with a hint to unset
    first so the new person gets onboarded.

The trust anchor is face recognition: the speaker must be visually
identified as the current primary at ``IDENTITY_ACTION_MIN_CONFIDENCE``
or higher. No passphrase — by design (we discussed this; the
shoulder-surfing risk and the recovery problem outweighed the
marginal security gain over face recognition alone).

This controller is dependency-injected the same way as
``OnboardingController`` so tests can drive it with fakes.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from onboarding import FaceObserver, ProfileStore, TTSChannel


log = logging.getLogger(__name__)


# Minimum identity_confidence required on the speaker's face for the
# verification gate to pass. Override at runtime via env. Defaults to
# 0.7 — strict enough that a low-confidence (e.g. partially-occluded)
# recognition can't authorize transfer/unset. The onboarding ticker
# and training-capture use lower thresholds because their failure
# mode (don't store a sample) is benign; this knob is for privileged
# actions where false-positive auth is the bad case.
def _load_min_confidence() -> float:
    raw = os.environ.get("IDENTITY_ACTION_MIN_CONFIDENCE", "0.7")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.7


IDENTITY_ACTION_MIN_CONFIDENCE: float = _load_min_confidence()


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------
#
# Deterministic regex matching. The phrasing set is narrow on purpose —
# these actions are privileged and we want low false-positive risk.
# A user who *almost* says "unset primary user" but phrases it weirdly
# gets routed through normal chat instead of accidentally clearing the
# primary designation.

_UNSET_PATTERNS = [
    re.compile(r"\bunset\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(r"\bremove\s+(?:me|the)\s+(?:as\s+)?(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(r"\bclear\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(r"\bi'?m\s+no\s+longer\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(r"\bi\s+am\s+no\s+longer\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(r"\bi'?m\s+not\s+(?:the\s+)?primary(?:\s+user)?\s+anymore\b", re.I),
    re.compile(r"\bstep\s+down\s+as\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
]


# Transfer patterns each capture the target name in group 1. Name
# tokens follow the same shape as introduction names (proper-cased
# alpha, with apostrophe / hyphen allowed).
_NAME = r"([A-Z][A-Za-z'\-]{1,63}|[a-z][a-z'\-]{1,63})"

_TRANSFER_PATTERNS = [
    re.compile(rf"\btransfer\s+(?:the\s+)?primary(?:\s+user)?\s+to\s+{_NAME}\b", re.I),
    re.compile(rf"\bmake\s+{_NAME}\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(rf"\bset\s+{_NAME}\s+as\s+(?:the\s+)?primary(?:\s+user)?\b", re.I),
    re.compile(rf"\b{_NAME}\s+is\s+(?:the\s+)?primary(?:\s+user)?\s+now\b", re.I),
]


def classify_intent(user_input: str) -> Optional[dict]:
    """Return ``{"type": "unset"}`` or ``{"type": "transfer", "target": <name>}``
    if the input matches a privileged identity-action phrase, else None.

    Order: unset is matched first so an utterance like "remove me as
    primary" is never accidentally parsed as a transfer to "me"."""
    if not user_input:
        return None
    text = user_input.strip()
    for pat in _UNSET_PATTERNS:
        if pat.search(text):
            return {"type": "unset"}
    for pat in _TRANSFER_PATTERNS:
        m = pat.search(text)
        if m:
            target = m.group(1).strip(".,!?")
            if target:
                return {"type": "transfer", "target": target}
    return None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class IdentityActionsController:
    """Handle privileged identity-control verbal intents. The runtime
    calls :py:meth:`try_handle` early in dispatch; the controller
    matches the intent, gates on face verification, and executes."""

    def __init__(
        self,
        *,
        face_observer: FaceObserver,
        profile_store: ProfileStore,
        tts: TTSChannel,
        min_confidence: float = IDENTITY_ACTION_MIN_CONFIDENCE,
        on_unset=None,
    ) -> None:
        self.face_observer = face_observer
        self.profile_store = profile_store
        self.tts = tts
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        # Optional callback fired after an unset (or transfer that
        # cleared the previous primary). Used by VaultRuntime to refresh
        # ``_awaiting_primary_user`` immediately.
        self._on_unset = on_unset

    # ------------------------------------------------------------------

    def try_handle(self, user_input: str, node_id: str) -> Optional[dict]:
        """Returns a response dict if ``user_input`` matched a privileged
        identity-action intent and was processed (regardless of whether
        the action succeeded or got refused); returns None if the input
        wasn't an identity-action intent at all, so the runtime falls
        through to normal routing."""
        intent = classify_intent(user_input)
        if intent is None:
            return None

        verification = self._verify_speaker(node_id)
        if not verification["ok"]:
            return self._refuse(node_id, intent, verification["reason"])

        if intent["type"] == "unset":
            return self._do_unset(node_id)
        if intent["type"] == "transfer":
            return self._do_transfer(node_id, intent["target"])
        return None

    # ------------------------------------------------------------------
    # Verification gate
    # ------------------------------------------------------------------

    def _verify_speaker(self, node_id: str) -> dict:
        """Confirm the speaker is the current primary user. The check
        chain returns the first failure reason it hits so the refuse
        path can render a useful message."""
        current_primary = self.profile_store.primary_user()
        if not current_primary:
            return {"ok": False, "reason": "no_primary_set"}
        face = self.face_observer.current_face(node_id)
        if face is None:
            return {"ok": False, "reason": "no_face_visible"}
        if face.identity != current_primary:
            return {"ok": False, "reason": "not_primary",
                    "seen_identity": face.identity}
        if (face.confidence or 0.0) < self.min_confidence:
            return {"ok": False, "reason": "low_confidence",
                    "confidence": face.confidence}
        return {"ok": True}

    def _refuse(self, node_id: str, intent: dict, reason: str) -> dict:
        message = {
            "no_primary_set": "There's no primary user set right now — nothing to change.",
            "no_face_visible": "Look at the camera and say that again — I need to see you to confirm.",
            "not_primary": "Only the primary user can change that.",
            "low_confidence": "I can't quite tell it's you — face the camera straight on and say that again.",
        }.get(reason, "I can't do that right now.")
        self.tts.say(node_id, message)
        return {
            "mode": "direct",
            "message": message,
            "identity_action": {
                "event": "refused",
                "intent": intent["type"],
                "reason": reason,
            },
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_unset(self, node_id: str) -> dict:
        previous = self.profile_store.primary_user()
        self.profile_store.unset_primary_user()
        if self._on_unset is not None:
            try:
                self._on_unset()
            except Exception as exc:
                log.warning("identity_actions on_unset callback failed: %s", exc)
        # Stay silent on unset per the spec (broadcasting "primary user
        # has been cleared" announces device state to the room).
        return {
            "mode": "direct",
            "message": "",
            "identity_action": {
                "event": "unset",
                "previous_primary": previous,
            },
        }

    def _do_transfer(self, node_id: str, target: str) -> dict:
        current = self.profile_store.primary_user()
        target_norm = _slug_identity(target)

        # Transfer-to-self is a no-op; acknowledge briefly.
        if current and current.lower() == target_norm:
            message = f"{target} is already the primary user."
            self.tts.say(node_id, message)
            return {
                "mode": "direct",
                "message": message,
                "identity_action": {
                    "event": "transfer_noop",
                    "target": target_norm,
                },
            }

        if not self.profile_store.has_profile(target_norm):
            # Unenrolled target — refuse with a hint. We don't auto-
            # initiate onboarding because the old primary is still
            # primary; the user has to explicitly unset first to
            # avoid an ambiguous transitional state where two people
            # think they're the active primary.
            message = (
                f"I don't recognize {target} yet. "
                f"Say 'unset primary user' and I'll onboard them when I see them."
            )
            self.tts.say(node_id, message)
            return {
                "mode": "direct",
                "message": message,
                "identity_action": {
                    "event": "transfer_unenrolled",
                    "target": target_norm,
                },
            }

        self.profile_store.designate_primary_user(target_norm)
        # Don't fire on_unset here — primary is still set (to the new
        # target). The runtime's existing refresh logic on completion
        # will pick up the new value.
        message = f"Done — {target} is now the primary user."
        self.tts.say(node_id, message)
        return {
            "mode": "direct",
            "message": message,
            "identity_action": {
                "event": "transferred",
                "previous_primary": current,
                "target": target_norm,
            },
        }


def _slug_identity(name: str) -> str:
    """Same normalization as onboarding._slug_identity. Duplicated here
    rather than imported to keep the module dependency surface minimal —
    onboarding imports nothing from identity_actions, and vice versa."""
    if not name:
        return "unknown"
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unknown"
