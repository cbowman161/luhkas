"""Concrete adapters that satisfy the onboarding Protocols.

These are **vault-owned** adapters — the data they read and write
(profile.json, identity_profile, known faces) is canonical to the
vault, and the camera_nodes are clients that query it. The transport
used to reach the per-node camera / audio HTTP services happens to be
``ScoutVaultBridge``, but the bridge is just the HTTP layer; nothing
about these adapters is scout-specific.

The state machine in ``onboarding.py`` is pure logic against
dependency-injected Protocols so it stays testable without hardware.
The adapters here are the runtime bridge: they speak HTTP to the
camera node for face state / training, HTTP to the audio node for TTS
playback, and JSON to the on-disk profile/identity stores.

Wiring (constructor injection) happens in ``vault_runtime.VaultRuntime``
during ``__init__``; tests don't touch this file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import requests

from onboarding import FaceSnapshot

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Face observation — vault-side adapter that reads camera_node state
# ---------------------------------------------------------------------------


class VaultFaceObserver:
    """``FaceObserver`` against a remote camera_node, owned by vault.

    Reads happen via the vault's HTTP bridge (currently named
    ``ScoutVaultBridge`` — that's a transport class, not a scout-
    specific concept; any camera_node speaks the same JSON over the
    same endpoints). Cheap on the camera side (meta is already
    aggregated for the UI), and the onboarding ticker calls these
    methods ~2× per second per active session — not a tight poll.

    ``confirm_pose`` is **stateful**: it tracks consecutive matching
    reads per ``(node_id, pose)`` and only returns True once the pose
    has been observed ``hold_required`` times in a row. This filters
    transient mid-rotation poses — a user turning their head all the
    way past "left" briefly registers as "left" but doesn't dwell.
    Resets to zero on any non-matching read, so the user has to
    actually hold the pose. Captures also reset the counter so the
    next pose starts from a clean slate.

    ``capture_reference`` hands off to the bridge's ``learn_face``,
    which POSTs to the camera_node's ``/learn_face`` endpoint. The
    pose-bucket the captured frame lands in is determined by the
    camera_node from the bbox at capture time, so we just have to time
    the call so it fires while the requested pose is held.
    """

    # Number of consecutive matching reads required before
    # confirm_pose returns True. At a 500ms tick interval that means
    # the user must hold the pose for ~1s — enough to filter transient
    # poses, short enough that the user doesn't feel they're being
    # ignored. Tunable per deployment if the camera frame rate or tick
    # cadence changes.
    DEFAULT_HOLD_REQUIRED = 3

    def __init__(self, bridge, *, hold_required: int = DEFAULT_HOLD_REQUIRED):
        # ``bridge`` is the vault's HTTP transport to camera/audio nodes
        # (today: ScoutVaultBridge; the naming is historical). All
        # methods called on it are camera-generic; none assume scout-
        # specific behavior.
        self.bridge = bridge
        self.hold_required = max(1, int(hold_required))
        # Per-(node_id, pose) running count of consecutive matching reads.
        # Threading: confirm_pose is called from the runtime's onboarding
        # ticker (single thread per node) so contention is per-node and
        # serialized; a plain dict is fine.
        self._hold_counts: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------

    def current_face(self, node_id: str) -> Optional[FaceSnapshot]:
        face = self._highest_confidence_face(node_id)
        if face is None:
            return None
        return FaceSnapshot(
            identity=str(face.get("identity") or "unknown"),
            confidence=float(face.get("identity_confidence") or 0.0),
            pose=str(face.get("reference_pose") or "unknown"),
            face_group_id=face.get("face_group_id") or face.get("vault_face_group_id"),
        )

    def confirm_pose(self, node_id: str, pose: str, identity_hint: str | None) -> bool:
        key = (node_id, pose)
        face = self._highest_confidence_face(node_id)
        if face is None:
            self._hold_counts.pop(key, None)
            return False
        current_pose = face.get("reference_pose")
        if current_pose != pose:
            self._hold_counts.pop(key, None)
            return False
        # Identity gating: must be the same face_group_id (or a
        # recognized identity match) we're onboarding. Pre-enrollment
        # the identity is "unknown" and we use face_group_id as the
        # stable subject reference.
        if identity_hint:
            ident = face.get("identity")
            fg = face.get("face_group_id") or face.get("vault_face_group_id")
            same_subject = (
                (ident and ident == identity_hint)
                or (fg and fg == identity_hint)
                or (not ident or str(ident) == "unknown")
            )
            if not same_subject:
                self._hold_counts.pop(key, None)
                return False
        # Pose matches and subject matches — count this read.
        count = self._hold_counts.get(key, 0) + 1
        self._hold_counts[key] = count
        if count >= self.hold_required:
            # Confirmed. Reset so re-entry to this pose (e.g. after
            # a failed capture retry) requires another full hold.
            self._hold_counts.pop(key, None)
            return True
        return False

    def capture_reference(self, node_id: str, identity: str, pose: str) -> bool:
        face = self._highest_confidence_face(node_id)
        if face is None:
            return False
        # learn_face takes a face_id (the tracked detection id) and a
        # name. The camera node enrolls the current frame as a sample
        # for ``identity``; the pose label is derived from the bbox.
        face_id = face.get("id") or face.get("face_group_id")
        if face_id is None:
            return False
        try:
            result = self.bridge.learn_face(identity)
        except Exception as exc:
            log.warning("VaultFaceObserver.capture_reference failed: %s", exc)
            return False
        ok = bool(isinstance(result, dict) and result.get("ok"))
        # On either success or failure, clear the hold counter for this
        # pose so a retry restarts the dwell requirement.
        self._hold_counts.pop((node_id, pose), None)
        return ok

    # ------------------------------------------------------------------

    def _highest_confidence_face(self, node_id: str) -> dict | None:
        """Pick the most confident face detection from camera meta, or
        the largest if none have identity_confidence (e.g. unknown
        faces pre-training). Returns None if no face is visible."""
        try:
            meta = self.bridge.camera_state_for_node(node_id)
        except Exception as exc:
            log.warning("camera_state_for_node(%s) failed: %s", node_id, exc)
            return None
        if not isinstance(meta, dict):
            return None
        faces = [
            det for det in meta.get("detections", [])
            if isinstance(det, dict) and det.get("label") == "face"
        ]
        if not faces:
            return None
        # Confident match first; fall back to largest bbox.
        def _score(det):
            conf = det.get("identity_confidence")
            if conf is not None:
                return (1, float(conf))
            bbox = det.get("bbox") or [0, 0, 0, 0]
            try:
                return (0, float(bbox[2]) * float(bbox[3]))
            except (TypeError, IndexError):
                return (0, 0.0)
        return max(faces, key=_score)


# ---------------------------------------------------------------------------
# TTS — POST to audio_node /tts
# ---------------------------------------------------------------------------


class NodeRegistryTTS:
    """``TTSChannel`` that routes spoken text to the audio service on
    ``node_id``. Falls back silently (logs only) if the node has no
    audio service registered — onboarding still returns the text in the
    response dict so the user-visible transcript carries it even when
    speakers aren't reachable."""

    AUDIO_SERVICE_KEY = "audio"  # see node/profile_loader.py: audio_node → "audio"
    TIMEOUT_SECONDS = 3.0

    def __init__(self, node_registry):
        self.node_registry = node_registry

    def say(self, node_id: str, text: str) -> None:
        if not text:
            return
        url = self._tts_url(node_id)
        if url is None:
            log.info("no audio service for %s; skipping TTS (text=%r)", node_id, text[:80])
            return
        try:
            requests.post(url, json={"text": text}, timeout=self.TIMEOUT_SECONDS)
        except Exception as exc:
            log.warning("TTS POST to %s failed: %s", url, exc)

    def _tts_url(self, node_id: str) -> str | None:
        if self.node_registry is None:
            return None
        try:
            base = self.node_registry.node_url(node_id, self.AUDIO_SERVICE_KEY)
        except Exception:
            return None
        if not base:
            return None
        return f"{base.rstrip('/')}/tts"


# ---------------------------------------------------------------------------
# Profile store — vault-owned, the canonical record of known users.
# Each camera_node consults this store (via the bridge) to translate
# face-recognition identity strings into display names, preferences,
# and primary-user status.
# ---------------------------------------------------------------------------


class VaultProfileStore:
    """``ProfileStore`` over the vault's canonical user data:

      - ``primary_user`` lives in ``identity_profile.json`` (the JSON
        the runtime reads in ``dispatch_guard_alert``).
      - ``display_name`` and arbitrary preferences live in the
        per-identity profiles under ``data/people/{identity}/profile.json``,
        written through ``remember(identity, type, key, value)``.

    Every camera_node shares this store (via the vault HTTP bridge), so
    enrollment at one node is immediately visible to recognition at
    another — there's no per-node identity drift to reconcile.

    Failures are logged but don't raise; onboarding's UX should
    degrade gracefully (it still verbally welcomed the user) rather
    than crash mid-conversation.
    """

    def __init__(self, bridge):
        self.bridge = bridge

    def set_display_name(self, identity: str, display_name: str) -> None:
        try:
            self.bridge.remember(
                identity, "fact", "display_name", display_name,
                source="onboarding", confidence=1.0,
            )
        except Exception as exc:
            log.warning("set_display_name(%s) failed: %s", identity, exc)

    def set_preference(self, identity: str, key: str, value: Any) -> None:
        try:
            self.bridge.remember(
                identity, "preference", key, value,
                source="onboarding", confidence=1.0,
            )
        except Exception as exc:
            log.warning("set_preference(%s, %s) failed: %s", identity, key, exc)

    def designate_primary_user(self, identity: str) -> None:
        try:
            self.bridge.update_identity_profile({"primary_user": identity})
        except Exception as exc:
            log.warning("designate_primary_user(%s) failed: %s", identity, exc)

    def unset_primary_user(self) -> None:
        """Clear the primary_user designation. The runtime treats a
        null/empty value identically to the key being absent — see
        VaultRuntime._check_awaiting_primary_user."""
        try:
            self.bridge.update_identity_profile({"primary_user": None})
        except Exception as exc:
            log.warning("unset_primary_user failed: %s", exc)

    def primary_user(self) -> str | None:
        try:
            profile = self.bridge.identity_profile or {}
        except Exception:
            return None
        primary = profile.get("primary_user")
        if primary is None:
            return None
        primary = str(primary).strip()
        return primary or None

    def has_profile(self, identity: str) -> bool:
        """True iff the vault person-memory store has an existing
        profile under ``identity``. Used by the transfer-primary handler
        to distinguish enrolled targets (immediate flip) from
        unenrolled ones (refuse with onboarding hint)."""
        if not identity:
            return False
        try:
            result = self.bridge.get_profile(identity)
        except Exception as exc:
            log.warning("has_profile(%s) failed: %s", identity, exc)
            return False
        if not isinstance(result, dict) or not result.get("ok"):
            return False
        profile = result.get("profile") or {}
        # An empty/never-touched profile dict counts as "no profile."
        # The remember() path always sets updated_at when a fact lands,
        # so its presence is a reliable signal that the identity has
        # been written to before.
        return bool(profile.get("updated_at") or profile.get("display_name") or profile.get("facts"))
