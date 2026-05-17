import json
import re
from typing import Any, Dict

from .models import LocalModel
from .config import PLANNER_MODEL
from .schemas import WorkOrder
from .validator import parse_json_contract, validate_work_order


REQUIRED_FILES = [
    {'path': 'src/api.py', 'purpose': 'API implementation'},
    {'path': 'artifacts/README.md', 'purpose': 'usage contract and test coverage guide'},
    {'path': 'tests/test_api.py', 'purpose': 'non-interactive verification tests'},
    {'path': 'commands.json', 'purpose': 'command triggers and args for the command agent'},
]

BACKGROUND_KEYWORDS = {
    'notification', 'notify', 'reminder', 'remind', 'schedule', 'scheduled',
    'timer', 'periodic', 'interval', 'watch', 'monitor', 'alert', 'due',
    'background', 'daemon', 'recurring', 'cron', 'polling', 'poll',
}


def needs_background_service(goal: str) -> bool:
    goal_lower = (goal or '').lower()
    return any(kw in goal_lower for kw in BACKGROUND_KEYWORDS)

SUCCESS_CRITERIA = [
    'test command exits with code 0',
    'test suite exercises public API endpoints and JSON responses',
    'implementation satisfies the user goal',
]


def safe_capability_name(goal: str) -> str:
    name = re.sub(r'[^a-z0-9]+', '_', (goal or '').lower()).strip('_')
    name = re.sub(r'_+', '_', name)[:50].strip('_')
    return name or 'generated_capability'


def deterministic_work_order(goal: str, notes: str = '') -> WorkOrder:
    """Return the canonical API-first package work order.

    Step 34 makes planning idempotent.  The package layout is a project
    invariant, so it should not depend on fragile LLM JSON formatting.  The LLM
    may enrich notes, but parse failures no longer create a separate "fallback"
    work order or poison later diagnostics with planner-failure wording.
    """
    files = list(REQUIRED_FILES)
    if needs_background_service(goal):
        files.append({
            'path': 'src/background.py',
            'purpose': 'background service that runs continuously to handle scheduled/periodic tasks',
        })
    return WorkOrder(
        goal=goal,
        capability_name=safe_capability_name(goal),
        entrypoint='src/api.py',
        files=files,
        test_command='python3 -m unittest discover -s tests',
        self_test_command='',
        success_criteria=list(SUCCESS_CRITERIA),
        notes=notes or 'Deterministic API-first work order',
    )


# Backwards-compatible alias for older imports/tests.  This no longer means
# "planner failed"; it returns the same canonical work order.
def fallback_work_order(goal: str, reason: str = '') -> WorkOrder:
    note = 'Deterministic API-first work order'
    if reason:
        note += '; planner model output was ignored/recovered: {}'.format(str(reason)[:300])
    return deterministic_work_order(goal, note)


class Planner:
    def __init__(self, model: LocalModel | None = None):
        self.model = model or LocalModel(model=PLANNER_MODEL)
        self.session_appender = None
        self.session_writer = None

    def _session_set(self, component: str, key: str, value: Any) -> None:
        if self.session_writer:
            try:
                self.session_writer(component, key, value)
            except Exception:
                pass

    def _session_append(self, component: str, key: str, value: Any) -> None:
        if self.session_appender:
            try:
                self.session_appender(component, key, value)
            except Exception:
                pass

    def create_work_order(self, goal: str, environment: Dict[str, Any]) -> WorkOrder:
        """Create a stable work order without relying on model JSON validity.

        The API-first package shape is fixed by architecture, so the planner's
        critical output must be deterministic and blackboard-stable.  We still
        ask the model for optional implementation notes, but any malformed JSON
        is treated as advisory noise and never changes the canonical file list.
        """
        base = deterministic_work_order(goal)
        raw = ''
        try:
            prompt = self._prompt(goal, environment)
            record = {
                'component': 'planner',
                'model': getattr(self.model, 'model', None),
                'path': 'work_order',
                'purpose': 'planning',
                'context': 'create_work_order',
                'attempt': 1,
                'prompt': prompt,
            }
            self._session_append('llm', 'prompts', record)
            self._session_set('llm', 'last_prompt', record)
            raw = self.model.generate(prompt)
            parsed = self._parse_planner_output(raw)
            if parsed:
                candidate = self._merge_with_canonical(goal, parsed)
                return validate_work_order(candidate)
        except Exception:
            # Planning must be reliable.  Generation/validation can still catch
            # real problems later, but malformed planner text should not force a
            # noisy fallback path.
            pass
        return base

    def _prompt(self, goal: str, environment: Dict[str, Any]) -> str:
        canonical = deterministic_work_order(goal).to_dict()
        return f'''
You are code_monkey, a local capability build planner.

Return ONLY valid JSON. No markdown. No prose. No comments.
All strings must be valid JSON strings.

Create implementation notes for this API-first work order.
Do not change entrypoint, files, or test_command.

Goal:
{goal}

Environment snapshot:
{environment}

Canonical work order JSON to preserve:
{json.dumps(canonical, indent=2)}

Return JSON with the same schema. You may only improve capability_name and notes.
'''.strip()

    def _parse_planner_output(self, raw: str) -> Dict[str, Any] | None:
        if not raw or not str(raw).strip():
            return None
        try:
            return parse_json_contract(raw, label='planner')
        except Exception:
            pass
        return self._best_effort_json(raw)

    def _best_effort_json(self, raw: str) -> Dict[str, Any] | None:
        """Recover a JSON object from common local-model planner drift."""
        text = str(raw).strip()
        if '```' in text:
            parts = text.split('```')
            fenced = []
            for i in range(1, len(parts), 2):
                block = parts[i].strip()
                lines = block.splitlines()
                if lines and lines[0].strip().lower() in {'json', 'javascript'}:
                    lines = lines[1:]
                fenced.append('\n'.join(lines))
            for block in fenced:
                parsed = self._decode_any_object(block)
                if parsed:
                    return parsed
        parsed = self._decode_any_object(text)
        if parsed:
            return parsed
        return None

    def _decode_any_object(self, text: str) -> Dict[str, Any] | None:
        decoder = json.JSONDecoder()
        starts = [m.start() for m in re.finditer(r'\{', text or '')]
        # Prefer later objects first because models often echo the schema before
        # returning their answer.
        for start in reversed(starts):
            try:
                obj, _ = decoder.raw_decode(text[start:])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _merge_with_canonical(self, goal: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        base = deterministic_work_order(goal).to_dict()
        name = parsed.get('capability_name')
        if isinstance(name, str) and name.strip():
            base['capability_name'] = safe_capability_name(name)
        notes = parsed.get('notes')
        if isinstance(notes, str) and notes.strip():
            base['notes'] = notes.strip()[:1000]
        else:
            base['notes'] = 'Deterministic API-first work order'
        # Keep the canonical path-keyed file set.  This prevents planner/model
        # drift from introducing duplicate or missing file records downstream.
        base['entrypoint'] = 'src/api.py'
        base['files'] = list(REQUIRED_FILES)
        base['test_command'] = 'python3 -m unittest discover -s tests'
        base['self_test_command'] = ''
        base['success_criteria'] = list(SUCCESS_CRITERIA)
        base['goal'] = goal
        return base
