from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple


FailureLesson = Dict[str, str]


def normalize_failure_signature(text: str) -> str:
    text = str(text or '').strip().lower()
    text = re.sub(r'/[^\s\"]+', '<path>', text)
    text = re.sub(r'line \d+', 'line N', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '<date>', text)
    text = re.sub(r'\b\d+\b', 'N', text)
    text = text.replace('generated python syntax error in ', 'syntax error in ')
    text = text.split('\n')[0]
    return text[:500] or 'unknown_failure'


def failure_text(report: Dict[str, Any]) -> str:
    if not report:
        return ''
    parts: List[str] = []
    if isinstance(report, dict):
        for key in ('stderr', 'stdout', 'error'):
            value = report.get(key)
            if value:
                parts.append(str(value))
        for failure in report.get('failures') or []:
            if isinstance(failure, dict):
                for key in ('stderr', 'stdout', 'error'):
                    value = failure.get(key)
                    if value:
                        parts.append(str(value))
        for result in report.get('results') or []:
            if isinstance(result, dict):
                for key in ('stderr', 'stdout', 'error'):
                    value = result.get(key)
                    if value:
                        parts.append(str(value))
    return '\n'.join(parts)


def extract_runtime_lessons(report: Dict[str, Any]) -> List[FailureLesson]:
    """Convert runtime/test failures into reusable scoped rules.

    This is deterministic on purpose. The LLM may still be used to repair, but
    the system owns the repeatable lesson extraction for common Python failures.
    """
    text = failure_text(report)
    lessons: List[FailureLesson] = []

    def add(scope: str, pattern: str, rule: str, example: str = '') -> None:
        lessons.append({
            'scope': scope or 'global',
            'failure_signature': pattern or normalize_failure_signature(rule),
            'lesson': rule.strip(),
            'example_error': (example or text)[:4000],
        })

    # NameError: name 'datetime' is not defined
    for match in re.finditer(r"NameError:\s+name ['\"]([^'\"]+)['\"] is not defined", text):
        name = match.group(1)
        scope = _scope_near(text, match.start())
        if name == 'datetime':
            add(
                scope,
                'missing_import_datetime',
                "If a file uses datetime.date or datetime.datetime, it must include 'import datetime' before use.",
            )
        elif name == 'os':
            add(scope, 'missing_import_os', "If a file uses os.*, it must include 'import os' before use.")
        elif name == 'tempfile':
            if scope == 'tests/test_api.py':
                add(scope, 'tests_do_not_use_tempfile_storage', "Tests must not use tempfile for normal capability storage. Run native DATA_DIR behavior and clean up by calling the capability delete/back-out API.")
            else:
                add(scope, 'avoid_tempfile_for_persistent_storage', "Persistent code should not use tempfile for normal runtime data. Use DATA_DIR derived from Path(__file__).resolve().parent / 'data'.")
        elif name == 'shutil':
            add(scope, 'missing_import_shutil', "If tests or code use shutil.*, include 'import shutil' before use.")
        else:
            add(scope, 'missing_name_' + _safe_key(name), "If a file uses name '{}', it must import or define it before use.".format(name))

    if 'FileExistsError' in text and 'os.makedirs' in text:
        add(
            'global',
            'makedirs_existing_directory',
            "When creating directories with os.makedirs, use exist_ok=True or check existence first so repeated tests do not fail.",
        )

    if 'FileNotFoundError' in text and ('open(' in text or 'os.listdir' in text or 'No such file or directory' in text):
        add(
            'global',
            'missing_directory_before_file_io',
            "Before writing, reading, or listing files in a directory, ensure the directory exists with os.makedirs(path, exist_ok=True) or an equivalent existence check.",
        )

    if 'AssertionError' in text and 'not found in' in text:
        add(
            'global',
            'return_values_must_match_tests',
            "Implementation return values and test expectations must match exactly. Prefer simple stable return strings/lists over extra formatting text.",
        )

    if 'AssertionError' in text and ('False is not true' in text or 'unexpectedly found' in text):
        add(
            'global',
            'native_storage_create_delete_consistency',
            "Create/list/delete must use the same native DATA_DIR path derived from Path(__file__).resolve().parent / 'data'. Delete must remove both persistent file records and any in-memory index entries.",
        )

    if 'FileNotFoundError' in text and ('/tmp' in text or 'reminders' in text):
        add(
            'global',
            'native_storage_no_tmp_or_cwd',
            "Persistent files must not use /tmp, cwd, or bare relative directories. Use SKILL_ROOT = Path(__file__).resolve().parent and DATA_DIR = SKILL_ROOT / 'data'.",
        )

    if 'ResourceWarning' in text and 'TemporaryDirectory' in text:
        add(
            'tests/test_api.py',
            'tests_do_not_use_tempfile_storage',
            "Tests must not use tempfile for normal capability storage. They must run native DATA_DIR behavior and clean up by calling the capability delete/back-out API.",
        )


    if 'NotADirectoryError' in text or 'not a directory' in text.lower():
        add(
            'src/api.py',
            'delete_file_vs_directory_correctly',
            "Do not use shutil.rmtree() on files. Delete file records with os.remove(path) or Path.unlink(). Use shutil.rmtree() only for directories after checking os.path.isdir(path) or Path(path).is_dir().",
        )

    if 'ResourceWarning' in text and 'unclosed file' in text:
        add(
            'global',
            'close_all_opened_files',
            "All opened files must be closed. Do not use inline open(path).read(). Use with open(...) as handle, or explicit open plus try/finally handle.close().",
        )

    if 'AssertionError' in text and ('True is not false' in text or 'False is not true' in text) and 'os.path.exists' in text:
        add(
            'src/api.py',
            'delete_api_must_remove_created_file',
            "The delete/back-out API must remove the exact same file path created by the add/create API, and update any in-memory state so existence checks pass after deletion.",
        )

    if 'unexpectedly found' in text and ('entries' in text or '.txt' in text):
        add(
            'src/api.py',
            'delete_updates_memory_and_storage',
            "Delete/remove/back-out must update both persistent storage and in-memory indexes/lists/dicts for the deleted record.",
        )

    if 'ModuleNotFoundError' in text:
        module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
        module = module_match.group(1) if module_match else 'unknown'
        add(
            'global',
            'missing_dependency_' + _safe_key(module),
            "If module '{}' is required, declare it as a dependency before testing instead of assuming it is installed.".format(module),
        )

    if 'ImportError' in text and 'src.api' in text:
        add(
            'tests/test_api.py',
            'test_import_src_main_from_workspace',
            "Tests run from the task workspace root and should import implementation with 'from src.api import ...'. Ensure src is importable and names exist.",
        )

    if not lessons and text.strip():
        add(
            'global',
            normalize_failure_signature(text),
            'Avoid repeating this runtime/test failure pattern: ' + text.strip().split('\n')[0][:500],
        )

    return _dedupe_lessons(lessons)


def extract_static_lessons(path: str, error: str) -> List[FailureLesson]:
    low = str(error or '').lower()
    scope = path or 'global'
    lessons: List[FailureLesson] = []

    def add(pattern: str, rule: str) -> None:
        lessons.append({
            'scope': scope,
            'failure_signature': pattern,
            'lesson': rule,
            'example_error': str(error or '')[:4000],
        })

    if 'syntax error' in low or 'expected' in low:
        add('static_syntax_structural_statement', 'Keep each Python statement syntactically complete. Avoid truncated calls, incomplete expressions, and copied terminal-control fragments.')
    if 'placeholder' in low or 'stub' in low or 'pass-only' in low:
        add('placeholder_or_stub_rejected', 'Do not use placeholders, TODOs, pass-only stubs, or fake tests. Implement real behavior before validation.')
    if 'cleanup' in low or 'delete' in low or 'remove' in low:
        add('missing_cleanup_contract', 'Every capability must include and test a delete/remove/cleanup/back-out path for records created during tests.')
    return _dedupe_lessons(lessons)


def enforce_lessons_on_code(path: str, content: str, lessons: Iterable[Dict[str, Any]]) -> List[str]:
    """Return deterministic lesson violations for this file content."""
    text = content or ''
    low = text.lower()
    violations: List[str] = []
    lesson_text = '\n'.join(str(l.get('lesson') or '') for l in lessons).lower()

    def has_import(module: str) -> bool:
        return re.search(r'(^|\n)\s*import\s+' + re.escape(module) + r'(\s|,|$)', text) is not None or re.search(r'(^|\n)\s*from\s+' + re.escape(module) + r'\s+import\s+', text) is not None

    if 'datetime' in lesson_text or 'datetime.' in text:
        if 'datetime.' in text and not has_import('datetime'):
            violations.append("Lesson violation: code uses datetime.* but does not include 'import datetime'.")

    if 'os.' in text and not has_import('os'):
        violations.append("Lesson violation: code uses os.* but does not include 'import os'.")

    if 'shutil.' in text and not has_import('shutil'):
        violations.append("Lesson violation: code uses shutil.* but does not include 'import shutil'.")

    if 'os.makedirs(' in text and 'exist_ok=true' not in low:
        # Accept an explicit existence guard near makedirs.
        if 'if not os.path.exists' not in low and 'if not os.path.isdir' not in low:
            violations.append("Lesson violation: os.makedirs should use exist_ok=True or an explicit existence check.")

    if path == 'src/api.py':
        if 'open(' in text or 'os.listdir' in text or 'os.remove' in text or 'json.dump' in text or 'json.load' in text:
            if '__file__' not in text or 'DATA_DIR' not in text:
                violations.append("Lesson violation: persistent file code must derive DATA_DIR from Path(__file__).resolve().parent / 'data'.")
        if 'os.getcwd(' in text or '/tmp' in text or 'tempfile.' in text:
            violations.append('Lesson violation: persistent runtime paths must not use cwd, /tmp, or tempfile.')


    if path == 'src/api.py':
        if 'shutil.rmtree' in text:
            has_dir_guard = 'os.path.isdir' in text or '.is_dir()' in text
            if not has_dir_guard:
                violations.append('Lesson violation: shutil.rmtree() requires an is-directory guard; use os.remove()/Path.unlink() for files.')
            if any(marker in text for marker in ['.txt', '.json', '.db', '.sqlite']) and 'os.remove(' not in text and '.unlink(' not in text:
                violations.append('Lesson violation: file-like records require os.remove()/Path.unlink() deletion, not only shutil.rmtree().')
        if 'open(' in text and ').read(' in text.replace(' ', ''):
            violations.append('Lesson violation: do not use inline open(...).read(); use with-open or close the handle explicitly.')
        if 'delete' in low or 'remove' in low or 'cleanup' in low:
            if any(marker in text for marker in ['.txt', '.json', '.db', '.sqlite']) and 'os.remove(' not in text and '.unlink(' not in text and 'json.dump' not in text:
                violations.append('Lesson violation: delete/back-out must remove the file or persistent record created by add/create.')

    if path == 'tests/test_api.py':
        if 'delete' not in low and 'remove' not in low and 'cleanup' not in low:
            violations.append('Lesson violation: tests must exercise the delete/remove/cleanup path.')
        if 'tempfile' in low:
            violations.append('Lesson violation: tests must not use tempfile for normal storage; run native DATA_DIR behavior.')
        if 'os.remove(' in text or 'shutil.rmtree' in text or '.unlink(' in text:
            violations.append('Lesson violation: tests must not manually delete capability records; call the delete/back-out API.')
        if 'storage_path' in low or 'storage_dir' in low or 'data_dir=' in low or 'tempdir' in low or 'temp_dir' in low:
            violations.append('Lesson violation: tests must not override storage paths; use the native skill storage contract.')

    return violations


def _scope_near(text: str, index: int) -> str:
    before = text[:index]
    # Prefer the last traceback File entry before this error.
    matches = list(re.finditer(r'File "([^"]+)"', before))
    if not matches:
        return 'global'
    path = matches[-1].group(1)
    if '/tests/test_api.py' in path or path.endswith('tests/test_api.py'):
        return 'tests/test_api.py'
    if '/src/api.py' in path or path.endswith('src/api.py'):
        return 'src/api.py'
    return 'global'


def _safe_key(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]+', '_', str(value)).strip('_').lower()[:80] or 'unknown'


def _dedupe_lessons(lessons: List[FailureLesson]) -> List[FailureLesson]:
    seen: set[Tuple[str, str]] = set()
    out: List[FailureLesson] = []
    for lesson in lessons:
        key = (lesson.get('scope', 'global'), lesson.get('failure_signature', 'unknown'))
        if key in seen:
            continue
        seen.add(key)
        out.append(lesson)
    return out
