from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .normalizer import extract_json_contract, normalize_generated_files
from .schemas import BuildFiles, WorkOrder


PLACEHOLDER_MARKERS = [
    'your code here',
    'add your implementation here',
    'add your tests here',
    'please fill in',
    'placeholder',
    'todo:',
    'not implemented',
    'implement me',
    'stub',
]


ASSERTION_MARKERS = [
    'assert ',
    'self.assert',
    'raise ',
    'if ',
    '!=',
    '==',
    ' in ',
    ' not in ',
]


def parse_json_contract(raw: str, label: str = 'model') -> Dict[str, Any]:
    return extract_json_contract(raw, label=label)


def parse_file_envelopes(raw: str) -> BuildFiles:
    # Backwards-compatible name. This now means "normalize generated files".
    return validate_build_files(normalize_generated_files(raw))


def validate_work_order(data: Dict[str, Any]) -> WorkOrder:
    required = [
        'goal',
        'capability_name',
        'entrypoint',
        'files',
        'test_command',
        'self_test_command',
        'success_criteria',
    ]
    for key in required:
        if key not in data:
            raise ValueError('Work order missing required field: {}'.format(key))
    if not isinstance(data['files'], list) or not data['files']:
        raise ValueError('Work order files must be a non-empty list')
    files = []
    for item in data['files']:
        if not isinstance(item, dict) or not item.get('path') or not item.get('purpose'):
            raise ValueError('Each work order file must have path and purpose')
        _validate_relative_task_path(str(item['path']))
        files.append({'path': str(item['path']), 'purpose': str(item['purpose'])})
    if not any(item.get('path') == 'artifacts/README.md' for item in files):
        files.append({'path': 'artifacts/README.md', 'purpose': 'usage contract and test coverage guide'})
    _validate_relative_task_path(str(data['entrypoint']))
    return WorkOrder(
        goal=str(data['goal']),
        capability_name=_safe_name(str(data['capability_name'])),
        entrypoint=str(data['entrypoint']),
        files=files,
        test_command=str(data['test_command']),
        self_test_command=str(data['self_test_command']),
        success_criteria=[str(x) for x in data['success_criteria']],
        notes=str(data.get('notes') or ''),
    )


def validate_build_files(build_files: BuildFiles) -> BuildFiles:
    if not build_files.files:
        raise ValueError('No files generated')
    seen = set()
    paths = {item.path for item in build_files.files}
    if 'src/api.py' not in paths:
        raise ValueError('Generated files must include src/api.py')
    if 'tests/test_api.py' not in paths:
        raise ValueError('Generated files must include tests/test_api.py')
    if 'artifacts/README.md' not in paths:
        raise ValueError('Generated files must include artifacts/README.md')

    parsed_by_path: Dict[str, ast.Module] = {}

    for item in build_files.files:
        _validate_relative_task_path(item.path)
        if item.path in seen:
            raise ValueError('Duplicate generated file: {}'.format(item.path))
        seen.add(item.path)
        content = item.content or ''
        if not content.strip():
            raise ValueError('Generated file is empty: {}'.format(item.path))
        _reject_placeholders(item.path, content)
        if item.path == 'artifacts/README.md':
            _validate_readme_single(content)
        if item.path.endswith('.py'):
            _reject_forbidden_text_before_ast(item.path, content)
            try:
                parsed_by_path[item.path] = ast.parse(content)
            except SyntaxError as exc:
                raise ValueError(
                    'Generated Python syntax error in {}: {}'.format(item.path, exc)
                ) from exc
            _reject_stub_functions(item.path, content, parsed_by_path[item.path])
            _reject_forbidden_python_subset(item.path, content, parsed_by_path[item.path])

    main = next(item.content for item in build_files.files if item.path == 'src/api.py')
    tests = next(item.content for item in build_files.files if item.path == 'tests/test_api.py')
    readme = next(item.content for item in build_files.files if item.path == 'artifacts/README.md')

    _validate_readme_contract(readme, main, tests)
    _validate_test_import_contract(tests)
    _validate_native_storage_contract(main, tests)
    _validate_cleanup_contract(main, tests)
    _validate_filesystem_domain_contract(
        main,
        tests,
        parsed_by_path.get('src/api.py'),
        parsed_by_path.get('tests/test_api.py'),
    )
    _validate_test_file_behavior(tests, parsed_by_path.get('tests/test_api.py'))
    return build_files


def validate_workspace_write(root: Path, relative_path: str) -> Path:
    _validate_relative_task_path(relative_path)
    target = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in target.parents and target != root_resolved:
        raise ValueError('Path escapes workspace: {}'.format(relative_path))
    return target


def _reject_placeholders(path: str, content: str) -> None:
    lowered = content.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        raise ValueError('Generated file contains placeholder content: {}'.format(path))

    meaningful = [
        line.strip() for line in content.splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]
    non_boilerplate = [
        line for line in meaningful
        if line not in {'pass', '...', 'return None'}
    ]
    if len(non_boilerplate) < 8:
        raise ValueError('Generated file appears incomplete or stub-like: {}'.format(path))


def _reject_stub_functions(path: str, content: str, tree: ast.Module) -> None:
    source_lines = content.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [stmt for stmt in node.body if not _is_docstring_stmt(stmt)]
        if not body:
            raise ValueError('Function {} in {} has no body'.format(node.name, path))
        if all(isinstance(stmt, (ast.Pass, ast.Expr)) and _is_ellipsis_expr(stmt) for stmt in body):
            raise ValueError('Function {} in {} is a pass/ellipsis stub'.format(node.name, path))
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            raise ValueError('Function {} in {} is a pass-only stub'.format(node.name, path))
        segment = _source_segment(source_lines, node)
        lowered = segment.lower()
        if node.name.startswith('run_') and 'pass' in lowered and len(body) <= 2:
            raise ValueError('Function {} in {} appears to be a stub'.format(node.name, path))




def _reject_forbidden_text_before_ast(path: str, content: str) -> None:
    """Lightweight text checks before ast.parse.

    Step 11 stops fighting common Python idioms such as f-strings and with-open.
    Those constructs are allowed if Python can parse them and runtime tests pass.
    This check only rejects obviously truncated or impossible-to-handle lines.
    """
    # Long-line rejection was a legacy guard from pre-API model output parsing.
    # Modern model/API responses may contain valid long Python literals, and AST
    # parsing plus runtime tests are the source of truth. Do not reject on length.
    return

def _reject_forbidden_python_subset(path: str, content: str, tree: ast.Module) -> None:
    """No-op compatibility hook.

    Earlier versions rejected common Python constructs. Step 11 allows natural
    Python and relies on normalization, AST parsing, and runtime tests instead.
    """
    return


README_REQUIRED_SECTIONS = [
    "# ",
    "## Purpose",
    "## Usage",
    "## Public API",
    "## Function Definitions",
    "## Imported Dependencies",
    "## Outputs and Return Values",
    "## Failure Modes",
    "## Data Storage",
    "## Cleanup / Delete Behavior",
    "## Test Coverage",
]


def _validate_readme_single(readme_content: str) -> None:
    """Validate README as a strict, machine-checkable spec.

    Earlier versions accepted fuzzy prose such as "return codes" or "tests".
    That caused repair loops because the model could not infer the exact shape
    the validator wanted. Step 19 makes README.md a schema-like markdown file
    with exact section headers.
    """
    text = readme_content or ''
    lowered = text.lower()
    stripped = text.strip()
    if not stripped.startswith('# '):
        raise ValueError('artifacts/README.md must start with a top-level # title')

    missing = []
    for section in README_REQUIRED_SECTIONS[1:]:
        if section not in text:
            missing.append(section)
    if missing:
        raise ValueError(
            'artifacts/README.md missing required section(s): ' + ', '.join(missing)
        )

    for section in README_REQUIRED_SECTIONS[1:]:
        body = _readme_section_body(text, section)
        if not body.strip():
            raise ValueError('artifacts/README.md section is empty: ' + section)
        # Keep README validation structural here. Semantic adequacy is handled
        # by the LLM semantic validator and AST cross-checks. Keyword checks are
        # intentionally avoided because valid prose may omit a preferred word.
        if len(body.strip()) < 10:
            raise ValueError('artifacts/README.md section is too short: ' + section)

    # The README must be useful enough to serve as the tester/analyzer oracle.
    if len([line for line in text.splitlines() if line.strip()]) < 18:
        raise ValueError('artifacts/README.md is too short to be a complete usage contract')

    public_api = _readme_section_body(text, '## Public API').lower()
    outputs = _readme_section_body(text, '## Outputs and Return Values').lower()
    tests = _readme_section_body(text, '## Test Coverage').lower()
    for behavior in ['add', 'list', 'delete']:
        if behavior not in public_api:
            raise ValueError('artifacts/README.md Public API must document {} behavior'.format(behavior))
        if behavior not in outputs:
            raise ValueError('artifacts/README.md Outputs and Return Values must document {} output'.format(behavior))
        if behavior not in tests:
            raise ValueError('artifacts/README.md Test Coverage must state that tests verify {} behavior'.format(behavior))


def _readme_section_body(text: str, section: str) -> str:
    marker = section + '\n'
    idx = text.find(marker)
    if idx < 0:
        # Allow section at EOF or with trailing spaces after header.
        m = re.search(r'^' + re.escape(section) + r'\s*$', text, flags=re.MULTILINE)
        if not m:
            return ''
        start = m.end()
    else:
        start = idx + len(marker)
    next_match = re.search(r'^##\s+', text[start:], flags=re.MULTILINE)
    if next_match:
        return text[start:start + next_match.start()]
    return text[start:]


def _readme_public_api_names(readme_content: str) -> list[str]:
    body = _readme_section_body(readme_content or '', '## Public API')
    names = set()
    for match in re.finditer(r'`([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', body):
        names.add(match.group(1))
    for match in re.finditer(r'\b(add_[a-zA-Z0-9_]+|list_[a-zA-Z0-9_]+|delete_[a-zA-Z0-9_]+|remove_[a-zA-Z0-9_]+|cleanup_[a-zA-Z0-9_]+)\b', body):
        names.add(match.group(1))
    return sorted(names)



def _python_defined_function_names(content: str) -> list[str]:
    try:
        tree = ast.parse(content or '')
    except SyntaxError:
        return []
    names = set()
    class_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if class_stack:
                names.add(class_stack[-1] + '.' + node.name)
            names.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if class_stack:
                names.add(class_stack[-1] + '.' + node.name)
            names.add(node.name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(names)


def _python_imported_dependency_names(content: str) -> list[str]:
    try:
        tree = ast.parse(content or '')
    except SyntaxError:
        return []
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or '').split('.')[0]
                if root:
                    names.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            if node.module:
                root = node.module.split('.')[0]
                if root:
                    names.add(root)
            for alias in node.names:
                if alias.name != '*':
                    names.add(alias.name)
    return sorted(names)


def _validate_readme_documents_source_contract(readme_content: str, main_content: str) -> None:
    function_body = _readme_section_body(readme_content or '', '## Function Definitions')
    import_body = _readme_section_body(readme_content or '', '## Imported Dependencies')
    function_low = function_body.lower()
    import_low = import_body.lower()

    functions = _python_defined_function_names(main_content or '')
    missing_functions = []
    for name in functions:
        # Require either the qualified method name or the bare function name.
        bare = name.split('.')[-1]
        if name.lower() not in function_low and bare.lower() not in function_low:
            missing_functions.append(name)
    if missing_functions:
        raise ValueError(
            'artifacts/README.md Function Definitions missing def(s): ' + ', '.join(missing_functions)
        )

    # Function Definitions should describe relationships, not only names.
    # The LLM semantic validator enforces this per function; this deterministic
    # check makes sure the section contains the relationship fields at all.
    if functions:
        function_low_compact = function_low.replace(' ', '')
        if 'calledby' not in function_low_compact:
            raise ValueError('artifacts/README.md Function Definitions must include Called by for each def')
        if 'calls:' not in function_low:
            raise ValueError('artifacts/README.md Function Definitions must include Calls for each def')

    imports = _python_imported_dependency_names(main_content or '')
    missing_imports = []
    for name in imports:
        if name.lower() not in import_low:
            missing_imports.append(name)
    if missing_imports:
        raise ValueError(
            'artifacts/README.md Imported Dependencies missing import(s): ' + ', '.join(missing_imports)
        )

def _validate_readme_contract(readme_content: str, main_content: str, test_content: str) -> None:
    _validate_readme_single(readme_content)
    _validate_readme_documents_source_contract(readme_content, main_content)
    readme = (readme_content or '').lower()
    main = main_content or ''
    tests = test_content or ''
    main_low = main.lower()
    tests_low = tests.lower()

    # Tests must exercise the behaviors promised in the README. This is a
    # lightweight coverage check; runtime tests remain the source of truth.
    coverage = _readme_section_body(readme_content or '', '## Test Coverage').lower()
    for term in ['add', 'list', 'delete']:
        if term in coverage and term not in tests_low:
            raise ValueError(
                'tests/test_api.py does not appear to test README Test Coverage behavior: ' + term
            )

    schema_public = set(_public_api_function_names_from_ast(main_content or ''))
    readme_public = set(_readme_public_api_names(readme_content))
    extra_readme_public = sorted(readme_public - schema_public)
    if extra_readme_public:
        raise ValueError(
            'artifacts/README.md Public API must only list schema() and schema-declared endpoints; move internal helper(s) to Function Definitions: ' + ', '.join(extra_readme_public)
        )
    for api_name in sorted(schema_public):
        if api_name in main and api_name not in tests:
            raise ValueError(
                'tests/test_api.py must exercise schema public API function: ' + api_name
            )

    if 'data_dir' in main_low and 'data_dir' not in readme:
        raise ValueError('artifacts/README.md must mention DATA_DIR when src/api.py exposes it')
    if 'data_dir' in tests_low and 'data_dir' not in readme:
        raise ValueError('artifacts/README.md must explain DATA_DIR because tests use it')

    # If README promises CLI behavior, tests should exercise subprocess.
    usage = _readme_section_body(readme_content or '', '## Usage').lower()
    if ('python' in usage or 'command' in usage or 'cli' in usage) and 'argparse' in main_low:
        if 'subprocess' not in tests_low:
            raise ValueError(
                'README describes CLI/command behavior, but tests do not exercise CLI behavior with subprocess'
            )

def _validate_test_import_contract(test_content: str) -> None:
    tests = test_content or ''
    low = tests.lower()
    if 'from src.api import' not in low and 'import src.api' not in low:
        raise ValueError('tests/test_api.py must import the implementation with: from src.api import ...')
    if 'os.' in tests and 'import os' not in low:
        raise ValueError('tests/test_api.py uses os but does not import os')
    if 'pathlib.' in tests and 'import pathlib' not in low:
        raise ValueError('tests/test_api.py uses pathlib but does not import pathlib')
    if 'datetime.' in tests and 'import datetime' not in low:
        raise ValueError('tests/test_api.py uses datetime but does not import datetime')
    if 'tempfile.' in tests and 'import tempfile' not in low:
        raise ValueError('tests/test_api.py uses tempfile but does not import tempfile')
    # Tests may mention the DATA_DIR storage contract in comments or assertions,
    # but they should normally exercise storage through the public API rather
    # than importing DATA_DIR directly.

def _validate_native_storage_contract(main_content: str, test_content: str) -> None:
    """Enforce native, promotion-safe runtime storage.

    Generated skills are first built in tasks/<task_id>/src and later may be
    promoted to skills/<skill_name>/. Persistent files must therefore be
    derived from the module's own location, not the current working directory,
    /tmp, or a test-supplied override. Tests must exercise the same native
    behavior the promoted skill will use and clean up through the skill API.
    """
    main = main_content or ''
    tests = test_content or ''
    main_low = main.lower()
    tests_low = tests.lower()

    file_like_markers = [
        'open(', 'write(', 'read(', 'json.dump', 'json.load',
        'os.listdir', 'os.remove', 'unlink(', 'mkdir(', 'makedirs(',
        'reminder', 'schedule', 'storage', 'data_dir', 'data/'
    ]
    needs_storage_contract = any(marker in main_low for marker in file_like_markers)
    if needs_storage_contract:
        if 'from pathlib import path' not in main_low and 'import pathlib' not in main_low:
            raise ValueError(
                'src/api.py must use pathlib.Path for promotion-safe native storage paths'
            )
        if 'path(__file__).resolve().parent' not in main_low and '__file__' not in main_low:
            raise ValueError(
                'src/api.py must derive SKILL_ROOT from Path(__file__).resolve().parent'
            )
        if 'data_dir' not in main_low:
            raise ValueError(
                "src/api.py must define DATA_DIR under SKILL_ROOT / 'data' for persistent files"
            )
        if "'data'" not in main and '"data"' not in main:
            raise ValueError(
                "src/api.py DATA_DIR must be based on a literal 'data' directory"
            )

    forbidden_main_patterns = [
        ('os.getcwd(', 'src/api.py must not use os.getcwd() for persistent runtime paths'),
        ('tempfile.', 'src/api.py must not use tempfile for normal persistent runtime data'),
        ('TemporaryDirectory', 'src/api.py must not use TemporaryDirectory for normal runtime data'),
        ('/tmp', 'src/api.py must not hardcode /tmp for persistent runtime paths'),
        ('"reminders"', 'src/api.py must not use a bare hardcoded reminders directory; put data under DATA_DIR'),
        ("'reminders'", 'src/api.py must not use a bare hardcoded reminders directory; put data under DATA_DIR'),
    ]
    for pattern, message in forbidden_main_patterns:
        if pattern.lower() in main_low:
            # Allow the word reminder(s) in text, but not as a bare directory variable.
            if pattern in ('"reminders"', "'reminders'"):
                if 'data_dir' in main_low:
                    continue
            raise ValueError(message)

    forbidden_test_patterns = [
        ('tempfile.', 'tests/test_api.py must not use tempfile for normal storage; test native DATA_DIR behavior'),
        ('TemporaryDirectory', 'tests/test_api.py must not override normal storage with TemporaryDirectory'),
        ('mkdtemp', 'tests/test_api.py must not override normal storage with mkdtemp'),
        ('os.getcwd(', 'tests/test_api.py must not rely on cwd for capability storage'),
        ('/tmp', 'tests/test_api.py must not hardcode /tmp'),
        ('os.remove(', 'tests/test_api.py must not manually delete capability records; call the delete/back-out API'),
        ('unlink(', 'tests/test_api.py must not manually unlink capability records; call the delete/back-out API'),
        ('shutil.rmtree', 'tests/test_api.py must not manually remove capability data; call the delete/back-out API'),
    ]
    for pattern, message in forbidden_test_patterns:
        if pattern.lower() in tests_low:
            raise ValueError(message)

    override_terms = ['storage_path', 'storage_dir', 'data_dir=', 'data_path', 'root_dir', 'tempdir', 'temp_dir']
    for term in override_terms:
        if term in tests_low:
            raise ValueError(
                'tests/test_api.py must not override storage paths; run the skill natively and clean up through its API'
            )

    if needs_storage_contract and 'data_dir' not in tests_low and 'data_dir' in main_low:
        # Tests should at least observe or clear native storage via module constants/API.
        # They do not have to import DATA_DIR directly if the public API covers it.
        pass


def _validate_cleanup_contract(main_content: str, test_content: str) -> None:
    main_lower = main_content.lower()
    test_lower = test_content.lower()

    cleanup_terms = ['delete', 'remove', 'cleanup', 'clean_up', 'backout', 'back_out']
    if not any(term in main_lower for term in cleanup_terms):
        raise ValueError(
            'src/api.py must include a delete/remove/cleanup/back-out function or command'
        )
    if not any(term in test_lower for term in cleanup_terms):
        raise ValueError(
            'tests/test_api.py must verify delete/remove/cleanup/back-out behavior'
        )

    creation_terms = ['add', 'create', 'schedule', 'insert', 'save', 'write']
    if not any(term in main_lower for term in creation_terms):
        raise ValueError('src/api.py must include a way to create a testable entry')
    if not any(term in test_lower for term in creation_terms):
        raise ValueError('tests/test_api.py must create a test entry before cleanup')

    verification_terms = ['list', 'get', 'load', 'read', 'exists', 'assert']
    if not any(term in test_lower for term in verification_terms):
        raise ValueError('tests/test_api.py must verify created entry state')



def _validate_filesystem_domain_contract(
    main_content: str,
    test_content: str,
    main_tree: Optional[ast.Module],
    test_tree: Optional[ast.Module],
) -> None:
    """Filesystem-specific correctness checks.

    These checks encode lessons from real failures:
    - files must be deleted with os.remove()/Path.unlink(), not shutil.rmtree()
    - shutil.rmtree() is only for directories and must be guarded by isdir()/is_dir()
    - inline open(...).read() leaks file handles; use with-open or explicit close
    - delete/back-out must remove the same persistent file path that add/create wrote
    """
    main = main_content or ''
    tests = test_content or ''
    main_low = main.lower()
    tests_low = tests.lower()

    _reject_inline_open_read('src/api.py', main)
    _reject_inline_open_read('tests/test_api.py', tests)

    if 'shutil.rmtree' in main_low:
        has_dir_guard = (
            'os.path.isdir' in main_low
            or '.is_dir()' in main_low
            or 'path.is_dir' in main_low
        )
        if not has_dir_guard:
            raise ValueError(
                'src/api.py uses shutil.rmtree without an is-directory guard. '
                'Use os.remove()/Path.unlink() for files and shutil.rmtree() only for directories.'
            )
        file_markers = ['.txt', '.json', '.db', '.sqlite', '.csv', '.log']
        if any(marker in main_low for marker in file_markers):
            has_file_delete = 'os.remove(' in main_low or '.unlink(' in main_low
            if not has_file_delete:
                raise ValueError(
                    'src/api.py deletes file-like records but does not use os.remove() '
                    'or Path.unlink(). Use shutil.rmtree() only for directories.'
                )

    if main_tree is not None:
        source = main.splitlines()
        for node in ast.walk(main_tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name.lower()
            if not any(word in name for word in ['delete', 'remove', 'cleanup', 'backout', 'back_out']):
                continue
            segment = _source_segment(source, node)
            low = segment.lower()
            if 'rmtree' in low and ('os.remove(' not in low and '.unlink(' not in low):
                raise ValueError(
                    'Function {} in src/api.py uses rmtree for cleanup but does not handle file deletion. '
                    'Delete file records with os.remove()/Path.unlink(), and use rmtree only for directories.'.format(node.name)
                )
            if any(marker in main_low for marker in ['.txt', '.json', '.db', '.sqlite', '.csv']) and not (
                'os.remove(' in low or '.unlink(' in low or 'json.dump' in low
            ):
                raise ValueError(
                    'Function {} in src/api.py must remove the same persistent file/record created by add/create.'.format(node.name)
                )

    if test_tree is not None:
        # Tests may inspect native files, but must not leak handles.
        # They must also verify cleanup through the public API rather than manual deletion.
        for node in ast.walk(test_tree):
            if isinstance(node, ast.Call) and _call_name(node.func).endswith('rmtree'):
                raise ValueError(
                    'tests/test_api.py must not use shutil.rmtree for capability records; call the skill delete/back-out API.'
                )


def _reject_inline_open_read(path: str, content: str) -> None:
    for number, line in enumerate((content or '').splitlines(), start=1):
        compact = line.replace(' ', '')
        if 'open(' in compact and ').read(' in compact:
            raise ValueError(
                '{} line {} uses inline open(...).read(); use with open(...) as handle or explicitly close the file.'.format(path, number)
            )
        if '=open(' in compact and 'close()' not in content:
            # If a file handle is manually opened anywhere, the whole file should contain a close.
            raise ValueError(
                '{} opens a file handle without an explicit close(); use with-open or try/finally close.'.format(path)
            )


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        if base:
            return base + '.' + func.attr
        return func.attr
    return ''
def _validate_test_file_behavior(test_content: str, tree: Optional[ast.Module]) -> None:
    lowered = test_content.lower()
    if 'unittest' not in lowered:
        raise ValueError(
            'tests/test_api.py must use unittest. You used pytest/bare assert style or omitted unittest. '
            'Required structure: import unittest; from src.api import ...; '
            'class TestPublicApi(unittest.TestCase): def test_...(self): self.assert...; '
            'do not import pytest or use pytest.mark.'
        )
    if 'pytest' in lowered:
        raise ValueError(
            'tests/test_api.py must not use pytest. Replace pytest imports/decorators/bare assert style with unittest.TestCase methods and self.assert* calls.'
        )
    if 'assert' not in lowered:
        raise ValueError('tests/test_api.py must contain assertions')
    if 'pass' in lowered and lowered.count('pass') >= lowered.count('assert'):
        raise ValueError('tests/test_api.py appears to contain placeholder pass tests')
    if tree:
        test_functions = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
        ]
        if not test_functions:
            raise ValueError('tests/test_api.py must define at least one test_ function')
        for node in test_functions:
            segment = ast.get_source_segment(test_content, node) or ''
            if 'assert' not in segment.lower():
                raise ValueError('Test function {} has no assertion'.format(node.name))


def _find_function(tree: ast.Module, name: str) -> Optional[ast.FunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(getattr(stmt, 'value', None), ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_ellipsis_expr(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, 'value', None), ast.Constant) and stmt.value.value is Ellipsis


def _source_segment(source_lines: Iterable[str], node: ast.AST) -> str:
    start = getattr(node, 'lineno', 1) - 1
    end = getattr(node, 'end_lineno', start + 1)
    lines = list(source_lines)
    return '\n'.join(lines[start:end])


def _validate_relative_task_path(path: str) -> None:
    if not path or path.startswith('/') or '\\' in path:
        raise ValueError('Invalid relative path: {}'.format(path))
    parts = Path(path).parts
    if any(part in {'..', ''} for part in parts):
        raise ValueError('Invalid relative path: {}'.format(path))
    if parts[0] not in {'src', 'tests', 'artifacts'}:
        raise ValueError(
            'Generated path must be under src/, tests/, or artifacts/: {}'.format(path)
        )


def _safe_name(name: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9_ -]+', '', name).strip().lower().replace(' ', '_')
    name = re.sub(r'_+', '_', name)
    return name or 'generated_capability'




def api_validator_rules_for_coder_prompt() -> str:
    """Concise shared contract for model prompts.

    Full enforcement remains in validator code. Prompts intentionally get the
    short version so local models do not drown in repeated rule text.
    """
    return """Shared contract summary:
- Output exactly one requested file block; no markdown fences, prose, placeholders, stubs, or special/control tokens.
- src/api.py is a Python-callable module only. No HTTP frameworks/routes/classes for public API.
- schema() is top-level and returns a dict with endpoints mapping public function names to metadata.
- Every schema endpoint is a same-name top-level function and returns ok/action/message/data/error through response helpers.
- Persistent files use SKILL_ROOT = Path(__file__).resolve().parent and DATA_DIR = SKILL_ROOT / 'data'. Do not use cwd/tmp/tempfile.
- tests/test_api.py uses unittest, imports from src.api, calls schema(), calls every schema endpoint, and inspects ok/action/message/data.
- README contains the required contract sections and must match src/api.py/schema().
"""

def validate_single_generated_file(item) -> None:
    """Validate one generated file before the full cross-file validation step.

    This intentionally checks only properties that can be judged for a single
    file: path safety, non-empty content, placeholder/stub rejection, Python
    syntax, and test-file assertions. Cross-file requirements like cleanup
    contracts are checked later by validate_build_files after every file exists.
    """
    _validate_relative_task_path(item.path)
    content = item.content or ''
    if not content.strip():
        raise ValueError('Generated file is empty: {}'.format(item.path))
    _reject_placeholders(item.path, content)
    if item.path == 'commands.json':
        _validate_commands_json(content)
        return  # JSON file — no further Python checks
    if item.path == 'artifacts/README.md':
        _validate_readme_single(content)
    if item.path.endswith('.py'):
        _reject_forbidden_text_before_ast(item.path, content)
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            raise ValueError(
                'Generated Python syntax error in {}: {}'.format(item.path, exc)
            ) from exc
        _reject_stub_functions(item.path, content, tree)
        _reject_forbidden_python_subset(item.path, content, tree)
        if item.path == 'src/api.py':
            _validate_api_source_contract(content)
        if item.path == 'tests/test_api.py':
            _validate_test_import_contract(content)
            _validate_test_file_behavior(content, tree)
            missing = _missing_response_keys(content)
            if missing:
                raise ValueError(_response_key_error(missing))
            # The error field is optional in tests. If the generated API exposes it,
            # tests may inspect it, but missing test assertions for error should not
            # block builds.

# ---------------------------------------------------------------------------
# commands.json validation
# ---------------------------------------------------------------------------

def _validate_commands_json(content: str) -> None:
    import json as _json
    try:
        data = _json.loads(content)
    except _json.JSONDecodeError as exc:
        raise ValueError('commands.json is not valid JSON: {}'.format(exc))
    if not isinstance(data, dict):
        raise ValueError('commands.json must be a JSON object')
    commands = data.get('commands')
    if not isinstance(commands, list):
        raise ValueError('commands.json must have a "commands" list')
    for i, cmd in enumerate(commands):
        triggers = cmd.get('triggers')
        if not isinstance(triggers, list) or not triggers:
            raise ValueError('commands.json command[{}] must have a non-empty "triggers" list'.format(i))
        if not cmd.get('action'):
            raise ValueError('commands.json command[{}] must have an "action"'.format(i))

# ---------------------------------------------------------------------------
# Step 22 API-first validation overrides
# ---------------------------------------------------------------------------

API_RESPONSE_KEYS = {'ok', 'action', 'message', 'data', 'error'}
TEST_REQUIRED_RESPONSE_KEYS = ('ok', 'action', 'message', 'data')


def _missing_response_keys(content: str) -> list[str]:
    text = content or ''
    return [key for key in TEST_REQUIRED_RESPONSE_KEYS if key not in text]


def _response_key_error(missing: list[str]) -> str:
    if not missing:
        return ''
    return 'tests/test_api.py must assert or inspect response field(s): ' + ', '.join(missing)


def _validate_readme_single(readme_content: str) -> None:
    """Step 22: README is still structured, but behavior is API-first.

    Keep this deterministic validator structural. The LLM semantic reviewer and
    AST/API validators check meaning. Avoid brittle keyword checks that cause
    false rejections of good README content.
    """
    text = readme_content or ''
    stripped = text.strip()
    if not stripped.startswith('# '):
        raise ValueError('artifacts/README.md must start with a top-level # title')
    missing = [section for section in README_REQUIRED_SECTIONS[1:] if section not in text]
    if missing:
        raise ValueError('artifacts/README.md missing required section(s): ' + ', '.join(missing))
    for section in README_REQUIRED_SECTIONS[1:]:
        body = _readme_section_body(text, section)
        if len(body.strip()) < 10:
            raise ValueError('artifacts/README.md section is empty or too short: ' + section)
    if len([line for line in text.splitlines() if line.strip()]) < 18:
        raise ValueError('artifacts/README.md is too short to be a complete usage contract')


def _schema_return_dict_from_ast(source: str) -> ast.Dict | None:
    """Return the AST dict node returned by schema(), if present."""
    try:
        tree = ast.parse(source or '')
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'schema':
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                return child.value
    return None


def _schema_return_value_from_ast(source: str) -> ast.AST | None:
    """Return the AST value returned by top-level schema(), if present."""
    try:
        tree = ast.parse(source or '')
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'schema':
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return child.value
    return None


def _schema_value_diagnostic(value: ast.AST | None) -> str:
    if value is None:
        return 'schema() did not return a value.'
    kind = type(value).__name__
    try:
        rendered = ast.unparse(value)
    except Exception:
        rendered = kind
    if len(rendered) > 500:
        rendered = rendered[:500] + '...'
    return 'schema() currently returns {}: {}'.format(kind, rendered)


_SCHEMA_DICT_ERROR = (
    'schema() in src/api.py must return a contract dict, not a list/string/tuple. '
    'The dict must contain package, endpoints, response_format, storage_notes, and notes. '
    'schema()["endpoints"] must be a dict mapping each public Python function name to metadata. '
    'Endpoint keys must be logical top-level Python function names, not HTTP route paths. '
    'Every endpoint key must have an exact same-name top-level function. '
    'Generic shape: {"package": "<task_package>", "endpoints": {"<endpoint_name>": {"args": ["<arg_name>"], "returns": "dict"}}, "response_format": ["ok", "action", "message", "data", "error"], "storage_notes": "<runtime storage>", "notes": "<task contract>"}. '
)


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _schema_endpoint_names_from_ast(source: str) -> list[str]:
    """Return public endpoint names declared by schema().

    v43+ rule: public API endpoints come only from schema(), not from helper
    functions.  v46 parses the schema AST directly so non-literal values such
    as bool/type objects or f-strings elsewhere in schema() do not make endpoint
    extraction fail.  The canonical shape is:

        {"endpoints": {"name": {"args": [...], "returns": "dict"}}}

    A list is still readable here for diagnostics/backward compatibility, but
    src/api.py validation below rejects list-based endpoints with a precise
    repair message.
    """
    schema_dict = _schema_return_dict_from_ast(source)
    if schema_dict is None:
        return []
    ignored_top_keys = {
        'name', 'package', 'skill', 'description', 'purpose', 'endpoints',
        'envelope', 'response', 'responses', 'response_format', 'storage',
        'storage_notes', 'data', 'notes', 'version',
    }
    endpoints_node: ast.AST | None = None
    for key_node, value_node in zip(schema_dict.keys, schema_dict.values):
        if _constant_string(key_node) == 'endpoints':
            endpoints_node = value_node
            break
    names: list[str] = []
    if isinstance(endpoints_node, ast.Dict):
        names = [name for key in endpoints_node.keys if (name := _constant_string(key))]
    elif isinstance(endpoints_node, (ast.List, ast.Tuple, ast.Set)):
        names = [name for item in endpoints_node.elts if (name := _constant_string(item))]
    elif endpoints_node is None:
        names = [
            name for key, value in zip(schema_dict.keys, schema_dict.values)
            if (name := _constant_string(key))
            and name not in ignored_top_keys
            and isinstance(value, ast.Dict)
        ]
    return sorted({name for name in names if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name)})

def _top_level_function_names_from_ast(source: str) -> list[str]:
    try:
        tree = ast.parse(source or '')
    except SyntaxError:
        return []
    return sorted({node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})


def _public_api_function_names_from_ast(source: str) -> list[str]:
    names: list[str] = []
    if 'schema' in _top_level_function_names_from_ast(source):
        names.append('schema')
    names.extend(_schema_endpoint_names_from_ast(source))
    return sorted(set(names))


def _function_has_dict_response_return(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    has_return = False
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            has_return = True
            value = child.value
            if value is None:
                return False
            if isinstance(value, ast.Dict):
                keys = set()
                for k in value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
                if API_RESPONSE_KEYS.issubset(keys):
                    return True
            # allow returning a variable produced by helper response() builder
            if isinstance(value, ast.Name):
                continue
            if isinstance(value, ast.Call):
                # e.g. return response(...). The source-level key check below
                # still requires response fields to exist in the module.
                continue
    return has_return


def _function_has_exception_envelope(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for child in ast.walk(node):
        if not isinstance(child, ast.Try):
            continue
        for handler in child.handlers:
            handled = handler.type is None
            if isinstance(handler.type, ast.Name) and handler.type.id == 'Exception':
                handled = True
            if isinstance(handler.type, ast.Tuple):
                handled = any(isinstance(elt, ast.Name) and elt.id == 'Exception' for elt in handler.type.elts)
            if not handled:
                continue
            for stmt in ast.walk(handler):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    keys = set()
                    ok_false = False
                    error_present = False
                    for key_node, value_node in zip(stmt.value.keys, stmt.value.values):
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            keys.add(key_node.value)
                            if key_node.value == 'ok' and isinstance(value_node, ast.Constant) and value_node.value is False:
                                ok_false = True
                            if key_node.value == 'error':
                                error_present = True
                    if {'ok', 'action', 'message', 'data', 'error'}.issubset(keys) and ok_false and error_present:
                        return True
    return False



def _function_uses_response_contract(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for child in ast.walk(node):
        if not isinstance(child, ast.Return):
            continue
        value = child.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in {'_cm_success', '_cm_failure'}:
                return True
    return False


def _validate_api_source_contract(api_content: str) -> None:
    """Validate src/api.py while it is generated as a single file.

    Step 23 closes a gap from Step 22: src/api.py could pass generic Python
    validation and only fail after README/test generation because schema() was
    checked solely during full package validation. This source-only check makes
    missing schema(), missing endpoint functions, and missing envelope fields
    repairable during the src/api.py generation loop.
    """
    source = api_content or ''
    low = source.lower()
    http_markers = [
        'from flask import', 'import flask', 'fastapi', 'django', 'bottle', 'sanic',
        'starlette', '@app.route', '@router.', 'jsonify(', 'request.get_json',
    ]
    if any(marker in low for marker in http_markers):
        raise ValueError(
            'src/api.py must expose a Python-callable interface only, not an HTTP service. '
            'Do not import/use Flask/FastAPI/Django or route decorators. '
            'schema()["endpoints"] keys must be top-level Python function names, not URL paths.'
        )
    if 'os.getcwd(' in low or '/tmp' in low or 'tempfile.' in low:
        raise ValueError("src/api.py API package must not use cwd, /tmp, or tempfile for persistent storage. Use the canonical storage contract instead: from pathlib import Path; SKILL_ROOT = Path(__file__).resolve().parent; DATA_DIR = SKILL_ROOT / 'data'; DATA_DIR.mkdir(parents=True, exist_ok=True); store persistent files under DATA_DIR. Do not mention os.getcwd() in code or schema()['storage_notes'].")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    top_level_names = _top_level_function_names_from_ast(source)
    if 'schema' not in top_level_names:
        raise ValueError('src/api.py must define schema() as a top-level public function. Do NOT wrap the API in a class. INVALID: class TodoAPI: def schema(self): ... VALID: def schema(): ...')
    schema_dict = _schema_return_dict_from_ast(source)
    if schema_dict is None:
        raise ValueError(_SCHEMA_DICT_ERROR + _schema_value_diagnostic(_schema_return_value_from_ast(source)))
    endpoints_node = None
    for key_node, value_node in zip(schema_dict.keys, schema_dict.values):
        if _constant_string(key_node) == 'endpoints':
            endpoints_node = value_node
            break
    if endpoints_node is None:
        raise ValueError('schema()["endpoints"] is required and must be a dictionary mapping endpoint names to metadata')
    if isinstance(endpoints_node, (ast.List, ast.Tuple, ast.Set)):
        raise ValueError('schema()["endpoints"] must be a dictionary mapping endpoint name to metadata, not a list. Use generic shape: "endpoints": {"<endpoint_name>": {"args": ["<arg_name>"], "returns": "dict"}}. ' + _schema_value_diagnostic(endpoints_node))
    if not isinstance(endpoints_node, ast.Dict):
        raise ValueError('schema()["endpoints"] must be a dictionary mapping endpoint name to metadata')
    endpoints = _schema_endpoint_names_from_ast(source)
    if not endpoints:
        raise ValueError('schema()["endpoints"] must contain at least one endpoint mapping using generic shape: {"<endpoint_name>": {"args": ["<arg_name>"], "returns": "dict"}}')
    route_like = [name for name in endpoints if name.startswith('/') or '<' in name or '>' in name]
    if route_like:
        raise ValueError(
            'schema()["endpoints"] keys must be logical Python function names, not HTTP route paths. '
            'Invalid route-like endpoint key(s): ' + ', '.join(route_like) + '. '
            'Use top-level function names such as create_item/list_items/delete_item, chosen for the task contract.'
        )
    missing_endpoints = [name for name in endpoints if name not in top_level_names]
    if missing_endpoints:
        required = '; '.join('def {}(...): return _cm_success/_cm_failure envelope'.format(name) for name in missing_endpoints)
        raise ValueError(
            'schema() declares endpoint(s) missing as top-level functions: ' + ', '.join(missing_endpoints) +
            '. You declared these names in schema()["endpoints"], so src/api.py must also implement them as exact same-name top-level functions. Required implementations: ' + required +
            '. Do not only return schema(); generate the full src/api.py with every endpoint implementation.'
        )
    response_keys = set(re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]\s*:", source))
    if not API_RESPONSE_KEYS.issubset(response_keys):
        raise ValueError('src/api.py must contain response dict fields: ok, action, message, data, error')
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith('_'):
            continue
        if node.name == 'schema':
            if not any(isinstance(child, ast.Return) and isinstance(child.value, ast.Dict) for child in ast.walk(node)):
                raise ValueError(_SCHEMA_DICT_ERROR + _schema_value_diagnostic(_schema_return_value_from_ast(source)))
            continue
        if node.name not in endpoints:
            continue
        if not _function_has_dict_response_return(node):
            raise ValueError('API endpoint {} in src/api.py must return the standard dict response envelope'.format(node.name))
        if not _function_uses_response_contract(node):
            raise ValueError('API endpoint {} in src/api.py must return through _cm_success() or _cm_failure()'.format(node.name))


def _validate_api_endpoint_contract(api_content: str, tests_content: str) -> None:
    source = api_content or ''
    tests = tests_content or ''
    low = source.lower()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    _validate_api_source_contract(source)
    public_names = _public_api_function_names_from_ast(source)
    endpoints = [name for name in public_names if name != 'schema']

    # Tests must call schema, all endpoint functions, and inspect the response
    # envelope. Collect every recognized issue before raising so repair prompts
    # can fix the whole batch rather than one validator complaint at a time.
    issues: list[str] = []
    if 'schema(' not in tests:
        issues.append('tests/test_api.py must call schema()')
    if re.search(r'assertEqual\s*\(\s*schema\s*\(\s*\)\s*,\s*\{', tests) or re.search(r'assertEqual\s*\(\s*response\s*,\s*\{', tests):
        issues.append('tests/test_api.py must not assert an exact schema dict; inspect schema structure flexibly')
    missing_calls = [name for name in endpoints if name + '(' not in tests]
    if missing_calls:
        issues.append('tests/test_api.py must call API endpoint(s): ' + ', '.join(missing_calls))

    missing = _missing_response_keys(tests)
    if missing:
        issues.append(_response_key_error(missing))
    if issues:
        raise ValueError('Validation issues to fix together:\n- ' + '\n- '.join(issues))
    # Do not require tests to mention optional error fields. Some skills may use
    # minimal success envelopes, and behavior validation is more important than
    # forcing a literal error assertion.

    if 'os.getcwd(' in low or '/tmp' in low or 'tempfile.' in low:
        raise ValueError("src/api.py API package must not use cwd, /tmp, or tempfile for persistent storage. Use the canonical storage contract instead: from pathlib import Path; SKILL_ROOT = Path(__file__).resolve().parent; DATA_DIR = SKILL_ROOT / 'data'; DATA_DIR.mkdir(parents=True, exist_ok=True); store persistent files under DATA_DIR. Do not mention os.getcwd() in code or schema()['storage_notes'].")


def _validate_readme_contract(readme_content: str, main_content: str, test_content: str) -> None:
    _validate_readme_single(readme_content)
    _validate_readme_documents_source_contract(readme_content, main_content)
    _validate_api_readme_contract(readme_content, main_content, test_content)


def _validate_api_readme_contract(readme_content: str, api_content: str, tests_content: str) -> None:
    readme_low = (readme_content or '').lower()
    outputs = _readme_section_body(readme_content or '', '## Outputs and Return Values').lower()
    public_api = _readme_section_body(readme_content or '', '## Public API').lower()
    functions = _readme_section_body(readme_content or '', '## Function Definitions').lower()
    tests_section = _readme_section_body(readme_content or '', '## Test Coverage').lower()

    public_names = _public_api_function_names_from_ast(api_content or '')
    if 'schema' not in public_names:
        raise ValueError('src/api.py must define schema()')
    for name in public_names:
        if name.lower() not in public_api and name.lower() not in functions:
            raise ValueError('artifacts/README.md must document API function: ' + name)
        if name.lower() not in outputs:
            raise ValueError('artifacts/README.md Outputs and Return Values must document API function: ' + name)

    for key in API_RESPONSE_KEYS:
        if key not in outputs and key not in readme_low:
            raise ValueError('artifacts/README.md must document response field: ' + key)

    for needed in ['schema', 'endpoint', 'response']:
        if needed not in readme_low:
            raise ValueError('artifacts/README.md must describe API-first contract including: ' + needed)

    for needed in ['schema', 'endpoint', 'response']:
        if needed not in tests_section:
            raise ValueError('artifacts/README.md Test Coverage must mention ' + needed)

    readme_public = set(_readme_public_api_names(readme_content))
    schema_public = set(public_names)
    extra_readme_public = sorted(readme_public - schema_public)
    if extra_readme_public:
        raise ValueError('artifacts/README.md Public API must only list schema() and schema-declared endpoints; move internal helper(s) to Function Definitions: ' + ', '.join(extra_readme_public))
    for api_name in sorted(schema_public):
        if api_name in (api_content or '') and api_name not in (tests_content or ''):
            raise ValueError('tests/test_api.py must exercise schema public API function: ' + api_name)


def _validate_test_import_contract(test_content: str) -> None:
    tests = test_content or ''
    low = tests.lower()
    if 'from src.api import' not in low and 'import src.api' not in low:
        raise ValueError('tests/test_api.py must import the implementation with: from src.api import ...')
    for module in ['os', 'pathlib', 'datetime', 'tempfile', 'json']:
        if module + '.' in tests and ('import ' + module) not in low:
            raise ValueError('tests/test_api.py uses {} but does not import {}'.format(module, module))
    # Tests may mention the DATA_DIR storage contract in comments or assertions,
    # but they should normally exercise storage through the public API rather
    # than importing DATA_DIR directly.


# Wrap validate_build_files so the API contract is enforced after existing checks.
_original_validate_build_files_step22 = validate_build_files


def validate_build_files(build_files: BuildFiles) -> BuildFiles:
    result = _original_validate_build_files_step22(build_files)
    api = next((item.content for item in build_files.files if item.path == 'src/api.py'), '')
    tests = next((item.content for item in build_files.files if item.path == 'tests/test_api.py'), '')
    _validate_api_endpoint_contract(api, tests)
    return result
