from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from .models import LocalModel
from .config import CODER_MODEL
from .normalizer import normalize_single_file
from .schemas import BuildFiles, GeneratedFile, WorkOrder
from .validator import validate_build_files, validate_single_generated_file, api_validator_rules_for_coder_prompt
from .semantic_validator import validate_readme_semantics
from .lesson_engine import enforce_lessons_on_code, extract_static_lessons, normalize_failure_signature


MAX_FILE_ATTEMPTS = 7
MAX_PREVIEW_CHARS = 5000


_SCHEMA_CONTRACT_REPAIR_RULES = """
PYTHON API CONTRACT (STRICT):
- Generated task artifacts expose a Python-callable interface only.
- Do NOT generate HTTP services, Flask/FastAPI/Django apps, route decorators, request handlers, jsonify, request objects, or route-path schema keys.
- schema() MUST return a Python dict, never a list/string/tuple.
- schema()["endpoints"] MUST be a dict mapping public Python function names to metadata.
- Endpoint keys are logical Python callable names, not URLs/routes. INVALID: "/add". VALID: "add_task".
- For EVERY key in schema()["endpoints"], the same file MUST define an exact same-name top-level function.
- Do NOT prefix schema endpoint functions with "_".
- Helper/internal functions may exist, but must NOT appear in schema()["endpoints"].
- Endpoint functions must be real implementations, not pass/stubs, and must return the standard envelope through _cm_success(...) or _cm_failure(...).
- Generation order for src/api.py: imports/constants, response helpers, internal helpers, ALL endpoint functions, then schema() last.
- Do not stop after writing schema(); schema declares the public contract, and the file is invalid until every declared endpoint function is implemented.
- Use this generic shape, filling task-specific names and args:
  def <endpoint_name>(<args>):
      ...
      return _cm_success("<endpoint_name>", "...", data)

  def schema():
      return {
          "package": "<task_specific_package>",
          "endpoints": {
              "<endpoint_name>": {"args": ["<arg_name>"], "returns": "dict"},
          },
          "response_format": ["ok", "action", "message", "data", "error"],
          "storage_notes": "<runtime storage behavior, if any>",
          "notes": "<task-specific API contract>",
      }
"""


class Coder:
    """LLM-backed file generator.

    Step 8 changes the build strategy from "generate all files in one large
    response" to "generate, normalize, validate, and repair exactly one file at
    a time". This prevents one bad long response from corrupting every file and
    makes the repair feedback much more precise.
    """

    def __init__(
        self,
        model: LocalModel | None = None,
        max_attempts: int = MAX_FILE_ATTEMPTS,
        progress=None,
        lesson_provider: Callable[[str | None, int], List[Dict[str, Any]]] | None = None,
        lesson_recorder: Callable[[str, str, str, str, str], None] | None = None,
    ):
        self.model = model or LocalModel(model=CODER_MODEL)
        self.max_attempts = max_attempts
        self.progress = progress
        self.lesson_provider = lesson_provider
        self.lesson_recorder = lesson_recorder
        self.session_writer = None
        self.session_appender = None

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

    def _generate_with_logged_prompt(self, prompt: str, *, path: str, purpose: str, context: str, attempt: int | None = None) -> str:
        """Call the coder model and persist the exact prompt for task diagnostics."""
        record = {
            'component': 'coder',
            'model': getattr(self.model, 'model', None),
            'path': path,
            'purpose': purpose,
            'context': context,
            'attempt': attempt,
            'prompt': prompt,
        }
        self._session_append('llm', 'prompts', record)
        self._session_set('llm', 'last_prompt', record)
        return self.model.generate(prompt)

    def _progress(self, message: str) -> None:
        if self.progress:
            try:
                self.progress(message)
            except Exception:
                pass

    def generate_files(self, work_order: WorkOrder, environment: Dict[str, Any]) -> BuildFiles:
        generated: List[GeneratedFile] = []
        per_file_notes: List[str] = []

        for file_spec in work_order.files:
            path = str(file_spec["path"])
            purpose = str(file_spec.get("purpose") or "required file")
            self._progress("Generating file: {}".format(path))
            self._session_set('coder', 'active_file', path)
            self._session_set('coder', 'active_file_purpose', purpose)
            current = BuildFiles(files=list(generated), notes="partial_generation")
            item, notes = self._generate_one_file(
                work_order=work_order,
                environment=environment,
                expected_path=path,
                purpose=purpose,
                current_files=current,
            )
            generated.append(item)
            self._session_append('coder', 'generated_files', {
                'path': path,
                'notes': notes,
                'bytes': len(item.content or ''),
            })
            self._session_set('validator', 'last_validated_file', path)
            per_file_notes.append("{}:{}".format(path, notes))
            partial = BuildFiles(files=list(generated), notes="; ".join(per_file_notes))
            self._session_set('builder', 'latest_generated_build_files', partial.to_dict())

        build_files = BuildFiles(files=generated, notes="; ".join(per_file_notes))
        # Final cross-file/package validation is performed by BuildManager after
        # it persists build_files.generated.json. This keeps failure diagnostics
        # from losing the generated source set when package validation fails.
        return build_files

    def repair_files(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        previous_files: BuildFiles,
        failure_report: Dict[str, Any],
    ) -> BuildFiles:
        repaired: List[GeneratedFile] = []
        per_file_notes: List[str] = []

        # Runtime/test repair is still file-by-file. Each call sees all current
        # files and the failure report, but may only return the one requested
        # file. This keeps output small while preserving enough context.
        for file_spec in work_order.files:
            path = str(file_spec["path"])
            self._progress("Repairing file after runtime failure: {}".format(path))
            self._session_set('coder', 'active_repair_file', path)
            self._session_set('coder', 'runtime_failure_summary', json.dumps(failure_report, indent=2, default=str)[:2000])
            existing = self._find_file(previous_files, path)
            current_context = BuildFiles(files=self._merge_repaired_context(repaired, previous_files), notes="runtime_context")
            item, notes = self._repair_one_file_from_runtime_failure(
                work_order=work_order,
                environment=environment,
                expected_path=path,
                purpose=str(file_spec.get("purpose") or "required file"),
                current_files=current_context,
                existing_file=existing,
                failure_report=failure_report,
            )
            repaired.append(item)
            per_file_notes.append("{}:{}".format(path, notes))
            partial = BuildFiles(files=list(repaired), notes="runtime_repair; " + "; ".join(per_file_notes))
            self._session_set('builder', 'latest_repaired_build_files', partial.to_dict())

        build_files = BuildFiles(files=repaired, notes="runtime_repair; " + "; ".join(per_file_notes))
        # BuildManager performs cross-file validation after it can persist the
        # repaired source set for diagnostics.
        return build_files

    def repair_files_from_validation_failure(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        previous_files: BuildFiles,
        validation_error: str,
    ) -> BuildFiles:
        error_text = str(validation_error)
        required_paths = [str(spec.get('path')) for spec in work_order.files]
        target_paths = [path for path in required_paths if path and path in error_text]
        if not target_paths:
            lowered = error_text.lower()
            target_paths = [
                path for path in required_paths
                if path and (Path(path).name.lower() in lowered or path.lower() in lowered)
            ]
        if not target_paths:
            target_paths = required_paths

        self._session_set('validator', 'artifact_validation_failure', {
            'error': error_text,
            'target_paths': target_paths,
            'current_files': previous_files.to_dict(),
        })

        repaired: List[GeneratedFile] = []
        per_file_notes: List[str] = []
        by_path = {item.path: item for item in previous_files.files}

        for spec in work_order.files:
            path = str(spec.get('path'))
            existing = by_path.get(path)
            if path not in target_paths and existing is not None:
                repaired.append(existing)
                per_file_notes.append(path + ':unchanged_after_artifact_validation')
                continue

            self._progress('Repairing artifact after validation failure: {}'.format(path))
            self._session_set('coder', 'active_repair_file', path)
            self._session_set('coder', 'repair_requested', {
                'path': path,
                'error': error_text,
                'current_content': existing.content if existing else '',
                'current_files': previous_files.to_dict(),
            })
            current_context = BuildFiles(files=self._merge_repaired_context(repaired, previous_files), notes='artifact_validation_context')
            prompt = self._single_file_static_repair_prompt(
                work_order=work_order,
                environment=environment,
                expected_path=path,
                purpose=str(spec.get('purpose') or 'required file'),
                current_files=current_context,
                previous_file=existing,
                validation_error=error_text,
                failures=[{
                    'attempt': 0,
                    'error': error_text,
                    'expected_path': path,
                    'content_preview': (existing.content if existing else '')[:1600],
                }],
                runtime_failure_report=None,
            )
            raw = self._generate_with_logged_prompt(
                prompt,
                path=path,
                purpose=str(spec.get('purpose') or 'required file'),
                context='artifact_validation_repair_initial',
                attempt=0,
            )
            item, notes = self._single_file_validate_repair_loop(
                work_order=work_order,
                environment=environment,
                expected_path=path,
                purpose=str(spec.get('purpose') or 'required file'),
                current_files=current_context,
                initial_raw=raw,
                context='artifact_validation_repair',
            )
            repaired.append(item)
            per_file_notes.append(path + ':' + notes)
            self._session_set('builder', 'latest_artifact_repair_files', BuildFiles(files=list(repaired), notes='artifact_validation_repair').to_dict())

        return BuildFiles(files=repaired, notes='artifact_validation_repair; ' + '; '.join(per_file_notes))

    def _merge_repaired_context(self, repaired: List[GeneratedFile], previous_files: BuildFiles) -> List[GeneratedFile]:
        by_path: Dict[str, GeneratedFile] = {item.path: item for item in previous_files.files}
        for item in repaired:
            by_path[item.path] = item
        ordered: List[GeneratedFile] = []
        seen = set()
        for item in list(repaired) + list(previous_files.files):
            if item.path in seen:
                continue
            seen.add(item.path)
            ordered.append(by_path[item.path])
        return ordered

    def _raw_file_block(self, item: GeneratedFile, content_type: str | None = None) -> str:
        ctype = content_type or ('text/markdown' if item.path.endswith('.md') else 'text/x-python')
        return (
            'FILE: ' + item.path + '\n'
            + 'CONTENT_TYPE: ' + ctype + '\n'
            + '---BEGIN CONTENT---\n'
            + (item.content or '')
            + '\n---END CONTENT---\n'
        )

    def _generate_one_file(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
    ) -> tuple[GeneratedFile, str]:
        if expected_path == 'artifacts/README.md':
            item = self._deterministic_readme_file(work_order, current_files)
            raw = self._raw_file_block(item, content_type='text/markdown')
            return self._single_file_validate_repair_loop(
                work_order=work_order,
                environment=environment,
                expected_path=expected_path,
                purpose=purpose,
                current_files=current_files,
                initial_raw=raw,
                context="initial_file_generation",
            )

        if expected_path == 'commands.json':
            item = self._deterministic_commands_json(work_order, current_files)
            return item, 'commands_json_deterministic'

        prompt = self._single_file_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            current_files=current_files,
        )
        raw = self._generate_with_logged_prompt(
            prompt,
            path=expected_path,
            purpose=purpose,
            context='initial_file_generation',
            attempt=1,
        )
        return self._single_file_validate_repair_loop(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            current_files=current_files,
            initial_raw=raw,
            context="initial_file_generation",
        )

    def _repair_one_file_from_runtime_failure(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
        existing_file: GeneratedFile | None,
        failure_report: Dict[str, Any],
    ) -> tuple[GeneratedFile, str]:
        if expected_path == 'artifacts/README.md':
            # README repairs are not special-cased. Start from the current
            # deterministic README, then use the normal validation/repair loop
            # so validator messages are routed back to the coder with the
            # current README content in context.
            item = self._deterministic_readme_file(work_order, current_files)
            raw = self._raw_file_block(item, content_type='text/markdown')
            return self._single_file_validate_repair_loop(
                work_order=work_order,
                environment=environment,
                expected_path=expected_path,
                purpose=purpose,
                current_files=current_files,
                initial_raw=raw,
                context="runtime_single_file_repair",
                runtime_failure_report=failure_report,
            )

        prompt = self._single_file_runtime_repair_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            current_files=current_files,
            existing_file=existing_file,
            failure_report=failure_report,
        )
        raw = self._generate_with_logged_prompt(
            prompt,
            path=expected_path,
            purpose=purpose,
            context='runtime_single_file_repair_initial',
            attempt=1,
        )
        return self._single_file_validate_repair_loop(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            current_files=current_files,
            initial_raw=raw,
            context="runtime_single_file_repair",
            runtime_failure_report=failure_report,
        )

    def _single_file_validate_repair_loop(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
        initial_raw: str,
        context: str,
        runtime_failure_report: Dict[str, Any] | None = None,
    ) -> tuple[GeneratedFile, str]:
        failures: List[Dict[str, Any]] = []
        raw = initial_raw
        last_file: GeneratedFile | None = None

        for attempt in range(1, self.max_attempts + 1):
            self._progress("Validating {} attempt {}".format(expected_path, attempt))
            self._session_set('coder', 'active_attempt', {
                'path': expected_path,
                'attempt': attempt,
                'context': context,
            })
            try:
                item = normalize_single_file(raw, expected_path)
                item = self._harden_generated_file_before_validation(
                    work_order=work_order,
                    expected_path=expected_path,
                    item=item,
                    current_files=current_files,
                )
                self._session_set('normalizer', 'last_file', {
                    'path': expected_path,
                    'content_preview': (item.content or '')[:1200],
                })
                item = self._review_against_lessons(
                    work_order=work_order,
                    environment=environment,
                    expected_path=expected_path,
                    purpose=purpose,
                    item=item,
                )
                if expected_path == 'src/api.py':
                    # Lesson review is model-driven and can reintroduce minor storage
                    # contract drift. Re-apply deterministic contract hardening before
                    # final lesson enforcement without inventing endpoints or behavior.
                    item = self._harden_generated_file_before_validation(
                        work_order=work_order,
                        expected_path=expected_path,
                        item=item,
                        current_files=current_files,
                    )
                self._enforce_lessons_before_validation(expected_path, item)
                validate_single_generated_file(item)
                self._semantic_validate_if_needed(
                    expected_path=expected_path,
                    item=item,
                    current_files=current_files,
                )
                last_file = item
                self._session_set('validator', 'last_result', {
                    'path': expected_path,
                    'status': 'passed',
                    'attempt': attempt,
                    'context': context,
                })
                self._progress("Static validation passed for {} on attempt {}".format(expected_path, attempt))
                if failures:
                    self._record_lessons_from_true_fix(
                        expected_path=expected_path,
                        failures=failures,
                        fixed_file=item,
                        context=context,
                    )
                return item, "{}_validated_on_attempt={}".format(context, attempt)
            except Exception as exc:
                self._session_set('validator', 'last_result', {
                    'path': expected_path,
                    'status': 'failed',
                    'attempt': attempt,
                    'context': context,
                    'error': str(exc),
                })
                self._session_append('validator', 'failures', {
                    'path': expected_path,
                    'attempt': attempt,
                    'error': str(exc),
                })
                self._progress("Static validation failed for {} on attempt {}: {}".format(expected_path, attempt, str(exc)))
                try:
                    last_file = normalize_single_file(raw, expected_path)
                    preview = (last_file.content or "")[:1600]
                except Exception:
                    preview = ""

                failures.append({
                    "attempt": attempt,
                    "error": str(exc),
                    "expected_path": expected_path,
                    "content_preview": preview,
                    "raw_preview": (raw or "")[:MAX_PREVIEW_CHARS],
                })

                if attempt >= self.max_attempts:
                    raise ValueError(
                        "File generation failed for {} after repair attempts.\n".format(expected_path)
                        + json.dumps({"failures": failures}, indent=2)
                        + "\n\nLAST RAW OUTPUT:\n{}".format(raw)
                    ) from exc

                self._progress("Requesting single-file repair for {}".format(expected_path))
                self._session_set('coder', 'repair_requested', {
                    'path': expected_path,
                    'attempt_after_failure': attempt,
                    'error': str(exc),
                    'current_content': last_file.content if last_file else '',
                    'current_files': current_files.to_dict(),
                })
                prompt = self._single_file_static_repair_prompt(
                    work_order=work_order,
                    environment=environment,
                    expected_path=expected_path,
                    purpose=purpose,
                    current_files=current_files,
                    previous_file=last_file,
                    validation_error=str(exc),
                    failures=failures,
                    runtime_failure_report=runtime_failure_report,
                )
                raw = self._generate_with_logged_prompt(
                    prompt,
                    path=expected_path,
                    purpose=purpose,
                    context=context + ':static_validation_repair',
                    attempt=attempt + 1,
                )

        raise RuntimeError("unreachable single-file generation loop exit")

    def _harden_generated_file_before_validation(
        self,
        work_order: WorkOrder,
        expected_path: str,
        item: GeneratedFile,
        current_files: BuildFiles,
    ) -> GeneratedFile:
        """Apply deterministic contract fixes before validation.

        Step 30: mechanically harden only src/api.py. Do not overwrite
        tests/test_api.py with a canned lifecycle template; tests must be
        generated by reading the README/API contract for this exact skill.
        """
        if expected_path == 'src/api.py':
            return self._harden_api_source_file(item)
        # Do not replace tests/test_api.py with a hardcoded lifecycle template.
        # The tester must use the already-generated README and src/api.py as
        # the behavior contract, then decide the best public-API tests for
        # this specific capability. Mechanical envelope rules are still
        # enforced by validation/repair prompts, but behavior is inferred.
        return item

    def _repair_common_api_syntax(self, source: str) -> str:
        """Repair common malformed API snippets before AST validation.

        Local models often get nested response dictionaries almost right but end
        a helper call with an extra closing brace, for example:
            return _cm_success({'ok': True, ...}}
        This pass only applies narrow, line-local repairs and never masks general
        syntax errors outside src/api.py.
        """
        if not source:
            return source
        fixed_lines = []
        changed = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('return _cm_success(') or stripped.startswith('return _cm_failure('):
                # Collapse the specific trailing "}}" typo into "})" while
                # preserving indentation and all earlier dict content.
                if line.rstrip().endswith('}}'):
                    line = line.rstrip()[:-1] + ')'
                    changed = True
                # Some generations produce "}})" for a helper call whose
                # single argument is a dict. That is one brace too many.
                if line.rstrip().endswith('}})'):
                    line = line.rstrip()[:-2] + '})'
                    changed = True
            fixed_lines.append(line)
        fixed = '\n'.join(fixed_lines).rstrip() + '\n'
        if changed:
            self._session_set('coder', 'deterministic_api_syntax_repair', {
                'path': 'src/api.py',
                'changed': True,
                'reason': 'repaired common extra-brace helper-call syntax before validation',
            })
        return fixed

    def _harden_api_source_file(self, item: GeneratedFile) -> GeneratedFile:
        source = item.content or ''
        source = self._repair_common_api_syntax(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # v44: do not invent fallback APIs. If model output remains
            # unparseable, preserve the failed candidate and let the normal
            # validation/repair loop ask the model for a full corrected file.
            self._session_set('coder', 'deterministic_api_syntax_repair', {
                'path': 'src/api.py',
                'changed': False,
                'reason': 'syntax remained invalid; no fallback API generated',
            })
            return GeneratedFile(path=item.path, content=source)

        response_keys = ['ok', 'action', 'message', 'data', 'error']

        class EnvelopeReturnTransformer(ast.NodeTransformer):
            def __init__(self):
                self.function_stack: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()
                return node

            def visit_Return(self, node: ast.Return):
                self.generic_visit(node)
                if not self.function_stack:
                    return node
                function_name = self.function_stack[-1]
                if function_name.startswith('_') or function_name == 'schema':
                    return node
                if not isinstance(node.value, ast.Dict):
                    return node
                existing = set()
                for key in node.value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        existing.add(key.value)
                defaults = {
                    'ok': ast.Constant(value=True),
                    'action': ast.Constant(value=function_name),
                    'message': ast.Constant(value=''),
                    'data': ast.Constant(value=None),
                    'error': ast.Constant(value=None),
                }
                for key in response_keys:
                    if key not in existing:
                        node.value.keys.append(ast.Constant(value=key))
                        node.value.values.append(defaults[key])
                return node

        tree = EnvelopeReturnTransformer().visit(tree)
        ast.fix_missing_locations(tree)
        hardened = ast.unparse(tree) + '\n'
        hardened = self._canonicalize_data_dir_storage(hardened)
        hardened = self._ensure_data_dir_created(hardened)
        hardened = self._apply_response_contract_helpers(hardened)
        if hardened != source:
            self._session_set('coder', 'deterministic_api_hardening', {
                'path': item.path,
                'changed': True,
                'reason': 'added response contract helpers, normalized endpoint returns, and DATA_DIR mkdir',
            })
        return GeneratedFile(path=item.path, content=hardened)

    def _canonicalize_data_dir_storage(self, source: str) -> str:
        """Normalize generated persistent storage to the named DATA_DIR contract.

        This is contract hardening, not fallback behavior: it only rewrites storage
        expressions the model already generated for the skill-local data directory.
        It does not invent endpoints or domain behavior.
        """
        if not source:
            return source
        text = source
        changed = False

        # Replace string-literal DATA_DIR misuse with the DATA_DIR variable.
        replacements = [
            ("os.path.join('DATA_DIR',", "os.path.join(DATA_DIR,"),
            ('os.path.join("DATA_DIR",', 'os.path.join(DATA_DIR,'),
            ("Path('DATA_DIR')", 'DATA_DIR'),
            ('Path("DATA_DIR")', 'DATA_DIR'),
        ]
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                changed = True

        # Convert os.path.join(DATA_DIR, name) to Path-native DATA_DIR / name.
        text2 = re.sub(r"os\.path\.join\(DATA_DIR,\s*([^\)]+)\)", r"DATA_DIR / \1", text)
        if text2 != text:
            text = text2
            changed = True

        # Replace repeated inline skill-local data path expressions with DATA_DIR.
        inline_patterns = [
            r"Path\(__file__\)\.resolve\(\)\.parent\s*/\s*['\"]data['\"]",
            r"\(Path\(__file__\)\.resolve\(\)\.parent\s*/\s*['\"]data['\"]\)",
        ]
        for pattern in inline_patterns:
            new_text = re.sub(pattern, 'DATA_DIR', text)
            if new_text != text:
                text = new_text
                changed = True

        needs_data_dir = 'DATA_DIR' in text or " / 'data'" in text or ' / "data"' in text
        if needs_data_dir and 'DATA_DIR =' not in text:
            lines = text.splitlines()
            # Ensure pathlib import exists.
            has_path_import = any(line.strip() == 'from pathlib import Path' for line in lines)
            if not has_path_import:
                insert_at = 0
                while insert_at < len(lines) and (lines[insert_at].startswith('import ') or lines[insert_at].startswith('from ')):
                    insert_at += 1
                lines.insert(insert_at, 'from pathlib import Path')
                changed = True

            # Insert SKILL_ROOT/DATA_DIR after imports.
            insert_at = 0
            while insert_at < len(lines) and (lines[insert_at].startswith('import ') or lines[insert_at].startswith('from ')):
                insert_at += 1
            while insert_at < len(lines) and not lines[insert_at].strip():
                insert_at += 1
            block = [
                'SKILL_ROOT = Path(__file__).resolve().parent',
                "DATA_DIR = SKILL_ROOT / 'data'",
            ]
            lines[insert_at:insert_at] = block + ['']
            text = '\n'.join(lines)
            changed = True

        # Prefer the canonical two-line definition if the model used an equivalent form.
        if 'DATA_DIR = Path(__file__).resolve().parent /' in text and 'SKILL_ROOT =' not in text:
            text = re.sub(
                r"DATA_DIR\s*=\s*Path\(__file__\)\.resolve\(\)\.parent\s*/\s*(['\"]data['\"])",
                "SKILL_ROOT = Path(__file__).resolve().parent\nDATA_DIR = SKILL_ROOT / \\1",
                text,
            )
            changed = True

        if changed:
            self._session_set('coder', 'deterministic_data_dir_canonicalization', {
                'path': 'src/api.py',
                'changed': True,
                'reason': 'normalized persistent storage references to the named DATA_DIR contract',
            })
        return text.rstrip() + '\n'

    def _ensure_data_dir_created(self, source: str) -> str:
        lines = source.splitlines()
        output = []
        inserted = False
        for line in lines:
            output.append(line)
            stripped = line.strip()
            if not inserted and stripped.startswith('DATA_DIR ='):
                rhs = stripped.split('=', 1)[1] if '=' in stripped else ''
                # Only add Path.mkdir for the canonical Path-based DATA_DIR contract.
                # If a model used os.path.join/os.getcwd, validator feedback should repair
                # that source instead of hardening it into a runtime error.
                if 'Path(' in rhs or 'SKILL_ROOT /' in rhs or "/ 'data'" in rhs or '/ "data"' in rhs:
                    output.append('DATA_DIR.mkdir(parents=True, exist_ok=True)')
                    inserted = True
        return '\n'.join(output).rstrip() + '\n'

    def _apply_response_contract_helpers(self, source: str) -> str:
        """Normalize public endpoint returns through response helpers.

        Step 33 replaces blanket try/except wrapping with an explicit response
        contract. Public endpoints should handle expected failures themselves
        and return through `_cm_success()` / `_cm_failure()`. This pass does not
        hide exceptions by default; it makes success/failure messages consistent
        and machine-checkable.
        """
        try:
            tree = ast.parse(source or '')
        except SyntaxError:
            return source

        helper_names = {'_cm_success', '_cm_failure'}
        existing_functions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def is_helper_call(value: ast.AST) -> bool:
            return (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in helper_names
            )

        def dict_key_values(value: ast.Dict) -> Dict[str, ast.AST]:
            out: Dict[str, ast.AST] = {}
            for key, item_value in zip(value.keys, value.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out[key.value] = item_value
            return out

        def string_or_default(node: ast.AST | None, default: str) -> ast.AST:
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
                return node
            return ast.Constant(value=default)

        def bool_literal(node: ast.AST | None) -> bool | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return node.value
            return None

        def helper_call_from_dict(function_name: str, call: ast.Call) -> ast.Return | None:
            if not call.args or not isinstance(call.args[0], ast.Dict):
                return None
            values = dict_key_values(call.args[0])
            ok_value = bool_literal(values.get('ok'))
            action_value = string_or_default(values.get('action'), function_name)
            data_value = values.get('data') or ast.Constant(value=None)
            if ok_value is False or (isinstance(call.func, ast.Name) and call.func.id == '_cm_failure'):
                message_value = string_or_default(values.get('message'), function_name + ' failed')
                error_value = values.get('error') or ast.Constant(value='operation_failed')
                return ast.Return(value=ast.Call(
                    func=ast.Name(id='_cm_failure', ctx=ast.Load()),
                    args=[action_value, message_value, error_value, data_value],
                    keywords=[],
                ))
            message_value = string_or_default(values.get('message'), function_name + ' completed')
            return ast.Return(value=ast.Call(
                func=ast.Name(id='_cm_success', ctx=ast.Load()),
                args=[action_value, message_value, data_value],
                keywords=[],
            ))

        def normalize_return(function_name: str, node: ast.Return) -> ast.Return:
            if is_helper_call(node.value):
                normalized = helper_call_from_dict(function_name, node.value)
                return normalized or node
            if not isinstance(node.value, ast.Dict):
                return node
            values = dict_key_values(node.value)
            ok_value = bool_literal(values.get('ok'))
            action_value = string_or_default(values.get('action'), function_name)
            data_value = values.get('data') or ast.Constant(value=None)
            if ok_value is False:
                message_value = string_or_default(values.get('message'), function_name + ' failed')
                error_value = values.get('error') or ast.Constant(value='operation_failed')
                return ast.Return(value=ast.Call(
                    func=ast.Name(id='_cm_failure', ctx=ast.Load()),
                    args=[action_value, message_value, error_value, data_value],
                    keywords=[],
                ))
            message_value = string_or_default(values.get('message'), function_name + ' completed')
            return ast.Return(value=ast.Call(
                func=ast.Name(id='_cm_success', ctx=ast.Load()),
                args=[action_value, message_value, data_value],
                keywords=[],
            ))

        class ResponseContractTransformer(ast.NodeTransformer):
            def __init__(self):
                self.function_stack: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()
                if node.name.startswith('_') or node.name == 'schema':
                    return node
                node.body = add_basic_parameter_failures(node.name, node.args, node.body)
                return node

            def visit_Return(self, node: ast.Return):
                self.generic_visit(node)
                if not self.function_stack:
                    return node
                function_name = self.function_stack[-1]
                if function_name.startswith('_') or function_name == 'schema':
                    return node
                return normalize_return(function_name, node)

        def add_basic_parameter_failures(function_name: str, args: ast.arguments, body: List[ast.stmt]) -> List[ast.stmt]:
            existing_source = ''.join(ast.unparse(stmt) for stmt in body[:4]) if body else ''
            checks: List[ast.stmt] = []
            for arg in args.args:
                name = arg.arg
                if name in {'self', 'cls'} or name.startswith('_'):
                    continue
                if name in existing_source and '_cm_failure' in existing_source:
                    continue
                condition = ast.BoolOp(op=ast.Or(), values=[
                    ast.Compare(left=ast.Name(id=name, ctx=ast.Load()), ops=[ast.Is()], comparators=[ast.Constant(value=None)]),
                    ast.BoolOp(op=ast.And(), values=[
                        ast.Call(func=ast.Name(id='isinstance', ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load()), ast.Name(id='str', ctx=ast.Load())], keywords=[]),
                        ast.UnaryOp(op=ast.Not(), operand=ast.Call(func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='strip', ctx=ast.Load()), args=[], keywords=[])),
                    ]),
                ])
                checks.append(ast.If(
                    test=condition,
                    body=[ast.Return(value=ast.Call(
                        func=ast.Name(id='_cm_failure', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=function_name),
                            ast.Constant(value=name + ' is required'),
                            ast.Constant(value='missing_' + name),
                            ast.Constant(value=None),
                        ],
                        keywords=[],
                    ))],
                    orelse=[],
                ))
            return checks + body

        tree = ResponseContractTransformer().visit(tree)

        # Canonicalize response helper definitions even when the model supplied
        # incomplete versions. This preserves endpoint behavior while enforcing
        # the response envelope contract; it does not invent capability logic.
        tree.body = [
            node for node in tree.body
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names)
        ]
        existing_functions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        if '_cm_success' not in existing_functions:
            success_def = ast.parse("""
def _cm_success(action, message, data=None):
    if message is None or message == '':
        message = str(action) + ' completed'
    return {'ok': True, 'action': str(action), 'message': str(message), 'data': data, 'error': None}
""").body[0]
            insert_at = 0
            while insert_at < len(tree.body) and isinstance(tree.body[insert_at], (ast.Import, ast.ImportFrom)):
                insert_at += 1
            tree.body.insert(insert_at, success_def)
        if '_cm_failure' not in existing_functions:
            failure_def = ast.parse("""
def _cm_failure(action, message, error, data=None):
    if message is None or message == '':
        message = str(action) + ' failed'
    if error is None or error == '':
        error = message
    return {'ok': False, 'action': str(action), 'message': str(message), 'data': data, 'error': str(error)}
""").body[0]
            insert_at = 0
            while insert_at < len(tree.body) and isinstance(tree.body[insert_at], (ast.Import, ast.ImportFrom)):
                insert_at += 1
            if insert_at < len(tree.body) and isinstance(tree.body[insert_at], ast.FunctionDef) and tree.body[insert_at].name == '_cm_success':
                insert_at += 1
            tree.body.insert(insert_at, failure_def)

        ast.fix_missing_locations(tree)
        return ast.unparse(tree).rstrip() + '\n'

    def _deterministic_test_file(self, work_order: WorkOrder, current_files: BuildFiles) -> GeneratedFile:
        api_file = self._find_file(current_files, 'src/api.py')
        source = api_file.content if api_file else ''
        details = self._source_contract_details(source)
        public_names = details.get('public_functions') or ['schema']
        endpoints = [name for name in public_names if name != 'schema']
        imports = ', '.join(public_names)
        if not imports:
            imports = 'schema'

        lines: List[str] = []
        lines.append('import unittest')
        lines.append('import uuid')
        lines.append('from src.api import ' + imports)
        lines.append('')
        lines.append("REQUIRED_ENVELOPE_FIELDS = {'ok', 'action', 'message', 'data', 'error'}")
        lines.append('')
        lines.append('')
        lines.append('class TestPublicApi(unittest.TestCase):')
        lines.append('    def assert_envelope(self, response):')
        lines.append('        self.assertIsInstance(response, dict)')
        lines.append('        for field in REQUIRED_ENVELOPE_FIELDS:')
        lines.append('            self.assertIn(field, response)')
        lines.append("        self.assertIsInstance(response['ok'], bool)")
        lines.append("        self.assertTrue(response['action'] is None or isinstance(response['action'], str))")
        lines.append("        self.assertTrue(response['message'] is None or isinstance(response['message'], str))")
        lines.append("        self.assertTrue(response['error'] is None or isinstance(response['error'], str))")
        lines.append('')
        lines.append('    def unique_text(self, prefix):')
        lines.append("        return prefix + '_' + uuid.uuid4().hex")
        lines.append('')
        lines.append('    def test_schema_returns_dict(self):')
        lines.append('        result = schema()')
        lines.append('        self.assertIsInstance(result, dict)')
        lines.append('')
        add_name = self._choose_endpoint(endpoints, ['add', 'create', 'save', 'set'])
        list_name = self._choose_endpoint(endpoints, ['list', 'all', 'get'])
        delete_name = self._choose_endpoint(endpoints, ['delete', 'remove', 'clear'])
        flow_endpoints = {name for name in [add_name, list_name, delete_name] if name}
        standalone_endpoints = [name for name in endpoints if name not in flow_endpoints]
        if standalone_endpoints:
            lines.append('    def test_public_endpoints_return_full_envelopes(self):')
            for endpoint in standalone_endpoints:
                args = self._sample_args_for_endpoint(details, endpoint)
                lines.append('        response = ' + endpoint + '(' + ', '.join(args) + ')')
                lines.append('        self.assert_envelope(response)')
            lines.append('')
        if add_name and list_name and delete_name:
            title_var = 'title'
            desc_var = 'description'
            lines.append('    def response_text(self, response):')
            lines.append("        return repr(response.get('data'))")
            lines.append('')
            lines.append('    def assert_list_shows_item(self, response, item_text):')
            lines.append('        self.assert_envelope(response)')
            lines.append('        self.assertIn(item_text, self.response_text(response))')
            lines.append('')
            lines.append('    def assert_list_hides_item(self, response, item_text):')
            lines.append('        self.assert_envelope(response)')
            lines.append('        self.assertNotIn(item_text, self.response_text(response))')
            lines.append('')
            lines.append('    def test_create_list_delete_flow(self):')
            lines.append("        title = self.unique_text('item')")
            lines.append("        description = self.unique_text('description')")
            add_args = self._sample_args_for_endpoint(details, add_name, title_var, desc_var)
            list_args = self._sample_args_for_endpoint(details, list_name, title_var, desc_var)
            delete_args = self._sample_args_for_endpoint(details, delete_name, title_var, desc_var)
            lines.append('        created = ' + add_name + '(' + ', '.join(add_args) + ')')
            lines.append('        self.assert_envelope(created)')
            lines.append("        self.assertTrue(created['ok'], created)")
            lines.append('        listed = ' + list_name + '(' + ', '.join(list_args) + ')')
            lines.append('        self.assert_list_shows_item(listed, title)')
            lines.append('        deleted = ' + delete_name + '(' + ', '.join(delete_args) + ')')
            lines.append('        self.assert_envelope(deleted)')
            lines.append("        self.assertTrue(deleted['ok'], deleted)")
            lines.append('        listed_after_delete = ' + list_name + '(' + ', '.join(list_args) + ')')
            lines.append('        self.assert_list_hides_item(listed_after_delete, title)')
            lines.append('')
        lines.append('')
        lines.append("if __name__ == '__main__':")
        lines.append('    unittest.main()')
        content = '\n'.join(lines) + '\n'
        self._session_set('coder', 'deterministic_test_generation', {
            'path': 'tests/test_api.py',
            'endpoints': endpoints,
            'reason': 'forced assert_envelope helper plus README-derived create-list-delete lifecycle assertions',
        })
        return GeneratedFile(path='tests/test_api.py', content=content)

    def _choose_endpoint(self, endpoints: List[str], tokens: List[str]) -> str | None:
        for token in tokens:
            for name in endpoints:
                if token in name.lower():
                    return name
        return None

    def _sample_args_for_endpoint(
        self,
        details: Dict[str, Any],
        endpoint: str,
        title_var: str = "self.unique_text('title')",
        desc_var: str = "self.unique_text('description')",
    ) -> List[str]:
        params = details.get('parameters', {}).get(endpoint) or []
        args: List[str] = []
        for index, param in enumerate(params):
            if param == 'none' or param.startswith('*'):
                continue
            low = param.lower()
            if any(token in low for token in ['title', 'name', 'id', 'key', 'reminder']):
                args.append(title_var)
            elif any(token in low for token in ['description', 'detail', 'body', 'text', 'message', 'content', 'note']):
                args.append(desc_var)
            elif any(token in low for token in ['due', 'date', 'time', 'when']):
                args.append("'2099-01-01T00:00:00'")
            elif any(token in low for token in ['count', 'limit', 'number', 'amount']):
                args.append('1')
            else:
                args.append("self.unique_text('value')")
        return args


    def _deterministic_readme_file(self, work_order: WorkOrder, current_files: BuildFiles) -> GeneratedFile:
        main_file = self._find_file(current_files, 'src/api.py')
        source = main_file.content if main_file else ''
        details = self._source_contract_details(source)
        title = (work_order.capability_name or 'Generated Capability').replace('_', ' ').title()
        public_names = details.get('public_functions') or ['schema']
        endpoint_names = [name for name in public_names if name != 'schema']
        if not endpoint_names:
            endpoint_names = ['public endpoint functions']

        lines = []
        lines.append('# ' + title)
        lines.append('')
        lines.append('## Purpose')
        lines.append('This API-first skill package implements the requested capability: ' + work_order.goal.strip())
        lines.append('It provides a small Python API in `src/api.py` so callers and tests can create, inspect, and remove capability data without using private helpers.')
        lines.append('')
        lines.append('## Usage')
        lines.append('Import the public functions from `src.api` and call them directly from Python.')
        lines.append('Call `schema()` first to discover available endpoint functions, parameters, response fields, and storage notes.')
        lines.append('Each endpoint returns a JSON-like dict response envelope and does not require command-line interaction.')
        lines.append('')
        lines.append('## Public API')
        for name in public_names:
            sig = details.get('signatures', {}).get(name, name + '()')
            if name == 'schema':
                lines.append('- `' + sig + '`: returns a dict describing the skill, endpoints, parameters, response envelope, and storage behavior.')
            else:
                lines.append('- `' + sig + '`: public API endpoint/action function. It returns a dict envelope with `ok`, `action`, `message`, `data`, and `error`.')
        lines.append('External callers and tests should use only these public API functions.')
        lines.append('')
        lines.append('## Function Definitions')
        for name in details.get('all_functions') or public_names:
            sig = details.get('signatures', {}).get(name, name + '()')
            bare = name.split('.')[-1]
            params = details.get('parameters', {}).get(name) or ['none']
            called_by = details.get('called_by', {}).get(bare) or ['external callers/tests or none']
            calls = details.get('calls', {}).get(name) or ['none']
            lines.append('- `' + sig + '`: function or method defined in `src/api.py` for the capability implementation.')
            lines.append('  Parameters: ' + ', '.join(params) + '.')
            if bare == 'schema':
                lines.append('  Returns/Outputs: returns a plain dict schema for the API-first package.')
            elif name in public_names or bare in public_names:
                lines.append('  Returns/Outputs: returns the standard dict response envelope with `ok`, `action`, `message`, `data`, and `error`.')
            else:
                lines.append('  Returns/Outputs: returns implementation data used by public endpoints, or `None` for initialization helpers.')
            lines.append('  Called by: ' + ', '.join(called_by) + '.')
            lines.append('  Calls: ' + ', '.join(calls) + '.')
        lines.append('')
        lines.append('## Imported Dependencies')
        imports = details.get('imports') or ['none']
        for name in imports:
            if name == 'none':
                lines.append('- `none`: `src/api.py` uses no imported modules.')
            else:
                lines.append('- `' + name + '`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.')
        lines.append('')
        lines.append('## Outputs and Return Values')
        lines.append('`schema()` returns a dict that documents the package name, endpoints, parameter expectations, response envelope, and storage notes.')
        for name in endpoint_names:
            lines.append('`' + name + '()` returns `{ "ok": bool, "action": str, "message": str, "data": dict/list/null, "error": str/null }`.')
        lines.append('Successful endpoint responses set `ok` to `True`, include an action name, a human-readable message, useful `data`, and `error` as `None`.')
        lines.append('Failed endpoint responses set `ok` to `False`, include the attempted action, a failure message, `data` as `None` or partial context, and an explanatory `error` string.')
        lines.append('')
        lines.append('## Failure Modes')
        lines.append('Invalid parameters, missing records, malformed identifiers, unavailable storage, or file I/O errors are reported through the response envelope rather than uncaught exceptions during normal endpoint use.')
        lines.append('Callers should check `ok` before using `data`, and should read `error` when `ok` is `False`.')
        lines.append('')
        lines.append('## Data Storage')
        lines.append('The skill uses native storage relative to the skill file location. Persistent data belongs under `src/data` through `SKILL_ROOT = Path(__file__).resolve().parent` and `DATA_DIR = SKILL_ROOT / "data"` when storage is needed.')
        lines.append('Storage must not depend on the current working directory, `/tmp`, `tempfile`, or a test-supplied override.')
        lines.append('')
        lines.append('## Cleanup / Delete Behavior')
        lines.append('Cleanup and back-out behavior is exposed through public delete/remove/cleanup API endpoints when the capability creates persistent entries.')
        lines.append('Tests and callers should create entries through the public add/create endpoint, verify they are visible through list/read endpoints, call the public delete/remove endpoint, and verify the entry is gone.')
        lines.append('')
        lines.append('## Behavioral Verification Contract')
        lines.append('For add/create + list/read + delete/remove style capabilities, tests must verify the full lifecycle described by the README and public API: create an item, list items and assert the created item is visible, delete/remove the item, then list again and assert the item is no longer visible.')
        lines.append('A test that only checks the response envelope is not sufficient for stateful capabilities.')
        lines.append('')
        lines.append('## Test Coverage')
        lines.append('Tests verify `schema()` documents the API-first package, every public endpoint can be called from `src.api`, and every endpoint response contains `ok`, `action`, `message`, `data`, and `error`.')
        lines.append('Tests verify add behavior, list behavior, delete/cleanup behavior, native DATA_DIR storage under the skill file location, and failure/empty-state behavior when applicable.')
        lines.append('Tests use only the public API and do not override storage paths or manually clean normal capability data except as emergency cleanup after the delete API has been asserted.')
        return GeneratedFile(path='artifacts/README.md', content='\n'.join(lines) + '\n')

    # Trigger word groups: maps token found in endpoint name → natural-language phrases.
    _TRIGGER_MAP = {
        'create': ['create {noun}', 'add {noun}', 'new {noun}'],
        'add':    ['add {noun}', 'create {noun}', 'new {noun}'],
        'save':   ['save {noun}', 'store {noun}'],
        'set':    ['set {noun}', 'update {noun}'],
        'update': ['update {noun}', 'edit {noun}', 'change {noun}'],
        'edit':   ['edit {noun}', 'update {noun}'],
        'delete': ['delete {noun}', 'remove {noun}'],
        'remove': ['remove {noun}', 'delete {noun}'],
        'clear':  ['clear {noun}'],
        'list':   ['list {noun}', 'show {noun}', 'my {noun}'],
        'show':   ['show {noun}', 'list {noun}'],
        'get':    ['get {noun}', 'show {noun}'],
        'check':  ['check {noun}', '{noun} status'],
        'status': ['{noun} status', 'check {noun}'],
        'search': ['search {noun}', 'find {noun}'],
        'find':   ['find {noun}', 'search {noun}'],
        'fetch':  ['fetch {noun}', 'get {noun}'],
        'run':    ['run {noun}'],
        'start':  ['start {noun}'],
        'stop':   ['stop {noun}'],
        'read':   ['read {noun}'],
        'write':  ['write {noun}'],
        'send':   ['send {noun}'],
        'load':   ['load {noun}'],
        'count':  ['count {noun}', 'how many {noun}'],
        'reset':  ['reset {noun}'],
    }

    _BACKGROUND_KEYWORDS = {
        'notification', 'notify', 'reminder', 'remind', 'schedule', 'scheduled',
        'timer', 'periodic', 'interval', 'watch', 'monitor', 'alert', 'due',
        'background', 'daemon', 'recurring', 'cron', 'polling', 'poll',
    }

    def _deterministic_commands_json(self, work_order: WorkOrder, current_files: BuildFiles) -> GeneratedFile:
        """Generate commands.json from the api.py AST — no LLM needed."""
        api_file = self._find_file(current_files, 'src/api.py')
        source = api_file.content if api_file else ''
        details = self._source_contract_details(source)
        public_names = [n for n in (details.get('public_functions') or []) if n != 'schema']

        capability_name = work_order.capability_name or 'generated_capability'
        goal_lower = (work_order.goal or '').lower()

        commands = []
        for fn_name in public_names:
            params = details.get('parameters', {}).get(fn_name) or []
            real_params = [p for p in params if p not in ('none',) and not p.startswith('*')]
            triggers = self._triggers_for_endpoint(fn_name, capability_name)
            if not triggers:
                triggers = [fn_name.replace('_', ' ')]
            args_spec = {}
            if real_params:
                # First param is usually the key identifier; include it in pattern
                triggers = [t + ' {' + real_params[0] + '}' if '{' not in t else t
                            for t in triggers[:2]]
                for p in real_params:
                    args_spec[p] = 'string'
            commands.append({
                'triggers': triggers,
                'action': fn_name,
                'args': args_spec,
                'description': fn_name.replace('_', ' '),
            })

        background_enabled = any(kw in goal_lower for kw in self._BACKGROUND_KEYWORDS)

        data = {
            'capability': capability_name,
            'description': (work_order.goal or '')[:200],
            'commands': commands,
            'background': {
                'enabled': background_enabled,
                'script': 'background.py' if background_enabled else None,
                'description': 'Runs background tasks for this capability.' if background_enabled else None,
            },
        }
        return GeneratedFile(path='commands.json', content=json.dumps(data, indent=2) + '\n')

    def _triggers_for_endpoint(self, fn_name: str, capability_name: str) -> List[str]:
        """Derive natural-language trigger phrases from an endpoint function name."""
        parts = fn_name.split('_')
        verb = parts[0] if parts else ''
        noun_parts = parts[1:]
        noun = ' '.join(noun_parts) if noun_parts else capability_name.replace('_', ' ')

        templates = self._TRIGGER_MAP.get(verb, [])
        triggers = []
        seen_words: set = set()
        for t in templates:
            phrase = t.replace('{noun}', noun)
            words = phrase.split()
            # Skip phrases with repeated words (e.g. "status status")
            if len(words) != len(set(words)):
                continue
            if phrase not in seen_words:
                seen_words.add(phrase)
                triggers.append(phrase)

        if not triggers:
            triggers = [fn_name.replace('_', ' ')]

        return triggers[:3]

    def _schema_endpoint_names_from_tree(self, tree: ast.AST) -> List[str]:
        """Extract schema()["endpoints"] keys even when schema contains f-strings.

        ast.literal_eval() fails on the whole schema dict if any unrelated value
        is dynamic, such as storage_notes: f"...{DATA_DIR}...". For README and
        test prompts we only need the endpoint-key contract, so inspect the AST
        shape directly and avoid evaluating the entire schema object.
        """
        schema_node = None
        for node in getattr(tree, 'body', []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'schema':
                schema_node = node
                break
        if schema_node is None:
            return []
        for child in ast.walk(schema_node):
            if not isinstance(child, ast.Return) or child.value is None:
                continue
            value = child.value
            if not isinstance(value, ast.Dict):
                continue
            for key_node, val_node in zip(value.keys, value.values):
                key_value = None
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    key_value = key_node.value
                if key_value != 'endpoints':
                    continue
                if isinstance(val_node, ast.Dict):
                    names = []
                    for endpoint_key in val_node.keys:
                        if isinstance(endpoint_key, ast.Constant) and isinstance(endpoint_key.value, str):
                            names.append(endpoint_key.value)
                    return sorted({name for name in names if name.isidentifier()})
                try:
                    endpoints = ast.literal_eval(val_node)
                except Exception:
                    return []
                if isinstance(endpoints, dict):
                    return sorted({str(k) for k in endpoints.keys() if isinstance(k, str) and str(k).isidentifier()})
                if isinstance(endpoints, (list, tuple, set)):
                    return sorted({str(x) for x in endpoints if isinstance(x, str) and str(x).isidentifier()})
        return []

    def _source_contract_details(self, source: str) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            'all_functions': [],
            'public_functions': [],
            'signatures': {},
            'parameters': {},
            'imports': [],
            'calls': {},
            'called_by': {},
        }
        if not source.strip():
            return details
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return details

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = (alias.name or '').split('.')[0]
                    if root:
                        imports.add(root)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                for alias in node.names:
                    if alias.name and alias.name != '*':
                        imports.add(alias.name)
        details['imports'] = sorted(imports)

        all_bare = set()
        class_stack: List[str] = []

        def arg_names(node: ast.AST) -> List[str]:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return []
            names = []
            for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                if arg.arg != 'self':
                    names.append(arg.arg)
            if node.args.vararg:
                names.append('*' + node.args.vararg.arg)
            if node.args.kwarg:
                names.append('**' + node.args.kwarg.arg)
            return names

        def signature(name: str, node: ast.AST) -> str:
            params = arg_names(node)
            return name + '(' + ', '.join(params) + ')'

        class DefVisitor(ast.NodeVisitor):
            def visit_ClassDef(visitor_self, node: ast.ClassDef) -> None:
                class_stack.append(node.name)
                visitor_self.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(visitor_self, node: ast.FunctionDef) -> None:
                qualified = class_stack[-1] + '.' + node.name if class_stack else node.name
                details['all_functions'].append(qualified)
                details['signatures'][qualified] = signature(qualified, node)
                details['parameters'][qualified] = arg_names(node) or ['none']
                all_bare.add(node.name)
                if not class_stack and node.name == 'schema':
                    details['public_functions'].append(node.name)
                    details['signatures'][node.name] = signature(node.name, node)
                    details['parameters'][node.name] = arg_names(node) or ['none']
                visitor_self.generic_visit(node)

            def visit_AsyncFunctionDef(visitor_self, node: ast.AsyncFunctionDef) -> None:
                visitor_self.visit_FunctionDef(node)

        DefVisitor().visit(tree)

        current: List[str] = []
        class_stack = []
        calls_by_qualified: Dict[str, set] = {}
        class CallVisitor(ast.NodeVisitor):
            def visit_ClassDef(visitor_self, node: ast.ClassDef) -> None:
                class_stack.append(node.name)
                visitor_self.generic_visit(node)
                class_stack.pop()

            def visit_FunctionDef(visitor_self, node: ast.FunctionDef) -> None:
                qualified = class_stack[-1] + '.' + node.name if class_stack else node.name
                current.append(qualified)
                calls_by_qualified.setdefault(qualified, set())
                visitor_self.generic_visit(node)
                current.pop()

            def visit_AsyncFunctionDef(visitor_self, node: ast.AsyncFunctionDef) -> None:
                visitor_self.visit_FunctionDef(node)

            def visit_Call(visitor_self, node: ast.Call) -> None:
                if current:
                    callee = None
                    if isinstance(node.func, ast.Name):
                        callee = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        callee = node.func.attr
                    if callee and callee in all_bare:
                        calls_by_qualified.setdefault(current[-1], set()).add(callee)
                visitor_self.generic_visit(node)

        CallVisitor().visit(tree)
        details['all_functions'] = sorted(set(details['all_functions']))
        schema_endpoints = self._schema_endpoint_names_from_tree(tree)
        top_level = {name.split('.')[-1] for name in details['all_functions'] if '.' not in name}
        for endpoint in schema_endpoints:
            if endpoint in top_level:
                details['public_functions'].append(endpoint)
        details['public_functions'] = sorted(set(details['public_functions']))
        details['calls'] = {k: sorted(v) for k, v in calls_by_qualified.items()}
        called_by: Dict[str, set] = {}
        for caller, callees in calls_by_qualified.items():
            for callee in callees:
                called_by.setdefault(callee, set()).add(caller)
        details['called_by'] = {k: sorted(v) for k, v in called_by.items()}
        return details

    def _semantic_validate_if_needed(
        self,
        expected_path: str,
        item: GeneratedFile,
        current_files: BuildFiles,
    ) -> None:
        # README has its own semantic validator. tests/test_api.py needs a
        # partial package-level contract check as soon as it is generated so
        # missing envelope assertions are repairable in the per-file loop
        # instead of surfacing only after all files are generated.
        if expected_path == 'artifacts/README.md':
            main_file = self._find_file(current_files, 'src/api.py')
            if not main_file:
                return
            tests_file = self._find_file(current_files, 'tests/test_api.py')
            self._progress('Semantic README validation for artifacts/README.md')
            self._session_set('validator', 'semantic_review', {
                'path': expected_path,
                'status': 'running',
            })
            validate_readme_semantics(
                readme_content=item.content or '',
                main_content=main_file.content or '',
                tests_content=(tests_file.content if tests_file else '') or '',
                model=self.model,
            )
            self._session_set('validator', 'semantic_review', {
                'path': expected_path,
                'status': 'passed',
            })
            return

        if expected_path == 'tests/test_api.py':
            main_file = self._find_file(current_files, 'src/api.py')
            readme_file = self._find_file(current_files, 'artifacts/README.md')
            if not main_file or not readme_file:
                return
            self._progress('Package contract validation for tests/test_api.py')
            self._session_set('validator', 'test_contract_review', {
                'path': expected_path,
                'status': 'running',
            })
            package_files_by_path = {existing.path: existing for existing in current_files.files}
            package_files_by_path[item.path] = item
            ordered_package_files = []
            for existing in list(current_files.files) + [item]:
                if existing.path in [seen_item.path for seen_item in ordered_package_files]:
                    continue
                ordered_package_files.append(package_files_by_path[existing.path])
            validate_build_files(BuildFiles(
                files=ordered_package_files,
                notes='partial_package_contract_validation',
            ))
            self._session_set('validator', 'test_contract_review', {
                'path': expected_path,
                'status': 'passed',
            })


    def _sanitize_prompt_text(self, text: str, limit: int | None = None) -> str:
        """Keep prompts useful without re-seeding known bad model artifacts."""
        text = text or ''
        bad_fragments = [
            '<｜begin▁of▁sentence｜>',
            '｜begin▁of▁sentence｜',
            '<|begin_of_sentence|>',
            '<｜end▁of▁sentence｜>',
            '｜end▁of▁sentence｜',
            '<|end_of_sentence|>',
        ]
        for frag in bad_fragments:
            text = text.replace(frag, '')
        text = text.replace('```python', '').replace('```', '')
        if limit is not None and len(text) > limit:
            return text[:limit] + '\n...[truncated for prompt brevity]...'
        return text

    def _compact_lessons_for_prompt(self, expected_path: str) -> str:
        lessons = self._lessons_for_path(expected_path)
        if not lessons:
            return 'None'
        compact = []
        for lesson in lessons[:8]:
            compact.append({
                'scope': lesson.get('scope'),
                'lesson': lesson.get('lesson'),
                'example_error': lesson.get('example_error'),
            })
        return json.dumps(compact, indent=2)

    def _test_skeleton_for_prompt(self, current_files: BuildFiles) -> str:
        main_file = self._find_file(current_files, 'src/api.py')
        if not main_file:
            return ''
        try:
            tree = ast.parse(main_file.content or '')
            endpoints = self._schema_endpoint_names_from_tree(tree)
        except Exception:
            endpoints = []
        imports = ['schema'] + [name for name in endpoints if name != 'schema']
        import_line = 'from src.api import ' + ', '.join(imports or ['schema'])
        return f"""Concrete unittest shape to fill, without changing framework:

import unittest
{import_line}

TEST_PLAN = "Exercise schema and every schema-declared endpoint through the public Python API."

class TestPublicApi(unittest.TestCase):
    def test_schema_contract(self):
        s = schema()
        self.assertIsInstance(s, dict)
        self.assertIn("endpoints", s)

    def test_public_api_lifecycle(self):
        # Call the public endpoints listed in schema()["endpoints"].
        # For every endpoint response, inspect ok, action, message, and data.
        pass

if __name__ == "__main__":
    unittest.main()

Replace the placeholder body with real calls and assertions. Do not keep pass."""

    def _context_files_for_prompt(self, current_files: BuildFiles, expected_path: str, limit: int = 12000) -> str:
        if not current_files.files:
            return ''
        selected = []
        for item in current_files.files:
            if expected_path == 'tests/test_api.py' and item.path not in {'src/api.py', 'artifacts/README.md'}:
                continue
            selected.append(item)
        if not selected:
            selected = list(current_files.files)
        text = self._full_files_for_prompt(BuildFiles(files=selected, notes=current_files.notes))
        return self._sanitize_prompt_text(text, limit=limit)

    def _single_file_prompt(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
    ) -> str:
        extra = ""
        if current_files.files:
            extra = "\nFiles already generated and validated:\n{}\n".format(
                self._full_files_for_prompt(current_files)
            )
        return self._base_single_file_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            heading="Generate exactly one required file.",
            extra=extra,
        )

    def _single_file_static_repair_prompt(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
        previous_file: GeneratedFile | None,
        validation_error: str,
        failures: List[Dict[str, Any]],
        runtime_failure_report: Dict[str, Any] | None = None,
    ) -> str:
        previous_text = self._file_for_prompt(previous_file) if previous_file else "No usable previous file content."
        previous = self._sanitize_prompt_text(previous_text, limit=9000)
        recent = [
            {
                "attempt": f.get("attempt"),
                "error": f.get("error"),
            }
            for f in failures[-3:]
        ]
        context = self._context_files_for_prompt(current_files, expected_path)
        focused_fix = self._focused_repair_instruction(expected_path, validation_error)
        extra_parts = [
            'Validation issues to fix together:\n' + self._sanitize_prompt_text(validation_error),
            focused_fix,
            'Previous/current file to edit:\n' + previous,
        ]
        if recent:
            extra_parts.append('Recent error history, errors only:\n' + json.dumps(recent, indent=2))
        if context:
            extra_parts.append('Other validated files for contract context:\n' + context)
        if runtime_failure_report:
            extra_parts.append('Runtime failure report:\n' + self._sanitize_prompt_text(json.dumps(runtime_failure_report, indent=2), limit=4000))
        if expected_path == 'tests/test_api.py':
            skeleton = self._test_skeleton_for_prompt(current_files)
            if skeleton:
                extra_parts.append(skeleton)
        extra_parts.append(f"""Repair mode:
- Edit the current file; do not rewrite from a different framework.
- Return exactly one FILE block for {expected_path}.
- Keep valid code; change only what is needed to satisfy all listed validation issues and the file-specific rules.
- No markdown fences, no prose, no special tokens.""")
        extra = '\n\n'.join(part for part in extra_parts if part)
        return self._base_single_file_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            heading="Repair exactly one generated file.",
            extra=extra,
        )

    def _single_file_runtime_repair_prompt(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        current_files: BuildFiles,
        existing_file: GeneratedFile | None,
        failure_report: Dict[str, Any],
    ) -> str:
        current = self._file_for_prompt(existing_file) if existing_file else "Missing file."
        extra = f"""
The complete set of files passed static validation but failed runtime tests.

Runtime/test failure report:
{json.dumps(failure_report, indent=2)}

Current version of the requested file:
{current}

All current files for context:
{self._full_files_for_prompt(current_files)}

RUNTIME REPAIR TASK:
- Return ONLY one FILE block for {expected_path}.
- The first non-empty line of your response must be: FILE: {expected_path}
- Treat the current file as the authoritative base. EDIT it; do not discard working code.
- Preserve every valid function, schema entry, import, constant, and response helper unless it directly causes the runtime failure.
- Fix the runtime failure AND keep all artifact-specific rules satisfied at the same time.
- Keep the public API, README, and tests consistent.
- Do not weaken tests to hide a real behavior mismatch.
- Do not return commentary, markdown fences, patches, or any other file.
"""
        return self._base_single_file_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            heading="Repair exactly one file after runtime verification failed.",
            extra=extra,
        )


    def _focused_repair_instruction(self, expected_path: str, error: str) -> str:
        err = (error or '').lower()
        if expected_path == 'tests/test_api.py':
            if 'unittest' in err:
                return """Focused fix:
Use unittest only. Required shape:
import unittest
from src.api import schema, <endpoints>
class TestPublicApi(unittest.TestCase): ...
if __name__ == "__main__": unittest.main()
Do not import pytest. Do not use top-level pytest-style test functions."""
            if 'response field' in err or 'ok' in err or 'message' in err or 'data' in err:
                return """Focused fix:
Fix ALL missing response-field inspections in one edit.
Every endpoint response tested must inspect the standard envelope fields: "ok", "action", "message", and "data".
Prefer a helper like _assert_response_envelope(response) that checks all four fields, then call it for each endpoint response.
Do not rewrite unrelated test structure."""
            if 'invalid character' in err or 'syntax error' in err:
                return """Focused fix:
Remove malformed/special-token text and return clean Python only. Do not copy corrupted fragments from the previous file."""
            if 'from src.api' in err or 'import' in err:
                return """Focused fix:
Use exactly this import style: from src.api import schema, <schema-declared endpoints>."""
        if expected_path == 'src/api.py':
            if 'data_dir' in err or 'storage' in err:
                return """Focused fix:
Define SKILL_ROOT = Path(__file__).resolve().parent and DATA_DIR = SKILL_ROOT / "data" once near the top. Use DATA_DIR / filename for persistent files."""
            if 'schema' in err:
                return """Focused fix:
Keep all endpoint implementations and add/repair top-level def schema(): returning a dict with an endpoints mapping."""
        if expected_path == 'artifacts/README.md':
            return 'Focused fix:\nMake the README accurately match src/api.py and schema()["endpoints"].'
        return 'Focused fix:\nFix the latest validation error without changing unrelated valid behavior.'

    def _artifact_rules_for_prompt(self, expected_path: str) -> str:
        if expected_path == 'src/api.py':
            return """API rules:
- Python-callable module only: no HTTP frameworks, route decorators, classes for public API, or handlers.
- Top-level functions only: _cm_success, _cm_failure, every public endpoint, and schema().
- Every endpoint returns _cm_success(...) or _cm_failure(...), each with ok/action/message/data/error.
- schema() returns a dict with package, endpoints, response_format, storage_notes, notes.
- schema()["endpoints"] maps exact top-level endpoint function names to {"args": [...], "returns": "dict"}.
- If persistent storage is needed, define once: SKILL_ROOT = Path(__file__).resolve().parent; DATA_DIR = SKILL_ROOT / "data"; DATA_DIR.mkdir(...). Store files under DATA_DIR.
- Prefer boring, explicit standard-library code over clever abstractions.
"""
        if expected_path == 'tests/test_api.py':
            return """Test rules:
- Use unittest only. No pytest import, pytest.mark, fixtures, bare assert-only style, or top-level test functions.
- Import only public API: from src.api import schema, <schema-declared endpoints>.
- Define class TestPublicApi(unittest.TestCase).
- Call schema(), inspect it flexibly, and call every schema-declared endpoint at least once.
- For endpoint responses, inspect ok, action, message, and data.
- Derive endpoint names, args, response shapes, and lifecycle from src/api.py and README context. Do not invent exact messages.
- For stateful APIs, create through public API, observe through public API, delete/cleanup through public API, then observe it is gone.
"""
        if expected_path == 'artifacts/README.md':
            return """README rules:
- Markdown only. Start with # title.
- Include exact sections: Purpose, Usage, Public API, Function Definitions, Imported Dependencies, Outputs and Return Values, Failure Modes, Data Storage, Cleanup / Delete Behavior, Test Coverage.
- Public API must list schema() and only schema-declared endpoint functions.
- Function Definitions may include helpers/internal functions.
- Accurately describe src/api.py, schema(), response data, runtime storage, and cleanup behavior.
"""
        if expected_path == 'src/background.py':
            return """Background service rules:
- This is a standalone long-running Python script, NOT imported as a module.
- It must import the public functions from src.api (e.g. from api import list_reminders).
- The main logic is a while True loop that sleeps between iterations (30-60 seconds).
- It should check for due items (by time, expiry, etc.) and act on them.
- To fire a brain notification, write a JSON file to ../../../data/bg_notifications/ named {timestamp}.json with {"capability": name, "message": str, "level": "info"|"success"|"warning"|"error"}.
- Use only the Python standard library.
- Handle exceptions inside the loop so a single failure does not crash the service.
- End the file with: if __name__ == "__main__": run()
- The file runs from its own directory (src/), so relative imports work.
"""
        return """General rules:
- Generate the requested file only and keep it consistent with the work order and existing files.
"""

    def _base_single_file_prompt(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        heading: str,
        extra: str,
    ) -> str:
        content_type = 'text/markdown' if expected_path.endswith('.md') else 'text/x-python'
        lessons_text = self._compact_lessons_for_prompt(expected_path)
        artifact_rules = self._artifact_rules_for_prompt(expected_path)
        return f"""
You are code_monkey. {heading}

Goal: {work_order.goal}
File: {expected_path}
Purpose: {purpose}
Run target after all files are valid: {work_order.test_command}

Output exactly this envelope and nothing else:
FILE: {expected_path}
CONTENT_TYPE: {content_type}
---BEGIN CONTENT---
<complete raw file content>
---END CONTENT---

Non-negotiable output rules:
- One complete file only; no markdown fences, prose, snippets, diffs, or patches.
- No placeholders, TODOs, pass-only stubs, examples-as-implementation, or special/control tokens.
- Python must parse with ast.parse.

File-specific instructions:
{artifact_rules}

Relevant lessons, if any:
{lessons_text}

Context:
{extra}
""".strip()


    def _review_against_lessons(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        item: GeneratedFile,
    ) -> GeneratedFile:
        lessons = self._lessons_for_path(expected_path)
        if not lessons:
            return item
        self._progress("Reviewing {} against {} lesson(s)".format(expected_path, len(lessons)))
        prompt = self._lesson_review_prompt(
            work_order=work_order,
            environment=environment,
            expected_path=expected_path,
            purpose=purpose,
            item=item,
            lessons=lessons,
        )
        try:
            raw = self._generate_with_logged_prompt(
                prompt,
                path=expected_path,
                purpose=purpose,
                context='lesson_review',
                attempt=None,
            )
            reviewed = normalize_single_file(raw, expected_path)
            if reviewed and reviewed.content and reviewed.content.strip():
                return reviewed
        except Exception as exc:
            self._progress("Lesson review failed for {}; keeping original: {}".format(expected_path, exc))
        return item

    def _lessons_for_path(self, expected_path: str) -> List[Dict[str, Any]]:
        if not self.lesson_provider:
            return []
        lessons: List[Dict[str, Any]] = []
        try:
            lessons.extend(self.lesson_provider(expected_path, 25))
        except Exception:
            pass
        try:
            lessons.extend(self.lesson_provider('global', 25))
        except Exception:
            pass
        seen = set()
        unique = []
        for lesson in lessons:
            key = lesson.get('id') or (lesson.get('scope'), lesson.get('failure_signature'))
            if key in seen:
                continue
            seen.add(key)
            unique.append(lesson)
        return unique[:40]


    def _enforce_lessons_before_validation(self, expected_path: str, item: GeneratedFile) -> None:
        lessons = self._lessons_for_path(expected_path)
        if not lessons:
            return
        violations = enforce_lessons_on_code(expected_path, item.content or '', lessons)
        if violations:
            raise ValueError('\n'.join(violations))

    def _lesson_review_prompt(
        self,
        work_order: WorkOrder,
        environment: Dict[str, Any],
        expected_path: str,
        purpose: str,
        item: GeneratedFile,
        lessons: List[Dict[str, Any]],
    ) -> str:
        content_type = 'text/markdown' if expected_path.endswith('.md') else 'text/x-python'
        return f"""
You are code_monkey's lesson reviewer.
Review exactly one generated file against past lessons learned.

Goal:
{work_order.goal}

Requested file path:
{expected_path}

Requested file purpose:
{purpose}

Environment snapshot:
{json.dumps(environment, indent=2)}

Lessons learned from previous true fixes:
{json.dumps(lessons, indent=2)}

Current generated file:
{self._file_for_prompt(item)}

Task:
- Decide whether the current file repeats any lesson.
- If it repeats a lesson, correct the file.
- If no lesson applies, return the file unchanged.
- Return ONLY one FILE block for {expected_path}.
- Do not include explanations before or after.

Output format:
FILE: {expected_path}
CONTENT_TYPE: {content_type}
---BEGIN CONTENT---
raw complete corrected file content here
---END CONTENT---
""".strip()

    def _record_lessons_from_true_fix(
        self,
        expected_path: str,
        failures: List[Dict[str, Any]],
        fixed_file: GeneratedFile,
        context: str,
    ) -> None:
        if not self.lesson_recorder:
            return
        recorded = set()
        for failure in failures:
            error = str(failure.get('error') or '').strip()
            if not error:
                continue
            extracted = extract_static_lessons(expected_path, error)
            if extracted:
                lesson_items = extracted
            else:
                lesson_items = [{
                    'scope': expected_path,
                    'failure_signature': self._failure_signature(error),
                    'lesson': self._lesson_text_from_error(expected_path, error),
                    'example_error': error,
                }]
            for lesson_item in lesson_items:
                signature = lesson_item.get('failure_signature') or self._failure_signature(error)
                scope = lesson_item.get('scope') or expected_path
                key = (scope, signature)
                if key in recorded:
                    continue
                recorded.add(key)
                lesson = lesson_item.get('lesson') or self._lesson_text_from_error(expected_path, error)
                try:
                    self.lesson_recorder(
                        scope,
                        signature,
                        lesson,
                        lesson_item.get('example_error') or error,
                        "fixed_during=" + context + "\n" + (fixed_file.content or '')[:2000],
                    )
                    self._progress("Recorded lesson for {}: {}".format(scope, signature))
                except Exception as exc:
                    self._progress("Could not record lesson for {}: {}".format(scope, exc))

    def _failure_signature(self, error: str) -> str:
        return normalize_failure_signature(error)[:300]

    def _lesson_text_from_error(self, path: str, error: str) -> str:
        low = error.lower()
        if 'context-manager' in low or 'with-statement' in low or 'with ' in low:
            return (
                "For " + path + ", avoid context-manager open statements when the "
                "validator rejects them. Use explicit handle = open(...), try/finally, "
                "and handle.close(), or let the sanitizer convert the pattern before validation."
            )
        if 'formatted string' in low or 'f-string' in low:
            return (
                "For " + path + ", avoid f-strings after this failure. Build text with "
                "plain concatenation or simple variables before writing or printing."
            )
        if 'split an expression' in low or 'expected' in low or 'syntax error' in low:
            return (
                "For " + path + ", avoid multiline expressions and nested calls. "
                "Assign intermediate variables and keep each Python statement structurally complete."
            )
        if 'placeholder' in low or 'stub' in low or 'pass-only' in low:
            return (
                "For " + path + ", do not use placeholders or pass-only stubs. "
                "Implement real behavior and tests before validation."
            )
        if 'tests/test_api.py' in path or path.startswith('tests/'):
            return (
                "For " + path + ", tests must import required modules, create a unique "
                "entry, verify it, call delete/remove/cleanup, and verify cleanup."
            )
        return "For " + path + ", avoid repeating this fixed failure: " + error[:500]

    def _find_file(self, build_files: BuildFiles, path: str) -> GeneratedFile | None:
        for item in build_files.files:
            if item.path == path:
                return item
        return None

    def _file_for_prompt(self, item: GeneratedFile | None) -> str:
        if not item:
            return "No file."
        return "FILE: {}\n---BEGIN CONTENT---\n{}\n---END CONTENT---".format(
            item.path,
            item.content,
        )

    def _full_files_for_prompt(self, build_files: BuildFiles | None) -> str:
        if not build_files or not build_files.files:
            return "No files."
        parts = [self._file_for_prompt(item) for item in build_files.files]
        return "\n\n".join(parts)
