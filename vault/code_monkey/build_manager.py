import json
from pathlib import Path
from typing import Any, Dict, List

from .coder import Coder
from .environment import snapshot
from .planner import Planner
from .runner import run_verification_commands
from .schemas import BuildFiles, WorkOrder
from .storage import Storage
from .validator import validate_build_files, validate_workspace_write
from .lesson_engine import extract_runtime_lessons, normalize_failure_signature, failure_text
from .workspace import create_workspace, ensure_workspace, read_json, write_json


MAX_RUNTIME_REPAIR_ATTEMPTS = 5


class BuildManager:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.storage = Storage(progress=self._emit if verbose else None)
        self.planner = Planner()
        self.coder = Coder(
            progress=self._emit if verbose else None,
            lesson_provider=self.storage.list_lessons,
            lesson_recorder=self.storage.record_lesson,
        )

    def _session_set(self, task_id: str, component: str, key: str, value: Any) -> None:
        self.storage.blackboard_set(task_id, component, key, value)
        try:
            root = ensure_workspace(task_id)
            write_json(root / 'session.json', self.storage.blackboard_snapshot(task_id))
        except Exception:
            pass

    def _session_append(self, task_id: str, component: str, key: str, value: Any, limit: int = 100) -> None:
        self.storage.blackboard_append(task_id, component, key, value, limit=limit)
        try:
            root = ensure_workspace(task_id)
            write_json(root / 'session.json', self.storage.blackboard_snapshot(task_id))
        except Exception:
            pass

    def _bind_coder_session(self, task_id: str) -> None:
        self.coder.session_writer = lambda component, key, value: self._session_set(task_id, component, key, value)
        self.coder.session_appender = lambda component, key, value: self._session_append(task_id, component, key, value)

    def _emit(self, message: str) -> None:
        if self.verbose:
            print("[code_monkey] " + str(message), flush=True)

    def submit(self, goal: str) -> Dict[str, Any]:
        self._emit('Creating task workspace')
        task_id, root = create_workspace()
        self._emit('Capturing environment snapshot')
        env = snapshot()
        write_json(root / 'environment.json', env)
        self.storage.create_task(task_id, goal, str(root))
        self._session_set(task_id, 'environment', 'snapshot', env)
        self._session_set(task_id, 'system', 'visible_summary', {
            'phase': 'created',
            'goal': goal,
            'workspace': str(root),
            'active_file': None,
            'verification_attempt': None,
        })
        return self.status(task_id)

    def plan(self, task_id: str) -> Dict[str, Any]:
        task = self._require_task(task_id)
        root = ensure_workspace(task_id)
        env = read_json(root / 'environment.json')
        self._emit('Planning work order')
        self.storage.update_task(task_id, 'planning', 'Creating work order')
        self._session_set(task_id, 'planner', 'phase', 'planning')
        self._session_set(task_id, 'planner', 'input_goal', task['goal'])
        self.planner.session_writer = lambda component, key, value: self._session_set(task_id, component, key, value)
        self.planner.session_appender = lambda component, key, value: self._session_append(task_id, component, key, value)
        work_order = self.planner.create_work_order(task['goal'], env)
        write_json(root / 'work_order.json', work_order.to_dict())
        self._session_set(task_id, 'planner', 'work_order', work_order.to_dict())
        self._session_set(task_id, 'system', 'visible_summary', {
            'phase': 'planned',
            'goal': task['goal'],
            'workspace': str(root),
            'files': [f.get('path') for f in work_order.files],
            'test_command': work_order.test_command,
        })
        self.storage.update_task(task_id, 'planned', 'Work order created')
        self._emit('Work order created')
        out = self.status(task_id)
        out['work_order'] = work_order.to_dict()
        return out

    def build(self, task_id: str) -> Dict[str, Any]:
        task = self._require_task(task_id)
        root = ensure_workspace(task_id)
        if not (root / 'work_order.json').exists():
            self.plan(task_id)
        env = read_json(root / 'environment.json')
        work_order = WorkOrder(**read_json(root / 'work_order.json'))

        self._emit('Starting file generation')
        self._bind_coder_session(task_id)
        self.storage.update_task(task_id, 'building', 'Generating and validating files')
        self._session_set(task_id, 'builder', 'required_files', work_order.files)
        self._session_set(task_id, 'builder', 'validated_files', [])
        build_files = None
        try:
            build_files = self.coder.generate_files(work_order, env)
            # Persist generated content before final package validation so failed
            # runs can print the README/code/tests that caused the failure.
            write_json(root / 'build_files.generated.json', build_files.to_dict())
            self._write_failed_candidate_files(root, build_files)

            for package_attempt in range(1, MAX_RUNTIME_REPAIR_ATTEMPTS + 1):
                try:
                    validate_build_files(build_files)
                    break
                except Exception as validation_exc:
                    self._session_set(task_id, 'validator', 'last_error', str(validation_exc))
                    self._session_set(task_id, 'validator', 'package_validation_attempt', package_attempt)
                    self._session_set(task_id, 'validator', 'package_validation_current_files', build_files.to_dict())
                    if package_attempt >= MAX_RUNTIME_REPAIR_ATTEMPTS:
                        raise
                    self.storage.update_task(task_id, 'repairing', 'Repairing generated files from validation failure')
                    self.storage.add_event(task_id, 'artifact_validation_failed', 'Generated artifact validation failed', {
                        'attempt': package_attempt,
                        'error': str(validation_exc),
                    })
                    build_files = self.coder.repair_files_from_validation_failure(
                        work_order=work_order,
                        environment=env,
                        previous_files=build_files,
                        validation_error=str(validation_exc),
                    )
                    write_json(root / 'build_files.generated.json', build_files.to_dict())
                    self._write_failed_candidate_files(root, build_files)
        except Exception as exc:
            if build_files is not None:
                self._session_set(task_id, 'builder', 'failed_candidate_files', [
                    'failed_candidates/latest/' + item.path for item in build_files.files
                ])
            self._session_set(task_id, 'validator', 'last_error', str(exc))
            self.storage.update_task(task_id, 'build_failed', 'File generation failed')
            self.storage.add_event(task_id, 'build_failed', 'File generation failed', {'error': str(exc)})
            out = self.status(task_id)
            out['error'] = str(exc)
            out['diagnostic_report'] = self.diagnostic_report(task_id, failed=True, error=str(exc))
            return out

        self._emit('Writing statically valid files to workspace')
        written = self._write_build_files(root, build_files)
        write_json(root / 'build_files.json', build_files.to_dict())
        self._session_set(task_id, 'builder', 'written_files', written)
        self._session_set(task_id, 'builder', 'build_notes', build_files.notes)
        self.storage.update_task(task_id, 'built', 'Files generated and passed static validation')
        self.storage.add_event(task_id, 'built_files', 'Generated files written to workspace', {
            'written_files': written,
            'build_notes': build_files.notes,
        })

        self._emit('Starting functional verification')
        try:
            verification = self._verify_with_repairs(task_id, root, work_order, env, build_files)
        except Exception as exc:
            out = self.status(task_id)
            out['error'] = str(exc)
            out['written_files'] = written
            out['build_notes'] = build_files.notes
            out['diagnostic_report'] = self.diagnostic_report(task_id, failed=True, error=str(exc))
            return out
        out = self.status(task_id)
        out['written_files'] = written
        out['build_notes'] = build_files.notes
        out['verification'] = verification
        if (root / 'build_files.json').exists():
            out['build_files'] = read_json(root / 'build_files.json')
        out['diagnostic_report'] = self.diagnostic_report(task_id, failed=False)
        return out

    def _verify_with_repairs(
        self,
        task_id: str,
        root: Path,
        work_order: WorkOrder,
        env: Dict[str, Any],
        build_files: BuildFiles,
    ) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        current_files = build_files

        for attempt in range(1, MAX_RUNTIME_REPAIR_ATTEMPTS + 1):
            self.storage.update_task(
                task_id,
                'testing',
                'Running verification attempt {}'.format(attempt),
            )
            self._session_set(task_id, 'tester', 'active_attempt', attempt)
            self._session_set(task_id, 'tester', 'test_command', work_order.test_command)
            self._session_set(task_id, 'tester', 'expected_cleanup_contract', {
                'rule': 'Tests must run native skill-relative DATA_DIR behavior, create entries, verify them, delete/back out through the skill API, and verify cleanup.',
                'storage_contract': 'src/api.py derives DATA_DIR from Path(__file__).resolve().parent / data; tests do not override storage.',
                'storage_paths': self._infer_storage_paths(current_files),
            })
            self._emit('Running verification attempt {}'.format(attempt))
            verification = run_verification_commands(
                root=root,
                test_command=work_order.test_command,
                self_test_command=work_order.self_test_command,
            )
            attempts.append({'attempt': attempt, 'verification': verification})
            write_json(root / 'verification.json', {
                'status': verification['status'],
                'attempts': attempts,
            })
            self._session_set(task_id, 'tester', 'last_verification', verification)
            self._session_append(task_id, 'tester', 'verification_history', {
                'attempt': attempt,
                'status': verification.get('status'),
                'summary': self._compact_verification(verification),
            })
            self.storage.add_event(task_id, 'verification_attempt', 'Verification attempt completed', {
                'attempt': attempt,
                'verification': verification,
            })

            if attempt > 1:
                self._record_lessons_for_disappeared_failures(
                    previous_report=attempts[-2].get('verification') or {},
                    current_report=verification,
                    fixed_files=current_files,
                )

            if verification.get('status') == 'success':
                self._emit('Verification passed on attempt {}'.format(attempt))
                if attempt > 1:
                    self._record_runtime_lessons(work_order, attempts[:-1], current_files)
                self._session_set(task_id, 'analyzer', 'result', 'verified')
                self._session_set(task_id, 'analyzer', 'reason', 'All verification commands passed.')
                self.storage.update_task(task_id, 'verified', 'Tests passed')
                self.storage.add_event(task_id, 'verified', 'Capability verified', {
                    'attempt': attempt,
                    'verification': verification,
                })
                return {'status': 'success', 'attempts': attempts}

            if attempt >= MAX_RUNTIME_REPAIR_ATTEMPTS:
                self._session_set(task_id, 'analyzer', 'result', 'failed')
                self._session_set(task_id, 'analyzer', 'reason', 'Verification failed after repair attempts.')
                self.storage.update_task(task_id, 'test_failed', 'Verification failed after repairs')
                self.storage.add_event(task_id, 'test_failed', 'Verification failed after repairs', {
                    'attempts': attempts,
                })
                raise ValueError(
                    'Verification failed after repair attempts.\n'
                    + json.dumps({'attempts': attempts}, indent=2)
                )

            self._emit('Verification failed; requesting runtime repair')
            self._session_set(task_id, 'analyzer', 'last_failure_summary', self._compact_verification(verification))
            self.storage.update_task(task_id, 'repairing', 'Repairing files from verification failure')
            try:
                current_files = self.coder.repair_files(
                    work_order=work_order,
                    environment=env,
                    previous_files=current_files,
                    failure_report=verification,
                )
                validate_build_files(current_files)
            except Exception as exc:
                self.storage.update_task(task_id, 'repair_failed', 'Runtime repair generation failed')
                self.storage.add_event(task_id, 'repair_failed', 'Runtime repair generation failed', {
                    'error': str(exc),
                    'previous_verification': verification,
                })
                raise

            self._emit('Writing repaired files to workspace')
            written = self._write_build_files(root, current_files)
            write_json(root / 'build_files.json', current_files.to_dict())
            self._session_set(task_id, 'builder', 'last_repaired_files', written)
            self._session_set(task_id, 'builder', 'last_repair_notes', current_files.notes)
            self.storage.add_event(task_id, 'repair_written', 'Repaired files written', {
                'attempt': attempt,
                'written_files': written,
                'build_notes': current_files.notes,
            })

        raise RuntimeError('unreachable verification loop exit')

    def _record_lessons_for_disappeared_failures(
        self,
        previous_report: Dict[str, Any],
        current_report: Dict[str, Any],
        fixed_files: BuildFiles,
    ) -> None:
        previous_lessons = extract_runtime_lessons(previous_report)
        if not previous_lessons:
            return
        current_text = failure_text(current_report).lower()
        current_signature = normalize_failure_signature(current_text)
        fixed_context = json.dumps(fixed_files.to_dict(), indent=2, default=str)[:4000]
        for lesson in previous_lessons:
            signature = lesson.get('failure_signature') or normalize_failure_signature(lesson.get('example_error', ''))
            example = lesson.get('example_error') or ''
            # If the same signature or exact first error line is still present,
            # do not call it fixed yet. Otherwise record it as a true fix.
            first_line = str(example).strip().split('\n')[0].lower()[:180]
            still_present = False
            if signature and signature in current_signature:
                still_present = True
            if first_line and first_line in current_text:
                still_present = True
            if still_present:
                continue
            try:
                self.storage.record_lesson(
                    lesson.get('scope') or 'global',
                    signature,
                    lesson.get('lesson') or 'Avoid repeating this fixed runtime/test failure.',
                    example,
                    fixed_context,
                )
                self._session_append('GLOBAL', 'lessons', 'recent', {
                    'scope': lesson.get('scope') or 'global',
                    'failure_signature': signature,
                    'lesson': lesson.get('lesson') or '',
                }, limit=50)
                self._emit('Recorded lesson: ' + str(signature))
            except Exception as exc:
                self._emit('Could not record runtime lesson: ' + str(exc))

    def _record_runtime_lessons(
        self,
        work_order: WorkOrder,
        failed_attempts: List[Dict[str, Any]],
        fixed_files: BuildFiles,
    ) -> None:
        if not failed_attempts:
            return
        # On final success, every remaining failure from the last failed attempt
        # is considered fixed and should become a reusable lesson.
        previous_report = failed_attempts[-1].get('verification') or {}
        self._record_lessons_for_disappeared_failures(
            previous_report=previous_report,
            current_report={'status': 'success', 'results': [], 'failures': []},
            fixed_files=fixed_files,
        )

    def _write_build_files(self, root: Path, build_files: BuildFiles) -> List[str]:
        written = []
        for item in build_files.files:
            target = validate_workspace_write(root, item.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content.rstrip() + '\n', encoding='utf-8')
            written.append(item.path)
        return written

    def _write_failed_candidate_files(self, root: Path, build_files: BuildFiles) -> List[str]:
        """Write latest failed generated candidates for post-failure debugging.

        These files live under failed_candidates/latest/ and do not masquerade
        as validated build outputs. They let users inspect src/api.py, tests,
        and README even when static validation rejects the package.
        """
        written = []
        base = root / 'failed_candidates' / 'latest'
        for item in build_files.files:
            target = (base / item.path).resolve()
            base_resolved = base.resolve()
            if base_resolved not in target.parents and target != base_resolved:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((item.content or '').rstrip() + '\n', encoding='utf-8')
            written.append('failed_candidates/latest/' + item.path)
        return written

    def _file_content_from_build_files(self, build_files_data: Dict[str, Any], path: str) -> str:
        for item in (build_files_data or {}).get('files') or []:
            if item.get('path') == path:
                return item.get('content') or ''
        return ''

    def diagnostic_report(self, task_id: str, failed: bool = False, error: str = '') -> Dict[str, Any]:
        """Return a compact end-of-run report for CLI diagnosis.

        On failure this includes the generated source artifacts and validation
        failures so the next repair can be based on evidence instead of guesses.
        On success it keeps the same shape but emphasizes step results.
        """
        task = self._require_task(task_id)
        root = ensure_workspace(task_id)
        events = self.storage.events(task_id)
        session = self.storage.blackboard_snapshot(task_id)

        def read_optional_json(name: str) -> Dict[str, Any]:
            path = root / name
            if not path.exists():
                return {}
            try:
                return read_json(path)
            except Exception as exc:
                return {'error': str(exc)}

        work_order = read_optional_json('work_order.json')
        build_files = read_optional_json('build_files.json')
        if not build_files:
            build_files = read_optional_json('build_files.generated.json')
        if not build_files and isinstance(session, dict):
            builder_snapshot = session.get('builder') or {}
            build_files = (
                builder_snapshot.get('latest_repaired_build_files')
                or builder_snapshot.get('latest_generated_build_files')
                or {}
            )
        verification = read_optional_json('verification.json')

        generated_files = []
        for item in (build_files.get('files') or []):
            path = item.get('path') or ''
            content = item.get('content') or ''
            generated_files.append({
                'path': path,
                'bytes': len(content),
                'content': content if failed else content[:1200],
            })

        validator_session = (session.get('validator') or {}) if isinstance(session, dict) else {}
        tester_session = (session.get('tester') or {}) if isinstance(session, dict) else {}

        step_results = []
        if work_order:
            step_results.append({
                'step': 'plan',
                'status': 'success',
                'files': [f.get('path') for f in work_order.get('files') or []],
                'test_command': work_order.get('test_command'),
            })
        validated = ((session.get('builder') or {}).get('validated_files') if isinstance(session, dict) else None) or []
        step_results.append({
            'step': 'generate_validate_files',
            'status': 'failed' if failed and not build_files else 'success' if build_files else task.get('state'),
            'notes': build_files.get('notes') or '',
            'validator_last_error': validator_session.get('last_error'),
            'validator_last_result': validator_session.get('last_result'),
            'failures': validator_session.get('failures') or [],
            'validated_files': validated,
        })
        if verification:
            step_results.append({
                'step': 'verification',
                'status': verification.get('status'),
                'attempts': verification.get('attempts') or [],
                'last_verification': tester_session.get('last_verification'),
            })

        return {
            'task_id': task_id,
            'status': 'failed' if failed else task.get('state'),
            'workspace': str(root),
            'error': error or None,
            'step_results': step_results,
            'generated_files': generated_files,
            'verification': verification or None,
            'events': events,
            'session_summary': {
                'builder': session.get('builder') if isinstance(session, dict) else None,
                'validator': validator_session,
                'tester': tester_session,
                'analyzer': session.get('analyzer') if isinstance(session, dict) else None,
            },
        }

    def submit_and_plan(self, goal: str) -> Dict[str, Any]:
        task = self.submit(goal)
        return self.plan(task['task_id'])

    def submit_and_build(self, goal: str) -> Dict[str, Any]:
        task = self.submit(goal)
        return self.build(task['task_id'])

    def status(self, task_id: str) -> Dict[str, Any]:
        task = self._require_task(task_id)
        root = ensure_workspace(task_id)
        out = {
            'task_id': task['task_id'],
            'state': task['state'],
            'workspace': task['workspace'],
            'message': task.get('message') or '',
        }
        # Keep status responses compact and avoid duplicating the environment.
        # The canonical environment snapshot is available at:
        #   session.components.environment.snapshot
        optional_json = [
            ('work_order', root / 'work_order.json'),
            ('build_files', root / 'build_files.json'),
            ('verification', root / 'verification.json'),
        ]
        for key, path in optional_json:
            if path.exists():
                try:
                    out[key] = read_json(path)
                except Exception as exc:
                    out[key] = {'error': str(exc)}
        out['session'] = self.storage.blackboard_snapshot(task_id)
        return out

    def list_tasks(self) -> Dict[str, Any]:
        tasks = self.storage.list_running_tasks()
        return {
            'tasks': tasks,
            'count': len(tasks),
            'scope': 'running',
            'message': 'Currently running tasks only. Use /tasks/{task_id} for any historical task details.',
        }

    def events(self, task_id: str) -> Dict[str, Any]:
        self._require_task(task_id)
        return {'task_id': task_id, 'events': self.storage.events(task_id)}

    def session(self, task_id: str) -> Dict[str, Any]:
        self._require_task(task_id)
        return self.storage.blackboard_snapshot(task_id)

    def lessons(self, scope: str | None = None) -> Dict[str, Any]:
        return {'lessons': self.storage.list_lessons(scope=scope, limit=100)}

    def _compact_verification(self, verification: Dict[str, Any]) -> Dict[str, Any]:
        failures = verification.get('failures') or []
        first = failures[0] if failures else {}
        text = str(first.get('stderr') or first.get('error') or first.get('stdout') or '')
        return {
            'status': verification.get('status'),
            'failure_count': len(failures),
            'first_failure': text[:1200],
        }

    def _infer_storage_paths(self, build_files: BuildFiles) -> List[str]:
        paths: List[str] = []
        for item in build_files.files:
            content = item.content or ''
            for marker in ['SKILL_ROOT', 'DATA_DIR', '__file__', 'data', 'os.getcwd', 'tempfile', '/tmp']:
                if marker in content and marker not in paths:
                    paths.append(marker)
        return paths

    def _require_task(self, task_id: str) -> Dict[str, Any]:
        task = self.storage.get_task(task_id)
        if not task:
            raise ValueError('Unknown task_id: {}'.format(task_id))
        return task
