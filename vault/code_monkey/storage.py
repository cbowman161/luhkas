import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DB_PATH, DATA_DIR

FINAL_NOTIFICATION_STATES = {'verified', 'build_failed', 'test_failed', 'failed', 'repair_failed', 'cancelled'}
SUCCESS_STATES = {'verified'}


class Storage:
    def __init__(self, db_path: Path = DB_PATH, progress=None):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.progress = progress
        self.conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            state TEXT NOT NULL,
            workspace TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            data TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            failure_signature TEXT NOT NULL,
            lesson TEXT NOT NULL,
            example_error TEXT,
            fix_context TEXT,
            times_seen INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(scope, failure_signature)
        )
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS blackboard (
            task_id TEXT NOT NULL,
            component TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(task_id, component, key)
        )
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL UNIQUE,
            goal TEXT NOT NULL,
            state TEXT NOT NULL,
            success INTEGER NOT NULL,
            message TEXT NOT NULL,
            workspace TEXT NOT NULL,
            data TEXT,
            read_at TEXT,
            created_at TEXT NOT NULL
        )
        ''')
        self.conn.commit()

    def create_task(self, task_id: str, goal: str, workspace: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT INTO tasks(task_id, goal, state, workspace, message, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (task_id, goal, 'created', workspace, 'Task created', now, now),
        )
        self.conn.commit()
        self.add_event(task_id, 'created', 'Task created', {'goal': goal, 'workspace': workspace})
        self.blackboard_set(task_id, 'system', 'task_id', task_id)
        self.blackboard_set(task_id, 'system', 'goal', goal)
        self.blackboard_set(task_id, 'system', 'workspace', workspace)
        self.blackboard_set(task_id, 'system', 'phase', 'created')

    def update_task(self, task_id: str, state: str, message: str = '') -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'UPDATE tasks SET state=?, message=?, updated_at=? WHERE task_id=?',
            (state, message, now, task_id),
        )
        self.conn.commit()
        self.blackboard_set(task_id, 'system', 'phase', state)
        self.blackboard_set(task_id, 'system', 'message', message or state)
        self.add_event(task_id, state, message or state, {})
        if state in FINAL_NOTIFICATION_STATES:
            self.add_notification(task_id, state=state, message=message or state, data={})

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute('SELECT * FROM tasks ORDER BY created_at DESC').fetchall()
        return [dict(r) for r in rows]

    def list_running_tasks(self) -> List[Dict[str, Any]]:
        """Return only tasks that have not reached a terminal state.

        /tasks is intended as a live activity view. Historical details remain
        available through /tasks/{task_id}, so completed/failed/cancelled tasks
        are intentionally excluded here.
        """
        final_states = tuple(FINAL_NOTIFICATION_STATES)
        placeholders = ','.join('?' for _ in final_states)
        rows = self.conn.execute(
            f'SELECT * FROM tasks WHERE state NOT IN ({placeholders}) ORDER BY created_at DESC',
            final_states,
        ).fetchall()
        return [dict(r) for r in rows]

    def add_event(self, task_id: str, event_type: str, message: str, data: Dict[str, Any]) -> None:
        if self.progress and event_type not in {'created'}:
            try:
                self.progress('{}: {}'.format(event_type, message))
            except Exception:
                pass
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            'INSERT INTO events(task_id, event_type, message, data, created_at) VALUES (?, ?, ?, ?, ?)',
            (task_id, event_type, message, json.dumps(data, default=str), now),
        )
        self.conn.commit()

    def events(self, task_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute('SELECT * FROM events WHERE task_id=? ORDER BY id', (task_id,)).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d['data'] = json.loads(d['data']) if d.get('data') else {}
            out.append(d)
        return out

    # ----------------------------
    # Notification board
    # ----------------------------

    def add_notification(self, task_id: str, state: str, message: str = '', data: Optional[Dict[str, Any]] = None) -> None:
        """Create one unread completion notification for a task.

        Notifications are intentionally completion-only. Running/queued tasks do
        not appear on the board, so checking notifications while another task is
        still running only shows tasks that have actually reached a final state.
        """
        task = self.get_task(task_id)
        if not task:
            return
        state = str(state or task.get('state') or '')
        if state not in FINAL_NOTIFICATION_STATES:
            return
        now = datetime.now(timezone.utc).isoformat()
        success = 1 if state in SUCCESS_STATES else 0
        payload = dict(data or {})
        payload.setdefault('state', state)
        self.conn.execute(
            '''INSERT OR IGNORE INTO notifications(
                   task_id, goal, state, success, message, workspace, data, read_at, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)''',
            (
                str(task_id),
                str(task.get('goal') or ''),
                state,
                success,
                str(message or task.get('message') or state),
                str(task.get('workspace') or ''),
                json.dumps(payload, default=str),
                now,
            ),
        )
        self.conn.commit()

    def list_notifications(
        self,
        unread_only: bool = True,
        limit: int = 50,
        mark_read: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return notification-board items.

        Default behavior is inbox-like: return unread completion notifications
        and mark exactly those returned notifications as read. Later checks only
        show newer/unread completions.
        """
        limit = max(1, min(int(limit or 50), 200))
        if unread_only:
            rows = self.conn.execute(
                '''SELECT * FROM notifications
                   WHERE read_at IS NULL
                   ORDER BY created_at ASC, id ASC
                   LIMIT ?''',
                (limit,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                '''SELECT * FROM notifications
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?''',
                (limit,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        ids: List[int] = []
        for row in rows:
            d = dict(row)
            try:
                d['data'] = json.loads(d['data']) if d.get('data') else {}
            except Exception:
                d['data'] = {}
            d['success'] = bool(d.get('success'))
            d['read'] = bool(d.get('read_at'))
            out.append(d)
            if mark_read and unread_only and not d.get('read_at'):
                ids.append(int(d['id']))
        if ids:
            now = datetime.now(timezone.utc).isoformat()
            placeholders = ','.join('?' for _ in ids)
            self.conn.execute(
                f'UPDATE notifications SET read_at=? WHERE id IN ({placeholders}) AND read_at IS NULL',
                (now, *ids),
            )
            self.conn.commit()
            for d in out:
                if int(d.get('id')) in ids:
                    d['read_at'] = now
                    d['read'] = True
        return out


    def list_recent_notifications(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return most recent notification-board items without changing read state."""
        return self.list_notifications(unread_only=False, limit=limit, mark_read=False)

    def count_unread_notifications(self) -> int:
        row = self.conn.execute(
            'SELECT COUNT(*) AS count FROM notifications WHERE read_at IS NULL'
        ).fetchone()
        return int(row['count']) if row else 0

    def mark_notification_read(self, notification_id: int) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            'UPDATE notifications SET read_at=? WHERE id=? AND read_at IS NULL',
            (now, int(notification_id)),
        )
        self.conn.commit()
        return bool(cur.rowcount)

    # ----------------------------
    # Durable async service queue helpers
    # ----------------------------

    def enqueue_task(self, task_id: str, message: str = 'Queued') -> None:
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f'Unknown task_id: {task_id}')
        if task.get('state') in {'verified', 'build_failed', 'test_failed', 'failed', 'cancelled'}:
            raise ValueError(f"Cannot enqueue final task {task_id} in state {task.get('state')}")
        self.update_task(task_id, 'queued', message)

    def recover_interrupted_tasks(self) -> int:
        """Re-queue tasks that were mid-flight when the service stopped."""
        interrupted = {'planning', 'building', 'built', 'testing', 'repairing', 'running', 'claimed'}
        placeholders = ','.join('?' for _ in interrupted)
        rows = self.conn.execute(
            f'SELECT task_id, state FROM tasks WHERE state IN ({placeholders})',
            tuple(interrupted),
        ).fetchall()
        for row in rows:
            self.update_task(row['task_id'], 'queued', f"Recovered from interrupted state: {row['state']}")
            self.add_event(row['task_id'], 'recovered', 'Task re-queued after service restart', {'previous_state': row['state']})
        return len(rows)

    def claim_next_task(self, worker_id: str = 'worker') -> Optional[Dict[str, Any]]:
        """Atomically claim one queued/runnable task for a worker."""
        now = datetime.now(timezone.utc).isoformat()
        runnable = ('queued', 'created', 'planned')
        try:
            self.conn.execute('BEGIN IMMEDIATE')
            row = self.conn.execute(
                """SELECT * FROM tasks
                   WHERE state IN (?, ?, ?)
                   ORDER BY updated_at ASC, created_at ASC
                   LIMIT 1""",
                runnable,
            ).fetchone()
            if not row:
                self.conn.execute('COMMIT')
                return None
            task_id = row['task_id']
            self.conn.execute(
                'UPDATE tasks SET state=?, message=?, updated_at=? WHERE task_id=?',
                ('claimed', f'Claimed by {worker_id}', now, task_id),
            )
            self.conn.execute('COMMIT')
        except Exception:
            try:
                self.conn.execute('ROLLBACK')
            except Exception:
                pass
            raise
        self.blackboard_set(task_id, 'service', 'worker_id', worker_id)
        self.add_event(task_id, 'claimed', f'Task claimed by {worker_id}', {'worker_id': worker_id})
        return self.get_task(task_id)

    def count_queued_tasks(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM tasks WHERE state IN ('queued', 'created', 'planned')"
        ).fetchone()
        return int(row['count']) if row else 0

    def count_active_tasks(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM tasks WHERE state IN ('claimed', 'planning', 'building', 'built', 'testing', 'repairing', 'running')"
        ).fetchone()
        return int(row['count']) if row else 0

    # ----------------------------
    # Blackboard / task session
    # ----------------------------

    def blackboard_set(self, task_id: str, component: str, key: str, value: Any) -> None:
        task_id = str(task_id)
        component = str(component or 'system')[:100]
        key = str(key or 'value')[:200]
        now = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(value, default=str)
        self.conn.execute(
            '''INSERT INTO blackboard(task_id, component, key, value, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(task_id, component, key)
               DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at''',
            (task_id, component, key, encoded, now),
        )
        self.conn.commit()

    def blackboard_append(self, task_id: str, component: str, key: str, value: Any, limit: int = 100) -> None:
        current = self.blackboard_get(task_id, component, key)
        if not isinstance(current, list):
            current = []
        current.append(value)
        if limit and len(current) > limit:
            current = current[-limit:]
        self.blackboard_set(task_id, component, key, current)

    def blackboard_get(self, task_id: str, component: str, key: str) -> Any:
        row = self.conn.execute(
            'SELECT value FROM blackboard WHERE task_id=? AND component=? AND key=?',
            (str(task_id), str(component), str(key)),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row['value'])
        except Exception:
            return row['value']

    def blackboard_snapshot(self, task_id: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            'SELECT component, key, value, updated_at FROM blackboard WHERE task_id=? ORDER BY component, key',
            (str(task_id),),
        ).fetchall()
        snapshot: Dict[str, Any] = {}
        updated_at: Dict[str, Any] = {}
        for row in rows:
            component = row['component']
            key = row['key']
            try:
                value = json.loads(row['value']) if row['value'] is not None else None
            except Exception:
                value = row['value']
            snapshot.setdefault(component, {})[key] = value
            updated_at.setdefault(component, {})[key] = row['updated_at']
        return {'task_id': task_id, 'components': snapshot, 'updated_at': updated_at}

    def record_lesson(
        self,
        scope: str,
        failure_signature: str,
        lesson: str,
        example_error: str = '',
        fix_context: str = '',
    ) -> None:
        scope = str(scope or 'global')[:200]
        failure_signature = str(failure_signature or 'unknown')[:500]
        lesson = str(lesson or '').strip()[:2000]
        if not lesson:
            lesson = 'Avoid repeating this failure: ' + failure_signature
        now = datetime.now(timezone.utc).isoformat()
        existing = self.conn.execute(
            'SELECT id, times_seen FROM lessons WHERE scope=? AND failure_signature=?',
            (scope, failure_signature),
        ).fetchone()
        if existing:
            self.conn.execute(
                '''UPDATE lessons
                   SET lesson=?, example_error=?, fix_context=?, times_seen=?, updated_at=?
                   WHERE id=?''',
                (
                    lesson,
                    str(example_error or '')[:4000],
                    str(fix_context or '')[:4000],
                    int(existing['times_seen']) + 1,
                    now,
                    existing['id'],
                ),
            )
        else:
            self.conn.execute(
                '''INSERT INTO lessons(
                       scope, failure_signature, lesson, example_error,
                       fix_context, times_seen, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    scope,
                    failure_signature,
                    lesson,
                    str(example_error or '')[:4000],
                    str(fix_context or '')[:4000],
                    1,
                    now,
                    now,
                ),
            )
        self.conn.commit()

    def list_lessons(self, scope: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 50), 200))
        if scope:
            rows = self.conn.execute(
                '''SELECT * FROM lessons
                   WHERE scope IN (?, 'global')
                   ORDER BY updated_at DESC
                   LIMIT ?''',
                (str(scope), limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                '''SELECT * FROM lessons
                   ORDER BY updated_at DESC
                   LIMIT ?''',
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
