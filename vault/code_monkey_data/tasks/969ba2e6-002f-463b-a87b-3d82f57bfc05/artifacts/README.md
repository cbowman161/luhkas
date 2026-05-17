# Implement A Python Http Api With Reminder Endpoint

## Purpose
This API-first skill package implements the requested capability: Implement a Python HTTP API with endpoints: create-reminder (POST), list-reminders (GET), delete-reminder (POST), and check-status (GET). Data fields: title (str), time (str), message (str), type (str). Persist reminders to src/data/ as JSON files. Background service will trigger notifications based on stored time data.
It provides a small Python API in `src/api.py` so callers and tests can create, inspect, and remove capability data without using private helpers.

## Usage
Import the public functions from `src.api` and call them directly from Python.
Call `schema()` first to discover available endpoint functions, parameters, response fields, and storage notes.
Each endpoint returns a JSON-like dict response envelope and does not require command-line interaction.

## Public API
- `check_status(title)`: public API endpoint/action function. It returns a dict envelope with `ok`, `action`, `message`, `data`, and `error`.
- `create_reminder(title, time, message, type)`: public API endpoint/action function. It returns a dict envelope with `ok`, `action`, `message`, `data`, and `error`.
- `delete_reminder(title)`: public API endpoint/action function. It returns a dict envelope with `ok`, `action`, `message`, `data`, and `error`.
- `list_reminders()`: public API endpoint/action function. It returns a dict envelope with `ok`, `action`, `message`, `data`, and `error`.
- `schema()`: returns a dict describing the skill, endpoints, parameters, response envelope, and storage behavior.
External callers and tests should use only these public API functions.

## Function Definitions
- `_cm_failure(action, message, error, data)`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: action, message, error, data.
  Returns/Outputs: returns implementation data used by public endpoints, or `None` for initialization helpers.
  Called by: check_status, create_reminder, delete_reminder, list_reminders.
  Calls: none.
- `_cm_success(action, message, data)`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: action, message, data.
  Returns/Outputs: returns implementation data used by public endpoints, or `None` for initialization helpers.
  Called by: check_status, create_reminder, delete_reminder, list_reminders.
  Calls: none.
- `check_status(title)`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: title.
  Returns/Outputs: returns the standard dict response envelope with `ok`, `action`, `message`, `data`, and `error`.
  Called by: external callers/tests or none.
  Calls: _cm_failure, _cm_success.
- `create_reminder(title, time, message, type)`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: title, time, message, type.
  Returns/Outputs: returns the standard dict response envelope with `ok`, `action`, `message`, `data`, and `error`.
  Called by: external callers/tests or none.
  Calls: _cm_failure, _cm_success.
- `delete_reminder(title)`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: title.
  Returns/Outputs: returns the standard dict response envelope with `ok`, `action`, `message`, `data`, and `error`.
  Called by: external callers/tests or none.
  Calls: _cm_failure, _cm_success.
- `list_reminders()`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: none.
  Returns/Outputs: returns the standard dict response envelope with `ok`, `action`, `message`, `data`, and `error`.
  Called by: external callers/tests or none.
  Calls: _cm_failure, _cm_success.
- `schema()`: function or method defined in `src/api.py` for the capability implementation.
  Parameters: none.
  Returns/Outputs: returns a plain dict schema for the API-first package.
  Called by: external callers/tests or none.
  Calls: none.

## Imported Dependencies
- `Path`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.
- `datetime`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.
- `json`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.
- `os`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.
- `pathlib`: imported by `src/api.py` for standard-library implementation, data modeling, validation, or native file storage support.

## Outputs and Return Values
`schema()` returns a dict that documents the package name, endpoints, parameter expectations, response envelope, and storage notes.
`check_status()` returns `{ "ok": bool, "action": str, "message": str, "data": dict/list/null, "error": str/null }`.
`create_reminder()` returns `{ "ok": bool, "action": str, "message": str, "data": dict/list/null, "error": str/null }`.
`delete_reminder()` returns `{ "ok": bool, "action": str, "message": str, "data": dict/list/null, "error": str/null }`.
`list_reminders()` returns `{ "ok": bool, "action": str, "message": str, "data": dict/list/null, "error": str/null }`.
Successful endpoint responses set `ok` to `True`, include an action name, a human-readable message, useful `data`, and `error` as `None`.
Failed endpoint responses set `ok` to `False`, include the attempted action, a failure message, `data` as `None` or partial context, and an explanatory `error` string.

## Failure Modes
Invalid parameters, missing records, malformed identifiers, unavailable storage, or file I/O errors are reported through the response envelope rather than uncaught exceptions during normal endpoint use.
Callers should check `ok` before using `data`, and should read `error` when `ok` is `False`.

## Data Storage
The skill uses native storage relative to the skill file location. Persistent data belongs under `src/data` through `SKILL_ROOT = Path(__file__).resolve().parent` and `DATA_DIR = SKILL_ROOT / "data"` when storage is needed.
Storage must not depend on the current working directory, `/tmp`, `tempfile`, or a test-supplied override.

## Cleanup / Delete Behavior
Cleanup and back-out behavior is exposed through public delete/remove/cleanup API endpoints when the capability creates persistent entries.
Tests and callers should create entries through the public add/create endpoint, verify they are visible through list/read endpoints, call the public delete/remove endpoint, and verify the entry is gone.

## Behavioral Verification Contract
For add/create + list/read + delete/remove style capabilities, tests must verify the full lifecycle described by the README and public API: create an item, list items and assert the created item is visible, delete/remove the item, then list again and assert the item is no longer visible.
A test that only checks the response envelope is not sufficient for stateful capabilities.

## Test Coverage
Tests verify `schema()` documents the API-first package, every public endpoint can be called from `src.api`, and every endpoint response contains `ok`, `action`, `message`, `data`, and `error`.
Tests verify add behavior, list behavior, delete/cleanup behavior, native DATA_DIR storage under the skill file location, and failure/empty-state behavior when applicable.
Tests use only the public API and do not override storage paths or manually clean normal capability data except as emergency cleanup after the delete API has been asserted.
