import sys
import time

from event_log import EventLog
from vault_runtime import VaultRuntime


def get_unread_notifications(event_log):
    """
    Notification-center compatibility wrapper.

    Preferred EventLog API:
        unread_notifications()
        mark_notifications_read(ids)

    If that API has not been added yet, this returns no notifications so main.py
    continues to work instead of crashing.
    """
    if not hasattr(event_log, "unread_notifications"):
        return []

    try:
        return event_log.unread_notifications() or []
    except Exception as exc:
        return [
            {
                "id": None,
                "level": "error",
                "message": f"Notification center error: {exc}",
                "job_id": None,
                "data": {},
            }
        ]


def mark_notifications_read(event_log, notifications):
    if not notifications:
        return

    if not hasattr(event_log, "mark_notifications_read"):
        return

    ids = [n.get("id") for n in notifications if n.get("id") is not None]

    if not ids:
        return

    try:
        event_log.mark_notifications_read(ids)
    except Exception:
        # Notification display should never break the CLI loop.
        pass


def show_notifications_if_any(event_log):
    notifications = get_unread_notifications(event_log)

    if not notifications:
        return

    print()
    print("🔔 Notifications:")

    for notification in notifications:
        level = notification.get("level") or "info"
        icon = {
            "success": "✅",
            "completed": "✅",
            "error": "❌",
            "failed": "❌",
            "warning": "⏸",
            "paused": "⏸",
            "info": "ℹ️",
        }.get(level, "🔔")

        message = notification.get("message") or "Task notification"
        job_id = notification.get("job_id")

        if job_id:
            print(f"{icon} {message} ({job_id})")
        else:
            print(f"{icon} {message}")

    mark_notifications_read(event_log, notifications)
    print()


def main():
    print("🧠 Brain_V2 Runtime Ready")
    print("Commands: updates, jobs, code monkey, new, exit\n")

    print("Warming up models...", flush=True)
    t0 = time.monotonic()
    runtime = VaultRuntime()
    elapsed = time.monotonic() - t0

    warmup = runtime.model_warmup or []
    ok = [r for r in warmup if r.get("ok")]
    failed = [r for r in warmup if not r.get("ok")]

    for r in ok:
        print(f"  ✓ {r['role']} ({r['model']})")
    for r in failed:
        print(f"  ✗ {r['role']} ({r.get('model', '?')}) — {r.get('error', 'unknown error')}")

    print(f"Models ready in {elapsed:.1f}s\n")

    event_log = runtime.event_log

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                # Empty input is still an interaction, so surface completed-task notifications.
                show_notifications_if_any(event_log)
                continue

            # Show notification-center messages when the user next interacts.
            # Do this after input() returns so background tasks do not interrupt the prompt.
            show_notifications_if_any(event_log)

            lowered = user_input.lower()

            if lowered in {"exit", "quit"}:
                print("👋 Exiting...")
                sys.exit(0)

            t0 = time.monotonic()
            response = runtime.handle(user_input)
            elapsed = time.monotonic() - t0

            has_display = response.get("has_display", True)

            if has_display:
                print(response["message"])
                display_content = response.get("display_content", "")
                if display_content:
                    print()
                    print(display_content)
            else:
                print(response.get("tts") or response["message"])

            if response.get("mode") == "async":
                if has_display:
                    print("I’ll log updates as the job progresses. Ask ‘updates’ to check them.")
                else:
                    print("Working on it. Say updates to check progress.")

            print(f"[{elapsed:.1f}s]")

            print()

        except KeyboardInterrupt:
            print("\n👋 Exiting...\n")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
