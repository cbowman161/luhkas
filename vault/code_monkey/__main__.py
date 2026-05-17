import argparse
import json
import sys

from .build_manager import BuildManager


def print_json(data):
    print(json.dumps(data, indent=2, sort_keys=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python3 -m code_monkey')
    sub = parser.add_subparsers(dest='command', required=True)

    submit = sub.add_parser('submit', help='create a build task')
    submit.add_argument('goal', help='capability goal')
    submit.add_argument('--plan', action='store_true', help='create work_order.json immediately')
    submit.add_argument('--build', action='store_true', help='plan and generate files immediately')

    plan = sub.add_parser('plan', help='create work_order.json for an existing task')
    plan.add_argument('task_id')

    build = sub.add_parser('build', help='generate files for an existing task')
    build.add_argument('task_id')

    status = sub.add_parser('status', help='show task status')
    status.add_argument('task_id')

    events = sub.add_parser('events', help='show task events')
    events.add_argument('task_id')

    session = sub.add_parser('session', help='show task blackboard/session state')
    session.add_argument('task_id')

    lessons = sub.add_parser('lessons', help='show lessons learned')
    lessons.add_argument('--scope', default=None, help='optional lesson scope/path')

    sub.add_parser('list', help='list tasks')

    service = sub.add_parser('service', help='run the long-running async coder service')
    service.add_argument('--host', default='127.0.0.1')
    service.add_argument('--port', type=int, default=8765)
    service.add_argument('--workers', type=int, default=2, help='maximum concurrent coder tasks')
    service.add_argument('--poll-seconds', type=float, default=1.0)


    args = parser.parse_args(argv)
    verbose = args.command in {'submit', 'plan', 'build'}
    manager = BuildManager(verbose=verbose)

    try:
        if args.command == 'submit':
            if args.build:
                print_json(manager.submit_and_build(args.goal))
            elif args.plan:
                print_json(manager.submit_and_plan(args.goal))
            else:
                print_json(manager.submit(args.goal))
            return 0
        if args.command == 'plan':
            print_json(manager.plan(args.task_id))
            return 0
        if args.command == 'build':
            print_json(manager.build(args.task_id))
            return 0
        if args.command == 'status':
            print_json(manager.status(args.task_id))
            return 0
        if args.command == 'events':
            print_json(manager.events(args.task_id))
            return 0
        if args.command == 'session':
            print_json(manager.session(args.task_id))
            return 0
        if args.command == 'lessons':
            print_json(manager.lessons(scope=args.scope))
            return 0
        if args.command == 'list':
            print_json(manager.list_tasks())
            return 0
        if args.command == 'service':
            from .service import run_service
            run_service(host=args.host, port=args.port, workers=args.workers, poll_seconds=args.poll_seconds)
            return 0
    except Exception as exc:
        print_json({'error': str(exc)})
        return 1

    parser.print_help()
    return 2


if __name__ == '__main__':
    sys.exit(main())
