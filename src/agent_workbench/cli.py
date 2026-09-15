"""Explicit local commands; importing the CLI never starts a model."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

from . import __version__


def _runtime(args):
    if not getattr(args, 'plugin', None):
        from .demo import build_demo_runtime
        return build_demo_runtime(Path(args.data_dir))
    module, sep, name = args.plugin.partition(':')
    if not sep or not module or not name:
        raise ValueError('plugin must name an installed module:factory')
    factory = getattr(importlib.import_module(module), name)
    runtime = factory(data_dir=Path(args.data_dir))
    runtime.services.allow_real_model = bool(args.real_model)
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(prog='agent-workbench')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('demo', 'serve', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--data-dir', default=str(Path.home() / '.agent-workbench' / 'demo'))
        if name != 'demo':
            command.add_argument('--plugin', required=True, help='Trusted installed Python module:factory')
            command.add_argument('--real-model', action='store_true', help='Allow configured real model calls (may incur charges)')
        if name == 'run':
            command.add_argument('--message', required=True)
            command.add_argument('--thread-id')
        else:
            command.add_argument('--chat-port', type=int, default=8786)
            command.add_argument('--diagnostics-port', type=int, default=8785)
    export = commands.add_parser('export')
    export.add_argument('run_id')
    export.add_argument('--data-dir', default=str(Path.home() / '.agent-workbench' / 'demo'))
    export.add_argument('--project-id', default='agent-workbench')
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'export':
            from .trace_export import build_trace_envelope
            filename = 'run-' + hashlib.sha256(args.run_id.encode()).hexdigest() + '.json'
            result = json.loads((Path(args.data_dir) / 'reports' / filename).read_text(encoding='utf-8'))
            trace = build_trace_envelope(result, project_id=args.project_id)
            if args.output.exists():
                raise ValueError('Output already exists; choose a new export path')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
            print('Trace exported. Imported traces do not establish independent trusted provenance.')
            return 0
        runtime = _runtime(args)
        if args.command in {'demo', 'serve'}:
            from .web.server import run_servers
            print(f'Chat: http://127.0.0.1:{args.chat_port}')
            print(f'Diagnostics: http://127.0.0.1:{args.diagnostics_port}')
            if args.command == 'demo':
                print('Offline reference demo. No real model or production system is used.')
            run_servers(runtime, chat_port=args.chat_port, diagnostics_port=args.diagnostics_port,
                        offline=args.command == 'demo')
            return 0
        result = runtime.start({'message': args.message, 'thread_id': args.thread_id})
        while result['state'].get('pending_interaction'):
            pending = result['state']['pending_interaction']
            print(json.dumps(pending, ensure_ascii=False, indent=2))
            if pending['type'] == 'clarification':
                response = {'answer': input('Answer: ')}
            else:
                decision = input('approve / reject / request_changes: ').strip()
                response = {'decision': decision}
                if decision == 'request_changes':
                    response['comment'] = input('Required changes: ')
            result = runtime.resume(result['state']['run_id'], pending['interaction_id'], response)
        print(result.get('report_text') or json.dumps(result['report'], ensure_ascii=False))
        return 0 if result['report'].get('status') == 'completed' else 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Provider/plugin exceptions may contain credentials or user data.
        print('Command failed: ' + type(exc).__name__)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
