"""Audit the explicit public file set; never traverse workspaces or credentials."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = ('README.md', 'README.en.md', 'LICENSE', 'SECURITY.md', 'CONTRIBUTING.md',
              'pyproject.toml', '.gitignore', '.gitattributes', 'MANIFEST.in')
PATTERNS = ('src/agent_workbench/**/*.py', 'src/agent_workbench/web/static/*.html',
            'src/agent_workbench/web/static/*.js', 'src/agent_workbench/web/static/*.css',
            'tests/**/*.py', 'tools/*.py', 'docs/*.md', '.github/workflows/*.yml')
FORBIDDEN_IMPORTS = {'agent', 'modules', 'core', 'scripts', 'commerce_eval'}
FORBIDDEN_TEXT = re.compile(r'(?i)\b(?:temu|aosom|vidaxl|costway)\b|[A-Z]:[\\/]Users[\\/]\d{4,}[\\/]')
SECRET = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|pypi-[A-Za-z0-9_-]{35,}|gh[pousr]_[A-Za-z0-9]{24,})\b')


def inventory():
    paths = {ROOT / name for name in ROOT_FILES}
    for pattern in PATTERNS:
        paths.update(ROOT.glob(pattern))
    files, errors = [], []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.is_symlink():
            errors.append({'file': relative, 'reason': 'missing_or_linked_file'})
            continue
        raw = path.read_bytes()
        content = raw.decode('utf-8-sig')
        if SECRET.search(content):
            errors.append({'file': relative, 'reason': 'credential_like_value'})
        # Boundary tests and this scanner legitimately name forbidden roots.
        if relative.startswith('src/'):
            if FORBIDDEN_TEXT.search(content):
                errors.append({'file': relative, 'reason': 'private_domain_or_personal_path'})
            if path.suffix == '.py':
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        modules = [node.module or '']
                    if any(name.split('.')[0] in FORBIDDEN_IMPORTS for name in modules):
                        errors.append({'file': relative, 'line': node.lineno, 'reason': 'private_import'})
        files.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    return {'manifest_version': 1, 'scope': 'explicit-public-files-only', 'files': files}, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    manifest, errors = inventory()
    path = ROOT / 'PUBLIC_FILES.json'
    if errors:
        print(json.dumps({'status': 'failed', 'findings': errors}, indent=2))
        return 1
    if args.write_manifest:
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    elif not path.exists() or json.loads(path.read_text(encoding='utf-8')) != manifest:
        print('Public manifest missing or stale; review changes before regenerating.')
        return 1
    print(json.dumps({'status': 'passed', 'file_count': len(manifest['files']),
                      'scope': manifest['scope']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
