"""Verify distribution payloads against the reviewed public-file manifest."""
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from audit_public_tree import ROOT, SECRET


def main():
    manifest = json.loads((ROOT / 'PUBLIC_FILES.json').read_text(encoding='utf-8'))
    expected = {item['path']: item['sha256'] for item in manifest['files']}
    reports = []
    for archive in sorted((ROOT / 'dist').iterdir()):
        if archive.suffix == '.whl':
            with zipfile.ZipFile(archive) as bundle:
                members = {name: bundle.read(name) for name in bundle.namelist() if not name.endswith('/')}
            mode = 'wheel'
        elif archive.name.endswith('.tar.gz'):
            with tarfile.open(archive) as bundle:
                entries = bundle.getmembers()
                assert all(entry.isfile() or entry.isdir() for entry in entries), 'Links are not allowed'
                members = {'/'.join(PurePosixPath(entry.name).parts[1:]): bundle.extractfile(entry).read()
                           for entry in entries if entry.isfile()}
            mode = 'sdist'
        else:
            continue
        package_count = 0
        for name, raw in members.items():
            assert '..' not in PurePosixPath(name).parts and not name.startswith('/'), 'Unsafe archive member'
            assert not SECRET.search(raw.decode('utf-8', errors='replace')), f'Credential-like content: {name}'
            source = 'src/' + name if mode == 'wheel' and name.startswith('agent_workbench/') else name
            if source in expected:
                assert hashlib.sha256(raw).hexdigest() == expected[source], f'Unreviewed bytes: {source}'
                if source.startswith('src/agent_workbench/'):
                    package_count += 1
                continue
            if mode == 'wheel':
                assert '.dist-info/' in name, f'Unexpected wheel file: {name}'
            else:
                assert name in {'PKG-INFO', 'setup.cfg', 'PUBLIC_FILES.json'} or name.startswith(
                    'src/agent_workbench.egg-info/'), f'Unexpected source archive file: {name}'
        required = sum(name.startswith('src/agent_workbench/') for name in expected)
        assert package_count == required, 'Package files are missing from the distribution'
        reports.append({'file': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                        'file_count': len(members), 'package_files': package_count})
    assert {item['file'].endswith('.whl') for item in reports} == {True, False}, 'Build wheel and sdist first'
    print(json.dumps({'status': 'passed', 'distributions': reports}, indent=2))


if __name__ == '__main__':
    main()
