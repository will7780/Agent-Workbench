import json
import runpy
from pathlib import Path


def test_manifest_order_and_hashes_are_platform_independent():
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / 'tools' / 'audit_public_tree.py'))
    current, errors = namespace['inventory']()
    assert not errors
    paths = [entry['path'] for entry in current['files']]
    assert paths == sorted(paths)
    assert current == json.loads((root / 'PUBLIC_FILES.json').read_text(encoding='utf-8'))
