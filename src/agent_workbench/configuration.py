"""Host configuration contracts; no application-specific configuration discovery."""
import copy
from typing import Protocol


class ConfigProvider(Protocol):
    def snapshot(self) -> dict: ...


class StaticConfigProvider:
    def __init__(self, values=None):
        self._values = copy.deepcopy(values or {})

    def snapshot(self):
        return copy.deepcopy(self._values)


def build_llm_config_availability_context(module_config):
    lines = []
    for module, values in (module_config or {}).items():
        if isinstance(values, dict):
            keys = sorted(str(k) for k, v in values.items() if v not in (None, '', [], {}))
            if keys:
                lines.append(f"- {module}: {', '.join(keys)}")
    if not lines:
        return ''
    return ('Runtime configuration supplies these parameter names. Ask only for missing '
            'required values. Current explicit user instructions override personal defaults, '
            'but never override mandatory host policies.\n' + '\n'.join(lines))
