"""Contract-driven required parameter checks."""


def check_required_params(registry, module, action, params):
    return [dict(module=module, action=action, param=name)
            for name in registry.get_required_params(module, action)
            if params.get(name) is None or params.get(name) == []
            or (isinstance(params.get(name), str) and not params[name].strip())]
