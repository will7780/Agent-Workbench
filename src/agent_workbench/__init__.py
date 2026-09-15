"""Agent Workbench: a plugin-based LangGraph runtime."""
__version__ = "0.1.0a1"


def __getattr__(name):
    if name in {'AgentRuntime', 'RuntimeServices'}:
        from .runtime import AgentRuntime, RuntimeServices
        return {'AgentRuntime': AgentRuntime, 'RuntimeServices': RuntimeServices}[name]
    if name == 'ToolRegistry':
        from .registry import ToolRegistry
        return ToolRegistry
    raise AttributeError(name)
