"""Single fail-closed adapter boundary, shared by graph and compatibility hosts."""
from typing import Protocol

from ..execution_mode import ExecutionMode
from ..permissions import (allows_read_only_execution, allows_local_write_execution,
                           allows_live_execution, requires_confirmation_for_level)
from ..registry import get_registry
from ..schemas import Observation
from .adapter_contracts import AdapterInput, AdapterOutput, blocked_output


class ToolExecutor(Protocol):
    def execute(self, request: AdapterInput) -> AdapterOutput: ...


class RegisteredToolExecutor:
    def __init__(self):
        self._handlers = {}

    def register(self, name, handler, *, modes=(ExecutionMode.DRY_RUN,)):
        for mode in modes:
            key = (ExecutionMode(mode), name)
            if key in self._handlers:
                raise ValueError("adapter_already_registered")
            self._handlers[key] = handler

    def execute(self, request):
        handler = self._handlers.get((request.execution_mode, request.adapter_name))
        if handler is None:
            return blocked_output(summary="Adapter unavailable", error_type="missing_adapter",
                                  message="No registered adapter for the requested mode")
        return handler(request)


def execute_adapter(adapter_input, registry=None):
    from ..artifact_review import verify_adapter_binding, artifact_blocked_observation
    from ..parameter_verification import _validate_schema
    reg = registry or get_registry()
    mode = ExecutionMode(adapter_input.execution_mode)
    module, action = adapter_input.module, adapter_input.action
    def blocked(reason):
        return blocked_output(summary="Execution blocked", error_type=reason,
                              message=reason).to_observation(adapter_input.step_id)
    valid, _ = reg.validate_module_action(module, action)
    if not valid:
        return blocked("unknown_capability")
    if reg.get_execution_mode_status(module, action, mode.value) != "available":
        return blocked("execution_mode_blocked")
    if _validate_schema(reg.get_parameter_contract(module, action), adapter_input.params):
        return blocked("parameter_validation_failed")
    permission = reg.get_risk_level(module, action)
    permitted = {ExecutionMode.READ_ONLY: allows_read_only_execution,
                 ExecutionMode.LOCAL_WRITE: allows_local_write_execution,
                 ExecutionMode.LIVE: allows_live_execution}.get(mode)
    if permitted and not permitted(permission):
        return blocked("blocked_by_permission")
    if mode == ExecutionMode.LIVE and requires_confirmation_for_level(permission) and not adapter_input.user_confirmed:
        return blocked("blocked_by_permission")
    error = verify_adapter_binding(adapter_input, reg)
    if error:
        return artifact_blocked_observation(adapter_input.step_id, error)
    executor = getattr(reg, "executor", None)
    if executor is None:
        return blocked("missing_adapter")
    adapter_input.adapter_name = reg.get_execution_adapter(module, action)
    try:
        output = executor.execute(adapter_input)
    except Exception:
        return blocked("adapter_execution_failed")
    if isinstance(output, AdapterOutput):
        return output.to_observation(adapter_input.step_id)
    if isinstance(output, Observation):
        return output
    return blocked("invalid_adapter_output")
