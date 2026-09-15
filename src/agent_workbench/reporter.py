from __future__ import annotations
from typing import Any, Dict, List, Optional
BLOCKED_ERROR_TYPES = frozenset({'blocked_by_permission', 'live_adapter_not_available', 'local_adapter_not_available', 'missing_adapter', 'execution_mode_blocked'})

def resolve_execution_modes(*, requested_execution_mode: str, execution_status: str, completed_steps: Optional[List[Dict[str, Any]]]=None, failed_steps: Optional[List[Dict[str, Any]]]=None, dry_run: Optional[bool]=None) -> Dict[str, str]:
    """
    推导 report 中的 requested / actual execution_mode。

    actual（execution_mode）在权限或 adapter 不可用时为 blocked；
    read-only/local-write/live 的业务执行失败仍保留 requested 模式。
    """
    requested = requested_execution_mode
    if dry_run is not None and (not requested_execution_mode):
        requested = 'dry_run' if dry_run else 'read_only'
    if execution_status == 'plan_only':
        return {'requested_execution_mode': requested, 'execution_mode': 'plan_only'}
    if execution_status in ('awaiting_confirmation', 'stopped'):
        return {'requested_execution_mode': requested, 'execution_mode': 'blocked'}
    actual = requested
    for step in failed_steps or []:
        err = step.get('error') or {}
        if err.get('type') in BLOCKED_ERROR_TYPES:
            actual = 'blocked'
            break
    return {'requested_execution_mode': requested, 'execution_mode': actual}

def resolve_execution_mode(*, requested_execution_mode: str='dry_run', execution_status: str, completed_steps: Optional[List[Dict[str, Any]]]=None, failed_steps: Optional[List[Dict[str, Any]]]=None, dry_run: Optional[bool]=None) -> str:
    """兼容旧调用：返回 actual execution_mode。"""
    if dry_run is not None and requested_execution_mode == 'dry_run' and (not dry_run):
        requested_execution_mode = 'read_only'
    modes = resolve_execution_modes(requested_execution_mode=requested_execution_mode, execution_status=execution_status, completed_steps=completed_steps, failed_steps=failed_steps)
    return modes['execution_mode']
