# -*- coding: utf-8 -*-
"""ExecutionAdapter 输入输出契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..execution_mode import ExecutionMode, resolve_pipeline_execution_mode
from ..permissions import PermissionLevel
from ..schemas import Observation, ObservationStatus


@dataclass
class AdapterInput:
    step_id: str
    module: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    execution_mode: ExecutionMode = ExecutionMode.DRY_RUN
    dry_run: bool = True
    user_confirmed: bool = False
    permission_level: PermissionLevel = PermissionLevel.L2_LOCAL_DRAFT_WRITE
    adapter_name: Optional[str] = None
    artifact_binding: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.execution_mode, str):
            self.execution_mode = ExecutionMode.coerce(self.execution_mode)
        if self.execution_mode != ExecutionMode.DRY_RUN:
            self.dry_run = self.execution_mode == ExecutionMode.DRY_RUN
        else:
            self.execution_mode = resolve_pipeline_execution_mode(dry_run=self.dry_run)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "module": self.module,
            "action": self.action,
            "params": self.params,
            "execution_mode": self.execution_mode.value,
            "dry_run": self.execution_mode == ExecutionMode.DRY_RUN,
            "user_confirmed": self.user_confirmed,
            "permission_level": self.permission_level.name,
            "adapter_name": self.adapter_name,
        }


@dataclass
class AdapterOutput:
    status: str
    summary: str
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    raw_output_ref: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    suggested_next_action: Optional[str] = None

    def to_observation(self, step_id: str) -> Observation:
        return Observation(
            step_id=step_id,
            status=self.status,
            summary=self.summary,
            raw_output_ref=self.raw_output_ref,
            artifacts=list(self.artifacts),
            error=self.error,
            suggested_next_action=self.suggested_next_action,
        )


def blocked_output(
    *,
    summary: str,
    error_type: str,
    message: str,
    suggested_next_action: str = "stop",
    retryable: bool = False,
) -> AdapterOutput:
    return AdapterOutput(
        status=ObservationStatus.FAILED.value,
        summary=summary,
        error={
            "type": error_type,
            "message": message,
            "retryable": retryable,
        },
        suggested_next_action=suggested_next_action,
    )
