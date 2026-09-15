# -*- coding: utf-8 -*-
"""Plan-First Agent 数据结构定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PlanStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    NEEDS_INFO = "needs_info"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_INPUT = "awaiting_input"
    PLAN_ONLY = "plan_only"


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class ReplannerAction(str, Enum):
    ASK_USER = "ask_user"
    RETRY_STEP = "retry_step"
    REQUIRE_CONFIRMATION = "require_confirmation"
    STOP = "stop"
    REPLAN = "replan"


@dataclass
class PlanStep:
    id: str
    module: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    risk_level: str = "low"
    requires_user_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "action": self.action,
            "params": self.params,
            "depends_on": self.depends_on,
            "expected_outputs": self.expected_outputs,
            "risk_level": self.risk_level,
            "requires_user_confirmation": self.requires_user_confirmation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        return cls(
            id=data["id"],
            module=data["module"],
            action=data["action"],
            params=dict(data.get("params") or {}),
            depends_on=list(data.get("depends_on") or []),
            expected_outputs=list(data.get("expected_outputs") or []),
            risk_level=data.get("risk_level", "low"),
            requires_user_confirmation=bool(data.get("requires_user_confirmation", False)),
        )


@dataclass
class Plan:
    plan_id: str
    plan_type: str
    goal: str
    risk_level: str
    requires_user_confirmation: bool
    created_from: str
    steps: List[PlanStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_type": self.plan_type,
            "goal": self.goal,
            "risk_level": self.risk_level,
            "requires_user_confirmation": self.requires_user_confirmation,
            "created_from": self.created_from,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            plan_id=data["plan_id"],
            plan_type=data["plan_type"],
            goal=data["goal"],
            risk_level=data.get("risk_level", "low"),
            requires_user_confirmation=bool(data.get("requires_user_confirmation", False)),
            created_from=data.get("created_from", "unknown"),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
        )


@dataclass
class Observation:
    step_id: str
    status: str
    summary: str
    raw_output_ref: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    suggested_next_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "summary": self.summary,
            "raw_output_ref": self.raw_output_ref,
            "artifacts": self.artifacts,
            "error": self.error,
            "suggested_next_action": self.suggested_next_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            step_id=data["step_id"],
            status=data["status"],
            summary=data["summary"],
            raw_output_ref=data.get("raw_output_ref"),
            artifacts=list(data.get("artifacts") or []),
            error=data.get("error"),
            suggested_next_action=data.get("suggested_next_action"),
        )


@dataclass
class ValidationResult:
    valid: bool
    plan_status: str
    errors: List[str] = field(default_factory=list)
    missing_info: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "plan_status": self.plan_status,
            "errors": self.errors,
            "missing_info": self.missing_info,
            "warnings": self.warnings,
        }


@dataclass
class ReplannerDecision:
    action: str
    reason: str
    step_id: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "step_id": self.step_id,
            "message": self.message,
        }
