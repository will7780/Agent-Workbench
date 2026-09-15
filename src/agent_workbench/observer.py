# -*- coding: utf-8 -*-
"""Observer — 压缩 adapter 结果为 step summary。"""

from __future__ import annotations

from typing import Any, Dict, List

from .schemas import Observation
from .context_manager import compact_running_summary
from .tools.observation_tools import compact_observation


class Observer:
    def observe(self, observation: Observation) -> Dict[str, Any]:
        compact = compact_observation(observation)
        return {
            "step_id": observation.step_id,
            "status": observation.status,
            "summary": observation.summary,
            "compact": compact,
            "artifact_refs": compact.get("artifact_refs", []),
            "error": observation.error,
            "suggested_next_action": observation.suggested_next_action,
        }

    def append_running_summary(self, current: str, step_summary: str) -> str:
        if not current:
            merged = step_summary
        else:
            merged = f"{current}\n{step_summary}"
        return compact_running_summary(merged)

    def collect_artifacts(
        self,
        existing: List[Dict[str, Any]],
        observation: Observation,
    ) -> List[Dict[str, Any]]:
        merged = list(existing)
        for art in observation.artifacts:
            merged.append(art)
        return merged
