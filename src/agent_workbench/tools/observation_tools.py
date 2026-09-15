# -*- coding: utf-8 -*-
"""观察工具 — 将 adapter 输出压缩为 step summary。"""

from __future__ import annotations

from typing import Any, Dict

from ..schemas import Observation


def compact_observation(observation: Observation) -> Dict[str, Any]:
    return {
        "step_id": observation.step_id,
        "status": observation.status,
        "summary": observation.summary,
        "artifact_count": len(observation.artifacts),
        "artifact_refs": [a.get("path") for a in observation.artifacts if a.get("path")],
        "has_error": observation.error is not None,
    }
