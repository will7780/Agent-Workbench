# -*- coding: utf-8 -*-
"""SkillProvider Protocol。load_resource() 复用 SkillLoadResult，不新增 ResourceResult。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from .skill_models import CompanyProfileSnapshot, SkillCatalogResult, SkillLoadResult, SkillRef


class SkillProvider(Protocol):
    provider_id: str

    def list_metadata(
        self,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillCatalogResult: ...

    def load_skill(
        self,
        skill_ref: SkillRef,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillLoadResult: ...

    def load_resource(
        self,
        skill_ref: SkillRef,
        resource_path: str,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillLoadResult: ...

    def health(
        self,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]: ...
