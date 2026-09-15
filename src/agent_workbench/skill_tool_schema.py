# -*- coding: utf-8 -*-
"""Company Skills Harness Tool schemas。不是业务工具，不进入 Adapter。"""

from __future__ import annotations

from typing import Any, Dict, List

SKILLS_LIST_TOOL = "skills__list"
SKILL_LOAD_TOOL = "skills__load"
SKILL_RESOURCE_LOAD_TOOL = "skills__resource_load"
SKILL_HARNESS_TOOL_NAMES = frozenset(
    {SKILLS_LIST_TOOL, SKILL_LOAD_TOOL, SKILL_RESOURCE_LOAD_TOOL}
)


def is_skill_harness_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip() in SKILL_HARNESS_TOOL_NAMES


def build_skill_harness_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": SKILLS_LIST_TOOL,
                "description": (
                    "List company Skill catalog metadata visible in the current Profile. "
                    "Returns name, description, version and checksum only; never Skill body. "
                    "Advisory operating methods only; does not grant tools or bypass Tool Guard."
                )[:1024],
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": SKILL_LOAD_TOOL,
                "description": (
                    "Load one Skill body by name after SkillGuard checks. "
                    "Optional version must match the Profile pin/catalog snapshot. "
                    "Does not expand tool permissions or execute adapters."
                )[:1024],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill name from the current catalog",
                        },
                        "version": {
                            "type": "string",
                            "description": "Optional exact version pin, e.g. 1.0.0",
                        },
                    },
                    "required": ["skill_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": SKILL_RESOURCE_LOAD_TOOL,
                "description": (
                    "Load one declared Skill resource (utf-8 text) after SkillGuard checks. "
                    "resource_path must already be declared on the Skill. Scripts are denied."
                )[:1024],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "Skill name that declared the resource",
                        },
                        "resource_path": {
                            "type": "string",
                            "description": "Declared relative resource path, e.g. references/notes.md",
                        },
                        "version": {
                            "type": "string",
                            "description": "Optional exact version pin, e.g. 1.0.0",
                        },
                    },
                    "required": ["skill_name", "resource_path"],
                    "additionalProperties": False,
                },
            },
        },
    ]
