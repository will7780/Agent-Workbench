# -*- coding: utf-8 -*-
"""基础设施/平台失败语义分类（Phase 11.6.1）。"""

from __future__ import annotations

from typing import Optional

INFRASTRUCTURE_ERROR_TYPES = frozenset(
    {
        "gui_launch_failed",
        "browser_unavailable",
        "network_unavailable",
        "platform_unreachable",
    }
)


def classify_infrastructure_error(message: str) -> Optional[str]:
    """
    将底层 GUI/浏览器/网络/平台错误映射为结构化 error.type。

    编排层失败与基础设施失败分离：仅当消息明确指向基础设施时返回类型。
    """
    text = (message or "").lower()
    if not text:
        return None

    if any(k in text for k in ("browser", "chrome", "playwright", "selenium", "浏览器")):
        if any(k in text for k in ("unavailable", "不可用", "not found", "无法启动", "未安装")):
            return "browser_unavailable"

    if any(
        k in text
        for k in (
            "network",
            "connection refused",
            "timed out",
            "timeout",
            "unreachable",
            "局域网",
            "网络",
            "无法连接",
        )
    ):
        return "network_unavailable"

    if any(
        k in text
        for k in (
            "platform",
            "平台",
            "登录失败",
            "403",
            "502",
            "503",
        )
    ):
        return "platform_unreachable"

    if any(k in text for k in ("gui", "launch_tool", "tkinter", "启动失败", "gui_launch")):
        return "gui_launch_failed"

    return None
