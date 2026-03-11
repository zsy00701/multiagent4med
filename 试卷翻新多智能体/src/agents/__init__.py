"""
医学试卷翻新多智能体系统 - 智能体模块

包含三个核心智能体：
- AnalystAgent: 结构分析智能体，负责解析试卷
- GeneratorAgent: 试题生成智能体，负责生成翻新题目
- AuditorAgent: 防照搬审核智能体，负责质量审核

审核规则引擎（audit_rules）独立于智能体，提供纯规则检查能力。
"""

from .analyst import AnalystAgent
from .generator import GeneratorAgent
from .auditor import AuditorAgent
from . import audit_rules

__all__ = [
    "AnalystAgent",
    "GeneratorAgent",
    "AuditorAgent",
    "audit_rules",
]
