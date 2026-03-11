"""
医学试卷翻新多智能体系统 - 翻新卷 Markdown 解析器

将系统输出的翻新卷 Markdown 文件反向解析为 GeneratedQuestion 列表，
以便独立审核工作流加载并逐题审核。
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

from .schemas import GeneratedQuestion, QuestionType, QualityScores, RenovationMethod

logger = logging.getLogger(__name__)

_TYPE_MAP = {v.value: v for v in QuestionType}

_RENOVATION_MAP = {
    "增加内容": RenovationMethod.ADD_CONTENT,
    "选项替换": RenovationMethod.OPTION_REPLACE,
    "同义替换": RenovationMethod.SYNONYM_REPLACE,
    "维度转换": RenovationMethod.DIMENSION_CONVERT,
}


def parse_generated_markdown(path: str | Path) -> List[GeneratedQuestion]:
    """解析翻新卷 Markdown 文件，返回 GeneratedQuestion 列表。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    blocks = re.split(r"^## 第 \d+ 题", text, flags=re.MULTILINE)
    if len(blocks) < 2:
        raise ValueError(f"未在 {path.name} 中找到题目（期望 '## 第 N 题' 格式）")

    questions: List[GeneratedQuestion] = []
    for idx, block in enumerate(blocks[1:], 1):
        try:
            q = _parse_question_block(block, idx)
            questions.append(q)
        except Exception as e:
            logger.warning(f"解析第 {idx} 题失败，跳过: {e}")

    logger.info(f"从 {path.name} 解析出 {len(questions)} 道题目")
    return questions


def _parse_question_block(block: str, idx: int) -> GeneratedQuestion:
    """解析单道题目的 Markdown 块。"""
    knowledge_point = _extract_field(block, r"\*\*【考点】\*\*\s*(.+)")
    type_str = _extract_field(block, r"\*\*【题干】\*\*\s*\((.+?)\)")
    answer = _extract_field(block, r"\*\*【正确答案】\*\*\s*(.+)")
    strategy = _extract_field(block, r"\*\*【翻新策略】\*\*\s*(.+)")
    explanation = _extract_multiline(block, r"\*\*【深度解析】\*\*")

    question_type = _TYPE_MAP.get(type_str, QuestionType.SINGLE_CHOICE)

    content = _extract_stem(block)
    options = _extract_options(block)
    quality_scores = _extract_quality_scores(block)

    renovation_method = None
    if strategy:
        for keyword, method in _RENOVATION_MAP.items():
            if keyword in strategy:
                renovation_method = method
                break

    return GeneratedQuestion(
        id=f"parsed_{idx}",
        type=question_type,
        content=content,
        knowledge_point=knowledge_point or "",
        options=options,
        answer=answer or "",
        explanation=explanation or "",
        explanation_with_citations=explanation or "",
        original_question_id=f"orig_{idx}",
        generation_attempt=1,
        passed_audit=False,
        renovation_method=renovation_method,
        renovation_strategy_detail=strategy,
        quality_scores=quality_scores,
    )


def _extract_field(block: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, block)
    return m.group(1).strip() if m else None


def _extract_stem(block: str) -> str:
    """提取题干内容（位于【题干】和【选项】/【正确答案】之间）。"""
    m = re.search(
        r"\*\*【题干】\*\*\s*\(.+?\)\s*\n(.*?)(?=\*\*【选项】|\*\*【正确答案】)",
        block,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*【题干】\*\*\s*\(.+?\)\s*\n(.*?)(?=---|\Z)", block, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_options(block: str) -> dict:
    """提取选项字典。"""
    options = {}
    m = re.search(
        r"\*\*【选项】\*\*\s*\n(.*?)(?=\*\*【正确答案】|\*\*【深度解析】|---|\Z)",
        block,
        re.DOTALL,
    )
    if not m:
        return options
    for line in m.group(1).strip().split("\n"):
        opt_match = re.match(r"[-*]\s*([A-Z])[.、．]\s*(.*)", line.strip())
        if opt_match:
            options[opt_match.group(1)] = opt_match.group(2).strip()
    return options


def _extract_multiline(block: str, start_pattern: str) -> Optional[str]:
    """提取多行字段内容（从标记行到下一个标记或分隔线）。"""
    m = re.search(
        start_pattern + r"\s*\n(.*?)(?=\*\*【|---|\Z)",
        block,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _extract_quality_scores(block: str) -> Optional[QualityScores]:
    """尝试从质量评分行解析评分。"""
    m = re.search(r"\*\*【质量评分】\*\*\s*(.+)", block)
    if not m:
        return None
    line = m.group(1)

    def _get(name: str) -> int:
        match = re.search(rf"{name}=(\d)", line)
        return int(match.group(1)) if match else 0

    # 兼容新旧两种维度名
    if "流畅性" in line:
        return QualityScores(
            fluency=_get("流畅性"),
            clarity=_get("清晰度"),
            conciseness=_get("简洁性"),
            relevance=_get("相关性"),
            consistency=_get("一致性"),
            answerability=_get("可回答性"),
            answer_consistency=_get("答案一致性"),
        )
    return QualityScores(
        fluency=_get("适当性"),
        clarity=_get("清晰度"),
        conciseness=0,
        relevance=_get("相关性"),
        consistency=0,
        answerability=0,
        answer_consistency=0,
    )
