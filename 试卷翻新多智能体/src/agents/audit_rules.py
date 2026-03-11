"""
医学试卷翻新多智能体系统 - 审核规则引擎

负责基于规则的快速检查：连续相同片段检测、Levenshtein 相似度、
题型/选项结构一致性校验、隐性题型漂移检测等。
"""

import re
import logging
from typing import List, Optional

from ..schemas import (
    Question,
    GeneratedQuestion,
    AuditResult,
    QualityScores,
    QuestionType,
    CLINICAL_CASE_TYPES,
)

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.65
TRUE_FALSE_OPTION_TEXTS = {"正确", "错误", "对", "错", "是", "否"}


# ── 文本相似度工具 ────────────────────────────────────────────


def find_long_matches(text1: str, text2: str, min_length: int) -> List[str]:
    """基于动态规划查找两个文本中所有 >= min_length 的最长公共子串。"""
    text1 = re.sub(r'\s+', '', text1)
    text2 = re.sub(r'\s+', '', text2)

    if not text1 or not text2:
        return []

    n, m = len(text1), len(text2)
    prev = [0] * (m + 1)
    matches: List[str] = []

    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] >= min_length:
                    is_boundary = (i == n or j == m or text1[i] != text2[j])
                    if is_boundary:
                        match_str = text1[i - curr[j]:i]
                        if not any(match_str in existing or existing in match_str for existing in matches):
                            matches.append(match_str)
        prev = curr

    return matches


def levenshtein_similarity(text1: str, text2: str) -> float:
    """计算归一化 Levenshtein 相似度，范围 0~1。"""
    text1 = re.sub(r'\s+', '', text1)
    text2 = re.sub(r'\s+', '', text2)

    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    if len(text1) < len(text2):
        text1, text2 = text2, text1

    previous = list(range(len(text2) + 1))
    for i, char1 in enumerate(text1, 1):
        current = [i]
        for j, char2 in enumerate(text2, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char1 != char2)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    distance = previous[-1]
    max_len = max(len(text1), len(text2))
    return 1 - distance / max_len


def looks_like_case_stem(content: str) -> bool:
    """判断题干是否具有病例题特征。"""
    text = re.sub(r'\s+', '', content)
    patterns = [
        r"(患者|病人).{0,8}\d+岁",
        r"\d+岁.{0,8}(患者|病人)",
        r"(男性|女性).{0,8}\d+岁",
        r"(主诉|现病史|既往史|入院|查体|体温|脉搏|血压|护理评估)",
    ]
    matched = sum(bool(re.search(pattern, text)) for pattern in patterns)
    # 匹配 2 个或以上模式即认为是病例题
    return matched >= 2


def is_allowed_fixed_option_copy(original: Question, orig_opt: str, gen_opt: str) -> bool:
    """判断判断题的固定选项是否允许完全相同。"""
    if original.type != QuestionType.TRUE_FALSE:
        return False
    return orig_opt == gen_opt and orig_opt in TRUE_FALSE_OPTION_TEXTS and gen_opt in TRUE_FALSE_OPTION_TEXTS


def count_subquestions(content: str) -> int:
    """统计题干中的小问数量。"""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    scored_markers = sum(bool(re.search(r'[（(]\s*\d+\s*分\s*[）)]', line)) for line in lines)
    if scored_markers > 0:
        return scored_markers
    markers = 0
    for line in lines:
        if any(tag in line for tag in ["第一问", "第二问", "第三问", "第四问"]):
            markers += 1
        elif line.endswith("？") or line.endswith("?(10分)") or line.endswith("？（10分）"):
            markers += 1
    return markers if markers > 0 else 1


# ── 规则检查主函数 ────────────────────────────────────────────


def rule_based_check(
    original: Question,
    generated: GeneratedQuestion,
    plagiarism_threshold: int,
) -> dict:
    """基于规则的快速检查：检测连续相同字符、完全相同选项、题型一致性等。"""
    issues: List[str] = []
    suggestions: List[str] = []
    stem_copied = False
    options_copied = False
    type_mismatch = False
    option_structure_mismatch = False
    hidden_type_drift = False
    subquestion_mismatch = False
    similarity_score = 0.0

    if original.type != generated.type:
        type_mismatch = True
        issues.append(f"题型不一致：原题为{original.type.value}，新题为{generated.type.value}")
        suggestions.append(f"必须保持题型为{original.type.value}，不得改变")

    # ── 判断题专项检测：是否被改成了选择题 ──
    if original.type == QuestionType.TRUE_FALSE:
        if len(generated.options) > 2:
            type_mismatch = True
            issues.append(
                f"判断题被改成了选择题！判断题只能有2个选项(A/B)，"
                f"但新题有{len(generated.options)}个选项"
            )
            suggestions.append("判断题必须只保留 A（正确）和 B（错误）两个选项")
        if generated.options and set(generated.options.keys()) != {"A", "B"}:
            type_mismatch = True
            issues.append(
                f"判断题选项标签被篡改：应为 A/B，实际为 {list(generated.options.keys())}"
            )
            suggestions.append("判断题选项标签必须是 A 和 B")
        content_flat = re.sub(r"\s+", "", generated.content)
        choice_pats = [r"以下哪", r"下列哪", r"哪项是", r"哪个是", r"哪一项",
                       r"不属于.*的是", r"属于.*的是", r"不正确的是", r"正确的是",
                       r"不包括", r"不恰当的是", r"错误的是"]
        for pat in choice_pats:
            if re.search(pat, content_flat):
                hidden_type_drift = True
                issues.append(f"判断题使用了选择题式问法（{pat}），题干必须是陈述句")
                suggestions.append("判断题题干应直接陈述一个事实，让考生判断对错，不能用「以下哪项」等选择式提问")
                break

    # ── 简答题/填空题专项检测：是否被改成了选择题 ──
    if original.type in {QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK}:
        if generated.options:
            type_mismatch = True
            issues.append(
                f"{original.type.value}被改成了选择题！"
                f"{original.type.value}不能有任何选项，"
                f"但新题有{len(generated.options)}个选项"
            )
            suggestions.append(f"{original.type.value}不能有选项，必须保持文字作答形式")

    if len(original.options) != len(generated.options):
        option_structure_mismatch = True
        issues.append(f"选项数量不一致：原题为{len(original.options)}个，新题为{len(generated.options)}个")
        suggestions.append(f"请保持选项数量为 {len(original.options)} 个")

    if original.options and generated.options and set(original.options.keys()) != set(generated.options.keys()):
        option_structure_mismatch = True
        issues.append("选项标签结构不一致")
        suggestions.append("请保持与原题相同的选项标签和作答结构")

    # 隐性题型漂移检测：只检查明显的病例化问题
    # 对于非病例题，如果新题有明显的病例化特征（3个或以上模式），才判定为漂移
    original_case_like = original.type in CLINICAL_CASE_TYPES
    if not original_case_like:
        # 非病例题：检查是否被病例化（使用更严格的阈值）
        text = re.sub(r'\s+', '', generated.content)
        patterns = [
            r"(患者|病人).{0,8}\d+岁",
            r"\d+岁.{0,8}(患者|病人)",
            r"(男性|女性).{0,8}\d+岁",
            r"(主诉|现病史|既往史|入院|查体|体温|脉搏|血压|护理评估)",
        ]
        matched = sum(bool(re.search(pattern, text)) for pattern in patterns)
        if matched >= 3 or bool(generated.patient_background):
            hidden_type_drift = True
            issues.append("非病例题被病例化，疑似发生题型漂移")
            suggestions.append("请保持非病例题的简洁形式，不要添加病例化描述")
    else:
        # 病例题：检查是否缺少病例化特征
        if not looks_like_case_stem(generated.content) and not generated.patient_background:
            hidden_type_drift = True
            issues.append("病例题缺少病例化特征，疑似题型漂移")
            suggestions.append("请保持病例题的病例化结构")

    original_subquestion_count = getattr(original, 'subquestion_count', 1)
    generated_subquestion_count = count_subquestions(generated.content)
    if original_subquestion_count != generated_subquestion_count:
        subquestion_mismatch = True
        issues.append(f"小问数量不一致：原题 {original_subquestion_count} 个，新题 {generated_subquestion_count} 个")
        suggestions.append("请保持与原题完全一致的小问数量，不得拆分或合并")

    stem_similarity = levenshtein_similarity(original.content, generated.content)
    all_orig = original.content + " " + " ".join(original.options.values())
    all_gen = generated.content + " " + " ".join(generated.options.values())
    overall_similarity = levenshtein_similarity(all_orig, all_gen)
    similarity_score = max(similarity_score, stem_similarity, overall_similarity)

    if stem_similarity >= SIMILARITY_THRESHOLD:
        issues.append(f"题干 Levenshtein 相似度过高：{stem_similarity:.2f}")
        suggestions.append("请重写题干句式与信息组织方式，避免贴近原文")

    if overall_similarity >= SIMILARITY_THRESHOLD:
        issues.append(f"整体 Levenshtein 相似度过高：{overall_similarity:.2f}")
        suggestions.append("请同步重写题干与选项，进一步拉开与原题的字符串差异")

    stem_matches = find_long_matches(original.content, generated.content, plagiarism_threshold)
    if stem_matches:
        stem_copied = True
        issues.append(f"题干存在照搬：发现 {len(stem_matches)} 处连续相同片段")
        suggestions.extend(f"请改写：'{m}'" for m in stem_matches[:3])
        similarity_score = max(similarity_score, 0.7)

    for key, orig_opt in original.options.items():
        gen_opt = generated.options.get(key, "")
        if is_allowed_fixed_option_copy(original, orig_opt, gen_opt):
            continue
        if orig_opt == gen_opt:
            options_copied = True
            issues.append(f"选项{key}与原题完全相同")
            suggestions.append(f"请重写选项{key}：'{orig_opt}'")
            similarity_score = max(similarity_score, 0.8)
        elif find_long_matches(orig_opt, gen_opt, plagiarism_threshold):
            options_copied = True
            issues.append(f"选项{key}存在照搬片段")
            similarity_score = max(similarity_score, 0.6)

    if len(find_long_matches(all_orig, all_gen, plagiarism_threshold + 2)) > 3:
        issues.append("整体相似度过高，存在多处照搬")
        similarity_score = max(similarity_score, 0.7)

    has_critical = (
        (stem_copied and options_copied)
        or similarity_score >= SIMILARITY_THRESHOLD
        or type_mismatch
        or option_structure_mismatch
        or hidden_type_drift
        or subquestion_mismatch
    )
    return {
        "has_critical_issue": has_critical,
        "stem_copied": stem_copied,
        "options_copied": options_copied,
        "type_mismatch": type_mismatch,
        "option_structure_mismatch": option_structure_mismatch,
        "hidden_type_drift": hidden_type_drift,
        "subquestion_mismatch": subquestion_mismatch,
        "similarity_score": similarity_score,
        "issues": issues,
        "suggestions": suggestions,
    }


# ── 早期拒绝 ─────────────────────────────────────────────────


def early_reject(rule_issues: dict, question_id: str) -> Optional[AuditResult]:
    """规则预检：严重问题直接拒绝，返回 AuditResult 或 None。"""

    _zero_quality = QualityScores(
        fluency=0, clarity=0, conciseness=0, relevance=0,
        consistency=0, answerability=0, answer_consistency=0,
    )

    def _reject(reason: str, knowledge_ok: bool = False) -> AuditResult:
        return AuditResult(
            passed=False,
            verdict="严重不合格",
            reason=reason,
            similarity_score=rule_issues["similarity_score"],
            issues=rule_issues["issues"],
            suggestions=rule_issues["suggestions"],
            stem_original=not rule_issues["stem_copied"],
            options_original=not rule_issues["options_copied"],
            knowledge_point_match=knowledge_ok,
            has_valid_citations=True,
            quality_scores=_zero_quality,
        )

    if rule_issues.get("type_mismatch"):
        logger.info(f"题目 {question_id} 题型不一致，直接拒绝")
        return _reject("题型不一致：新题必须与原题保持相同题型")

    if rule_issues.get("option_structure_mismatch"):
        logger.info(f"题目 {question_id} 选项结构不一致，直接拒绝")
        return _reject("选项结构不一致：新题必须与原题保持相同选项数量和作答形式", knowledge_ok=True)

    if rule_issues.get("hidden_type_drift"):
        logger.info(f"题目 {question_id} 存在隐性题型漂移，直接拒绝")
        return _reject("题干结构与原题不一致，疑似发生隐性题型变化", knowledge_ok=True)

    if rule_issues.get("subquestion_mismatch"):
        logger.info(f"题目 {question_id} 小问数量不一致，直接拒绝")
        return _reject("小问数量不一致：翻新题必须与原题保持相同的小问数量", knowledge_ok=True)

    if rule_issues["has_critical_issue"] and rule_issues["similarity_score"] >= SIMILARITY_THRESHOLD:
        logger.info(f"题目 {question_id} 规则预检未通过")
        result = _reject("规则检查未通过：相似度过高，题干和选项存在照搬", knowledge_ok=True)
        result.suggestions = result.suggestions + [
            "请大幅改写题干，避免连续相同片段",
            "请重新设计选项，确保与原题差异足够",
        ]
        return result

    return None
