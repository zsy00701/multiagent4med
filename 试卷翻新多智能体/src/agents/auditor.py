"""
医学试卷翻新多智能体系统 - 质量审核智能体 (QGEval 2.0)

基于 QGEval 2.0 框架的七维评估体系，对翻新题目进行深度剖析、打分和纠错。
规则检查逻辑已拆分至 audit_rules 模块。
"""

import re
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from ..llm_client import LLMClient, get_llm_client
from ..schemas import (
    Question,
    GeneratedQuestion,
    AuditResult,
    QualityScores,
    CLINICAL_CASE_TYPES,
    QuestionType,
)
from ..config import settings
from .audit_rules import (
    SIMILARITY_THRESHOLD,
    rule_based_check,
    early_reject,
)

logger = logging.getLogger(__name__)


class AuditResponse(BaseModel):
    """LLM 审核响应结构（QGEval 2.0）"""
    verdict: str = Field(..., description="综合评判：通过/需修改/严重不合格")
    similarity_score: float = Field(..., description="相似度评分 0-1")
    stem_original: bool = Field(..., description="题干是否原创")
    options_original: bool = Field(..., description="选项是否原创")
    knowledge_point_match: bool = Field(..., description="考点是否一致")
    has_valid_citations: bool = Field(..., description="引用是否有效")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="修改建议")
    reason: str = Field(..., description="审核结论")
    quality_scores: dict = Field(default_factory=dict, description="QGEval 2.0 七维质量评分")
    deduction_details: List[str] = Field(default_factory=list, description="扣分项详述")


AUDITOR_SYSTEM_PROMPT = """你是医学试卷质量审核专家（QGEval 2.0 评估专家）。
你的任务是接收医学/护理翻新题目，严格审核其医学事实准确性、题型规范性和质量评分。评估必须客观、尖锐且具有建设性。

## 🔴 第一优先级：医学事实准确性审查

在进行任何评分之前，你必须首先审查以下内容：
1. **正确答案是否真的正确？** 答案所依据的医学知识是否符合教科书和临床指南？
2. **干扰项是否真的错误？** 是否存在把正确的医学知识当做干扰项的情况？如果干扰项其实也是正确的，该题有多个正确答案，必须判定为不合格。
3. **题干中的医学描述是否准确？** 包括：病理机制、药理作用、解剖位置、护理操作、检查方法、正常值范围等。
4. **是否存在编造的医学概念？** 如不存在的药名、不存在的术式、虚构的分类方式等。
5. **数值是否准确？** 如体温、血压、实验室检查的正常范围、药物剂量等。

⚠️ 只要发现任何医学事实性错误，直接判定为「严重不合格」，无论其他维度评分多高。

## ⚠️ 案例分析题/论述题/简答题的审核宽容度

对于案例分析题、论述题、简答题等主观题型，审核标准需适当灵活：
1. **参考答案是要点式的**，不要求措辞完美，只要医学方向正确即可
2. **护理措施的表述灵活性**：如「遵医嘱使用血管活性药物」和「配合医生使用升压药物」都是可接受的表述，不应因措辞差异判定为事实错误
3. **数值的合理范围**：如饮水量「2000-3000ml」是护理教科书常见推荐范围，除非明显违反禁忌（如心衰/肾衰患者限水），不应判定为事实错误
4. **引用标注**：简答题/论述题的参考答案中引用标注不够精准时，只要内容本身正确，应判「需修改」而非「严重不合格」
5. **仅对以下情况判定「严重不合格」**：明显违反医学常识（如把禁忌症当适应症）、编造不存在的药物/术式、护理操作方向性错误

## 🔴 第二优先级：题型格式规范性审查

1. **判断题**：题干必须是陈述句（可判断对错），不能使用「以下哪项」「哪个是」「不属于...的是」等选择题式问法。选项只能是 A.正确 / B.错误。
2. **单选题**：必须有且仅有一个正确答案，选项数量与原题一致。
3. **多选题**：答案必须包含 2 个或以上字母，选项数量与原题一致。
4. **简答题/填空题**：不能有任何选择选项。
5. **病例题**：必须包含完整的病例背景。

## 核心评估框架 (QGEval 2.0)

请依据以下 7 个维度对题目进行 1-5 分的评估：

1. **流畅性 (fluency)**：
   - 5分：句子结构完整，用词精准自然，无语法问题
   - 1分：文本支离破碎，充满严重语法错误

2. **清晰度 (clarity)**：
   - 5分：题目意图明确、焦点突出，无任何歧义
   - 1分：表述混乱，用词矛盾，完全无法理解询问焦点

3. **简洁性 (conciseness)**：
   - 5分：用词精炼，无冗余修饰语或重复信息
   - 1分：被大量无关信息淹没，严重浪费认知资源

4. **相关性 (relevance)**：
   - 5分：紧密围绕上下文核心事实提出
   - 1分：完全偏离上下文，或属于把病例题干删了丝毫不影响答题的"伪病例题"

5. **一致性 (consistency)**：
   - 5分：内部逻辑自洽，医学事实准确，所有条件在语义和逻辑上完全兼容
   - 3分：基本正确但某些细节可商榷
   - 1分：存在医学事实错误或原则性逻辑错误（如女性前列腺癌、把正确医学知识当干扰项）

6. **可回答性 (answerability)**：
   - 5分：问题明确聚焦，存在唯一确定的标准答案
   - 3分：答案基本确定但表述有小瑕疵
   - 1分：没有标准答案，或多个选项都可以是正确答案

7. **答案一致性 (answer_consistency)**：
   - 5分：正确答案确定无争议，所有干扰项确实是错误的
   - 3分：答案基本正确但个别干扰项的错误程度不够明显
   - 1分：正确答案有误，或干扰项中存在也正确的选项

## 同时检查

1. 题型是否与原题完全一致（特别注意判断题不能写成选择题形式）。
2. 考点是否一致。
3. 相似度是否 < 0.65（翻新后的题目必须与原题有显著差异）。
4. 选项结构是否一致。

## 综合评判标准

**选择题/判断题（客观题）**：
- **严重不合格**：存在明确的医学事实错误 / 题型格式错误 / 任何维度 <= 1 / 均分 < 2.0
- **需修改**：存在维度 < 3 但无维度 <= 1，且无事实性错误
- **通过**：所有维度 >= 3 且均分 >= 3.5，且无事实性错误

**案例分析题/论述题/简答题（主观题）**：
- **严重不合格**：存在方向性的医学错误（如把禁忌症当适应症、编造药物/术式） / 题型格式严重错误 / 均分 < 2.0
- **需修改**：护理措施表述不够规范但方向正确 / 引用不够精准 / 格式小问题
- **通过**：医学方向正确，护理措施合理，各维度均分 >= 3.0

返回 JSON 格式：
{
  "verdict": "通过/需修改/严重不合格",
  "similarity_score": 0.0-1.0,
  "stem_original": true/false,
  "options_original": true/false,
  "knowledge_point_match": true/false,
  "has_valid_citations": true/false,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "reason": "审核结论",
  "quality_scores": {
    "fluency": 1-5,
    "clarity": 1-5,
    "conciseness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5,
    "answerability": 1-5,
    "answer_consistency": 1-5
  },
  "deduction_details": ["仅列出得分低于4分的维度，指出题目中具体哪一句话、哪个选项触发了扣分"]
}

注意：stem_original 和 options_original 必须是布尔值（true/false），不能是字符串。
"""


class AuditorAgent:
    """防照搬审核智能体（QGEval 2.0 评估专家）"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        plagiarism_threshold: int = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.plagiarism_threshold = plagiarism_threshold or settings.generation.plagiarism_threshold

    # ── 公共入口 ──────────────────────────────────────────────

    def audit(self, original: Question, generated: GeneratedQuestion) -> AuditResult:
        """同步审核"""
        logger.info(f"审核题目 {generated.id}")
        rule_issues = rule_based_check(original, generated, self.plagiarism_threshold)

        early = early_reject(rule_issues, generated.id)
        if early:
            return early

        fast_pass = self._fast_pass_result(original, generated, rule_issues)
        if fast_pass is not None:
            return fast_pass

        prompt = self._build_audit_prompt(original, generated)
        try:
            response = self.llm.chat(
                prompt=prompt,
                system_prompt=AUDITOR_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=self._resolve_max_tokens(original),
                json_mode=True,
            )
            return self._merge_and_decide(response, rule_issues, generated.id)
        except Exception as e:
            logger.error(f"审核题目 {generated.id} 失败: {e}")
            return self._error_result(e)

    async def audit_async(self, original: Question, generated: GeneratedQuestion) -> AuditResult:
        """异步审核"""
        logger.info(f"异步审核题目 {generated.id}")
        rule_issues = rule_based_check(original, generated, self.plagiarism_threshold)

        early = early_reject(rule_issues, generated.id)
        if early:
            return early

        fast_pass = self._fast_pass_result(original, generated, rule_issues)
        if fast_pass is not None:
            return fast_pass

        prompt = self._build_audit_prompt(original, generated)
        try:
            response = await self.llm.achat(
                prompt=prompt,
                system_prompt=AUDITOR_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=self._resolve_max_tokens(original),
                json_mode=True,
            )
            return self._merge_and_decide(response, rule_issues, generated.id)
        except Exception as e:
            logger.error(f"异步审核题目 {generated.id} 失败: {e}")
            return self._error_result(e)

    # ── 决策逻辑 ─────────────────────────────────────────────

    def _merge_and_decide(self, response: str, rule_issues: dict, question_id: str) -> AuditResult:
        """合并规则检查与 LLM 审核结果，做出最终决策"""
        try:
            data = self.llm.extract_json_from_response(response)
            for field in ("stem_original", "options_original", "knowledge_point_match", "has_valid_citations"):
                if isinstance(data.get(field), str):
                    data[field] = data[field].lower() in ("true", "是", "原创", "一致", "有效")
            parsed = AuditResponse.model_validate(data)
        except Exception as e:
            logger.error(f"解析审核响应失败: {e}, 原始响应: {response[:500]}")
            raise

        all_issues = list(set(rule_issues["issues"] + parsed.issues))
        all_suggestions = list(set(rule_issues["suggestions"] + parsed.suggestions))
        all_deductions = list(parsed.deduction_details)

        final_similarity = max(parsed.similarity_score, rule_issues["similarity_score"])

        quality_scores = None
        if parsed.quality_scores:
            try:
                quality_scores = QualityScores.model_validate(parsed.quality_scores)
            except Exception as e:
                logger.warning(f"质量评分解析失败: {e}")

        similarity_ok = final_similarity < SIMILARITY_THRESHOLD
        knowledge_ok = parsed.knowledge_point_match
        structure_ok = not rule_issues.get("option_structure_mismatch", False)
        quality_ok = quality_scores.passed if quality_scores else True

        should_pass = similarity_ok and knowledge_ok and structure_ok and quality_ok

        # 综合评判
        if should_pass:
            verdict = "通过"
        elif quality_scores and quality_scores.verdict == "严重不合格":
            verdict = "严重不合格"
        elif not similarity_ok or not structure_ok:
            verdict = "严重不合格"
        else:
            verdict = quality_scores.verdict if quality_scores else "需修改"

        if not should_pass and quality_scores:
            low = quality_scores.low_dims
            if low:
                all_issues.append(f"QGEval 评分不达标：{', '.join(low)}")
                all_suggestions.append(f"请重点提升以下维度：{', '.join(low)}")

        result = AuditResult(
            passed=should_pass,
            verdict=verdict,
            reason=parsed.reason,
            similarity_score=final_similarity,
            issues=all_issues,
            suggestions=all_suggestions,
            stem_original=parsed.stem_original and not rule_issues["stem_copied"],
            options_original=parsed.options_original and not rule_issues["options_copied"],
            knowledge_point_match=parsed.knowledge_point_match,
            has_valid_citations=parsed.has_valid_citations,
            quality_scores=quality_scores,
            deduction_details=all_deductions,
        )
        logger.info(
            f"题目 {question_id} 审核结果：{verdict}"
            + (f" (QGEval 均分: {quality_scores.average:.1f})" if quality_scores else "")
        )
        return result

    def _fast_pass_result(
        self, original: Question, generated: GeneratedQuestion, rule_issues: dict
    ) -> Optional[AuditResult]:
        """低风险客观题走规则快速审核，减少额外 token。"""
        complex_types = CLINICAL_CASE_TYPES | {
            QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK,
            QuestionType.CASE_ANALYSIS, QuestionType.A3, QuestionType.A4,
        }
        if original.type in complex_types:
            return None
        if rule_issues["issues"]:
            return None
        if rule_issues["similarity_score"] >= 0.40:
            return None
        if not generated.quality_scores or not generated.quality_scores.passed:
            return None

        return AuditResult(
            passed=True,
            verdict="通过",
            reason="规则快速审核通过",
            similarity_score=rule_issues["similarity_score"],
            issues=[],
            suggestions=[],
            stem_original=True,
            options_original=True,
            knowledge_point_match=True,
            has_valid_citations=True,
            quality_scores=generated.quality_scores,
        )

    @staticmethod
    def _error_result(e: Exception) -> AuditResult:
        return AuditResult(
            passed=False,
            verdict="严重不合格",
            reason=f"审核过程出错: {e}",
            similarity_score=0.5,
            issues=["审核过程发生错误"],
            suggestions=["请重新生成"],
        )

    # ── Prompt 构建 ───────────────────────────────────────────

    def _build_audit_prompt(self, original: Question, generated: GeneratedQuestion) -> str:
        orig_options = "\n".join(f"  {k}. {v}" for k, v in original.options.items())
        gen_options = "\n".join(f"  {k}. {v}" for k, v in generated.options.items())

        parts = [
            "## 原始题目\n",
            f"**题型**：{original.type.value}",
            f"**考点**：{original.knowledge_point}",
            f"**题干**：{original.content}",
            f"**选项**：\n{orig_options}",
            f"**答案**：{original.answer}",
            "\n## 翻新题目（待审核）\n",
            f"**题型**：{generated.type.value}",
            f"**翻新策略**：{generated.renovation_strategy_detail or (generated.renovation_method.value if generated.renovation_method else '未指定')}",
            f"**题干**：{generated.content}",
            f"**选项**：\n{gen_options}",
            f"**答案**：{generated.answer}",
            f"**解析**：{self._trim_text(generated.explanation_with_citations, 220)}",
        ]

        if generated.patient_background:
            pb = generated.patient_background
            parts.append(f"\n**虚拟患者背景**：{pb.gender}，{pb.age}，主诉：{pb.chief_complaint}")
        elif generated.type in CLINICAL_CASE_TYPES:
            parts.append("\n**注意：本题为病例题但缺少虚拟患者背景结构化数据**")

        if generated.quality_scores:
            qs = generated.quality_scores
            parts.append(
                f"\n**生成时自检评分（QGEval 2.0）**："
                f"流畅性={qs.fluency} 清晰度={qs.clarity} 简洁性={qs.conciseness} "
                f"相关性={qs.relevance} 一致性={qs.consistency} "
                f"可回答性={qs.answerability} 答案一致性={qs.answer_consistency}"
            )

        has_options = bool(original.options)

        parts.extend([
            "\n## 审核任务\n",
            "请按照以下优先级顺序严格审核：",
            "",
            "### 🔴 第一步：医学事实准确性（一票否决）",
            "- 正确答案在医学上是否确实正确？有无教科书/指南依据？",
            "- 每个干扰项是否确实错误？是否存在干扰项其实也是对的（导致多个正确答案）？",
            "- 题干中的医学描述（药理、病理、解剖、护理操作等）是否准确？",
            "- 是否存在编造的概念、错误的数据或混淆的疾病特征？",
            "- ⚠️ 发现任何事实错误 → 直接判「严重不合格」",
            "",
            "### 🔴 第二步：题型格式规范性",
            "- 判断题：题干是否为陈述句？是否误用了「以下哪项」等选择题问法？",
            "- 选择题：选项数量、标签(A/B/C/D)是否与原题一致？单选是否只有一个正确答案？",
            "- 多选题：答案是否包含多个字母？",
            "- 简答题/填空题：是否错误地添加了选项？",
            "",
            "### 第三步：原创性与差异度",
            f"- 题干/选项是否有实质性改写？是否存在连续照搬（>{self.plagiarism_threshold}字相同）？",
            "- Levenshtein 相似度是否 < 0.65？",
            "",
            "### 第四步：QGEval 2.0 七维评分（1-5分）",
            "- 流畅性 / 清晰度 / 简洁性 / 相关性 / 一致性 / 可回答性 / 答案一致性",
            "- 特别注意「一致性」维度要考察医学事实准确性",
            "- 特别注意「答案一致性」维度要确认干扰项确实错误",
            "",
            "### 第五步：其他",
            "- 考点是否与原题一致",
            "- 病例背景（仅A2/A3/A4型题）是否完整",
            "- 引用标注是否合理",
            "\n## 输出格式要求\n",
            "请严格按照以下 JSON 格式输出审核结果：",
            "{",
            '  "verdict": "通过/需修改/严重不合格",',
            '  "similarity_score": 0.0-1.0,',
            '  "stem_original": true/false,',
            '  "options_original": {0},'.format(
                "true/false" if has_options else "true"
            ),
            '  "knowledge_point_match": true/false,',
            '  "has_valid_citations": true/false,',
            '  "issues": ["问题1", "问题2"],',
            '  "suggestions": ["建议1", "建议2"],',
            '  "reason": "审核结论",',
            '  "quality_scores": {',
            '    "fluency": 1-5,',
            '    "clarity": 1-5,',
            '    "conciseness": 1-5,',
            '    "relevance": 1-5,',
            '    "consistency": 1-5,',
            '    "answerability": 1-5,',
            '    "answer_consistency": 1-5',
            '  },',
            '  "deduction_details": ["仅列出得分<4的维度，尖锐指出具体哪句话/哪个选项触发扣分"]',
            "}",
            "\n注意：stem_original 和 options_original 必须是布尔值（true/false），不能是字符串或其他类型。",
        ])
        return "\n".join(parts)

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) <= max_chars else text[:max_chars] + '...'

    @staticmethod
    def _resolve_max_tokens(original: Question) -> int:
        if original.type in CLINICAL_CASE_TYPES or original.subquestion_count > 1:
            return 3072
        return 1536
