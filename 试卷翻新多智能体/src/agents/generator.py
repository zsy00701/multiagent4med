"""
医学试卷翻新多智能体系统 - 试题生成智能体

核心智能体：负责根据原题和 RAG 上下文生成翻新题目。
遵循《泌尿外科/护理出题规范》，通过"增加内容"、"选项替换"、"同义替换"、"维度转换"生成高质量题目。
"""

import logging
import random
import re
import time
from typing import Optional, List
from pydantic import BaseModel, Field

from ..llm_client import LLMClient, get_llm_client
from ..schemas import (
    Question, GeneratedQuestion, RAGContext, Citation, AuditResult,
    RenovationMethod, PatientBackground, QualityScores,
    QuestionType, CLINICAL_CASE_TYPES,
)

logger = logging.getLogger(__name__)


class GeneratedQuestionResponse(BaseModel):
    """LLM 生成题目的响应结构（字段带默认值以容错 LLM 不完整输出）"""
    renovation_method: str = Field(default="同义替换", description="采用的翻新策略")
    renovation_strategy_detail: str = Field(default="策略C（同义替换）", description="本次使用的详细翻新策略组合")
    content: str = Field(default="", description="新题干（含完整病例背景）")
    options: dict = Field(default_factory=dict, description="新选项")
    answer: str = Field(default="", description="正确答案")
    explanation_with_citations: str = Field(default="", description="深度解析：含正确项逻辑及错误项排除原因")
    citations: List[dict] = Field(default_factory=list, description="引用来源")
    patient_background: Optional[dict] = Field(None, description="虚拟患者背景（病例题必填）")
    quality_self_check: Optional[dict] = Field(None, description="五维质量自检评分")


GENERATOR_SYSTEM_PROMPT = """你是护理出题专家，专门负责围绕指定知识点**从零创作全新试题**。

🎯 核心目标：你收到的原题仅仅是一个"知识点线索"——告诉你本题要考什么。你的任务是**完全抛开原题的表述**，结合参考资料中的专业内容，像一位资深命题组长一样**独立命制一道全新的题**。最终结果：把新题和原题放在一起，任何人都不会觉得它们出自同一套卷子。

硬约束（不可违反，按优先级排列）：
1. 🔴 **医学事实必须准确**（最高优先级！）：正确答案必须有明确的医学依据，干扰项必须确实是错误的。绝对不能为了追求差异化而编造不存在的医学概念、捏造数据、歪曲药理机制或混淆疾病特征。如果你对某个医学事实不确定，请使用参考资料中有据可查的内容。
2. 题型绝对不变。
3. 考点（知识点）不变。
4. 选项数量与作答形式不变。
5. **字符串相似度必须 < 0.35**，**禁止连续4字以上照搬原文**。
6. 病例题才可改病例参数，非病例题不得病例化。
7. 原题有几个小问，翻新后必须仍是几个。

🔥 翻新方法论（必须至少运用 3 种）：
1. **临床情境嵌入（最重要！）**：每道题都必须嵌入一个具体的临床护理场景，让题目读起来像在真实的病房/手术室/护理站中发生的。常用场景框架：
   - 「责任护士在执行XX操作时...」「护士交班时发现...」「术后护理巡视中...」
   - 「科室进行护理教学查房，讨论XX的护理要点时...」「护士对患者进行出院健康宣教时...」
   - 「夜班护士监测到XX异常指标，应首先...」「护理质控检查中发现...」
2. **切入角度翻转**：原题从"定义/概念"切入 → 你从"临床应用/操作规范/并发症处理"切入；原题问"是什么" → 你问"为什么/怎么做/什么情况下不能做"。
3. **知识维度迁移**：同一个知识点有多个可考维度（病因、机制、表现、诊断、治疗、护理、预防、健康教育）。原题考了维度A → 你换到维度B去考。
4. **参考资料融合**：认真阅读提供的参考资料，从中提取原题未涉及的专业细节（数据、指南建议、操作要点、注意事项），融入新题中。
5. **提问逻辑重构**：正向→反向（正确→错误/不当）、单一→综合、直接→情境化。
6. **全部选项推倒重来**：每个选项都必须重新设计内容。干扰项要从不同的错误角度设计。
7. **专业术语升级**：用更精准、更学术的医学/护理术语替代原题中的通俗表达。

🔴 医学事实准确性红线（比差异化更重要！）：
- 正确答案必须在医学教科书或临床指南中有明确依据
- 干扰项必须是真正错误的，不能把正确的医学知识当干扰项
- 不能为了让题目看起来不同而编造药理机制、混淆疾病分类、虚构检查数据
- 涉及具体数值（正常值范围、剂量、时间等）必须准确
- 如果不确定某个知识点，优先使用参考资料中明确记载的内容
- 选择题：确保有且只有一个正确答案（单选）或确定的多个正确答案（多选）
- 判断题：陈述的事实必须明确可判断对错，不能模棱两可

❌ 不合格示例（会被审核直接打回）：
- "急性肾盂肾炎最常见的致病菌是" → "急性肾盂肾炎最主要的病原菌是"（❌ 换了两个词，句式完全一样）
- "A. 大肠埃希菌" → "A. 大肠杆菌"（❌ 别名替换，没有实质变化）
- "关于膀胱癌的描述，正确的是" → "有关膀胱癌的叙述，正确的是"（❌ 只是同义词互换）
- 题干结构和原题一样，只是把个别词换了（❌ 这不叫翻新，这叫抄袭）

✅ 合格示例（新题必须达到这个程度的差异）：
- 原题："急性肾盂肾炎最常见的致病菌是" → "引发成人上尿路感染时，责任护士进行健康宣教应重点提及哪种病原体的预防"（✅ 换了切入角度：从纯病因→护理宣教场景；句式完全不同）
- 原题选项："A. 大肠埃希菌 B. 变形杆菌 C. 葡萄球菌 D. 链球菌" → "A. 革兰阴性杆菌中最常定植于肠道的菌种 B. 以脲酶活性著称的条件致病菌 C. 引起皮肤软组织感染的球菌 D. 常导致风湿热的链状排列球菌"（✅ 用功能描述替代菌名，考察更深层理解）
- 原题："下列关于膀胱癌的描述，正确的是" → "某护理查房中讨论泌尿系肿瘤的护理评估要点，以下哪项体现了循证护理的正确认识"（✅ 嵌入护理场景，换问法，更高阶）

🚫🚫🚫 题型锁定红线（违反即判定为废题）🚫🚫🚫
- 判断题 → 绝对不能变成选择题！判断题只有 A（正确）和 B（错误）两个选项，翻新后仍然只能有 A、B 两个选项，不能出现 C、D 选项。
  ⚠️ 判断题的题干必须是一个「可以判断对错的陈述句」，绝对禁止使用选择题式问法！
  ❌ 禁止出现："以下哪项"、"下列哪项"、"哪项是"、"哪个是"、"不属于...的是"、"属于...的是"、"不正确的是"、"正确的是"等选择式提问
  ✅ 正确写法：直接陈述一个事实或观点，让考生判断该陈述是否正确
  ✅ 示例："黄体酮缓解肾绞痛的机制是松弛平滑肌" → 判断正误
  ❌ 错误写法："以下关于黄体酮的描述，哪项是正确的？" → 这是选择题问法，不是判断题！
- 简答题 → 绝对不能变成选择题！简答题没有选项，翻新后仍然不能有任何选项，必须保持文字作答形式。
- 填空题 → 绝对不能变成选择题！填空题翻新后仍然是填空，不能添加选项。
- 选择题 → 不能变成判断题或简答题！有选项的题翻新后必须保留选项。
- 单选题 → 不能变成多选题！多选题 → 不能变成单选题！

📌 临床语境要求（重要！所有题型都适用）：
**必须在题干中嵌入临床护理工作场景**，让题目有「现场感」，而不是干巴巴的教科书定义。
用护理操作、查房讨论、术后巡视、健康宣教等场景引入知识点。

✅ 鼓励的写法（有临床现场感）：
- 选择题：「责任护士在执行TURP术后膀胱冲洗时，发现引流液颜色加深，首先应采取的措施是...」
- 选择题：「泌尿外科护士在进行用药指导时，关于α受体阻滞剂的注意事项，正确的是...」
- 判断题：「责任护士在执行术后护理巡视时，持续膀胱冲洗液的温度应维持在25-30℃以减少膀胱痉挛。」
- 判断题：「护理查房中讨论肾上腺手术护理时，提到术后24h引流液超过200ml应警惕活动性出血。」
- 多选题：「护士在为前列腺癌根治术后留置导尿管的患者制定护理计划时，正确的做法包括...」

❌ 避免的写法：
- 干巴巴的教科书定义，没有任何临床场景（如「前列腺增生最常见的症状是排尿困难。」）
- 过于冗长的完整病史采集格式（如写满一大段「患者男，XX岁，主诉+现病史+既往史+查体」），非病例题不能构造完整虚拟患者

⚠️ 小问格式要求（案例分析题、简答题等）：
- 如果原题有 N 个小问，新题必须严格保持 N 个小问
- 每个小问必须独立成行，使用明确标记：
  * 方式1：第一问、第二问、第三问...
  * 方式2：(10分)、(15分)、(20分)... 在行尾标注分数
  * 方式3：1. 2. 3. 编号开头
- 不要合并小问，不要拆分小问，不要增减小问数量

可用策略（每题必须组合 3~4 种，且必须包含 F 和 G）：
- A 病例参数微调（仅限病例题：完全更换患者年龄、性别、主诉、症状体征、检查数据等）
- B 干扰项重构（用全新的迷惑角度重新设计全部错误选项，不能只是换说法）
- C 深度改写（彻底重组句式和措辞，不是简单同义词替换）
- D 问法转换（切换提问方向和考察维度：正→反、病因→表现、措施→禁忌、概念→应用等）
- E 情境重塑（用全新的护理场景/临床背景承载同一知识点，让题目有"现场感"）
- F 全面翻新（以知识点为锚，从零编写一道新题，不参考原题的句式结构）
- G RAG知识融合（从参考资料中提取原题未涉及的专业细节，如指南数据、操作规范、注意事项等，融入新题中提升专业深度）
将使用的策略组合写入 `renovation_strategy_detail`。

严格输出以下 JSON 格式（注意 options 必须是对象/字典，citations 必须是对象数组，answer 必须是字符串，patient_background 必须是对象或 null）：
{
  "renovation_method": "策略名称",
  "renovation_strategy_detail": "详细策略说明",
  "content": "新题干",
  "options": {"A": "选项A内容", "B": "选项B内容", "C": "选项C内容", "D": "选项D内容"},
  "answer": "A",
  "explanation_with_citations": "解析内容，含引用标注[1][2]",
  "citations": [{"source": "来源文件名", "page": "页码", "content_snippet": "引用片段"}],
  "patient_background": null,
  "quality_self_check": {"fluency": 4, "clarity": 4, "conciseness": 4, "relevance": 4, "consistency": 4, "answerability": 4, "answer_consistency": 4}
}

注意：
- 多选题的 answer 用字母拼接，如 "ABCD"，不要用数组
- options 不要用数组，必须用字典
- patient_background 必须是对象（包含 gender, age, chief_complaint 等字段）或 null，不能是字符串

质控（QGEval 2.0）：七维评分每项 >=3，均分 >=3.5。
"""

_RENOVATION_METHOD_MAP = {
    "增加内容": RenovationMethod.ADD_CONTENT,
    "细节丰满": RenovationMethod.ADD_CONTENT,
    "选项替换": RenovationMethod.OPTION_REPLACE,
    "干扰项重构": RenovationMethod.OPTION_REPLACE,
    "同义替换": RenovationMethod.SYNONYM_REPLACE,
    "问法转换": RenovationMethod.SYNONYM_REPLACE,
}


def _normalize_renovation_method(method_text: str) -> Optional[RenovationMethod]:
    if not method_text:
        return None

    for keyword, method in _RENOVATION_METHOD_MAP.items():
        if keyword in method_text:
            return method
    return None


class GeneratorAgent:
    """试题生成智能体"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()

    def generate(
        self,
        original_question: Question,
        rag_context: RAGContext,
        audit_feedback: Optional[AuditResult] = None,
        attempt: int = 1,
    ) -> GeneratedQuestion:
        """同步生成翻新题目"""
        logger.info(f"生成题目 {original_question.id} (尝试 {attempt})")
        prompt = self._build_prompt(original_question, rag_context, audit_feedback)
        t0 = time.monotonic()
        try:
            response = self.llm.chat(
                prompt=prompt,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                temperature=0.95,
                max_tokens=self._resolve_max_tokens(original_question),
                json_mode=True,
            )
            elapsed = time.monotonic() - t0
            return self._parse_and_build(response, original_question, attempt, elapsed)
        except Exception as e:
            logger.error(f"生成题目 {original_question.id} 失败: {e}")
            raise

    async def generate_async(
        self,
        original_question: Question,
        rag_context: RAGContext,
        audit_feedback: Optional[AuditResult] = None,
        attempt: int = 1,
    ) -> GeneratedQuestion:
        """异步生成翻新题目"""
        logger.info(f"异步生成题目 {original_question.id} (尝试 {attempt})")
        prompt = self._build_prompt(original_question, rag_context, audit_feedback)
        t0 = time.monotonic()
        try:
            response = await self.llm.achat(
                prompt=prompt,
                system_prompt=GENERATOR_SYSTEM_PROMPT,
                temperature=0.95,
                max_tokens=self._resolve_max_tokens(original_question),
                json_mode=True,
            )
            elapsed = time.monotonic() - t0
            return self._parse_and_build(response, original_question, attempt, elapsed)
        except Exception as e:
            logger.error(f"异步生成题目 {original_question.id} 失败: {e}")
            raise

    @staticmethod
    def _preprocess_llm_data(data: dict) -> dict:
        """预处理 LLM 返回的 JSON 数据，修正常见格式偏差"""
        if "options" in data and isinstance(data["options"], list):
            opts = {}
            for item in data["options"]:
                item_str = str(item).strip()
                match = re.match(r"^([A-Z])[.、．]\s*(.*)", item_str)
                if match:
                    opts[match.group(1)] = match.group(2)
                else:
                    opts[item_str] = item_str
            data["options"] = opts

        if "answer" in data and isinstance(data["answer"], list):
            data["answer"] = "".join(str(a).strip().upper() for a in data["answer"])

        if "citations" in data and isinstance(data["citations"], list):
            fixed = []
            for c in data["citations"]:
                if isinstance(c, dict):
                    fixed.append(c)
                elif isinstance(c, str):
                    fixed.append({"source": c, "content_snippet": c})
            data["citations"] = fixed

        # Fix patient_background if it's a string instead of dict
        if "patient_background" in data and isinstance(data["patient_background"], str):
            logger.warning(f"patient_background 是字符串而非字典，将其设为 None: {data['patient_background'][:50]}")
            data["patient_background"] = None

        return data

    def _parse_and_build(self, response: str, original: Question, attempt: int, elapsed: float) -> GeneratedQuestion:
        """解析 LLM 响应并构建 GeneratedQuestion"""
        data = self.llm.extract_json_from_response(response)
        data = self._preprocess_llm_data(data)
        parsed = GeneratedQuestionResponse.model_validate(data)

        if not parsed.content.strip():
            raise ValueError("LLM 返回的题干内容为空，需要重试")
        if not parsed.answer.strip():
            raise ValueError("LLM 返回的答案为空，需要重试")

        normalized_options = self._normalize_options(original, parsed.options)

        # 对于非病例题，如果 LLM 错误地返回了 patient_background，强制设为 None
        patient_bg_for_validation = parsed.patient_background
        if original.type not in CLINICAL_CASE_TYPES and parsed.patient_background:
            logger.warning(f"非病例题 {original.id} 错误地返回了 patient_background，已自动忽略")
            patient_bg_for_validation = None

        # 自动修复案例分析题小问数量（多了就裁剪末尾）
        fixed_content = parsed.content
        if original.subquestion_count > 1:
            gen_count = self._count_subquestions(fixed_content)
            if gen_count > original.subquestion_count:
                trimmed = self._trim_excess_subquestions(fixed_content, original.subquestion_count)
                if trimmed:
                    logger.warning(
                        f"题目 {original.id} 小问数 {gen_count} > 原题 {original.subquestion_count}，"
                        f"已自动裁剪多余小问"
                    )
                    fixed_content = trimmed

        self._validate_generated_structure(
            original=original,
            content=fixed_content,
            options=normalized_options,
            answer=parsed.answer,
            patient_background=patient_bg_for_validation,
        )

        citations = [
            Citation(
                source=c.get("source", "未知"),
                page=c.get("page"),
                content_snippet=c.get("content_snippet"),
            )
            for c in parsed.citations
        ]

        renovation_method = _normalize_renovation_method(parsed.renovation_method)

        patient_bg = None
        if patient_bg_for_validation:
            try:
                patient_bg = PatientBackground.model_validate(patient_bg_for_validation)
            except Exception as e:
                logger.warning(f"患者背景解析失败，忽略: {e}")

        quality_scores = None
        if parsed.quality_self_check:
            try:
                quality_scores = QualityScores.model_validate(parsed.quality_self_check)
            except Exception as e:
                logger.warning(f"质量自检评分解析失败，忽略: {e}")

        # 保持原题型不变
        new_type = original.type

        generated = GeneratedQuestion(
            id=f"{original.id}_v{attempt}",
            type=new_type,
            content=fixed_content,
            knowledge_point=original.knowledge_point,
            options=normalized_options,
            answer=parsed.answer,
            explanation=parsed.explanation_with_citations,
            citations=citations,
            explanation_with_citations=parsed.explanation_with_citations,
            original_question_id=original.id,
            generation_attempt=attempt,
            passed_audit=False,
            category=original.category,
            difficulty=original.difficulty,
            section_title=original.section_title,
            display_number=original.display_number,
            subquestion_count=original.subquestion_count,
            renovation_method=renovation_method,
            renovation_strategy_detail=parsed.renovation_strategy_detail,
            patient_background=patient_bg,
            quality_scores=quality_scores,
            generation_time_seconds=round(elapsed, 2),
        )
        logger.info(f"题目 {original.id} 生成成功 (策略: {parsed.renovation_method}, 用时: {elapsed:.1f}s)")
        return generated

    @staticmethod
    def _normalize_options(original: Question, parsed_options: dict) -> dict:
        """按原题结构归一化选项，避免题型被隐式改写。"""
        # 简答题/填空题：强制无选项
        if original.type in {QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK}:
            if parsed_options:
                logger.warning(
                    f"{original.type.value} {original.id} LLM 错误返回了选项，已强制清空"
                )
            return {}

        if not original.options:
            return {}

        # 判断题：强制只保留 A/B，回退到原题选项
        if original.type == QuestionType.TRUE_FALSE:
            if not parsed_options:
                return dict(original.options)
            normalized = {str(key).strip().upper(): value for key, value in parsed_options.items()}
            if set(normalized.keys()) != {"A", "B"}:
                logger.warning(
                    f"判断题 {original.id} LLM 返回了非法选项 {list(normalized.keys())}，"
                    f"强制回退为原题选项 A/B"
                )
                return dict(original.options)
            return {"A": normalized["A"], "B": normalized["B"]}

        if not parsed_options:
            return {}

        normalized = {str(key).strip().upper(): value for key, value in parsed_options.items()}
        if set(normalized.keys()) == set(original.options.keys()):
            return {key: normalized[key] for key in original.options.keys()}

        return parsed_options

    def _validate_generated_structure(
        self,
        original: Question,
        content: str,
        options: dict,
        answer: str,
        patient_background: Optional[dict],
    ) -> None:
        """严格校验翻新题结构，确保题型不被隐式改变。"""
        requires_options = bool(original.options)

        # ── 判断题专项校验 ──
        if original.type == QuestionType.TRUE_FALSE:
            choice_patterns = [
                r"以下哪", r"下列哪", r"哪项是", r"哪个是", r"哪一项",
                r"不属于.*的是", r"属于.*的是", r"不正确的是", r"正确的是",
                r"不包括", r"不恰当的是", r"错误的是", r"不符合.*的是",
            ]
            content_flat = re.sub(r"\s+", "", content)
            for pat in choice_patterns:
                if re.search(pat, content_flat):
                    raise ValueError(
                        f"判断题题干使用了选择题式问法（匹配到「{pat}」），"
                        f"判断题必须是陈述句，不能用「以下哪项」「不属于...的是」等选择式提问！"
                    )
            if len(options) != 2:
                raise ValueError(
                    f"判断题被改成了选择题！判断题只能有 2 个选项(A/B)，"
                    f"但 LLM 返回了 {len(options)} 个选项：{list(options.keys())}"
                )
            if set(options.keys()) != {"A", "B"}:
                raise ValueError(
                    f"判断题选项标签必须是 A 和 B，"
                    f"但 LLM 返回了 {list(options.keys())}，疑似题型漂移为选择题"
                )

        # ── 简答题/填空题专项校验 ──
        if original.type in {QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK}:
            if options:
                raise ValueError(
                    f"{original.type.value}被改成了选择题！"
                    f"{original.type.value}不能有任何选项，"
                    f"但 LLM 返回了 {len(options)} 个选项，必须保持文字作答形式"
                )

        # ── 通用选项校验 ──
        if requires_options and not options:
            raise ValueError("当前题型必须保留选项，但 LLM 返回的选项为空")

        if not requires_options and options:
            raise ValueError(
                f"当前题型({original.type.value})不应生成选项，"
                f"但 LLM 返回了 {len(options)} 个选项，疑似发生题型漂移为选择题"
            )

        if requires_options and len(options) != len(original.options):
            raise ValueError(f"选项数量不一致：原题 {len(original.options)} 个，新题 {len(options)} 个")

        if requires_options and set(options.keys()) != set(original.options.keys()):
            raise ValueError("选项标签不一致，疑似改变了作答结构")

        if original.type in CLINICAL_CASE_TYPES:
            if not patient_background:
                raise ValueError("病例题缺少结构化 patient_background，需重试")
            if not self._looks_like_case_stem(content):
                raise ValueError("病例题题干缺少病例化特征，疑似题型漂移")
        else:
            if patient_background:
                raise ValueError("非病例题返回了 patient_background，疑似被改成病例题")
            text = re.sub(r"\s+", "", content)
            patterns = [
                r"(患者|病人).{0,8}\d+岁",
                r"\d+岁.{0,8}(患者|病人)",
                r"(男性|女性).{0,8}\d+岁",
                r"(主诉|现病史|既往史|入院|查体|体温|脉搏|血压|护理评估)",
            ]
            matched = sum(bool(re.search(pattern, text)) for pattern in patterns)
            if matched >= 3:
                raise ValueError("非病例题题干出现明显病例化描述，疑似发生 A1/A2 等题型漂移")

        if not self._answer_matches_type(original, answer):
            raise ValueError(f"答案格式与题型不匹配：{original.type.value} -> {answer}")

        generated_subquestion_count = self._count_subquestions(content)
        if original.subquestion_count != generated_subquestion_count:
            raise ValueError(
                f"小问数量不一致：原题 {original.subquestion_count} 个，新题 {generated_subquestion_count} 个"
            )

    @staticmethod
    def _looks_like_case_stem(content: str) -> bool:
        text = re.sub(r"\s+", "", content)
        patterns = [
            r"(患者|病人).{0,8}\d+岁",
            r"\d+岁.{0,8}(患者|病人)",
            r"(男性|女性).{0,8}\d+岁",
            r"(主诉|现病史|既往史|入院|查体|体温|脉搏|血压|护理评估)",
        ]
        matched = sum(bool(re.search(pattern, text)) for pattern in patterns)
        # 匹配 2 个或以上模式即认为是病例题
        # 这个阈值在非病例题检测时会更严格（通过上下文判断）
        return matched >= 2

    @staticmethod
    def _count_subquestions(content: str) -> int:
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        scored_markers = sum(bool(re.search(r'[（(]\s*\d+\s*分\s*[）)]', line)) for line in lines)
        if scored_markers > 0:
            return scored_markers
        markers = 0
        for line in lines:
            if any(tag in line for tag in ["第一问", "第二问", "第三问", "第四问", "第五问"]):
                markers += 1
            elif re.match(r'^[（(]?\s*[1-9]\s*[）)]\s*', line):
                markers += 1
            elif re.match(r'^\d+\s*[、.．]\s*\S', line) and '？' in line:
                markers += 1
            elif line.endswith("？") or line.endswith("?"):
                markers += 1
        return markers if markers > 0 else 1

    @staticmethod
    def _trim_excess_subquestions(content: str, target_count: int) -> Optional[str]:
        """当生成的小问数超过目标时，尝试裁剪末尾多余的小问。"""
        lines = content.split('\n')
        score_pattern = re.compile(r'[（(]\s*\d+\s*分\s*[）)]')
        marker_positions = []
        for i, line in enumerate(lines):
            if score_pattern.search(line.strip()):
                marker_positions.append(i)

        if len(marker_positions) > target_count:
            cut_line = marker_positions[target_count]
            trimmed = '\n'.join(lines[:cut_line]).strip()
            check = GeneratorAgent._count_subquestions(trimmed)
            if check == target_count:
                return trimmed

        question_mark_positions = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.endswith("？") or stripped.endswith("?"):
                question_mark_positions.append(i)

        if len(question_mark_positions) > target_count:
            cut_line = question_mark_positions[target_count] + 1
            trimmed = '\n'.join(lines[:cut_line]).strip()
            check = GeneratorAgent._count_subquestions(trimmed)
            if check == target_count:
                return trimmed

        return None

    @staticmethod
    def _answer_matches_type(original: Question, answer: str) -> bool:
        normalized = re.sub(r"[\s,，、；;.]", "", answer.strip().upper())

        if not original.options:
            return bool(answer.strip())

        question_type = original.type
        if question_type in {QuestionType.MULTIPLE_CHOICE, QuestionType.X}:
            return bool(re.fullmatch(r"[A-Z]{2,}", normalized))

        if question_type == QuestionType.TRUE_FALSE:
            return normalized in {"A", "B"}

        return bool(re.fullmatch(r"[A-Z]", normalized))

    def _build_prompt(
        self,
        original_question: Question,
        rag_context: RAGContext,
        audit_feedback: Optional[AuditResult],
    ) -> str:
        options_str = "\n".join(f"  {k}. {v}" for k, v in original_question.options.items())
        strategy_combo = self._choose_strategy_combo(original_question)

        prompt_parts = [
            "## 原始题目\n",
            f"**所属章节**：{original_question.section_title or '未标注'}",
            f"**原题号**：{original_question.display_number or original_question.id}",
            f"**题型**：{original_question.type.value}",
            f"**考点**：{original_question.knowledge_point}",
            f"**题干**：{original_question.content}",
        ]
        if original_question.options:
            prompt_parts.append(f"**选项**：\n{options_str}")
        prompt_parts.append(f"**答案**：{original_question.answer}")

        if original_question.explanation:
            prompt_parts.append(f"**解析**：{self._trim_text(original_question.explanation, 220)}")

        # 翻新策略建议
        prompt_parts.append("\n## 翻新策略建议\n")
        strategy_hint = self._suggest_strategy(original_question, strategy_combo)
        prompt_parts.append(strategy_hint)

        # 是否需要构建虚拟患者背景
        needs_patient_bg = original_question.type in CLINICAL_CASE_TYPES
        if needs_patient_bg:
            prompt_parts.append("\n**⚠️ 本题需要构建完整的虚拟患者背景（patient_background 必填）**")

        prompt_parts.append("\n## 参考资料（用于引用标注）\n")
        prompt_parts.append(self._compact_rag_context(rag_context))

        if audit_feedback and not audit_feedback.passed:
            prompt_parts.append("\n## 上次生成的问题（请针对性改进）\n")
            prompt_parts.append("**审核结果**：未通过")
            prompt_parts.append(f"**原因**：{audit_feedback.reason}")
            if audit_feedback.issues:
                prompt_parts.append("**具体问题**：")
                prompt_parts.extend(f"  - {issue}" for issue in audit_feedback.issues[:3])
            if audit_feedback.suggestions:
                prompt_parts.append("**修改建议**：")
                prompt_parts.extend(f"  - {s}" for s in audit_feedback.suggestions[:3])
            if audit_feedback.quality_scores:
                qs = audit_feedback.quality_scores
                prompt_parts.append("**上次质量评分（QGEval 2.0）**：")
                prompt_parts.append(
                    f"  流畅性={qs.fluency} 清晰度={qs.clarity} 简洁性={qs.conciseness} "
                    f"相关性={qs.relevance} 一致性={qs.consistency} "
                    f"可回答性={qs.answerability} 答案一致性={qs.answer_consistency}"
                )

        prompt_parts.extend([
            "\n## 任务\n",
            "⚠️ **重要**：原题只是告诉你「要考什么知识点」。你要做的是：**先阅读参考资料**，理解这个知识点的完整内涵，然后**完全抛开原题**，像拿到一个「出题任务单」一样独立命题。",
            "",
            "你的目标：新题和原题之间的相似度必须极低——不仅措辞不同，连提问的角度、切入点、选项设计逻辑都应该不同。",
            "",
            "要求：",
            "0. **结构锁定**：保持原题在章节中的位置不变，保持原题号与原有展示结构",
            "1. **知识点锁定**：考点与原题完全相同",
            "2. **题型锁定**：题型必须与原题完全相同，绝对不得改变",
            "3. **低相似度锁定（最高优先级！）**：字符串相似度必须低于 0.35，禁止连续 4 字以上与原文相同。做到这一点的唯一方法是：不要试图「改写」原题，而是从知识点出发**重新构思一道题**",
            "4. **选项锁定**：选项数量、作答方式、答案形式必须与原题一致",
            "5. **病例题规则**：仅当原题本身是 A2/A3/A4/病例题时，才可修改病例基础信息并填写 patient_background",
            "6. **策略执行**：按以下策略组合生成，并在 renovation_strategy_detail 中原样写出：{0}".format(" + ".join(strategy_combo)),
            "7. **RAG资料融合**：从参考资料中挑选原题未提及的专业细节（指南推荐、操作规范、数值标准、并发症处理等），融入题干或选项中",
            "8. **深度解析**：包含正确项逻辑 + 每个错误项排除原因，解析中标注参考资料来源[1][2]",
            "9. **质量自检**：完成 QGEval 2.0 七维评分，每项 ≥ 3 分，均分 ≥ 3.5 分",
            "",
            "🔴 **医学事实准确性自检（最高优先级，任一项不通过必须重写！）**：",
            "- [ ] 正确答案在医学教科书或参考资料中有明确依据吗？",
            "- [ ] 每个干扰项确实是错误的吗？（不能把正确的医学知识当干扰项）",
            "- [ ] 题干中的医学描述（病理、药理、解剖、护理操作等）是否准确？",
            "- [ ] 涉及的数值（正常值、剂量、时间等）是否与教科书一致？",
            "- [ ] 解析中对正确答案的论证和对错误选项的排除是否逻辑严密？",
            "",
            "🔥 **创新力度自检（任一项不通过必须重写）**：",
            "- [ ] 提问角度是否与原题不同？",
            "- [ ] 题干是否引入了原题中没有的专业信息？",
            "- [ ] 选项的设计逻辑是否与原题不同？",
            "- [ ] 新题是否能独立成题，读起来像一道全新的题？",
        ])

        # 题型专属强约束指令
        if original_question.type == QuestionType.TRUE_FALSE:
            prompt_parts.append(
                "\n🚫 **判断题红线**：本题是判断题！\n"
                "- options 必须为 {\"A\": \"正确\", \"B\": \"错误\"}，不能出现 C/D/E 选项\n"
                "- answer 只能是 \"A\" 或 \"B\"\n"
                "- 题干必须是「带临床护理场景的陈述句」，让考生判断对错\n"
                "- ❌ 禁止选择题式提问：「以下哪项」「哪个是」「不属于...的是」\n"
                "- ✅ 正确：「责任护士在执行术后膀胱冲洗护理时，持续膀胱冲洗液的温度应维持在25-30℃。」\n"
                "- ✅ 正确：「护理查房中讨论用药护理时，提到非那雄胺服用3个月后PSA值下降约50%属于正常药物反应。」"
            )
        elif original_question.type in {QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK}:
            prompt_parts.append(
                f"\n🚫 **{original_question.type.value}红线**：本题是{original_question.type.value}，"
                f"options 必须为空字典 {{}}，绝对不能生成任何 A/B/C/D 选项！answer 必须是完整的文字答案！"
            )
        elif original_question.type in {QuestionType.MULTIPLE_CHOICE, QuestionType.X}:
            orig_answer = original_question.answer.strip().upper()
            ans_len = len(re.sub(r"[\s,，、；;.]", "", orig_answer))
            prompt_parts.append(
                f"\n🚫 **多选题红线**：本题是多选题（X型题），answer 必须是**多个字母**组合（如 \"ABD\"、\"ACD\"），"
                f"绝对不能只写一个字母！原题答案含 {ans_len} 个选项字母，新题答案也应包含 2 个或以上字母。"
            )

        if original_question.type in CLINICAL_CASE_TYPES or (
            original_question.type == QuestionType.SHORT_ANSWER and original_question.subquestion_count > 1
        ):
            prompt_parts.append(
                "\n📋 **案例分析题/论述题专属约束（极其重要！）**：\n"
                "1. **每个小问必须保留完整病例背景**：不能只在第一问写病例，后续小问省略。每个小问开头都要有完整的病例情境，让考生无需翻看前文即可独立作答。\n"
                "2. **RAG引用准确性（一票否决！）**：\n"
                "   - 参考资料可能来自不同疾病/手术的护理内容，你必须甄别哪些资料与本题的具体疾病/手术相关\n"
                "   - ❌ 禁止跨疾病搬运：不能把「肾损伤」的卧床要求应用到「输尿管镜碎石术后」\n"
                "   - ❌ 禁止跨术式搬运：不能把「膀胱肿瘤术后」的护理措施应用到「泌尿系结石术后」\n"
                "   - ✅ 只引用与本题具体疾病/手术直接相关的内容，不确定的宁可不引用\n"
                "3. **护士职责范围**：\n"
                "   - 所有护理措施必须在护士执业范围内，涉及医疗决策的必须加「遵医嘱」前缀\n"
                "   - ❌ 错误：「必要时使用血管活性药物」（护士不能自行决定用药）\n"
                "   - ✅ 正确：「遵医嘱使用血管活性药物维持血压」\n"
                "   - ❌ 错误：「给予抗生素治疗」\n"
                "   - ✅ 正确：「遵医嘱给予抗感染治疗，观察用药反应」\n"
                "4. **数值引用谨慎**：\n"
                "   - 护理指标的具体数值（饮水量、卧床时间等）必须与本题疾病/手术直接匹配\n"
                "   - 如不确定具体数值，使用「适当增加饮水量」而非「每日2000-3000ml」这种硬性数字\n"
                "5. **参考答案要点格式**：answer 字段写成要点式，每个要点独立成行，逻辑清晰"
            )

        # 如果有多个小问，添加详细的格式说明和示例
        if original_question.subquestion_count > 1:
            n = original_question.subquestion_count
            prompt_parts.append(
                f"\n🚫🚫🚫 **小问数量锁定（最高优先级，违反即废题！）** 🚫🚫🚫\n"
                f"原题恰好包含 **{n}** 个小问，新题也必须恰好 **{n}** 个小问！\n"
                f"- 不得多于 {n} 个，不得少于 {n} 个\n"
                f"- 每个小问独立成段落，段落末尾标注分数如 （10分）\n"
                f"- 生成后请自己数一下：content 中出现了几个 '？' 或几个 '（N分）'，必须正好是 {n} 个\n"
                f"- 如果你发现自己写了 {n + 1} 个小问，请删掉多余的那个再输出"
            )
        else:
            prompt_parts.append("10. **单一问题**：本题只有 1 个问题，不要拆分成多个小问")

        prompt_parts.append("\n请输出 JSON 格式的新题目。")
        return "\n".join(prompt_parts)

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text if len(text) <= max_chars else text[:max_chars] + '...'

    @staticmethod
    def _compact_rag_context(rag_context: RAGContext, max_chunks: int = 3, max_chars: int = 1500) -> str:
        if not rag_context.chunks:
            return "无相关参考资料"

        parts = []
        used_chars = 0
        for idx, chunk in enumerate(rag_context.chunks[:max_chunks], 1):
            source = chunk.get("metadata", {}).get("source", "未知来源")
            page = chunk.get("metadata", {}).get("page", "")
            content = chunk.get("content", chunk.get("document", ""))
            content = re.sub(r'\s+', ' ', content).strip()
            remain = max_chars - used_chars
            if remain <= 0:
                break
            content = content[:remain]
            page_info = f" 第{page}页" if page else ""
            part = f"[{idx}] {source}{page_info}：{content}"
            parts.append(part)
            used_chars += len(content)
        return "\n".join(parts) if parts else "无相关参考资料"

    @staticmethod
    def _resolve_max_tokens(question: Question) -> int:
        if question.type in CLINICAL_CASE_TYPES or question.subquestion_count > 1:
            return 4096
        if question.type in {QuestionType.SHORT_ANSWER, QuestionType.CASE_ANALYSIS}:
            return 4096
        return 2560

    @staticmethod
    def _choose_strategy_combo(question: Question) -> List[str]:
        """根据题型选择翻新策略组合：F + G 始终必选，再随机加 1~2 个辅助策略。"""
        selected = {"策略F（全面翻新）", "策略G（RAG知识融合）"}

        if question.type == QuestionType.TRUE_FALSE:
            extra_pool = [
                "策略C（深度改写）",
                "策略D（问法转换）",
                "策略E（情境重塑）",
            ]
            extra_weights = [2, 5, 3]
        elif question.type in {QuestionType.SHORT_ANSWER, QuestionType.FILL_IN_BLANK}:
            extra_pool = [
                "策略C（深度改写）",
                "策略D（问法转换）",
                "策略E（情境重塑）",
            ]
            extra_weights = [2, 5, 4]
        elif question.type in CLINICAL_CASE_TYPES:
            extra_pool = [
                "策略A（病例参数微调）",
                "策略B（干扰项重构）",
                "策略D（问法转换）",
                "策略E（情境重塑）",
            ]
            extra_weights = [3, 4, 4, 4]
        else:
            extra_pool = [
                "策略B（干扰项重构）",
                "策略C（深度改写）",
                "策略D（问法转换）",
                "策略E（情境重塑）",
            ]
            extra_weights = [5, 2, 5, 4]

        extra_count = random.randint(1, min(2, len(extra_pool)))
        pool_with_weights = list(zip(extra_pool, extra_weights))
        while len(selected) < extra_count + 2:
            remaining = [(s, w) for s, w in pool_with_weights if s not in selected]
            if not remaining:
                break
            strategies, w_list = zip(*remaining)
            chosen = random.choices(strategies, weights=w_list, k=1)[0]
            selected.add(chosen)

        preferred_order = {
            "策略A（病例参数微调）": 0,
            "策略B（干扰项重构）": 1,
            "策略C（深度改写）": 2,
            "策略D（问法转换）": 3,
            "策略E（情境重塑）": 4,
            "策略F（全面翻新）": 5,
            "策略G（RAG知识融合）": 6,
        }
        return sorted(selected, key=lambda item: preferred_order.get(item, 99))

    @staticmethod
    def _suggest_strategy(question: Question, strategy_combo: List[str]) -> str:
        """输出本题的随机策略建议。"""
        lines = [
            f"原题为 {question.type.value}。",
            "本次必须保持原题型不变，不得升级、降级或改造成其他题型。",
            f"随机选中的策略组合：{' + '.join(strategy_combo)}。",
            "",
            "🔥 **创新力度要求**：",
            "- 不要从原题出发「改」，而是从知识点出发「写」一道新题",
            "- **必须嵌入临床护理场景**（如：执行操作时、护理查房中、交班时、术后巡视中等），不能只是干巴巴的教科书定义",
            "- 必须变换提问角度或考察维度（不能和原题用同样的问法）",
            "- 必须从参考资料中引入原题没有的专业细节",
            "- 选项要从全新的干扰角度设计，不能只是把原选项换个说法",
        ]

        # ── 判断题专项约束 ──
        if question.type == QuestionType.TRUE_FALSE:
            lines.append("\n🚫🚫🚫 **判断题题型锁定（最高优先级）** 🚫🚫🚫")
            lines.append("- 这是判断题，绝对不能变成选择题！")
            lines.append("- options 必须且只能是：{\"A\": \"正确\", \"B\": \"错误\"}")
            lines.append("- 绝对不能出现 C、D、E 等选项")
            lines.append("- answer 只能是 \"A\" 或 \"B\"")
            lines.append("- ⚠️ 判断题的题干必须是一个「陈述句」，让考生判断这句话对不对")
            lines.append("- ❌ 绝对禁止选择题式问法：「以下哪项」「下列哪项」「哪项是」「不属于...的是」「属于...的是」")
            lines.append("- ✅ 正确格式：带临床场景的陈述句。")
            lines.append("  例1：「责任护士在执行TURP术后膀胱冲洗时，冲洗液温度应维持在25-30℃以减少膀胱痉挛。」")
            lines.append("  例2：「泌尿外科术后护理中，发现引流液由淡红色突变为鲜红色时，应首先考虑冲洗速度过快所致。」")
            lines.append("- ❌ 错误格式1：「以下关于膀胱冲洗的描述，哪项正确？」（选择题问法！）")
            lines.append("- ❌ 错误格式2：「膀胱冲洗液的温度应维持在25-30℃。」（太干巴巴，缺少临床场景）")

        # ── 简答题专项约束 ──
        if question.type == QuestionType.SHORT_ANSWER:
            lines.append("\n🚫🚫🚫 **简答题题型锁定（最高优先级）** 🚫🚫🚫")
            lines.append("- 这是简答题，绝对不能变成选择题！")
            lines.append("- options 必须为空字典 {}")
            lines.append("- 绝对不能生成 A、B、C、D 等任何选项")
            lines.append("- answer 必须是完整的文字答案")
            lines.append("- 翻新方式：改写题干的表述和问法，但保持文字作答形式")

        # ── 填空题专项约束 ──
        if question.type == QuestionType.FILL_IN_BLANK:
            lines.append("\n🚫🚫🚫 **填空题题型锁定（最高优先级）** 🚫🚫🚫")
            lines.append("- 这是填空题，绝对不能变成选择题！")
            lines.append("- options 必须为空字典 {}")
            lines.append("- 绝对不能生成 A、B、C、D 等任何选项")
            lines.append("- 翻新方式：改写题干，但保持填空作答形式")

        # 策略 F + G 说明
        lines.append("\n**策略F（全面翻新）**：不要在原题基础上修改，而是只提取知识点，然后从零写一道新题。题型不变，但题干结构、提问方式、选项内容全部推倒重来。")
        lines.append("\n**策略G（RAG知识融合）**：仔细阅读下方的参考资料，从中找到原题没有涉及的专业细节（如具体数据、指南标准、操作步骤、注意事项、分类方式等），把这些内容融入新题的题干或选项中，让新题有「教科书深度」。")

        lines.append("\n📌 **临床语境要求**：")
        lines.append("- 必须嵌入临床护理场景，如「责任护士在执行XX操作时...」「术后护理巡视中发现...」「护理查房讨论XX时...」")
        lines.append("- 让题目读起来像真实的临床护理工作场景，而不是干巴巴的教科书定义")
        lines.append("- ⚠️ 非病例题不能构造完整虚拟患者（禁止写「患者男/女，XX岁，主诉XX，因XX入院」这种完整病史采集格式）")

        if question.type in CLINICAL_CASE_TYPES:
            lines.append("\n病例题可微调非关键基础信息，但必须保留核心护理指征、阳性症状和关键体征。")

        if question.options:
            lines.append(f"\n原题共有 {len(question.options)} 个选项，新题也必须保持 {len(question.options)} 个选项。")

        if question.type in {QuestionType.MULTIPLE_CHOICE, QuestionType.X}:
            lines.append("多选题必须继续保持多选作答方式，不能改为单选。")

        return "\n".join(lines)

    def batch_generate(
        self,
        questions: List[Question],
        rag_contexts: List[RAGContext],
    ) -> List[GeneratedQuestion]:
        """批量生成题目（同步）"""
        results = []
        for q, ctx in zip(questions, rag_contexts):
            try:
                results.append(self.generate(q, ctx))
            except Exception as e:
                logger.error(f"批量生成中题目 {q.id} 失败: {e}")
        return results
