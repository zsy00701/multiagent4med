"""
医学试卷翻新多智能体系统 - 数据模型定义

使用 Pydantic 定义严格的数据交换标准，确保数据完整性和类型安全。
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── 枚举类型 ──────────────────────────────────────────────────

class QuestionType(str, Enum):
    """题目类型枚举"""
    SINGLE_CHOICE = "单选题"
    MULTIPLE_CHOICE = "多选题"
    TRUE_FALSE = "判断题"
    SHORT_ANSWER = "简答题"
    CASE_ANALYSIS = "病例分析题"
    FILL_IN_BLANK = "填空题"
    A1 = "A1型题"
    A2 = "A2型题"
    A3 = "A3型题"
    A4 = "A4型题"
    B1 = "B1型题"
    X = "X型题"


class RenovationMethod(str, Enum):
    """翻新策略枚举"""
    ADD_CONTENT = "增加内容"
    OPTION_REPLACE = "选项替换"
    SYNONYM_REPLACE = "同义替换"
    DIMENSION_CONVERT = "维度转换"


# 需要构建完整虚拟患者背景的题型
CLINICAL_CASE_TYPES = {
    QuestionType.A2, QuestionType.A3, QuestionType.A4,
    QuestionType.CASE_ANALYSIS,
}

# A1/单选可升维为 A2 病例模拟题
UPGRADABLE_TO_CASE = {
    QuestionType.A1, QuestionType.SINGLE_CHOICE,
}


# ── 子模型 ────────────────────────────────────────────────────

class Citation(BaseModel):
    """引用来源模型"""
    source: str = Field(..., description="来源文件名")
    page: Optional[str] = Field(None, description="页码或章节")
    chunk_id: Optional[str] = Field(None, description="文本块ID")
    content_snippet: Optional[str] = Field(None, description="引用内容片段")


class PatientBackground(BaseModel):
    """虚拟患者背景（A2/A3/A4/病例分析题必填）"""
    gender: str = Field(..., description="性别")
    age: str = Field(..., description="年龄，如 '45岁'")
    chief_complaint: str = Field(..., description="主诉")
    positive_symptoms: List[str] = Field(default_factory=list, description="阳性症状列表")
    positive_signs: List[str] = Field(default_factory=list, description="阳性体征列表")
    abnormal_reports: List[str] = Field(default_factory=list, description="异常检查检验报告")
    treatment_history: Optional[str] = Field(None, description="诊治经过")
    negative_symptoms: List[str] = Field(default_factory=list, description="阴性症状（有鉴别诊断价值）")
    negative_signs: List[str] = Field(default_factory=list, description="阴性体征（有鉴别诊断价值）")
    specific_tests: List[str] = Field(default_factory=list, description="与考点密切相关的特异性检验结果")


class QualityScores(BaseModel):
    """QGEval 2.0 七维质量评分（1-5分）"""
    fluency: int = Field(0, ge=0, le=5, description="流畅性：句子结构完整，用词精准自然，无语法问题")
    clarity: int = Field(0, ge=0, le=5, description="清晰度：题目意图明确、焦点突出，无任何歧义")
    conciseness: int = Field(0, ge=0, le=5, description="简洁性：用词精炼，无冗余修饰语或重复信息")
    relevance: int = Field(0, ge=0, le=5, description="相关性：紧密围绕上下文核心事实提出")
    consistency: int = Field(0, ge=0, le=5, description="一致性：内部逻辑自洽，所有条件在语义和逻辑上完全兼容")
    answerability: int = Field(0, ge=0, le=5, description="可回答性：问题明确聚焦，存在可从上下文中直接找到或推断出的具体答案")
    answer_consistency: int = Field(0, ge=0, le=5, description="答案一致性：存在确定的答案，且非答案的选项一定是错误的")

    @property
    def all_scores(self) -> list:
        return [
            self.fluency, self.clarity, self.conciseness, self.relevance,
            self.consistency, self.answerability, self.answer_consistency,
        ]

    @property
    def average(self) -> float:
        valid = [s for s in self.all_scores if s > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def passed(self) -> bool:
        valid = [s for s in self.all_scores if s > 0]
        return bool(valid) and self.average >= 3.5 and all(s >= 3 for s in valid)

    @property
    def verdict(self) -> str:
        """综合评判：通过 / 需修改 / 严重不合格"""
        valid = [s for s in self.all_scores if s > 0]
        if not valid:
            return "严重不合格"
        if any(s <= 1 for s in valid) or self.average < 2.0:
            return "严重不合格"
        if self.average >= 3.5 and all(s >= 3 for s in valid):
            return "通过"
        return "需修改"

    @property
    def low_dims(self) -> list:
        """返回得分低于 4 分的维度名称及分数"""
        dim_names = {
            "fluency": "流畅性", "clarity": "清晰度", "conciseness": "简洁性",
            "relevance": "相关性", "consistency": "一致性",
            "answerability": "可回答性", "answer_consistency": "答案一致性",
        }
        return [
            f"{label}({getattr(self, field)})"
            for field, label in dim_names.items()
            if 0 < getattr(self, field) < 4
        ]


# ── 核心数据模型 ──────────────────────────────────────────────

class Question(BaseModel):
    """题目数据模型 - 核心数据结构"""
    id: str = Field(..., description="题目唯一标识符")
    type: QuestionType = Field(..., description="题目类型")
    content: str = Field(..., description="题干内容")
    knowledge_point: str = Field(..., description="考察的知识点")
    options: Dict[str, str] = Field(default_factory=dict, description="选项字典")
    answer: str = Field(..., description="正确答案")
    explanation: str = Field(default="", description="答案解析")

    citations: List[Citation] = Field(default_factory=list, description="参考资料引用列表")
    explanation_with_citations: str = Field(default="", description="带引用标注的解析")

    difficulty: Optional[str] = Field(None, description="难度等级")
    category: Optional[str] = Field(None, description="所属分类/章节")
    source_exam: Optional[str] = Field(None, description="来源试卷")
    section_title: Optional[str] = Field(None, description="原试卷中的章节标题")
    display_number: Optional[str] = Field(None, description="题目在所属章节中的显示题号")
    subquestion_count: int = Field(1, description="题目包含的小问数量")


class GeneratedQuestion(Question):
    """生成的新题目，增加翻新相关元数据"""
    original_question_id: str = Field(..., description="对应的原题ID")
    generation_attempt: int = Field(1, description="生成尝试次数")
    passed_audit: bool = Field(False, description="是否通过审核")
    similarity_score: Optional[float] = Field(None, description="与原题的相似度评分")

    renovation_method: Optional[RenovationMethod] = Field(None, description="采用的翻新策略")
    renovation_strategy_detail: Optional[str] = Field(None, description="本次使用的详细翻新策略组合")
    patient_background: Optional[PatientBackground] = Field(None, description="虚拟患者背景")
    quality_scores: Optional[QualityScores] = Field(None, description="五维质量评分")
    generation_time_seconds: Optional[float] = Field(None, description="出题用时（秒）")


class AuditResult(BaseModel):
    """审核结果模型"""
    passed: bool = Field(..., description="是否通过审核")
    verdict: str = Field("通过", description="综合评判：通过 / 需修改 / 严重不合格")
    reason: str = Field(..., description="审核结论说明")
    similarity_score: float = Field(0.0, description="相似度评分 (0-1)")

    issues: List[str] = Field(default_factory=list, description="发现的具体问题")
    suggestions: List[str] = Field(default_factory=list, description="修改建议列表")

    stem_original: bool = Field(True, description="题干是否原创")
    options_original: bool = Field(True, description="选项是否原创")
    knowledge_point_match: bool = Field(True, description="考点是否一致")
    has_valid_citations: bool = Field(True, description="引用是否有效")

    quality_scores: Optional[QualityScores] = Field(None, description="QGEval 2.0 七维质量评分")
    deduction_details: List[str] = Field(default_factory=list, description="扣分项详述（得分<4的维度）")


class RAGContext(BaseModel):
    """RAG检索上下文"""
    query: str = Field(..., description="检索查询")
    chunks: List[Dict] = Field(default_factory=list, description="检索到的文本块")
    
    def to_context_string(self) -> str:
        """将检索结果转换为上下文字符串"""
        if not self.chunks:
            return "无相关参考资料"
        
        context_parts = []
        for i, chunk in enumerate(self.chunks, 1):
            source = chunk.get("metadata", {}).get("source", "未知来源")
            page = chunk.get("metadata", {}).get("page", "")
            content = chunk.get("content", chunk.get("document", ""))
            
            page_info = f" (第{page}页)" if page else ""
            context_parts.append(f"[参考资料{i} - {source}{page_info}]\n{content}")
        
        return "\n\n".join(context_parts)


class ExamPaper(BaseModel):
    """试卷模型"""
    title: str = Field(..., description="试卷标题")
    questions: List[Question] = Field(default_factory=list, description="题目列表")
    total_score: Optional[int] = Field(None, description="总分")
    time_limit: Optional[int] = Field(None, description="考试时长(分钟)")
    header_lines: List[str] = Field(default_factory=list, description="标题下方需保留的头部信息行")
    
    # 元数据
    source_file: Optional[str] = Field(None, description="来源文件")
    created_at: Optional[str] = Field(None, description="创建时间")
    version: int = Field(1, description="版本号")


class GeneratedExamPaper(BaseModel):
    """生成的翻新试卷"""
    title: str = Field(..., description="试卷标题")
    questions: List[GeneratedQuestion] = Field(default_factory=list, description="生成的题目列表")
    original_exam_title: str = Field(..., description="原始试卷标题")
    version: int = Field(1, description="翻新版本号")
    
    # 统计信息
    total_questions: int = Field(0, description="题目总数")
    passed_audit_count: int = Field(0, description="通过审核的题目数")
    average_similarity: float = Field(0.0, description="平均相似度")
    header_lines: List[str] = Field(default_factory=list, description="从原试卷继承的头部信息行")
    source_file: Optional[str] = Field(None, description="原始试卷文件")
    
    # 元数据
    created_at: Optional[str] = Field(None, description="生成时间")
    generation_config: Optional[Dict] = Field(None, description="生成配置")


class WorkflowState(BaseModel):
    """工作流状态追踪"""
    current_step: str = Field("initialized", description="当前步骤")
    total_questions: int = Field(0, description="总题目数")
    processed_questions: int = Field(0, description="已处理题目数")
    failed_questions: List[str] = Field(default_factory=list, description="失败的题目ID列表")
    retry_counts: Dict[str, int] = Field(default_factory=dict, description="各题目重试次数")
    
    # 错误追踪
    errors: List[Dict] = Field(default_factory=list, description="错误记录")
    
    def add_error(self, question_id: str, error: str, step: str) -> None:
        """添加错误记录"""
        self.errors.append({
            "question_id": question_id,
            "error": error,
            "step": step
        })
