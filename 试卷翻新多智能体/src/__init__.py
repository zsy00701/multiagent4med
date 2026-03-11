"""
医学试卷翻新多智能体系统 (Medical Exam Refurbishment System)

基于 RAG 的医学试卷翻新系统，能够读取原始医学试卷，结合外部知识库，
生成多套翻新试卷。核心原则：考点一致、表达原创、严禁照搬、拒绝幻觉。

主要模块：
- config: 配置管理
- llm_client: LLM 客户端封装
- file_loader: 多格式文件加载器
- rag_engine: RAG 向量检索引擎
- schemas: Pydantic 数据模型
- workflow: 工作流编排
- agents: 智能体模块
"""

from .config import settings, Settings
from .llm_client import LLMClient, get_llm_client
from .file_loader import FileLoader, read_docx_full_text, load_knowledge_base
from .rag_engine import RAGEngine, get_rag_engine
from .schemas import (
    Question,
    QuestionType,
    GeneratedQuestion,
    AuditResult,
    ExamPaper,
    GeneratedExamPaper,
    RAGContext,
    Citation,
    RenovationMethod,
    PatientBackground,
    QualityScores,
    CLINICAL_CASE_TYPES,
    UPGRADABLE_TO_CASE,
)

__version__ = "1.1.0"
__author__ = "Medical Exam AI Team"

__all__ = [
    # 配置
    "settings",
    "Settings",
    
    # LLM
    "LLMClient",
    "get_llm_client",
    
    # 文件处理
    "FileLoader",
    "read_docx_full_text",
    "load_knowledge_base",
    
    # RAG
    "RAGEngine",
    "get_rag_engine",
    
    # 数据模型
    "Question",
    "QuestionType",
    "GeneratedQuestion",
    "AuditResult",
    "ExamPaper",
    "GeneratedExamPaper",
    "RAGContext",
    "Citation",
    "RenovationMethod",
    "PatientBackground",
    "QualityScores",
    "CLINICAL_CASE_TYPES",
    "UPGRADABLE_TO_CASE",
]
