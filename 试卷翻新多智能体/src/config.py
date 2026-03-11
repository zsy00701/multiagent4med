"""
医学试卷翻新多智能体系统 - 配置管理

集中管理所有配置项，支持从环境变量和 .env 文件加载配置。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class LLMConfig(BaseSettings):
    """LLM 配置"""
    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}
    
    provider: str = Field(default="openai", alias="LLM_PROVIDER")
    api_key: str = Field(default="", alias="LLM_API_KEY")
    base_url: str = Field(default="https://api.moonshot.cn/v1", alias="LLM_BASE_URL")
    model: str = Field(default="kimi-k2.5", alias="LLM_MODEL")
    analyst_model: Optional[str] = Field(default=None, alias="ANALYST_LLM_MODEL")
    generator_model: Optional[str] = Field(default=None, alias="GENERATOR_LLM_MODEL")
    auditor_model: Optional[str] = Field(default=None, alias="AUDITOR_LLM_MODEL")
    auto_role_routing: bool = Field(default=False, alias="LLM_AUTO_ROLE_ROUTING")
    temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096)
    timeout: int = Field(default=300, alias="LLM_TIMEOUT")  # 增加到 5 分钟
    proxy: Optional[str] = Field(default=None, alias="LLM_PROXY")

    def resolve_model(self, role: Optional[str] = None) -> str:
        role_map = {
            "analyst": self.analyst_model,
            "generator": self.generator_model,
            "auditor": self.auditor_model,
        }
        explicit_model = role_map.get(role)
        if explicit_model:
            return explicit_model
        if not self.auto_role_routing or role is None:
            return self.model
        return self._resolve_auto_role_model(role)

    def _resolve_auto_role_model(self, role: str) -> str:
        if role == "generator":
            return self.model

        normalized_model = self.model.lower()
        normalized_base_url = self.base_url.lower()

        if self.provider == "gemini":
            if "gemini-2.5-pro" in normalized_model:
                return "gemini-2.5-flash"
            if "gemini-1.5-pro" in normalized_model:
                return "gemini-1.5-flash"
            return self.model

        if self.provider == "openai" and "openai.com" in normalized_base_url:
            if normalized_model.startswith("gpt-4.1"):
                return "gpt-4.1-mini"
            if normalized_model.startswith("gpt-4o"):
                return "gpt-4o-mini"
            return self.model

        return self.model


class EmbeddingConfig(BaseSettings):
    """Embedding 模型配置"""
    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}
    
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )
    device: str = Field(default="cpu")


class ChromaConfig(BaseSettings):
    """ChromaDB 配置"""
    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}
    
    persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="medical_knowledge", alias="CHROMA_COLLECTION_NAME")


class RAGConfig(BaseSettings):
    """RAG 检索配置"""
    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}
    
    chunk_size: int = Field(default=500, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="RAG_CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="RAG_TOP_K")
    knowledge_weight: float = Field(default=0.7, alias="RAG_KNOWLEDGE_WEIGHT")


class GenerationConfig(BaseSettings):
    """生成配置"""
    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}

    num_versions: int = Field(default=3, alias="NUM_EXAM_VERSIONS")
    max_retry_attempts: int = Field(default=5, alias="MAX_RETRY_ATTEMPTS")
    plagiarism_threshold: int = Field(default=8, alias="PLAGIARISM_THRESHOLD")
    enable_audit: bool = Field(default=True, alias="ENABLE_AUDIT")


class PathConfig:
    """路径配置"""
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # 自动检测项目根目录
            self.root = Path(__file__).parent.parent
        else:
            self.root = project_root
        
        # 数据目录
        self.data_dir = self.root / "data"
        self.input_dir = self.data_dir
        self.knowledge_base_dir = self.data_dir / "knowledge_base"
        self.output_dir = self.data_dir / "output"
        self.chroma_dir = self.data_dir / "chroma_db"
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_input_exam_path(self, filename: str) -> Path:
        """获取输入试卷路径"""
        return self.input_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """获取输出文件路径"""
        return self.output_dir / filename
    
    def list_knowledge_base_files(self) -> list[Path]:
        """列出知识库中的所有文件"""
        if not self.knowledge_base_dir.exists():
            return []
        
        supported_extensions = {".pdf", ".docx", ".doc", ".txt"}
        files = []
        for ext in supported_extensions:
            files.extend(self.knowledge_base_dir.glob(f"*{ext}"))
        return sorted(files)


class Settings:
    """全局配置管理器"""
    _instance: Optional["Settings"] = None
    
    def __new__(cls) -> "Settings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.llm = LLMConfig()
        self.embedding = EmbeddingConfig()
        self.chroma = ChromaConfig()
        self.rag = RAGConfig()
        self.generation = GenerationConfig()
        self.paths = PathConfig()
        
        # 日志级别
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        self._initialized = True
    
    def validate(self) -> list[str]:
        """验证配置完整性，返回错误列表"""
        errors = []
        
        if not self.llm.api_key:
            errors.append("LLM_API_KEY 未配置")
        
        if not self.paths.knowledge_base_dir.exists():
            errors.append(f"知识库目录不存在: {self.paths.knowledge_base_dir}")
        
        return errors
    
    def print_config(self) -> None:
        """打印当前配置（隐藏敏感信息）"""
        print("=" * 50)
        print("当前配置")
        print("=" * 50)
        print(f"LLM Provider: {self.llm.provider}")
        print(f"LLM Model: {self.llm.model}")
        print(f"Analyst Model: {self.llm.resolve_model('analyst')}")
        print(f"Generator Model: {self.llm.resolve_model('generator')}")
        print(f"Auditor Model: {self.llm.resolve_model('auditor')}")
        if self.llm.provider == "openai":
            print(f"LLM Base URL: {self.llm.base_url}")
        print(f"LLM API Key: {'*' * 8 + self.llm.api_key[-4:] if self.llm.api_key else '未配置'}")
        if self.llm.proxy:
            print(f"LLM Proxy: {self.llm.proxy}")
        print(f"Embedding Model: {self.embedding.model_name}")
        print(f"ChromaDB Dir: {self.chroma.persist_dir}")
        print(f"生成版本数: {self.generation.num_versions}")
        print(f"最大重试次数: {self.generation.max_retry_attempts}")
        print(f"知识库文件数: {len(self.paths.list_knowledge_base_files())}")
        print("=" * 50)


# 全局配置实例
settings = Settings()
