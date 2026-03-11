"""
医学试卷翻新多智能体系统 - RAG 引擎

基于 ChromaDB 和 sentence-transformers 的本地向量检索引擎。
负责知识库的索引构建和相似度检索。
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .config import settings
from .file_loader import load_knowledge_base, DocumentChunk, FileLoader
from .schemas import RAGContext

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG 检索引擎"""
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        初始化 RAG 引擎

        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: 集合名称
            embedding_model: Embedding 模型名称
        """
        self.persist_dir = Path(persist_dir or settings.chroma.persist_dir)
        self.collection_name = collection_name or settings.chroma.collection_name
        self.embedding_model_name = embedding_model or settings.embedding.model_name

        # 确保目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 初始化 Embedding 模型
            logger.info(f"加载 Embedding 模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=settings.embedding.device
            )
        except Exception as e:
            logger.error(f"加载 Embedding 模型失败: {e}")
            raise RuntimeError(f"无法加载 Embedding 模型 {self.embedding_model_name}: {e}")

        try:
            # 初始化 ChromaDB 客户端
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 获取或创建集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )

            logger.info(f"RAG 引擎初始化完成，集合中已有 {self.collection.count()} 条记录")
        except Exception as e:
            logger.error(f"初始化 ChromaDB 失败: {e}")
            raise RuntimeError(f"无法初始化 ChromaDB: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值，用于检测文件变化"""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_indexed_files(self) -> Dict[str, str]:
        """获取已索引的文件及其哈希值"""
        metadata_path = self.persist_dir / "indexed_files.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_indexed_files(self, indexed: Dict[str, str]) -> None:
        """保存已索引文件的记录"""
        metadata_path = self.persist_dir / "indexed_files.json"
        with open(metadata_path, "w") as f:
            json.dump(indexed, f, ensure_ascii=False, indent=2)
    
    def ingest_knowledge_base(
        self,
        knowledge_base_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        force_rebuild: bool = False
    ) -> int:
        """
        导入知识库，构建向量索引

        Args:
            knowledge_base_dir: 知识库目录
            chunk_size: 块大小
            chunk_overlap: 块重叠
            force_rebuild: 是否强制重建索引

        Returns:
            新增的文档块数量
        """
        kb_dir = knowledge_base_dir or settings.paths.knowledge_base_dir
        chunk_size = chunk_size or settings.rag.chunk_size
        chunk_overlap = chunk_overlap or settings.rag.chunk_overlap

        if not kb_dir.exists():
            logger.warning(f"知识库目录不存在: {kb_dir}")
            return 0

        # 获取已索引的文件
        indexed_files = {} if force_rebuild else self._get_indexed_files()

        # 如果强制重建，清空集合
        if force_rebuild:
            logger.info("强制重建索引，清空现有集合")
            try:
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.error(f"清空集合失败: {e}")
                raise

        # 检查需要索引的文件
        new_chunks_count = 0
        updated_indexed = indexed_files.copy()
        file_loader = FileLoader()

        for ext in [".docx", ".doc", ".pdf", ".txt"]:
            for file_path in kb_dir.glob(f"*{ext}"):
                file_key = str(file_path)

                try:
                    current_hash = self._compute_file_hash(file_path)
                except Exception as e:
                    logger.error(f"计算文件哈希失败 {file_path.name}: {e}")
                    continue

                # 检查文件是否已索引且未变化
                if file_key in indexed_files and indexed_files[file_key] == current_hash:
                    logger.debug(f"文件未变化，跳过: {file_path.name}")
                    continue

                logger.info(f"索引文件: {file_path.name}")

                # 如果文件已存在但有变化，先删除旧的索引
                if file_key in indexed_files:
                    try:
                        self._delete_file_chunks(file_path.name)
                    except Exception as e:
                        logger.warning(f"删除旧索引失败 {file_path.name}: {e}")

                # 直接加载单个文件并切分
                try:
                    file_chunks = file_loader.load_with_chunks(
                        file_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    if file_chunks:
                        self._add_chunks(file_chunks)
                        new_chunks_count += len(file_chunks)
                        logger.info(f"成功索引 {file_path.name}，新增 {len(file_chunks)} 个块")
                    else:
                        logger.warning(f"文件 {file_path.name} 未产生任何文档块")

                    updated_indexed[file_key] = current_hash

                except Exception as e:
                    logger.error(f"索引文件失败 {file_path.name}: {e}")
                    continue

        # 保存索引记录
        try:
            self._save_indexed_files(updated_indexed)
        except Exception as e:
            logger.error(f"保存索引记录失败: {e}")

        logger.info(f"知识库索引完成，新增 {new_chunks_count} 个块，总计 {self.collection.count()} 个块")
        return new_chunks_count
    
    def _delete_file_chunks(self, source_name: str) -> None:
        """删除指定来源的所有文档块"""
        try:
            # ChromaDB 不直接支持按 metadata 删除，需要先查询再删除
            results = self.collection.get(
                where={"source": source_name}
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"删除旧索引: {source_name}, {len(results['ids'])} 个块")
        except Exception as e:
            logger.warning(f"删除旧索引失败: {e}")
    
    def _add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """添加文档块到向量库"""
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk.source}_{chunk.chunk_id or i}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "page": chunk.page or 0,
                "chunk_id": chunk.chunk_id or i,
                **(chunk.metadata or {})
            })

        try:
            # 批量计算 Embedding
            logger.debug(f"计算 {len(documents)} 个文档块的 Embedding")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()

            # 添加到集合
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        except Exception as e:
            logger.error(f"添加文档块到向量库失败: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        n: int = None,
        filter_source: Optional[str] = None
    ) -> RAGContext:
        """
        检索相关文档

        Args:
            query: 查询文本
            n: 返回结果数量
            filter_source: 可选，过滤特定来源

        Returns:
            RAGContext 包含检索结果
        """
        n = n or settings.rag.top_k

        if self.collection.count() == 0:
            logger.warning("向量库为空，无法检索")
            return RAGContext(query=query, chunks=[])

        try:
            # 计算查询向量
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            ).tolist()

            # 构建查询参数
            query_params: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n,
                "include": ["documents", "metadatas", "distances"]
            }

            if filter_source:
                query_params["where"] = {"source": filter_source}

            # 执行查询
            results = self.collection.query(**query_params)

            # 转换结果格式
            chunks = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0

                    chunks.append({
                        "content": doc,
                        "document": doc,  # 兼容字段
                        "metadata": {
                            "source": metadata.get("source", "未知"),
                            "page": metadata.get("page"),
                            "chunk_id": metadata.get("chunk_id"),
                            "similarity": 1 - distance  # 转换为相似度
                        }
                    })

            logger.debug(f"检索 '{query[:50]}...' 返回 {len(chunks)} 条结果")
            return RAGContext(query=query, chunks=chunks)
        except Exception as e:
            logger.error(f"RAG检索过程出错: {e}，返回空结果")
            return RAGContext(query=query, chunks=[])
    
    def get_context_for_question(
        self,
        question_content: str,
        knowledge_point: str,
        n: int = None,
        knowledge_weight: float = None
    ) -> RAGContext:
        """
        为特定题目获取相关上下文

        Args:
            question_content: 题目内容
            knowledge_point: 考点
            n: 结果数量
            knowledge_weight: 知识点权重 (0-1)，越高越侧重知识点匹配，None 则使用默认值 0.7

        Returns:
            RAGContext
        """
        # 策略：先基于知识点检索，再结合题目内容
        # 这样可以找到知识点相关但题目不同的内容

        # 使用默认值
        if knowledge_weight is None:
            knowledge_weight = 0.7

        try:
            if knowledge_weight >= 0.9:
                # 高权重：主要基于知识点
                query = knowledge_point
            elif knowledge_weight >= 0.5:
                # 中权重：知识点优先，题目内容辅助
                query = f"{knowledge_point}\n\n相关内容：{question_content[:150]}"
            else:
                # 低权重：知识点和题目内容并重
                query = f"{knowledge_point}\n{question_content[:200]}"

            logger.debug(f"RAG检索策略 - 知识点权重: {knowledge_weight}, 查询: {query[:100]}...")
            return self.retrieve(query, n)
        except Exception as e:
            logger.error(f"RAG检索失败: {e}，返回空上下文")
            return RAGContext(query=knowledge_point, chunks=[])
    
    def reset(self) -> None:
        """重置向量库"""
        logger.warning("重置向量库")
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._save_indexed_files({})
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        indexed_files = self._get_indexed_files()
        return {
            "total_chunks": self.collection.count(),
            "indexed_files": len(indexed_files),
            "files": list(indexed_files.keys()),
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name
        }


# 全局 RAG 引擎实例
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """获取全局 RAG 引擎实例"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
