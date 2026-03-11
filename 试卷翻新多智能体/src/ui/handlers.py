"""事件处理函数"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Any

import gradio as gr
import pandas as pd

from src.config import settings
from src.ui.wrapper import GradioExamRefurbisher
from src.ui.utils import validate_file, cleanup_temp_files, format_file_size


logger = logging.getLogger(__name__)


def process_exam(
    file,
    num_versions: int,
    max_retries: int,
    rag_top_k: int,
    knowledge_weight: float,
    plagiarism_threshold: int,
    force_rebuild: bool,
    use_async: bool,
    enable_audit: bool,
    output_format: List[str],
    include_answers: List[str],
    progress=gr.Progress()
) -> List[Any]:
    """处理试卷生成请求

    Args:
        file: 上传的文件
        num_versions: 生成版本数
        max_retries: 最大重试次数
        rag_top_k: RAG 检索数量
        knowledge_weight: 知识点权重
        plagiarism_threshold: 抄袭阈值
        force_rebuild: 是否强制重建 RAG
        use_async: 是否使用异步模式
        enable_audit: 是否启用审核功能
        output_format: 输出格式 (Markdown/PDF/Markdown + PDF)
        include_answers: 生成版本列表
        progress: Gradio 进度对象

    Returns:
        [markdown_content, json_data, download_files, stats_df, log_content, kb_status, kb_files]
    """
    file_path = None

    try:
        # 验证文件
        is_valid, msg = validate_file(file)
        if not is_valid:
            raise gr.Error(msg)

        # 创建临时目录
        temp_dir = Path("data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 清理旧文件
        cleanup_temp_files(temp_dir, max_age_hours=24)

        # 保存上传文件
        file_path = temp_dir / Path(file.name).name
        shutil.copy(file.name, file_path)

        logger.info(f"文件已上传: {file_path}")

        # 准备参数
        params = {
            'num_versions': int(num_versions),
            'max_retries': int(max_retries),
            'rag_top_k': int(rag_top_k),
            'knowledge_weight': float(knowledge_weight),
            'plagiarism_threshold': int(plagiarism_threshold),
            'force_rebuild_rag': force_rebuild,
            'use_async': use_async,
            'enable_audit': enable_audit,
            'output_format': output_format,
            'include_answers': include_answers
        }

        # 创建包装器并运行
        wrapper = GradioExamRefurbisher(
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )

        result = wrapper.run_with_progress(file_path, params)

        # 格式化返回结果
        return format_results(result, wrapper)

    except gr.Error:
        raise
    except Exception as e:
        logger.exception("处理失败")
        raise gr.Error(f"处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if file_path and file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")


def format_results(result, wrapper: GradioExamRefurbisher) -> List[Any]:
    """格式化结果为 Gradio 组件可用的格式

    Args:
        result: RefurbishResult 对象
        wrapper: GradioExamRefurbisher 实例

    Returns:
        [markdown_content, json_data, download_files, stats_df, log_content, kb_status, kb_files]
    """
    # 获取日志
    log_content = wrapper.get_logs()

    # 获取知识库状态
    kb_status, kb_files = get_kb_status()

    if not result.success:
        error_msg = "# ❌ 生成失败\n\n" + "\n".join(f"- {err}" for err in result.errors)
        return [
            error_msg,
            {},
            None,
            pd.DataFrame(),
            log_content,
            kb_status,
            kb_files
        ]

    # 读取生成的文件
    output_dir = settings.paths.output_dir
    markdown_files = sorted(output_dir.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    json_files = sorted(output_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    pdf_files = sorted(output_dir.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)
    docx_files = sorted(output_dir.glob("*.docx"), key=lambda x: x.stat().st_mtime, reverse=True)
    xlsx_files = sorted(output_dir.glob("*.xlsx"), key=lambda x: x.stat().st_mtime, reverse=True)
    txt_files = sorted(output_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)

    # Markdown 内容（显示第一个文件）
    if markdown_files:
        markdown_content = markdown_files[0].read_text(encoding='utf-8')
    else:
        markdown_content = "# ⚠️ 未找到生成的 Markdown 文件"

    # JSON 数据
    if json_files:
        json_data = json.loads(json_files[0].read_text(encoding='utf-8'))
    else:
        json_data = {}

    # 下载文件列表（包含所有生成的文件）
    num_versions = result.stats.get('versions_generated', 1)
    download_files = (
        markdown_files[:num_versions * 2] +
        json_files[:num_versions] +
        pdf_files[:num_versions * 2] +
        docx_files[:num_versions * 2] +
        xlsx_files[:num_versions * 2] +
        txt_files[:num_versions * 2]
    )

    # 统计信息
    stats_data = [
        ["原始题目数", str(result.stats.get('original_questions', 0))],
        ["生成版本数", str(result.stats.get('versions_generated', 0))],
        ["生成题目总数", str(result.stats.get('total_questions_generated', 0))],
        ["通过审核数", str(result.stats.get('total_passed_audit', 0))],
        ["通过率", f"{result.stats.get('pass_rate', 0)*100:.1f}%"],
        ["平均相似度", f"{result.stats.get('average_similarity', 0):.2f}"],
        ["失败题目数", str(result.stats.get('failed_questions', 0))]
    ]
    stats_df = pd.DataFrame(stats_data, columns=["指标", "值"])

    return [
        markdown_content,
        json_data,
        download_files if download_files else None,
        stats_df,
        log_content,
        kb_status,
        kb_files
    ]


def get_kb_status() -> tuple[str, pd.DataFrame]:
    """获取知识库状态

    Returns:
        (状态摘要, 文件列表 DataFrame)
    """
    try:
        kb_dir = settings.paths.knowledge_base_dir
        if not kb_dir.exists():
            return "知识库目录不存在", pd.DataFrame()

        # 获取所有文件
        files = []
        for ext in ['.pdf', '.docx', '.doc', '.txt']:
            files.extend(kb_dir.glob(f"*{ext}"))

        # 读取索引状态
        indexed_file = settings.paths.chroma_dir / "indexed_files.json"
        indexed_files = set()
        if indexed_file.exists():
            try:
                indexed_data = json.loads(indexed_file.read_text(encoding='utf-8'))
                indexed_files = set(indexed_data.keys())
            except Exception as e:
                logger.warning(f"读取索引文件失败: {e}")

        # 构建文件列表
        files_data = []
        for f in sorted(files, key=lambda x: x.name):
            status = "✅ 已索引" if str(f) in indexed_files else "⏳ 未索引"
            files_data.append([
                f.name,
                format_file_size(f.stat().st_size),
                status
            ])

        files_df = pd.DataFrame(files_data, columns=["文件名", "大小", "状态"])

        # 状态摘要
        status = f"📚 知识库文件: {len(files)} 个 | ✅ 已索引: {len(indexed_files)} 个"

        return status, files_df

    except Exception as e:
        logger.exception("获取知识库状态失败")
        return f"❌ 获取状态失败: {str(e)}", pd.DataFrame()
