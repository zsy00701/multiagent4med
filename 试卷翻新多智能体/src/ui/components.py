"""UI 组件定义"""

import gradio as gr
from src.config import settings


def create_param_panel():
    """创建参数配置面板

    Returns:
        参数组件列表
    """
    with gr.Group():
        gr.Markdown("### ⚙️ 参数配置")

        num_versions = gr.Slider(
            minimum=1,
            maximum=10,
            value=settings.generation.num_versions,
            step=1,
            label="生成版本数",
            info="生成多少套翻新试卷"
        )

        max_retries = gr.Slider(
            minimum=1,
            maximum=5,
            value=settings.generation.max_retry_attempts,
            step=1,
            label="最大重试次数",
            info="单题生成失败时的最大重试次数"
        )

        rag_top_k = gr.Slider(
            minimum=1,
            maximum=10,
            value=settings.rag.top_k,
            step=1,
            label="RAG 检索数量",
            info="从知识库检索的相关文档数量"
        )

        knowledge_weight = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=settings.rag.knowledge_weight,
            step=0.1,
            label="知识点权重",
            info="0.7-0.8: 侧重知识点 | 0.5-0.6: 知识点和内容并重"
        )

        plagiarism_threshold = gr.Slider(
            minimum=5,
            maximum=15,
            value=settings.generation.plagiarism_threshold,
            step=1,
            label="抄袭阈值",
            info="连续相同字符数超过此值视为抄袭"
        )

        force_rebuild = gr.Checkbox(
            label="强制重建 RAG 索引",
            value=False,
            info="重新索引知识库文件（首次运行或知识库更新时使用）"
        )

        use_async = gr.Checkbox(
            label="使用异步模式",
            value=True,
            info="异步模式可提高处理速度"
        )

        enable_audit = gr.Checkbox(
            label="启用审核功能",
            value=settings.generation.enable_audit,
            info="禁用后生成的题目将不经过审核直接通过（可加快速度）"
        )

        gr.Markdown("### 📄 输出选项")

        output_format = gr.CheckboxGroup(
            choices=["Markdown (.md)", "Word (.docx)", "PDF (.pdf)", "Excel (.xlsx)", "纯文本 (.txt)"],
            value=["Markdown (.md)"],
            label="输出格式",
            info="可同时选择多种格式"
        )

        include_answers = gr.CheckboxGroup(
            choices=["纯题目版（无答案）", "完整版（含答案和解析）"],
            value=["完整版（含答案和解析）"],
            label="生成版本",
            info="可同时生成两种版本"
        )

    return [
        num_versions,
        max_retries,
        rag_top_k,
        knowledge_weight,
        plagiarism_threshold,
        force_rebuild,
        use_async,
        enable_audit,
        output_format,
        include_answers
    ]


def create_result_panel():
    """创建结果展示面板

    Returns:
        结果组件列表
    """
    with gr.Tabs():
        with gr.Tab("📄 生成结果"):
            markdown_output = gr.Markdown(
                value="等待生成...",
                label="试卷预览"
            )
            with gr.Accordion("JSON 数据", open=False):
                json_output = gr.JSON(label="详细数据")
            download_files = gr.File(
                label="下载文件",
                file_count="multiple"
            )

        with gr.Tab("📊 统计信息"):
            stats_table = gr.Dataframe(
                label="统计数据",
                headers=["指标", "值"],
                datatype=["str", "str"]
            )

        with gr.Tab("📝 实时日志"):
            log_output = gr.Textbox(
                label="日志输出",
                lines=20,
                max_lines=30,
                interactive=False
            )

        with gr.Tab("📚 知识库状态"):
            kb_status = gr.Textbox(
                label="状态摘要",
                interactive=False
            )
            kb_files = gr.Dataframe(
                label="索引文件列表",
                headers=["文件名", "大小", "状态"],
                datatype=["str", "str", "str"]
            )
            refresh_kb_btn = gr.Button("🔄 刷新知识库状态", size="sm")

    return [
        markdown_output,
        json_output,
        download_files,
        stats_table,
        log_output,
        kb_status,
        kb_files,
        refresh_kb_btn
    ]
