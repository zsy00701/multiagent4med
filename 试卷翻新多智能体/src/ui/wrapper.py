"""ExamRefurbisher 包装器"""

import io
import json
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

from rich.console import Console

from src.workflow import ExamRefurbisher, RefurbishResult
from src.config import settings
from src.ui.utils import (
    generate_markdown_without_answers,
    generate_markdown_with_answers,
    markdown_to_pdf,
    generate_word_document,
    generate_excel_workbook,
    generate_text_file
)


class GradioExamRefurbisher:
    """Gradio UI 的 ExamRefurbisher 包装器

    提供进度回调和日志捕获功能
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        """初始化包装器

        Args:
            progress_callback: 进度回调函数，接收 (progress, desc) 参数
        """
        self.progress_callback = progress_callback
        self.log_buffer = io.StringIO()
        self.console = Console(file=self.log_buffer, force_terminal=False, width=100)
        self.logger = logging.getLogger(__name__)

    def run_with_progress(
        self,
        file_path: Path,
        params: Dict[str, Any]
    ) -> RefurbishResult:
        """运行工作流并实时更新进度

        Args:
            file_path: 试卷文件路径
            params: 参数字典

        Returns:
            RefurbishResult 对象
        """
        # 更新进度: 初始化
        self._update_progress(0.0, "初始化系统...")

        # 创建 ExamRefurbisher 实例
        refurbisher = ExamRefurbisher(
            num_versions=params['num_versions'],
            max_retries=params['max_retries'],
            use_async=params['use_async'],
            enable_audit=params.get('enable_audit', True)
        )

        # 临时覆盖配置
        original_rag_config = {
            'top_k': settings.rag.top_k,
            'knowledge_weight': settings.rag.knowledge_weight
        }
        original_gen_config = {
            'plagiarism_threshold': settings.generation.plagiarism_threshold
        }

        try:
            # 应用临时配置
            settings.rag.top_k = params['rag_top_k']
            settings.rag.knowledge_weight = params['knowledge_weight']
            settings.generation.plagiarism_threshold = params['plagiarism_threshold']

            # 更新进度: 初始化 RAG
            self._update_progress(0.1, "初始化知识库...")

            # 运行工作流
            result = refurbisher.run(
                input_exam_path=file_path,
                output_dir=settings.paths.output_dir,
                force_rebuild_rag=params['force_rebuild_rag']
            )

            # 生成额外的输出文件
            if result.success:
                self._update_progress(0.95, "生成输出文件...")
                self._generate_output_files(result, params)

            # 更新进度: 完成
            self._update_progress(1.0, "生成完成！")

            return result

        finally:
            # 恢复原始配置
            settings.rag.top_k = original_rag_config['top_k']
            settings.rag.knowledge_weight = original_rag_config['knowledge_weight']
            settings.generation.plagiarism_threshold = original_gen_config['plagiarism_threshold']

    def _update_progress(self, progress: float, desc: str):
        """更新进度

        Args:
            progress: 进度值 (0.0-1.0)
            desc: 进度描述
        """
        # 记录到日志
        self.console.print(f"[{progress*100:.0f}%] {desc}")

        # 调用回调
        if self.progress_callback:
            try:
                self.progress_callback(progress, desc=desc)
            except Exception as e:
                self.logger.warning(f"进度回调失败: {e}")

    def get_logs(self) -> str:
        """获取捕获的日志

        Returns:
            日志文本
        """
        return self.log_buffer.getvalue()

    def clear_logs(self):
        """清空日志缓冲区"""
        self.log_buffer = io.StringIO()
        self.console = Console(file=self.log_buffer, force_terminal=False, width=100)

    def _generate_output_files(self, result: RefurbishResult, params: Dict[str, Any]):
        """生成额外的输出文件

        Args:
            result: 翻新结果
            params: 参数字典
        """
        output_dir = settings.paths.output_dir
        output_formats = params.get('output_format', ['Markdown (.md)'])
        include_answers = params.get('include_answers', ['完整版（含答案和解析）'])

        for exam in result.generated_exams:
            base_name = exam.title.replace(' ', '_')

            # 读取 JSON 数据
            json_path = output_dir / f"{base_name}.json"
            if not json_path.exists():
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                exam_data = json.load(f)

            # 生成纯题目版
            if "纯题目版（无答案）" in include_answers:
                self._generate_single_version(
                    exam_data, base_name, output_dir, output_formats,
                    include_answers=False, suffix="_纯题目版"
                )

            # 生成完整版（含答案）
            if "完整版（含答案和解析）" in include_answers:
                self._generate_single_version(
                    exam_data, base_name, output_dir, output_formats,
                    include_answers=True, suffix=""
                )

    def _generate_single_version(
        self,
        exam_data: dict,
        base_name: str,
        output_dir: Path,
        output_formats: List[str],
        include_answers: bool,
        suffix: str
    ):
        """生成单个版本的多种格式文件

        Args:
            exam_data: 试卷数据
            base_name: 基础文件名
            output_dir: 输出目录
            output_formats: 输出格式列表
            include_answers: 是否包含答案
            suffix: 文件名后缀
        """
        # Markdown
        if "Markdown (.md)" in output_formats:
            md_path = output_dir / f"{base_name}{suffix}.md"
            if include_answers:
                md_content = generate_markdown_with_answers(exam_data)
            else:
                md_content = generate_markdown_without_answers(exam_data)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            self.console.print(f"[green]✓ 已生成: {md_path.name}[/green]")

        # Word
        if "Word (.docx)" in output_formats:
            docx_path = output_dir / f"{base_name}{suffix}.docx"
            if generate_word_document(exam_data, docx_path, include_answers):
                self.console.print(f"[green]✓ 已生成: {docx_path.name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ Word 生成失败（需要: pip install python-docx）[/yellow]")

        # PDF
        if "PDF (.pdf)" in output_formats:
            pdf_path = output_dir / f"{base_name}{suffix}.pdf"
            if include_answers:
                md_content = generate_markdown_with_answers(exam_data)
            else:
                md_content = generate_markdown_without_answers(exam_data)

            if markdown_to_pdf(md_content, pdf_path):
                self.console.print(f"[green]✓ 已生成: {pdf_path.name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ PDF 生成失败（需要: pip install markdown2 pdfkit 或 reportlab）[/yellow]")

        # Excel
        if "Excel (.xlsx)" in output_formats:
            xlsx_path = output_dir / f"{base_name}{suffix}.xlsx"
            if generate_excel_workbook(exam_data, xlsx_path, include_answers):
                self.console.print(f"[green]✓ 已生成: {xlsx_path.name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ Excel 生成失败（需要: pip install openpyxl）[/yellow]")

        # 纯文本
        if "纯文本 (.txt)" in output_formats:
            txt_path = output_dir / f"{base_name}{suffix}.txt"
            if generate_text_file(exam_data, txt_path, include_answers):
                self.console.print(f"[green]✓ 已生成: {txt_path.name}[/green]")
            else:
                self.console.print(f"[yellow]⚠ 文本文件生成失败[/yellow]")

