"""
医学试卷翻新多智能体系统 - 工作流编排

ExamRefurbisher 类：总控流水线，协调各智能体完成试卷翻新任务。
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from itertools import zip_longest
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .config import settings
from .file_loader import FileLoader
from .rag_engine import RAGEngine, get_rag_engine
from .llm_client import get_llm_client
from .schemas import (
    Question,
    GeneratedQuestion,
    GeneratedExamPaper,
    ExamPaper,
    AuditResult,
    WorkflowState
)
from .agents import AnalystAgent, GeneratorAgent, AuditorAgent
from .docx_formatter import DocxFormatter

logger = logging.getLogger(__name__)
console = Console()

MAX_CONCURRENCY = 3  # 降低并发度以避免 API 限流（429 错误）


@dataclass
class RefurbishResult:
    """翻新结果"""
    success: bool
    original_exam: ExamPaper
    generated_exams: List[GeneratedExamPaper]
    stats: Dict[str, Any]
    errors: List[Dict]


class ExamRefurbisher:
    """
    试卷翻新工作流编排器
    
    协调 Analyst、Generator、Auditor 三个智能体，
    完成从原卷解析到生成多套翻新卷的完整流程。
    """
    
    def __init__(
        self,
        num_versions: int = None,
        max_retries: int = None,
        use_async: bool = True,
        max_concurrency: int = MAX_CONCURRENCY,
        enable_audit: bool = True,
    ):
        self.num_versions = num_versions or settings.generation.num_versions
        self.max_retries = max_retries or settings.generation.max_retry_attempts
        self.use_async = use_async
        self.max_concurrency = max_concurrency
        self.enable_audit = enable_audit

        self.file_loader = FileLoader()
        self.rag_engine: Optional[RAGEngine] = None
        self.docx_formatter = DocxFormatter()

        analyst_llm = get_llm_client(role="analyst")
        generator_llm = get_llm_client(role="generator")
        self.analyst = AnalystAgent(analyst_llm)
        self.generator = GeneratorAgent(generator_llm)

        # 审核智能体变为可选
        if self.enable_audit:
            auditor_llm = get_llm_client(role="auditor")
            self.auditor = AuditorAgent(auditor_llm)
        else:
            self.auditor = None
            console.print("[yellow]审核功能已禁用，生成的题目将不经过审核直接通过[/yellow]")

        self.state = WorkflowState()
    
    def initialize_rag(self, force_rebuild: bool = False) -> int:
        console.print("[bold blue]初始化 RAG 知识库...[/bold blue]")
        self.rag_engine = get_rag_engine()
        
        kb_files = settings.paths.list_knowledge_base_files()
        if not kb_files:
            console.print("[yellow]警告：知识库目录为空[/yellow]")
            return 0
        
        console.print(f"发现 {len(kb_files)} 个知识库文件")
        new_chunks = self.rag_engine.ingest_knowledge_base(force_rebuild=force_rebuild)
        stats = self.rag_engine.get_stats()
        console.print(f"[green]知识库就绪，共 {stats['total_chunks']} 个文档块[/green]")
        return new_chunks
    
    # ── 运行入口 ──────────────────────────────────────────────
    
    def run(
        self,
        input_exam_path: str | Path,
        output_dir: Optional[str | Path] = None,
        force_rebuild_rag: bool = False,
    ) -> RefurbishResult:
        if self.use_async:
            return asyncio.run(self.run_async(input_exam_path, output_dir, force_rebuild_rag))
        return self._run_sync(input_exam_path, output_dir, force_rebuild_rag)
    
    def _run_sync(self, input_exam_path, output_dir=None, force_rebuild_rag=False) -> RefurbishResult:
        input_path = Path(input_exam_path)
        output_path = Path(output_dir) if output_dir else settings.paths.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        errors: List[Dict] = []
        generated_exams: List[GeneratedExamPaper] = []
        
        try:
            self.state.current_step = "初始化RAG"
            self.initialize_rag(force_rebuild_rag)
            
            self.state.current_step = "解析原卷"
            console.print(f"\n[bold blue]解析原始试卷: {input_path.name}[/bold blue]")
            exam_text = self.file_loader.load(input_path)
            original_exam = self.analyst.analyze(exam_text, input_path.name)
            self.state.total_questions = len(original_exam.questions)
            console.print(f"[green]解析完成，共 {self.state.total_questions} 道题目[/green]")
            self._print_exam_structure(original_exam)
            
            self.state.current_step = "生成翻新卷"
            for version in range(1, self.num_versions + 1):
                console.print(f"\n[bold cyan]生成第 {version}/{self.num_versions} 套翻新卷[/bold cyan]")
                gen_questions, ver_errors = self._generate_version_sync(original_exam, version)
                errors.extend(ver_errors)
                exam = self._build_exam(original_exam, gen_questions, version)
                generated_exams.append(exam)
                self._save_exam(original_exam, exam, output_path)
            
            stats = self._compute_stats(original_exam, generated_exams)
            self._print_summary(stats)
            return RefurbishResult(True, original_exam, generated_exams, stats, errors)
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            console.print(f"[red]错误: {e}[/red]")
            return RefurbishResult(False, ExamPaper(title="", questions=[]), [], {}, [{"error": str(e)}])
    
    async def run_async(self, input_exam_path, output_dir=None, force_rebuild_rag=False) -> RefurbishResult:
        input_path = Path(input_exam_path)
        output_path = Path(output_dir) if output_dir else settings.paths.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        errors: List[Dict] = []
        generated_exams: List[GeneratedExamPaper] = []
        
        try:
            self.state.current_step = "初始化RAG"
            self.initialize_rag(force_rebuild_rag)
            
            self.state.current_step = "解析原卷"
            console.print(f"\n[bold blue]解析原始试卷: {input_path.name}[/bold blue]")
            exam_text = self.file_loader.load(input_path)
            original_exam = await self.analyst.analyze_async(exam_text, input_path.name)
            self.state.total_questions = len(original_exam.questions)
            console.print(f"[green]解析完成，共 {self.state.total_questions} 道题目[/green]")
            self._print_exam_structure(original_exam)
            
            self.state.current_step = "生成翻新卷"
            semaphore = asyncio.Semaphore(self.max_concurrency)
            
            for version in range(1, self.num_versions + 1):
                console.print(f"\n[bold cyan]生成第 {version}/{self.num_versions} 套翻新卷[/bold cyan]")
                gen_questions, ver_errors = await self._generate_version_async(
                    original_exam, version, semaphore
                )
                errors.extend(ver_errors)
                exam = self._build_exam(original_exam, gen_questions, version)
                generated_exams.append(exam)
                self._save_exam(original_exam, exam, output_path)
            
            stats = self._compute_stats(original_exam, generated_exams)
            self._print_summary(stats)
            return RefurbishResult(True, original_exam, generated_exams, stats, errors)
            
        except Exception as e:
            logger.error(f"异步工作流执行失败: {e}")
            console.print(f"[red]错误: {e}[/red]")
            return RefurbishResult(False, ExamPaper(title="", questions=[]), [], {}, [{"error": str(e)}])
    
    # ── 版本生成 ──────────────────────────────────────────────
    
    def _generate_version_sync(
        self, original_exam: ExamPaper, version: int
    ) -> tuple[List[GeneratedQuestion], List[Dict]]:
        generated_questions: List[GeneratedQuestion] = []
        errors: List[Dict] = []
        
        with self._make_progress() as progress:
            task = progress.add_task("[cyan]生成试题...", total=len(original_exam.questions))
            for q in original_exam.questions:
                try:
                    gen_q = self._process_question_sync(q, version)
                    if gen_q:
                        generated_questions.append(gen_q)
                        self.state.processed_questions += 1
                    else:
                        errors.append({"question_id": q.id, "error": "题目在最大重试后仍未通过审核，已阻断整套试卷输出"})
                        self.state.failed_questions.append(q.id)
                except Exception as e:
                    logger.error(f"处理题目 {q.id} 失败: {e}")
                    errors.append({"question_id": q.id, "label": self._question_label(q), "error": str(e)})
                    self.state.failed_questions.append(q.id)
                    break
                progress.update(task, advance=1)
        return generated_questions, errors
    
    async def _generate_version_async(
        self, original_exam: ExamPaper, version: int, semaphore: asyncio.Semaphore
    ) -> tuple[List[GeneratedQuestion], List[Dict]]:
        errors: List[Dict] = []
        
        with self._make_progress() as progress:
            task = progress.add_task("[cyan]并发生成试题...", total=len(original_exam.questions))
            
            async def _bounded(q: Question):
                async with semaphore:
                    result = await self._process_question_async(q, version, progress, task)
                    await asyncio.sleep(0.5)
                    return result
            
            results = await asyncio.gather(
                *(_bounded(q) for q in original_exam.questions),
                return_exceptions=True,
            )
        
        generated_map: Dict[int, GeneratedQuestion] = {}
        failed_indices: List[int] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                q_id = original_exam.questions[i].id
                logger.error(f"题目 {q_id} 处理失败: {result}")
                failed_indices.append(i)
            elif result is not None:
                generated_map[i] = result
                self.state.processed_questions += 1
            else:
                failed_indices.append(i)

        if failed_indices:
            console.print(
                f"[yellow]有 {len(failed_indices)} 道题目首轮失败，开始串行补救重试...[/yellow]"
            )
            for idx in failed_indices:
                q = original_exam.questions[idx]
                q_label = self._question_label(q)
                try:
                    logger.info(f"补救重试题目 {q_label}")
                    result = await self._retry_loop(q, version, use_async=True)
                    if result is not None:
                        generated_map[idx] = result
                        self.state.processed_questions += 1
                        if q.id in self.state.failed_questions:
                            self.state.failed_questions.remove(q.id)
                        console.print(f"[green]  ✓ 题目 {q_label} 补救成功[/green]")
                    else:
                        errors.append({
                            "question_id": q.id, "label": q_label,
                            "error": "题目在补救重试后仍未生成合格题目",
                        })
                        self.state.failed_questions.append(q.id)
                        console.print(f"[red]  ✗ 题目 {q_label} 补救失败[/red]")
                except Exception as e:
                    logger.error(f"题目 {q_label} 补救重试失败: {e}")
                    errors.append({
                        "question_id": q.id, "label": q_label,
                        "error": str(e),
                    })
                    self.state.failed_questions.append(q.id)
                    console.print(f"[red]  ✗ 题目 {q_label} 补救失败: {e}[/red]")

        generated_questions = [generated_map[i] for i in sorted(generated_map.keys())]
        return generated_questions, errors
    
    # ── 单题处理 ──────────────────────────────────────────────
    
    def _process_question_sync(self, question: Question, version: int) -> Optional[GeneratedQuestion]:
        return self._retry_loop(question, version, use_async=False)
    
    async def _process_question_async(
        self, question: Question, version: int, progress: Progress, task_id
    ) -> Optional[GeneratedQuestion]:
        result = await self._retry_loop(question, version, use_async=True)
        progress.update(task_id, advance=1)
        return result
    
    def _retry_loop(self, question: Question, version: int, use_async: bool):
        """统一的重试循环，根据 use_async 切换调用方式"""
        if self.rag_engine is None:
            raise RuntimeError("RAG 引擎未初始化，请先调用 initialize_rag()")
        question_label = self._question_label(question)
        rag_context = self.rag_engine.get_context_for_question(
            question.content, question.knowledge_point,
            knowledge_weight=settings.rag.knowledge_weight,
        )

        async def _async_impl():
            attempt = 1
            audit_feedback = None
            last_generated = None
            while attempt <= self.max_retries:
                try:
                    generated = await self.generator.generate_async(question, rag_context, audit_feedback, attempt)
                    last_generated = generated

                    if not self.enable_audit:
                        generated.passed_audit = True
                        generated.similarity_score = 0.0
                        return generated

                    audit_result = await self.auditor.audit_async(question, generated)
                    generated.passed_audit = audit_result.passed
                    generated.similarity_score = audit_result.similarity_score
                    if audit_result.passed:
                        return generated
                    audit_feedback = audit_result
                    attempt += 1
                    self.state.retry_counts[question.id] = attempt
                except Exception as e:
                    logger.error(f"题目 {question_label} 异步尝试 {attempt} 失败: {e}")
                    attempt += 1
            if last_generated is not None:
                logger.warning(f"题目 {question_label} 耗尽重试，使用最后一次生成结果（未通过审核）")
                last_generated.passed_audit = False
                return last_generated
            raise ValueError(self._build_failure_message(question, audit_feedback))

        if use_async:
            return _async_impl()

        attempt = 1
        audit_feedback = None
        last_generated = None
        while attempt <= self.max_retries:
            try:
                generated = self.generator.generate(question, rag_context, audit_feedback, attempt)
                last_generated = generated

                if not self.enable_audit:
                    generated.passed_audit = True
                    generated.similarity_score = 0.0
                    return generated

                audit_result = self.auditor.audit(question, generated)
                generated.passed_audit = audit_result.passed
                generated.similarity_score = audit_result.similarity_score
                if audit_result.passed:
                    return generated
                audit_feedback = audit_result
                attempt += 1
                self.state.retry_counts[question.id] = attempt
            except Exception as e:
                logger.error(f"题目 {question_label} 尝试 {attempt} 失败: {e}")
                attempt += 1
        if last_generated is not None:
            logger.warning(f"题目 {question_label} 耗尽重试，使用最后一次生成结果（未通过审核）")
            last_generated.passed_audit = False
            return last_generated
        raise ValueError(self._build_failure_message(question, audit_feedback))
    
    @staticmethod
    def _make_progress() -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )

    @staticmethod
    def _clean_exam_title(exam: ExamPaper) -> str:
        """从原始文件名或解析标题中提取干净的试卷标题。"""
        import re
        if exam.source_file:
            title = Path(exam.source_file).stem
        else:
            title = exam.title
        title = re.sub(r'[（(]\s*[）)]\s*', '', title)
        title = re.sub(r'\s{2,}', ' ', title).strip()
        return title

    @staticmethod
    def _question_label(question: Question) -> str:
        section = question.section_title or "未分区"
        number = question.display_number or question.id
        return f"{section} 第{number}题"

    @staticmethod
    def _build_failure_message(question: Question, audit_feedback: Optional[AuditResult]) -> str:
        label = ExamRefurbisher._question_label(question)
        if audit_feedback is None:
            return f"{label} 在最大重试后仍未生成合格题目"

        issue_text = "；".join(audit_feedback.issues[:3]) if audit_feedback.issues else "无具体问题明细"
        return (
            f"{label} 在最大重试后仍未通过审核：{audit_feedback.reason}"
            f"；主要问题：{issue_text}"
        )

    @staticmethod
    def _print_exam_structure(exam: ExamPaper) -> None:
        section_counts: Dict[str, int] = {}
        for question in exam.questions:
            key = question.section_title or question.type.value
            section_counts[key] = section_counts.get(key, 0) + 1

        table = Table(title="原卷结构概览")
        table.add_column("章节")
        table.add_column("题数", justify="right")
        for section, count in section_counts.items():
            table.add_row(section, str(count))
        console.print(table)
    
    def _build_exam(
        self, original: ExamPaper, questions: List[GeneratedQuestion], version: int
    ) -> GeneratedExamPaper:
        self._validate_exam_structure(original, questions)
        unpassed_questions = [q.display_number or q.id for q in questions if not q.passed_audit]
        if unpassed_questions:
            raise ValueError(f"存在未通过审核的题目，拒绝导出试卷：{', '.join(unpassed_questions)}")
        passed_count = sum(1 for q in questions if q.passed_audit)
        avg_similarity = (
            sum(q.similarity_score or 0 for q in questions) / max(len(questions), 1)
        )
        clean_title = self._clean_exam_title(original)
        return GeneratedExamPaper(
            title=f"{clean_title} (翻新卷{version})",
            questions=questions,
            original_exam_title=clean_title,
            version=version,
            total_questions=len(questions),
            passed_audit_count=passed_count,
            average_similarity=avg_similarity,
            header_lines=original.header_lines,
            source_file=original.source_file,
            created_at=datetime.now().isoformat(),
            generation_config={"num_versions": self.num_versions, "max_retries": self.max_retries},
        )

    @staticmethod
    def _validate_exam_structure(original: ExamPaper, questions: List[GeneratedQuestion]) -> None:
        """严格校验翻新卷结构必须与原卷一致。

        这里不假设固定题量，而是按当前原卷动态对齐：
        - 顶层题目总数一致
        - 章节顺序一致
        - 每题题型一致
        - 每题显示题号一致
        - 每题小问数量一致
        """
        if len(questions) != len(original.questions):
            raise ValueError(
                f"翻新卷题量不一致：原卷 {len(original.questions)} 题，生成卷 {len(questions)} 题"
            )

        for index, (original_question, generated_question) in enumerate(zip(original.questions, questions), 1):
            if original_question.type != generated_question.type:
                raise ValueError(
                    f"第 {index} 题题型不一致：原卷为 {original_question.type.value}，生成卷为 {generated_question.type.value}"
                )

            if original_question.section_title != generated_question.section_title:
                raise ValueError(
                    f"第 {index} 题章节不一致：原卷为 {original_question.section_title}，生成卷为 {generated_question.section_title}"
                )

            if original_question.display_number != generated_question.display_number:
                raise ValueError(
                    f"第 {index} 题显示题号不一致：原卷为 {original_question.display_number}，生成卷为 {generated_question.display_number}"
                )

            if original_question.subquestion_count != generated_question.subquestion_count:
                raise ValueError(
                    f"第 {index} 题小问数量不一致：原卷为 {original_question.subquestion_count}，生成卷为 {generated_question.subquestion_count}"
                )
    
    def _save_exam(self, original_exam: ExamPaper, exam: GeneratedExamPaper, output_dir: Path) -> None:
        """保存翻新试卷。结构不合格或审核未通过的试卷不会走到这里。"""
        # 生成规范的文件名
        filename = self._generate_safe_filename(exam, original_exam)

        # 保存 Markdown 格式
        md_path = output_dir / f"{filename}.md"
        md_content = self._generate_markdown(exam)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 保存 DOCX 格式（仅题目，不含答案和解析）
        docx_path = output_dir / f"{filename}.docx"
        self.docx_formatter.convert_to_docx(
            exam,
            docx_path,
            include_answers=False,
            include_explanations=False
        )

        console.print(f"[green]已保存: {md_path.name}, {docx_path.name}[/green]")

    @staticmethod
    def _generate_safe_filename(exam: GeneratedExamPaper, original_exam: ExamPaper) -> str:
        """
        生成安全且规范的文件名

        格式：原试卷名_翻新版本号_生成时间
        例如：护理学基础试卷_v1_20260310_143025
        """
        import re
        from datetime import datetime

        from pathlib import Path

        if original_exam.source_file:
            original_title = Path(original_exam.source_file).stem
        else:
            original_title = original_exam.title
        original_title = re.sub(r'[\s\-_]*试卷[\s\-_]*$', '', original_title)
        original_title = re.sub(r'[\s\-_]*试题[\s\-_]*$', '', original_title)

        # 移除或替换不安全的文件名字符
        safe_title = re.sub(r'[<>:"/\\|?*\s]+', '_', original_title)
        safe_title = re.sub(r'_+', '_', safe_title)  # 合并多个下划线
        safe_title = safe_title.strip('_')  # 移除首尾下划线

        # 限制长度
        if len(safe_title) > 30:
            safe_title = safe_title[:30].rstrip('_')

        # 生成时间戳
        if exam.created_at:
            try:
                dt = datetime.fromisoformat(exam.created_at.replace('Z', '+00:00'))
                timestamp = dt.strftime('%Y%m%d_%H%M%S')
            except:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 组合文件名：原试卷名_v版本号_时间戳
        filename = f"{safe_title}_v{exam.version}_{timestamp}"

        return filename

    @staticmethod
    def _generate_structure_report(original_exam: ExamPaper, exam: GeneratedExamPaper) -> str:
        """生成原卷 vs 翻新卷的结构一致性报告。"""
        def section_counts(questions: List[Question | GeneratedQuestion]) -> Dict[str, int]:
            stats: Dict[str, int] = {}
            for question in questions:
                key = question.section_title or "未分区"
                stats[key] = stats.get(key, 0) + 1
            return stats

        original_sections = section_counts(original_exam.questions)
        generated_sections = section_counts(exam.questions)
        total_match = len(original_exam.questions) == len(exam.questions)
        section_match = original_sections == generated_sections

        lines = [
            f"# {exam.title} 结构校验报告",
            "",
            f"- 原始试卷：{original_exam.title}",
            f"- 翻新试卷：{exam.title}",
            f"- 原卷题量：{len(original_exam.questions)}",
            f"- 翻新卷题量：{len(exam.questions)}",
            f"- 题量一致：{'是' if total_match else '否'}",
            f"- 章节题量一致：{'是' if section_match else '否'}",
            "",
            "## 章节统计",
            "",
        ]

        all_sections = list(dict.fromkeys(list(original_sections.keys()) + list(generated_sections.keys())))
        for section in all_sections:
            lines.append(
                f"- {section}：原卷 {original_sections.get(section, 0)} 题 / 翻新卷 {generated_sections.get(section, 0)} 题"
            )

        lines.extend([
            "",
            "## 逐题比对",
            "",
        ])

        for index, pair in enumerate(zip_longest(original_exam.questions, exam.questions), 1):
            original_question, generated_question = pair
            if original_question is None:
                lines.append(f"- 第 {index} 题：翻新卷多出题目，章节={generated_question.section_title} 题号={generated_question.display_number}")
                continue
            if generated_question is None:
                lines.append(f"- 第 {index} 题：翻新卷缺少题目，原卷章节={original_question.section_title} 题号={original_question.display_number}")
                continue

            checks = {
                '章节': original_question.section_title == generated_question.section_title,
                '题号': original_question.display_number == generated_question.display_number,
                '题型': original_question.type == generated_question.type,
                '小问数': original_question.subquestion_count == generated_question.subquestion_count,
                '选项数': len(original_question.options) == len(generated_question.options),
            }
            status = '通过' if all(checks.values()) else '异常'
            detail = '，'.join(f"{name}={'是' if ok else '否'}" for name, ok in checks.items())
            lines.append(
                f"- 第 {index} 题：{status} | 原卷[{original_question.section_title} / {original_question.display_number} / {original_question.type.value}] "
                f"→ 翻新[{generated_question.section_title} / {generated_question.display_number} / {generated_question.type.value}] | {detail}"
            )

        lines.extend([
            "",
            "## 结论",
            "",
            "- 若逐题比对均为“通过”，说明翻新卷在题量、章节、题型、题号和大小题结构上与原卷一致。",
            "- 若出现“异常”，应直接判定该翻新卷结构不合格，需要重新生成。",
        ])
        return "\n".join(lines)
    
    def _generate_markdown(self, exam: GeneratedExamPaper) -> str:
        """生成 Markdown 格式的试卷（符合出题规范输出格式）"""
        lines = [
            f"# {exam.title}",
            "",
            f"- 原始试卷: {exam.original_exam_title}",
            f"- 版本号: {exam.version}",
            f"- 生成时间: {exam.created_at}",
            f"- 题目总数: {exam.total_questions}",
            f"- 通过审核: {exam.passed_audit_count}",
            "",
            "---",
            "",
        ]
        
        for i, q in enumerate(exam.questions, 1):
            lines.append(f"## 第 {i} 题")
            lines.append("")

            # 考点标注
            lines.append(f"**【考点】** {q.knowledge_point}")
            lines.append("")

            # 翻新策略
            if q.renovation_strategy_detail or q.renovation_method:
                strategy_text = q.renovation_strategy_detail or q.renovation_method.value
                lines.append(f"**【翻新策略】** {strategy_text}")
                lines.append("")

            # 题目正文
            lines.append(f"**【题干】** ({q.type.value})")
            lines.append("")
            lines.append(q.content)
            lines.append("")
            
            # 选项列表
            if q.options:
                lines.append("**【选项】**")
                lines.append("")
                for key, value in q.options.items():
                    lines.append(f"- {key}. {value}")
                lines.append("")
            
            # 标准答案
            lines.append(f"**【正确答案】** {q.answer}")
            lines.append("")
            
            # 深度解析
            if q.explanation_with_citations:
                lines.append("**【深度解析】**")
                lines.append("")
                lines.append(q.explanation_with_citations)
                lines.append("")

            # 质量评分 (QGEval 2.0)
            if q.quality_scores:
                qs = q.quality_scores
                lines.append(
                    f"**【质量评分】** 流畅性={qs.fluency} | 清晰度={qs.clarity} | "
                    f"简洁性={qs.conciseness} | 相关性={qs.relevance} | 一致性={qs.consistency} | "
                    f"可回答性={qs.answerability} | 答案一致性={qs.answer_consistency} "
                    f"| 均分={qs.average:.1f}"
                )
                lines.append("")
            
            # 出题用时
            if q.generation_time_seconds:
                lines.append(f"**【出题用时】** {q.generation_time_seconds:.1f} 秒")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def _compute_stats(
        self,
        original: ExamPaper,
        generated: List[GeneratedExamPaper]
    ) -> Dict[str, Any]:
        """计算统计信息（含质量评分和翻新策略统计）"""
        total_generated = sum(e.total_questions for e in generated)
        total_passed = sum(e.passed_audit_count for e in generated)
        avg_similarity = sum(e.average_similarity for e in generated) / max(len(generated), 1)
        
        # 质量评分统计
        all_questions = [q for e in generated for q in e.questions]
        quality_scores_list = [q.quality_scores for q in all_questions if q.quality_scores]
        avg_quality = 0.0
        dim_averages = {}
        if quality_scores_list:
            avg_quality = sum(qs.average for qs in quality_scores_list) / len(quality_scores_list)
            for dim in ("fluency", "clarity", "conciseness", "relevance", "consistency", "answerability", "answer_consistency"):
                vals = [getattr(qs, dim) for qs in quality_scores_list if getattr(qs, dim) > 0]
                dim_averages[dim] = sum(vals) / len(vals) if vals else 0
        
        # 翻新策略分布
        strategy_counts: Dict[str, int] = {}
        for q in all_questions:
            method = q.renovation_method.value if q.renovation_method else "未指定"
            strategy_counts[method] = strategy_counts.get(method, 0) + 1
        
        # 平均出题用时
        times = [q.generation_time_seconds for q in all_questions if q.generation_time_seconds]
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            "original_questions": len(original.questions),
            "versions_generated": len(generated),
            "total_questions_generated": total_generated,
            "total_passed_audit": total_passed,
            "pass_rate": total_passed / max(total_generated, 1),
            "average_similarity": avg_similarity,
            "failed_questions": len(self.state.failed_questions),
            "retry_stats": self.state.retry_counts,
            "average_quality_score": avg_quality,
            "quality_dim_averages": dim_averages,
            "strategy_distribution": strategy_counts,
            "average_generation_time": avg_time,
        }
    
    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """打印汇总信息"""
        console.print("\n")
        
        table = Table(title="翻新结果汇总")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")
        
        table.add_row("原始题目数", str(stats["original_questions"]))
        table.add_row("生成版本数", str(stats["versions_generated"]))
        table.add_row("生成题目总数", str(stats["total_questions_generated"]))
        table.add_row("通过审核数", str(stats["total_passed_audit"]))
        table.add_row("通过率", f"{stats['pass_rate']:.1%}")
        table.add_row("平均相似度", f"{stats['average_similarity']:.2f}")
        table.add_row("失败题目数", str(stats["failed_questions"]))
        
        if stats.get("average_quality_score"):
            table.add_row("平均质量评分", f"{stats['average_quality_score']:.1f}/5.0")
        if stats.get("average_generation_time"):
            table.add_row("平均出题用时", f"{stats['average_generation_time']:.1f}s")
        
        console.print(table)
        
        # 翻新策略分布
        if stats.get("strategy_distribution"):
            strat_table = Table(title="翻新策略分布")
            strat_table.add_column("策略", style="cyan")
            strat_table.add_column("数量", style="green")
            for method, count in stats["strategy_distribution"].items():
                strat_table.add_row(method, str(count))
            console.print(strat_table)
        
        # QGEval 2.0 七维质量评分详情
        if stats.get("quality_dim_averages"):
            dim_names = {
                "fluency": "流畅性",
                "clarity": "清晰度",
                "conciseness": "简洁性",
                "relevance": "相关性",
                "consistency": "一致性",
                "answerability": "可回答性",
                "answer_consistency": "答案一致性",
            }
            qual_table = Table(title="QGEval 2.0 七维质量评分均值")
            qual_table.add_column("维度", style="cyan")
            qual_table.add_column("均分", style="green")
            for dim, avg in stats["quality_dim_averages"].items():
                qual_table.add_row(dim_names.get(dim, dim), f"{avg:.1f}/5.0")
            console.print(qual_table)
