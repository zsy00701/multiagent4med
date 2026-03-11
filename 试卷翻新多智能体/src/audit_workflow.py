"""
医学试卷翻新多智能体系统 - 独立审核工作流

加载原卷 + 翻新卷 → 逐题配对审核 → 输出 QGEval 2.0 详细评分报告。
与生成流程 (workflow.py) 完全解耦。
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import settings
from .file_loader import FileLoader
from .llm_client import get_llm_client
from .md_parser import parse_generated_markdown
from .schemas import Question, GeneratedQuestion, AuditResult
from .agents import AnalystAgent, AuditorAgent

logger = logging.getLogger(__name__)
console = Console()


class AuditWorkflow:
    """独立审核工作流：加载原卷和翻新卷，逐题审核并输出报告。"""

    def __init__(self, use_async: bool = True):
        self.use_async = use_async
        self.file_loader = FileLoader()

        analyst_llm = get_llm_client(role="analyst")
        auditor_llm = get_llm_client(role="auditor")
        self.analyst = AnalystAgent(analyst_llm)
        self.auditor = AuditorAgent(auditor_llm)

    def run(
        self,
        original_path: str | Path,
        generated_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> List[Dict[str, Any]]:
        """执行独立审核。"""
        if self.use_async:
            return asyncio.run(self.run_async(original_path, generated_path, output_path))
        return self._run_sync(original_path, generated_path, output_path)

    def _run_sync(self, original_path, generated_path, output_path=None) -> List[Dict[str, Any]]:
        original_path = Path(original_path)
        generated_path = Path(generated_path)

        console.print(f"\n[bold blue]解析原始试卷: {original_path.name}[/bold blue]")
        exam_text = self.file_loader.load(original_path)
        original_exam = self.analyst.analyze(exam_text, original_path.name)
        console.print(f"[green]原卷解析完成，共 {len(original_exam.questions)} 道题目[/green]")

        console.print(f"\n[bold blue]解析翻新卷: {generated_path.name}[/bold blue]")
        generated_questions = parse_generated_markdown(generated_path)
        console.print(f"[green]翻新卷解析完成，共 {len(generated_questions)} 道题目[/green]")

        results = self._audit_all_sync(original_exam.questions, generated_questions)

        report_path = self._resolve_output_path(output_path, original_path, generated_path)
        self._save_report(results, report_path, original_path.name, generated_path.name)
        self._print_summary(results)
        return results

    async def run_async(self, original_path, generated_path, output_path=None) -> List[Dict[str, Any]]:
        original_path = Path(original_path)
        generated_path = Path(generated_path)

        console.print(f"\n[bold blue]解析原始试卷: {original_path.name}[/bold blue]")
        exam_text = self.file_loader.load(original_path)
        original_exam = await self.analyst.analyze_async(exam_text, original_path.name)
        console.print(f"[green]原卷解析完成，共 {len(original_exam.questions)} 道题目[/green]")

        console.print(f"\n[bold blue]解析翻新卷: {generated_path.name}[/bold blue]")
        generated_questions = parse_generated_markdown(generated_path)
        console.print(f"[green]翻新卷解析完成，共 {len(generated_questions)} 道题目[/green]")

        results = await self._audit_all_async(original_exam.questions, generated_questions)

        report_path = self._resolve_output_path(output_path, original_path, generated_path)
        self._save_report(results, report_path, original_path.name, generated_path.name)
        self._print_summary(results)
        return results

    # ── 逐题审核 ──────────────────────────────────────────────

    def _audit_all_sync(
        self,
        originals: List[Question],
        generated: List[GeneratedQuestion],
    ) -> List[Dict[str, Any]]:
        results = []
        pairs = self._pair_questions(originals, generated)
        console.print(f"\n[bold cyan]开始逐题审核（共 {len(pairs)} 题）...[/bold cyan]")

        for i, (orig, gen) in enumerate(pairs, 1):
            console.print(f"  审核第 {i}/{len(pairs)} 题...", end=" ")
            try:
                audit_result = self.auditor.audit(orig, gen)
                results.append(self._pack_result(i, orig, gen, audit_result))
                console.print(f"[{'green' if audit_result.passed else 'red'}]{audit_result.verdict}[/]")
            except Exception as e:
                logger.error(f"审核第 {i} 题失败: {e}")
                results.append(self._pack_error(i, orig, gen, e))
                console.print(f"[red]出错: {e}[/red]")
        return results

    async def _audit_all_async(
        self,
        originals: List[Question],
        generated: List[GeneratedQuestion],
    ) -> List[Dict[str, Any]]:
        pairs = self._pair_questions(originals, generated)
        console.print(f"\n[bold cyan]开始逐题审核（共 {len(pairs)} 题）...[/bold cyan]")

        async def _audit_one(idx: int, orig: Question, gen: GeneratedQuestion):
            try:
                audit_result = await self.auditor.audit_async(orig, gen)
                return self._pack_result(idx, orig, gen, audit_result)
            except Exception as e:
                logger.error(f"审核第 {idx} 题失败: {e}")
                return self._pack_error(idx, orig, gen, e)

        tasks = [_audit_one(i, o, g) for i, (o, g) in enumerate(pairs, 1)]
        results = await asyncio.gather(*tasks)

        for r in results:
            verdict = r.get("verdict", "出错")
            color = "green" if r.get("passed") else "red"
            console.print(f"  第 {r['index']} 题: [{color}]{verdict}[/]")

        return list(results)

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _pair_questions(
        originals: List[Question],
        generated: List[GeneratedQuestion],
    ) -> list:
        if len(originals) != len(generated):
            console.print(
                f"[yellow]警告：原卷 {len(originals)} 题 vs 翻新卷 {len(generated)} 题，"
                f"按较少的一方配对[/yellow]"
            )
        return list(zip(originals, generated))

    @staticmethod
    def _pack_result(idx: int, orig: Question, gen: GeneratedQuestion, audit: AuditResult) -> Dict[str, Any]:
        return {
            "index": idx,
            "question_id": orig.id,
            "question_type": orig.type.value,
            "knowledge_point": orig.knowledge_point,
            "passed": audit.passed,
            "verdict": audit.verdict,
            "similarity_score": audit.similarity_score,
            "stem_original": audit.stem_original,
            "options_original": audit.options_original,
            "knowledge_point_match": audit.knowledge_point_match,
            "quality_scores": {
                "fluency": audit.quality_scores.fluency if audit.quality_scores else 0,
                "clarity": audit.quality_scores.clarity if audit.quality_scores else 0,
                "conciseness": audit.quality_scores.conciseness if audit.quality_scores else 0,
                "relevance": audit.quality_scores.relevance if audit.quality_scores else 0,
                "consistency": audit.quality_scores.consistency if audit.quality_scores else 0,
                "answerability": audit.quality_scores.answerability if audit.quality_scores else 0,
                "answer_consistency": audit.quality_scores.answer_consistency if audit.quality_scores else 0,
            },
            "average_score": audit.quality_scores.average if audit.quality_scores else 0,
            "issues": audit.issues,
            "suggestions": audit.suggestions,
            "deduction_details": audit.deduction_details,
            "reason": audit.reason,
        }

    @staticmethod
    def _pack_error(idx: int, orig: Question, gen: GeneratedQuestion, error: Exception) -> Dict[str, Any]:
        return {
            "index": idx,
            "question_id": orig.id,
            "question_type": orig.type.value,
            "knowledge_point": orig.knowledge_point,
            "passed": False,
            "verdict": "严重不合格",
            "similarity_score": 0,
            "quality_scores": {},
            "average_score": 0,
            "issues": [f"审核过程出错: {error}"],
            "suggestions": ["请重新生成"],
            "deduction_details": [],
            "reason": f"审核出错: {error}",
        }

    @staticmethod
    def _resolve_output_path(output_path, original_path: Path, generated_path: Path) -> Path:
        if output_path:
            return Path(output_path)
        return generated_path.parent / f"{generated_path.stem}_审核报告.md"

    # ── 报告生成 ──────────────────────────────────────────────

    def _save_report(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        original_name: str,
        generated_name: str,
    ) -> None:
        lines = [
            f"# QGEval 2.0 审核报告",
            "",
            f"- 原始试卷: {original_name}",
            f"- 翻新试卷: {generated_name}",
            f"- 审核时间: {datetime.now().isoformat()}",
            f"- 题目总数: {len(results)}",
            f"- 通过: {sum(1 for r in results if r['passed'])}",
            f"- 需修改: {sum(1 for r in results if r['verdict'] == '需修改')}",
            f"- 严重不合格: {sum(1 for r in results if r['verdict'] == '严重不合格')}",
            "",
            "---",
            "",
        ]

        dim_labels = [
            ("fluency", "流畅性"), ("clarity", "清晰度"), ("conciseness", "简洁性"),
            ("relevance", "相关性"), ("consistency", "一致性"),
            ("answerability", "可回答性"), ("answer_consistency", "答案一致性"),
        ]

        for r in results:
            verdict_icon = {"通过": "PASS", "需修改": "WARN", "严重不合格": "FAIL"}.get(r["verdict"], "?")
            lines.append(f"## 第 {r['index']} 题 [{verdict_icon}]")
            lines.append("")
            lines.append(f"- **题型**: {r['question_type']}")
            lines.append(f"- **考点**: {r['knowledge_point']}")
            lines.append(f"- **综合评判**: {r['verdict']}")
            lines.append(f"- **相似度**: {r['similarity_score']:.2f}")
            lines.append("")

            qs = r.get("quality_scores", {})
            if qs:
                lines.append("**【雷达图得分】**")
                lines.append("")
                for key, label in dim_labels:
                    score = qs.get(key, 0)
                    bar = "█" * score + "░" * (5 - score)
                    lines.append(f"- {label}: {bar} {score}/5")
                lines.append(f"- **均分**: {r.get('average_score', 0):.1f}/5.0")
                lines.append("")

            deductions = r.get("deduction_details", [])
            if deductions:
                lines.append("**【扣分项详述】**")
                lines.append("")
                for d in deductions:
                    lines.append(f"- {d}")
                lines.append("")
            else:
                lines.append("**【扣分项详述】** 无")
                lines.append("")

            if r.get("issues"):
                lines.append("**【发现的问题】**")
                lines.append("")
                for issue in r["issues"]:
                    lines.append(f"- {issue}")
                lines.append("")

            if r.get("suggestions"):
                lines.append("**【修改建议】**")
                lines.append("")
                for s in r["suggestions"]:
                    lines.append(f"- {s}")
                lines.append("")

            lines.append(f"**【审核结论】** {r['reason']}")
            lines.append("")
            lines.append("---")
            lines.append("")

        # 汇总统计
        lines.extend(self._build_summary_section(results, dim_labels))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"\n[green]审核报告已保存: {output_path}[/green]")

    @staticmethod
    def _build_summary_section(results: List[Dict], dim_labels: list) -> List[str]:
        lines = [
            "## 汇总统计",
            "",
        ]
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        lines.append(f"- 通过率: {passed}/{total} ({passed / max(total, 1):.0%})")

        all_scores = [r.get("quality_scores", {}) for r in results if r.get("quality_scores")]
        if all_scores:
            lines.append("")
            lines.append("**各维度均值：**")
            lines.append("")
            for key, label in dim_labels:
                vals = [s.get(key, 0) for s in all_scores if s.get(key, 0) > 0]
                avg = sum(vals) / len(vals) if vals else 0
                lines.append(f"- {label}: {avg:.1f}/5.0")
            overall = [r.get("average_score", 0) for r in results if r.get("average_score", 0) > 0]
            if overall:
                lines.append(f"- **总均分**: {sum(overall) / len(overall):.1f}/5.0")

        return lines

    # ── 终端摘要 ──────────────────────────────────────────────

    def _print_summary(self, results: List[Dict[str, Any]]) -> None:
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        need_fix = sum(1 for r in results if r["verdict"] == "需修改")
        rejected = sum(1 for r in results if r["verdict"] == "严重不合格")

        table = Table(title="QGEval 2.0 审核汇总")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="green")
        table.add_row("题目总数", str(total))
        table.add_row("通过", f"[green]{passed}[/green]")
        table.add_row("需修改", f"[yellow]{need_fix}[/yellow]")
        table.add_row("严重不合格", f"[red]{rejected}[/red]")
        table.add_row("通过率", f"{passed / max(total, 1):.0%}")

        avg_scores = [r.get("average_score", 0) for r in results if r.get("average_score", 0) > 0]
        if avg_scores:
            table.add_row("QGEval 均分", f"{sum(avg_scores) / len(avg_scores):.1f}/5.0")

        console.print("")
        console.print(table)
