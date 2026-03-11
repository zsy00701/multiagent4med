#!/usr/bin/env python3
"""
批量 QGEval 评估：对所有翻新试卷运行独立审核，并输出汇总报告。
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.audit_workflow import AuditWorkflow

console = Console()


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_path=False)
        ],
    )


TOPIC_KEYWORDS = ["肾上腺", "前列腺增生", "前列腺癌", "泌尿系结石"]

ORIGINAL_MAP = {
    "肾上腺": "新护士理论考试答卷  (肾上腺、膀胱）.docx",
    "前列腺增生": "新护士理论考试答卷（前列腺增生）.docx",
    "前列腺癌": "新护士理论考试答卷（前列腺癌）.docx",
    "泌尿系结石": "新护士理论考试答卷（泌尿系结石）.docx",
}


def find_generated_versions(output_dir: Path, topic: str) -> dict[str, Path]:
    """找到某个主题下每个版本号对应的最新翻新卷 (.md)。"""
    candidates = sorted(output_dir.glob("*.md"))
    versions = {}  # v1 -> latest path
    for f in candidates:
        if topic not in f.name:
            continue
        if "_审核报告" in f.name:
            continue
        for vn in ["_v1_", "_v2_", "_v3_"]:
            if vn in f.name:
                label = vn.strip("_")
                if label not in versions or f.name > versions[label].name:
                    versions[label] = f
    return versions


async def run_single_audit(workflow: AuditWorkflow, original: Path, generated: Path) -> dict:
    """运行单对审核，返回汇总结果。"""
    results = await workflow.run_async(original, generated)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    need_fix = sum(1 for r in results if r.get("verdict") == "需修改")
    rejected = sum(1 for r in results if r.get("verdict") == "严重不合格")
    avg_scores = [r.get("average_score", 0) for r in results if r.get("average_score", 0) > 0]
    avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0

    dim_keys = ["fluency", "clarity", "conciseness", "relevance",
                "consistency", "answerability", "answer_consistency"]
    dim_avgs = {}
    for key in dim_keys:
        vals = [r["quality_scores"].get(key, 0) for r in results
                if r.get("quality_scores") and r["quality_scores"].get(key, 0) > 0]
        dim_avgs[key] = sum(vals) / len(vals) if vals else 0

    # 翻新专项指标（要求相似度 < 0.65，题干/选项原创、考点匹配）
    sim_vals = [r.get("similarity_score", 0) for r in results if "similarity_score" in r]
    avg_similarity = sum(sim_vals) / len(sim_vals) if sim_vals else 0
    stem_orig = sum(1 for r in results if r.get("stem_original", False)) / max(total, 1)
    opts_orig = sum(1 for r in results if r.get("options_original", False)) / max(total, 1)
    kp_match = sum(1 for r in results if r.get("knowledge_point_match", False)) / max(total, 1)
    sim_ok = sum(1 for s in sim_vals if s < 0.65) / len(sim_vals) if sim_vals else 0

    return {
        "original": original.name,
        "generated": generated.name,
        "total": total,
        "passed": passed,
        "need_fix": need_fix,
        "rejected": rejected,
        "pass_rate": passed / max(total, 1),
        "avg_score": avg,
        "dim_avgs": dim_avgs,
        "avg_similarity": avg_similarity,
        "similarity_ok_rate": sim_ok,
        "stem_original_rate": stem_orig,
        "options_original_rate": opts_orig,
        "knowledge_point_match_rate": kp_match,
        "details": results,
    }


def save_summary_report(all_results: list[dict], output_path: Path):
    """生成汇总对比报告。"""
    dim_labels = [
        ("fluency", "流畅性"), ("clarity", "清晰度"), ("conciseness", "简洁性"),
        ("relevance", "相关性"), ("consistency", "一致性"),
        ("answerability", "可回答性"), ("answer_consistency", "答案一致性"),
    ]

    lines = [
        "# QGEval 2.0 批量审核汇总报告",
        "",
        f"- 审核时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 审核试卷数: {len(all_results)}",
        "",
        "---",
        "",
        "## 总览表",
        "",
        "| 试卷 | 题数 | 通过 | 需修改 | 不合格 | 通过率 | QGEval均分 | 相似度 | 相似达标率 | 题干原创 | 选项原创 | 考点匹配 |",
        "|------|------|------|--------|--------|--------|------------|--------|------------|----------|----------|----------|",
    ]

    for r in all_results:
        short_name = r["generated"].replace("新护士理论考试答卷", "").split("_2026")[0]
        sim = r.get("avg_similarity", 0)
        sim_ok = r.get("similarity_ok_rate", 0)
        stem = r.get("stem_original_rate", 0)
        opts = r.get("options_original_rate", 0)
        kp = r.get("knowledge_point_match_rate", 0)
        lines.append(
            f"| {short_name} | {r['total']} | {r['passed']} | {r['need_fix']} | "
            f"{r['rejected']} | {r['pass_rate']:.0%} | {r['avg_score']:.1f}/5.0 | "
            f"{sim:.2f} | {sim_ok:.0%} | {stem:.0%} | {opts:.0%} | {kp:.0%} |"
        )

    lines.extend(["", "---", ""])

    lines.append("## 各维度均分对比")
    lines.append("")
    header = "| 试卷 | " + " | ".join(label for _, label in dim_labels) + " | 综合 |"
    sep = "|------|" + "|".join(["------"] * (len(dim_labels) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in all_results:
        short_name = r["generated"].replace("新护士理论考试答卷", "").split("_2026")[0]
        dims = " | ".join(f"{r['dim_avgs'].get(k, 0):.1f}" for k, _ in dim_labels)
        lines.append(f"| {short_name} | {dims} | {r['avg_score']:.1f} |")

    lines.extend(["", "---", ""])

    by_topic = defaultdict(list)
    for r in all_results:
        for kw in TOPIC_KEYWORDS:
            if kw in r["generated"]:
                by_topic[kw].append(r)
                break

    lines.append("## 按主题汇总")
    lines.append("")
    for topic, rs in by_topic.items():
        avg_pass = sum(r["pass_rate"] for r in rs) / len(rs) if rs else 0
        avg_score = sum(r["avg_score"] for r in rs) / len(rs) if rs else 0
        avg_sim = sum(r.get("avg_similarity", 0) for r in rs) / len(rs) if rs else 0
        avg_sim_ok = sum(r.get("similarity_ok_rate", 0) for r in rs) / len(rs) if rs else 0
        lines.append(f"### {topic}")
        lines.append(f"- 版本数: {len(rs)}")
        lines.append(f"- 平均通过率: {avg_pass:.0%}")
        lines.append(f"- 平均 QGEval 均分: {avg_score:.1f}/5.0")
        lines.append(f"- 平均相似度: {avg_sim:.2f}")
        lines.append(f"- 相似达标率: {avg_sim_ok:.0%}")
        lines.append("")

    lines.extend(["", "---", ""])
    lines.append("## 翻新专项指标说明")
    lines.append("")
    lines.append("- **相似度**：翻新题与原题语义相似度 (0-1)，越低越好")
    lines.append("- **相似达标率**：相似度 < 0.65 的题目占比（翻新要求有显著差异）")
    lines.append("- **题干原创**：题干非照搬原题的比例")
    lines.append("- **选项原创**：选项非照搬原题的比例")
    lines.append("- **考点匹配**：考点与原题一致的比例")
    lines.append("")

    lines.extend(["---", ""])
    grand_avg = sum(r["avg_score"] for r in all_results) / len(all_results) if all_results else 0
    grand_pass = sum(r["pass_rate"] for r in all_results) / len(all_results) if all_results else 0
    grand_sim = sum(r.get("avg_similarity", 0) for r in all_results) / len(all_results) if all_results else 0
    grand_sim_ok = sum(r.get("similarity_ok_rate", 0) for r in all_results) / len(all_results) if all_results else 0
    grand_stem = sum(r.get("stem_original_rate", 0) for r in all_results) / len(all_results) if all_results else 0
    grand_opts = sum(r.get("options_original_rate", 0) for r in all_results) / len(all_results) if all_results else 0
    grand_kp = sum(r.get("knowledge_point_match_rate", 0) for r in all_results) / len(all_results) if all_results else 0
    lines.append("## 全局统计")
    lines.append(f"- 总审核版本数: {len(all_results)}")
    lines.append(f"- 全局平均通过率: {grand_pass:.0%}")
    lines.append(f"- 全局 QGEval 均分: {grand_avg:.1f}/5.0")
    lines.append(f"- 全局平均相似度: {grand_sim:.2f}（越低越好，目标 < 0.65）")
    lines.append(f"- 全局相似达标率: {grand_sim_ok:.0%}")
    lines.append(f"- 全局题干原创率: {grand_stem:.0%}")
    lines.append(f"- 全局选项原创率: {grand_opts:.0%}")
    lines.append(f"- 全局考点匹配率: {grand_kp:.0%}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[bold green]汇总报告已保存: {output_path}[/bold green]")


async def main():
    setup_logging("INFO")

    console.print(Panel.fit(
        "[bold blue]QGEval 2.0 批量审核[/bold blue]\n"
        "[dim]对所有翻新试卷进行独立质量评估[/dim]",
        border_style="blue",
    ))

    errors = settings.validate()
    if errors:
        console.print("[bold red]环境检查失败：[/bold red]")
        for error in errors:
            console.print(f"  - {error}")
        sys.exit(1)

    data_dir = settings.paths.input_dir
    output_dir = settings.paths.output_dir

    pairs = []
    for topic, orig_name in ORIGINAL_MAP.items():
        orig_path = data_dir / orig_name
        if not orig_path.exists():
            console.print(f"[yellow]跳过 {topic}: 原卷不存在 {orig_path}[/yellow]")
            continue
        versions = find_generated_versions(output_dir, topic)
        if not versions:
            console.print(f"[yellow]跳过 {topic}: 未找到翻新卷[/yellow]")
            continue
        for vn in sorted(versions):
            pairs.append((topic, orig_path, versions[vn]))
            console.print(f"  [cyan]{topic}[/cyan] {vn}: {versions[vn].name}")

    console.print(f"\n[bold]共 {len(pairs)} 对待审核[/bold]\n")

    if not pairs:
        console.print("[red]没有找到任何待审核的配对[/red]")
        sys.exit(1)

    workflow = AuditWorkflow(use_async=True)
    all_results = []

    for i, (topic, orig_path, gen_path) in enumerate(pairs, 1):
        console.print(Panel(
            f"[bold]{i}/{len(pairs)}  {topic} — {gen_path.name}[/bold]",
            border_style="cyan",
        ))
        t0 = time.time()
        result = await run_single_audit(workflow, orig_path, gen_path)
        elapsed = time.time() - t0
        sim = result.get("avg_similarity", 0)
        sim_ok = result.get("similarity_ok_rate", 0)
        console.print(
            f"  [green]完成[/green] — 通过率 {result['pass_rate']:.0%}, "
            f"均分 {result['avg_score']:.1f}/5.0, 相似度 {sim:.2f} (达标{sim_ok:.0%}), 耗时 {elapsed:.0f}s\n"
        )
        all_results.append(result)

    summary_path = output_dir / f"QGEval_批量审核汇总_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_summary_report(all_results, summary_path)

    table = Table(title="QGEval 2.0 批量审核汇总")
    table.add_column("试卷", style="cyan")
    table.add_column("通过率", style="green")
    table.add_column("均分", style="yellow")
    table.add_column("相似度", style="magenta")
    table.add_column("相似达标", style="blue")
    for r in all_results:
        short = r["generated"].replace("新护士理论考试答卷", "").split("_2026")[0]
        sim = r.get("avg_similarity", 0)
        sim_ok = r.get("similarity_ok_rate", 0)
        table.add_row(short, f"{r['pass_rate']:.0%}", f"{r['avg_score']:.1f}/5.0",
                      f"{sim:.2f}", f"{sim_ok:.0%}")
    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
