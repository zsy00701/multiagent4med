#!/usr/bin/env python3
"""
医学试卷翻新多智能体系统 - 独立审核入口 (QGEval 2.0)

使用方法:
    python run_audit.py --original <原卷路径> --generated <翻新卷路径> [--output <报告路径>]

示例:
    python run_audit.py --original data/原卷.docx --generated data/output/翻新版1.md
    python run_audit.py --original data/原卷.docx --generated data/output/翻新版1.md --output report.md
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

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


def main():
    parser = argparse.ArgumentParser(
        description="医学试卷独立审核工具 (QGEval 2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --original 原卷.docx --generated 翻新版1.md
  %(prog)s --original 原卷.docx --generated 翻新版1.md --output 审核报告.md
  %(prog)s --original 原卷.docx --generated 翻新版1.md --sync
        """,
    )

    parser.add_argument(
        "--original", "-o",
        type=str, required=True,
        help="原始试卷路径 (.docx)",
    )
    parser.add_argument(
        "--generated", "-g",
        type=str, required=True,
        help="翻新卷路径 (.md)",
    )
    parser.add_argument(
        "--output", "-r",
        type=str, default=None,
        help="审核报告输出路径（默认为翻新卷同目录下 *_审核报告.md）",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="使用同步模式（默认异步）",
    )
    parser.add_argument(
        "--log-level",
        type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    console.print(Panel.fit(
        "[bold blue]医学试卷独立审核工具[/bold blue]\n"
        "[dim]QGEval 2.0 Framework[/dim]",
        border_style="blue",
    ))

    errors = settings.validate()
    if errors:
        console.print("[bold red]环境检查失败：[/bold red]")
        for error in errors:
            console.print(f"  - {error}")
        sys.exit(1)

    original_path = Path(args.original)
    generated_path = Path(args.generated)

    if not original_path.is_absolute() and not original_path.exists():
        original_path = settings.paths.input_dir / original_path

    if not generated_path.is_absolute() and not generated_path.exists():
        generated_path = settings.paths.output_dir / generated_path

    if not original_path.exists():
        console.print(f"[red]找不到原卷: {original_path}[/red]")
        sys.exit(1)
    if not generated_path.exists():
        console.print(f"[red]找不到翻新卷: {generated_path}[/red]")
        sys.exit(1)

    console.print(f"\n[bold]审核配置：[/bold]")
    console.print(f"  原始试卷: {original_path}")
    console.print(f"  翻新试卷: {generated_path}")
    console.print(f"  运行模式: {'同步' if args.sync else '异步'}")

    try:
        workflow = AuditWorkflow(use_async=not args.sync)
        results = workflow.run(
            original_path=original_path,
            generated_path=generated_path,
            output_path=args.output,
        )

        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        if passed == total:
            console.print(f"\n[bold green]全部 {total} 题通过审核！[/bold green]")
        else:
            console.print(f"\n[bold yellow]{passed}/{total} 题通过审核[/bold yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]审核失败: {e}[/red]")
        logging.exception("详细错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
