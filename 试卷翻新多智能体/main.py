#!/usr/bin/env python3
"""
医学试卷翻新多智能体系统 - 主程序入口

使用方法:
    python main.py --input <试卷路径> [--versions N] [--output <输出目录>]

示例:
    python main.py --input "新护士理论考试答卷（前列腺增生）.docx" --versions 3
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.workflow import ExamRefurbisher

console = Console()


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志系统"""
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False
            )
        ]
    )


def check_environment() -> bool:
    """检查运行环境"""
    errors = settings.validate()
    
    if errors:
        console.print("[bold red]环境检查失败：[/bold red]")
        for error in errors:
            console.print(f"  - {error}")
        return False
    
    return True


def list_available_exams() -> list[Path]:
    """列出可用的试卷文件"""
    exam_dir = settings.paths.input_dir
    exams = []
    
    for ext in [".docx", ".doc"]:
        exams.extend(exam_dir.glob(f"*{ext}"))
    
    return sorted(exams)


def interactive_mode() -> tuple[Path, int]:
    """交互式选择试卷"""
    console.print("\n[bold cyan]可用的试卷文件：[/bold cyan]")
    
    exams = list_available_exams()
    
    if not exams:
        console.print("[yellow]未找到试卷文件，请将 .docx 文件放入 data/ 目录[/yellow]")
        sys.exit(1)
    
    for i, exam in enumerate(exams, 1):
        console.print(f"  {i}. {exam.name}")
    
    while True:
        try:
            choice = console.input("\n[cyan]请选择试卷编号 (输入数字): [/cyan]")
            idx = int(choice) - 1
            if 0 <= idx < len(exams):
                selected_exam = exams[idx]
                break
            else:
                console.print("[red]无效的选择，请重试[/red]")
        except ValueError:
            console.print("[red]请输入有效的数字[/red]")
    
    while True:
        try:
            versions = console.input("[cyan]生成几套翻新卷? (默认 3): [/cyan]") or "3"
            num_versions = int(versions)
            if num_versions > 0:
                break
            else:
                console.print("[red]请输入正整数[/red]")
        except ValueError:
            console.print("[red]请输入有效的数字[/red]")
    
    return selected_exam, num_versions


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="医学试卷翻新多智能体系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --input exam.docx                    # 使用默认设置翻新试卷
  %(prog)s --input exam.docx --versions 5       # 生成 5 套翻新卷
  %(prog)s --input exam.docx --output ./results # 指定输出目录
  %(prog)s --interactive                        # 交互式模式
  %(prog)s --rebuild-rag                        # 重建知识库索引
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入试卷路径 (.docx)"
    )
    
    parser.add_argument(
        "--versions", "-v",
        type=int,
        default=None,
        help=f"生成的翻新卷数量 (默认: {settings.generation.num_versions})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="启用交互式模式"
    )
    
    parser.add_argument(
        "--rebuild-rag",
        action="store_true",
        help="强制重建 RAG 知识库索引"
    )
    
    parser.add_argument(
        "--sync",
        action="store_true",
        help="使用同步模式（而非异步）"
    )

    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="禁用审核功能，生成的题目将不经过审核直接通过"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="显示当前配置并退出"
    )
    
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="跳过确认，直接开始执行"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level)
    
    # 打印标题
    console.print(Panel.fit(
        "[bold blue]医学试卷翻新多智能体系统[/bold blue]\n"
        "[dim]Medical Exam Refurbishment System v1.0[/dim]",
        border_style="blue"
    ))
    
    # 显示配置
    if args.show_config:
        settings.print_config()
        sys.exit(0)
    
    # 环境检查
    if not check_environment():
        console.print("\n[yellow]提示：请检查 .env 文件中的配置[/yellow]")
        sys.exit(1)
    
    # 确定输入试卷
    if args.interactive or not args.input:
        input_path, num_versions = interactive_mode()
    else:
        input_path = Path(args.input)
        num_versions = args.versions or settings.generation.num_versions
        
        # 如果是相对路径，在 data 目录下查找
        if not input_path.is_absolute() and not input_path.exists():
            input_path = settings.paths.input_dir / input_path
    
    # 检查文件是否存在
    if not input_path.exists():
        console.print(f"[red]错误：找不到试卷文件 {input_path}[/red]")
        sys.exit(1)
    
    # 确定输出目录
    output_dir = Path(args.output) if args.output else settings.paths.output_dir
    
    # 打印配置摘要
    console.print("\n[bold]运行配置：[/bold]")
    console.print(f"  输入试卷: {input_path}")
    console.print(f"  输出目录: {output_dir}")
    console.print(f"  生成版本数: {num_versions}")
    console.print(f"  使用模式: {'同步' if args.sync else '异步'}")
    console.print(f"  审核功能: {'禁用' if args.no_audit else '启用'}")
    console.print(f"  重建RAG: {'是' if args.rebuild_rag else '否'}")
    
    # 确认执行
    if not args.yes:
        try:
            confirm = console.input("\n[cyan]确认开始? (Y/n): [/cyan]") or "Y"
            if confirm.lower() not in ["y", "yes", ""]:
                console.print("[yellow]已取消[/yellow]")
                sys.exit(0)
        except EOFError:
            pass
    
    # 创建工作流并执行
    console.print("\n" + "=" * 50)
    
    try:
        refurbisher = ExamRefurbisher(
            num_versions=num_versions,
            use_async=not args.sync,
            enable_audit=not args.no_audit
        )

        result = refurbisher.run(
            input_exam_path=input_path,
            output_dir=output_dir,
            force_rebuild_rag=args.rebuild_rag
        )
        
        if result.success:
            console.print("\n[bold green]✓ 试卷翻新完成！[/bold green]")
            console.print(f"[green]输出目录: {output_dir}[/green]")
            
            # 列出生成的文件
            console.print("\n[bold]生成的文件：[/bold]")
            for f in sorted(output_dir.glob("*.md")):
                console.print(f"  - {f.name}")
        else:
            console.print("\n[bold red]✗ 试卷翻新失败[/bold red]")
            if result.errors:
                console.print("[red]错误信息：[/red]")
                for err in result.errors[:5]:
                    console.print(f"  - {err}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]发生错误: {e}[/red]")
        logging.exception("详细错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
