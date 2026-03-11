#!/usr/bin/env python3
"""
医学试卷翻新系统 - Gradio Web UI

使用方法:
    python gradio_app.py

访问地址:
    http://127.0.0.1:7860
"""

import sys
from pathlib import Path

import gradio as gr

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.components import create_param_panel, create_result_panel
from src.ui.handlers import process_exam, get_kb_status


def create_app():
    """创建 Gradio 应用"""

    with gr.Blocks(title="医学试卷翻新系统") as demo:
        # 标题
        gr.Markdown(
            """
            # 🏥 医学试卷翻新多智能体系统

            基于 RAG 的医学试卷自动翻新工具，支持上传试卷、配置参数并生成多套翻新版本。
            """
        )

        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传试卷")

                file_input = gr.File(
                    label="选择试卷文件",
                    file_types=[".docx", ".doc", ".pdf", ".txt"],
                    file_count="single"
                )

                file_info = gr.Textbox(
                    label="文件信息",
                    interactive=False,
                    visible=False
                )

                # 参数配置面板
                params = create_param_panel()

                # 开始生成按钮
                submit_btn = gr.Button(
                    "🚀 开始生成",
                    variant="primary",
                    size="lg"
                )

                # 示例文件
                gr.Markdown(
                    """
                    ---
                    ### 💡 使用提示

                    1. 上传试卷文件（支持 Word、PDF、TXT）
                    2. 调整参数配置（可使用默认值）
                    3. 点击"开始生成"按钮
                    4. 在右侧查看生成结果

                    **首次使用**: 建议勾选"强制重建 RAG 索引"
                    """
                )

            # 右侧结果展示区
            with gr.Column(scale=2):
                results = create_result_panel()

        # 绑定事件
        submit_btn.click(
            fn=process_exam,
            inputs=[file_input, *params],
            outputs=results[:-1],  # 除了刷新按钮
            queue=True
        )

        # 刷新知识库按钮
        refresh_kb_btn = results[-1]
        refresh_kb_btn.click(
            fn=get_kb_status,
            inputs=[],
            outputs=[results[5], results[6]]  # kb_status, kb_files
        )

        # 文件上传时显示信息
        def show_file_info(file):
            if file is None:
                return gr.update(visible=False)
            file_path = Path(file.name)
            info = f"📄 {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)"
            return gr.update(value=info, visible=True)

        file_input.change(
            fn=show_file_info,
            inputs=[file_input],
            outputs=[file_info]
        )

        # 页面加载时获取知识库状态
        demo.load(
            fn=get_kb_status,
            inputs=[],
            outputs=[results[5], results[6]]
        )

    return demo


def main():
    """主函数"""
    demo = create_app()

    # 启动服务器
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
