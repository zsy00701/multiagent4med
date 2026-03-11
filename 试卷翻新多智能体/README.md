# 医学试卷翻新多智能体系统

基于大语言模型的医学/护理试卷自动翻新系统，采用"分析—生成—审核"三智能体协作架构，结合 RAG 知识检索和 QGEval 2.0 七维质量评估框架，实现试卷的高质量自动翻新。

## 核心特性

- **三智能体协作**：Analyst（结构分析）→ Generator（题目生成）→ Auditor（质量审核），流水线式处理
- **RAG 知识增强**：基于 ChromaDB 向量检索，从医学知识库中提取相关内容辅助出题，避免幻觉
- **QGEval 2.0 评估**：七维质量打分体系（流畅性、清晰度、简洁性、相关性、一致性、可回答性、答案一致性）
- **严格题型保持**：判断题不会变成选择题，简答题不会变成选择题，题目数量与原卷完全一致
- **医学事实守护**：多层校验机制防止医学事实性错误，一票否决制
- **多版本生成**：一份原卷可生成多个不同翻新版本，支持异步并发加速
- **多格式输出**：同时输出 Markdown 和 Word (.docx) 格式

## 项目结构

```
试卷翻新多智能体/
├── main.py                # 命令行入口
├── gradio_app.py          # Web UI 入口（Gradio）
├── run_audit.py           # 单卷独立审核
├── run_batch_audit.py     # 批量审核 + 汇总报告
├── requirements.txt       # Python 依赖
├── .env.example           # 环境变量模板
│
├── data/
│   ├── *.docx             # 原始试卷
│   ├── knowledge_base/    # RAG 知识库源文件
│   ├── chroma_db/         # 向量索引（自动生成）
│   └── output/            # 翻新试卷 + 审核报告
│
└── src/
    ├── config.py           # 配置管理
    ├── schemas.py          # 数据模型（Question、QualityScores 等）
    ├── llm_client.py       # LLM 客户端（支持 OpenAI 兼容 API）
    ├── file_loader.py      # 文件加载（DOCX/PDF/TXT）
    ├── rag_engine.py       # RAG 向量检索引擎
    ├── workflow.py          # 翻新主工作流
    ├── audit_workflow.py    # 独立审核工作流
    ├── md_parser.py         # Markdown 翻新卷解析
    ├── docx_formatter.py    # Word 格式化输出
    │
    ├── agents/
    │   ├── analyst.py       # 分析智能体：试卷结构解析
    │   ├── generator.py     # 生成智能体：翻新题目生成
    │   ├── auditor.py       # 审核智能体：QGEval 质量审核
    │   └── audit_rules.py   # 规则引擎：相似度/题型/抄袭检测
    │
    └── ui/                  # Gradio Web UI 组件
        ├── components.py
        ├── handlers.py
        ├── utils.py
        └── wrapper.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 LLM API 密钥：

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

支持的 LLM 服务：DeepSeek、Kimi (Moonshot)、智谱 GLM-4、通义千问、OpenAI 等任何 OpenAI 兼容 API。

### 3. 准备知识库（可选）

将医学教材、护理手册等 PDF/DOCX/TXT 文件放入 `data/knowledge_base/` 目录，系统首次运行时会自动构建向量索引。

### 4. 运行

**命令行模式：**

```bash
# 翻新单份试卷，生成 3 个版本
python main.py --input data/原卷.docx --versions 3

# 交互式选择试卷
python main.py --interactive

# 强制重建知识库索引
python main.py --input data/原卷.docx --rebuild-rag
```

**Web UI 模式：**

```bash
python gradio_app.py
# 浏览器访问 http://127.0.0.1:7860
```

## 质量评估（QGEval 2.0）

系统内置独立的质量评估工具，基于 QGEval 2.0 框架对翻新试卷进行七维打分。

### 单卷审核

```bash
python run_audit.py --original data/原卷.docx --generated data/output/翻新版1.md
```

### 批量审核

```bash
python run_batch_audit.py
```

自动匹配所有原卷与翻新卷，输出汇总报告，包含：

| 指标 | 说明 |
|------|------|
| QGEval 七维评分 | 流畅性、清晰度、简洁性、相关性、一致性、可回答性、答案一致性（1-5 分） |
| 通过率 | 通过审核的题目占比 |
| 相似度 | 翻新题与原题的语义相似度（目标 < 0.65） |
| 题干原创率 | 题干非照搬原题的比例 |
| 选项原创率 | 选项非照搬原题的比例 |
| 考点匹配率 | 翻新题考点与原题一致的比例 |

## 工作流程

```
原始试卷 (.docx)
    │
    ▼
┌─────────────┐
│  FileLoader  │  加载文件，提取文本
└──────┬──────┘
       ▼
┌─────────────┐
│   Analyst    │  解析试卷结构 → 题型、考点、选项、答案
└──────┬──────┘
       ▼
┌─────────────┐
│  RAG Engine  │  检索知识库，获取相关医学知识片段
└──────┬──────┘
       ▼
┌─────────────┐
│  Generator   │  逐题翻新（临床语境嵌入 / RAG 融合 / 同义替换）
└──────┬──────┘
       ▼
┌─────────────┐
│   Auditor    │  QGEval 七维评分 + 医学事实校验 + 题型一致性检查
└──────┬──────┘
       ▼
  翻新试卷 (.md + .docx) + 审核报告
```

## 翻新策略

系统综合运用多种策略确保翻新后的题目与原题有显著差异：

- **RAG 知识融合**：从知识库中检索相关内容，用新的医学知识点替换或扩展原题
- **临床语境嵌入**：为题目添加护理场景（如"责任护士在执行 XX 操作时..."），增加临床真实感
- **同义替换与重构**：保持考点不变，用不同的表述方式重新组织题干和选项
- **维度转换**：从不同角度考察同一知识点（如从"是什么"转为"为什么"或"怎么做"）

## 配置说明

所有配置通过 `.env` 文件管理，主要配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | openai | LLM 服务提供商 |
| `LLM_API_KEY` | — | API 密钥 |
| `LLM_BASE_URL` | — | API 地址 |
| `LLM_MODEL` | deepseek-chat | 默认模型 |
| `ANALYST_LLM_MODEL` | — | 分析智能体专用模型（可选） |
| `GENERATOR_LLM_MODEL` | — | 生成智能体专用模型（可选） |
| `AUDITOR_LLM_MODEL` | — | 审核智能体专用模型（可选） |
| `LLM_TEMPERATURE` | 0.7 | 生成温度 |
| `NUM_EXAM_VERSIONS` | 3 | 默认翻新版本数 |
| `MAX_RETRY_ATTEMPTS` | 5 | 单题最大重试次数 |
| `RAG_TOP_K` | 5 | RAG 检索返回数量 |
| `RAG_KNOWLEDGE_WEIGHT` | 0.7 | 知识点匹配权重 |
| `PLAGIARISM_THRESHOLD` | 15 | 连续相同字符数上限 |

## 支持的题型

单选题、多选题、判断题、简答题、填空题、病例分析题、论述题、配对题，以及 A1/A2/A3/A4/B1/X 型题。
