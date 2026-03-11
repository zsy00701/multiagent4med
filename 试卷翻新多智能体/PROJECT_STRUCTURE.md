# 项目结构说明

## 目录结构

```
试卷翻新多智能体/
│
├── README.md                           # 项目说明（快速开始）
├── ARCHITECTURE.md                     # 架构文档（系统设计详解）
├── OPTIMIZATION_SUMMARY.md             # 优化总结（配置参数）
├── PROJECT_CLEANUP.md                  # 项目整理记录
├── DOCX_FORMATTER_INTEGRATION.md       # DOCX 格式转换器集成说明
├── PROJECT_STRUCTURE.md                # 本文件（项目结构说明）
│
├── requirements.txt                    # Python 依赖列表
├── .env.example                        # 环境变量配置示例
├── .gitignore                          # Git 忽略文件配置
│
├── main.py                             # 命令行入口（CLI 模式）
├── gradio_app.py                       # Web UI 入口（图形界面）
│
├── src/                                # 源代码目录
│   ├── __init__.py                     # 包初始化文件
│   │
│   ├── config.py                       # 配置管理（LLM、RAG、路径等）
│   ├── schemas.py                      # 数据模型（Pydantic）
│   ├── llm_client.py                   # LLM 客户端封装
│   ├── rag_engine.py                   # RAG 检索引擎
│   ├── file_loader.py                  # 文件加载器（DOCX/PDF/TXT）
│   ├── docx_formatter.py               # DOCX 格式转换器
│   ├── workflow.py                     # 工作流编排（核心控制器）
│   │
│   ├── agents/                         # 智能体模块
│   │   ├── __init__.py
│   │   ├── analyst.py                  # 分析智能体（解析试卷）
│   │   ├── generator.py                # 生成智能体（生成新题）
│   │   └── auditor.py                  # 审核智能体（质量审核）
│   │
│   └── ui/                             # UI 组件（Gradio）
│       ├── __init__.py
│       ├── components.py               # UI 组件定义
│       ├── handlers.py                 # 事件处理器
│       ├── utils.py                    # UI 工具函数
│       └── wrapper.py                  # UI 包装器
│
├── data/                               # 数据目录
│   ├── knowledge_base/                 # 知识库文档（PDF/DOCX/TXT）
│   ├── output/                         # 输出结果（生成的试卷）
│   └── chroma_db/                      # 向量数据库（ChromaDB）
│
└── docs/                               # 详细文档
    ├── CHANGELOG.md                    # 更新日志
    ├── SECURITY_NOTICE.md              # 安全说明
    ├── USAGE_GUIDE.md                  # 使用指南
    ├── WEB_UI_GUIDE.md                 # Web UI 指南
    ├── 本次更新说明.md
    ├── 知识点匹配策略说明.md
    └── README_OLD.md                   # 旧版 README 备份
```

## 核心文件说明

### 入口文件

#### main.py
- **功能**：命令行模式入口
- **用途**：通过终端运行系统
- **启动**：`python main.py`

#### gradio_app.py
- **功能**：Web UI 模式入口
- **用途**：通过浏览器图形界面操作
- **启动**：`python gradio_app.py`
- **访问**：http://localhost:7860

### 配置文件

#### .env
- **功能**：环境变量配置（敏感信息）
- **内容**：
  - `LLM_API_KEY` - LLM API 密钥
  - `LLM_PROVIDER` - LLM 提供商（openai/gemini）
  - `LLM_MODEL` - 模型名称
  - `LLM_BASE_URL` - API 基础 URL
  - 其他配置参数
- **注意**：不要提交到 Git

#### requirements.txt
- **功能**：Python 依赖列表
- **安装**：`pip install -r requirements.txt`
- **主要依赖**：
  - pydantic - 数据验证
  - openai - LLM 客户端
  - chromadb - 向量数据库
  - python-docx - DOCX 处理
  - gradio - Web UI
  - rich - 终端美化

### 源代码模块

#### src/config.py
- **功能**：配置管理
- **类**：
  - `LLMConfig` - LLM 配置
  - `EmbeddingConfig` - 嵌入模型配置
  - `ChromaConfig` - ChromaDB 配置
  - `RAGConfig` - RAG 检索配置
  - `GenerationConfig` - 生成配置
  - `PathConfig` - 路径配置
  - `Settings` - 全局配置管理器（单例）

#### src/schemas.py
- **功能**：数据模型定义
- **模型**：
  - `Question` - 原始题目
  - `GeneratedQuestion` - 生成的题目
  - `ExamPaper` - 原始试卷
  - `GeneratedExamPaper` - 生成的试卷
  - `RAGContext` - RAG 上下文
  - `AuditResult` - 审核结果
  - `QualityScores` - 质量评分
  - `QuestionType` - 题型枚举
  - `RenovationMethod` - 翻新策略枚举

#### src/llm_client.py
- **功能**：LLM 客户端封装
- **类**：`LLMClient`
- **方法**：
  - `chat()` / `achat()` - 同步/异步调用
  - `chat_with_schema()` - 结构化输出
  - `extract_json_from_response()` - JSON 提取
- **支持**：OpenAI、Gemini、Kimi 等

#### src/rag_engine.py
- **功能**：RAG 检索引擎
- **类**：`RAGEngine`
- **方法**：
  - `ingest_knowledge_base()` - 导入知识库
  - `get_context_for_question()` - 检索相关知识
  - `_hybrid_search()` - 混合检索
- **技术**：ChromaDB + Sentence-Transformers

#### src/file_loader.py
- **功能**：文件加载器
- **类**：`FileLoader`
- **支持格式**：
  - DOCX - Word 文档
  - PDF - PDF 文档
  - TXT - 纯文本

#### src/docx_formatter.py
- **功能**：DOCX 格式转换器
- **类**：`DocxFormatter`
- **方法**：
  - `convert_to_docx()` - 转换为 DOCX
  - `_add_title()` - 添加标题
  - `_add_question()` - 添加题目
  - `_add_answer_section()` - 添加答案（可选）
  - `_add_explanation_section()` - 添加解析（可选）
- **特点**：
  - 保持原始试卷格式
  - 支持中文字体（宋体）
  - 自动格式化

#### src/workflow.py
- **功能**：工作流编排（核心控制器）
- **类**：`ExamRefurbisher`
- **方法**：
  - `run()` / `run_async()` - 主入口
  - `initialize_rag()` - 初始化 RAG
  - `_generate_version_async()` - 并发生成版本
  - `_process_question_async()` - 处理单个题目
  - `_retry_loop()` - 重试逻辑
  - `_save_exam()` - 保存试卷（JSON/MD/DOCX）
- **特点**：
  - 异步并发处理（MAX_CONCURRENCY = 10）
  - 自动重试机制（最多 3 次）
  - 进度条显示
  - 统计信息汇总

### 智能体模块

#### src/agents/analyst.py
- **功能**：分析智能体
- **职责**：
  - 解析原始试卷
  - 提取题目信息
  - 识别题型和知识点
  - 处理 A3/A4 共用题干
- **输入**：试卷文本
- **输出**：`ExamPaper` 对象

#### src/agents/generator.py
- **功能**：生成智能体
- **职责**：
  - 生成新题目
  - RAG 知识检索
  - 情景多样化
  - 保持题型不变
- **输入**：原题 + RAG 上下文
- **输出**：`GeneratedQuestion` 对象
- **策略**：
  - 改变数值
  - 改变情景
  - 改变表述
  - 增加内容

#### src/agents/auditor.py
- **功能**：审核智能体
- **职责**：
  - 质量审核
  - 相似度检测（< 75%）
  - 知识点一致性检查
  - 五维质量评分
- **输入**：原题 + 生成题
- **输出**：`AuditResult` 对象
- **标准**：
  - 知识点一致（一票否决）
  - 相似度 < 75%
  - 医学准确性
  - 表达清晰度

### UI 模块

#### src/ui/components.py
- **功能**：UI 组件定义
- **内容**：Gradio 界面组件

#### src/ui/handlers.py
- **功能**：事件处理器
- **内容**：按钮点击、文件上传等事件处理

#### src/ui/utils.py
- **功能**：UI 工具函数
- **内容**：格式化、验证等辅助函数

#### src/ui/wrapper.py
- **功能**：UI 包装器
- **内容**：将核心功能包装为 UI 可调用接口

## 数据目录

### data/knowledge_base/
- **用途**：存放知识库文档
- **格式**：PDF、DOCX、TXT
- **内容**：医学教材、指南、文献等
- **处理**：自动导入到 ChromaDB 向量数据库

### data/output/
- **用途**：存放生成的试卷
- **格式**：
  - `.md` - Markdown 格式（含完整信息：题目、答案、解析、评分）
  - `.docx` - Word 文档（仅题目和选项，不含答案）
- **命名**：`试卷名称_翻新版N.{md|docx}`

### data/chroma_db/
- **用途**：ChromaDB 向量数据库
- **内容**：知识库文档的向量表示
- **自动生成**：首次运行时自动创建

## 文档目录

### 根目录文档

#### README.md
- **内容**：项目说明、快速开始
- **受众**：新用户

#### ARCHITECTURE.md
- **内容**：系统架构详解
- **受众**：开发者、维护者

#### OPTIMIZATION_SUMMARY.md
- **内容**：优化历史、配置参数
- **受众**：运维人员

### docs/ 目录

#### USAGE_GUIDE.md
- **内容**：详细使用指南
- **受众**：普通用户

#### WEB_UI_GUIDE.md
- **内容**：Web UI 使用说明
- **受众**：图形界面用户

#### CHANGELOG.md
- **内容**：版本更新记录
- **受众**：所有用户

## 工作流程

### 1. 初始化
```
用户启动 → 加载配置 → 初始化 RAG → 导入知识库
```

### 2. 解析原卷
```
上传试卷 → FileLoader 加载 → Analyst 解析 → ExamPaper 对象
```

### 3. 生成新题
```
遍历题目 → RAG 检索 → Generator 生成 → Auditor 审核 → 重试（如需）
```

### 4. 保存结果
```
GeneratedExamPaper → 保存 JSON → 保存 Markdown → 保存 DOCX
```

## 代码统计

```
总代码行数：4333 行

核心模块：
- workflow.py:       ~515 行  (工作流编排)
- agents/:           ~1200 行 (智能体)
  - analyst.py:      ~400 行
  - generator.py:    ~400 行
  - auditor.py:      ~400 行
- llm_client.py:     ~380 行  (LLM 客户端)
- rag_engine.py:     ~300 行  (RAG 引擎)
- schemas.py:        ~250 行  (数据模型)
- docx_formatter.py: ~203 行  (格式转换)
- ui/:               ~800 行  (UI 组件)
- 其他:              ~685 行  (配置、加载器等)
```

## 技术栈

### 核心技术
- **Python 3.10+** - 编程语言
- **Pydantic** - 数据验证
- **asyncio** - 异步编程

### LLM 集成
- **OpenAI API** - GPT-4o, GPT-4o-mini
- **Google Gemini** - gemini-1.5-pro
- **Kimi API** - kimi-k2.5（推荐）

### RAG 技术
- **ChromaDB** - 向量数据库
- **Sentence-Transformers** - 文本嵌入
- **混合检索** - 知识点 + 内容相似度

### 文件处理
- **python-docx** - DOCX 读写
- **pypdf** - PDF 解析
- **标准库** - TXT 处理

### UI 框架
- **Gradio** - Web UI
- **Rich** - 终端美化

### 工具库
- **tenacity** - 重试机制
- **tiktoken** - Token 计数
- **python-dotenv** - 环境变量

## 配置参数

### LLM 配置
```python
LLM_PROVIDER=openai          # LLM 提供商
LLM_API_KEY=your-key         # API 密钥
LLM_BASE_URL=https://...     # API 地址
LLM_MODEL=gpt-4o             # 模型名称
LLM_TEMPERATURE=0.7          # 温度参数
LLM_TIMEOUT=300              # 超时时间（秒）
```

### 生成配置
```python
NUM_EXAM_VERSIONS=3          # 生成试卷套数
MAX_RETRY_ATTEMPTS=3         # 最大重试次数
MAX_CONCURRENCY=10           # 并发度
```

### RAG 配置
```python
RAG_CHUNK_SIZE=500           # 文档块大小
RAG_CHUNK_OVERLAP=50         # 块重叠大小
RAG_TOP_K=5                  # 检索 Top-K
RAG_KNOWLEDGE_WEIGHT=0.7     # 知识点权重
```

### 审核配置
```python
PLAGIARISM_THRESHOLD=75      # 相似度阈值（%）
```

## 性能指标

- **处理速度**：50 道题约 5-10 分钟
- **并发度**：10（可调整）
- **通过率**：> 95%（经过重试）
- **相似度**：< 75%

## 使用场景

### 1. 命令行模式
```bash
python main.py
```
- 适合：批量处理、自动化脚本
- 特点：快速、高效

### 2. Web UI 模式
```bash
python gradio_app.py
```
- 适合：交互式操作、可视化
- 特点：友好、直观

## 输出说明

### Markdown 格式
- **用途**：人类阅读、文档归档
- **内容**：完整信息（题目、答案、解析、评分、翻新策略等）

### DOCX 格式
- **用途**：考试使用、打印分发
- **内容**：仅题目和选项（不含答案和解析）
- **格式**：简洁清晰，便于考试

## 维护建议

### 1. 定期更新知识库
- 添加最新医学文献
- 更新临床指南
- 补充教材内容

### 2. 监控生成质量
- 检查通过率
- 分析失败原因
- 调整审核标准

### 3. 优化性能
- 调整并发度
- 优化 Prompt
- 升级模型版本

### 4. 备份数据
- 备份 .env 配置
- 备份知识库文档
- 备份生成结果

## 常见问题

### 1. JSON 解析失败
- **原因**：max_tokens 不足
- **解决**：已优化为 12288

### 2. 生成速度慢
- **原因**：并发度低
- **解决**：已提升到 10

### 3. 通过率低
- **原因**：审核标准过严
- **解决**：已放宽到 75%

### 4. 题型发生变化
- **原因**：已修复
- **解决**：保持原题型不变

## 版本信息

- **当前版本**：2.0.0
- **最后更新**：2026-03-06
- **维护团队**：Medical Exam AI Team

---

**文档创建时间**：2026-03-06
**创建者**：Claude Sonnet 4.5
