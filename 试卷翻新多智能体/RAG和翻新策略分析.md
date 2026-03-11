# RAG 和翻新策略分析报告

## 1. RAG（检索增强生成）使用情况

### RAG 是否有用？✅ **有用**

#### 工作原理
1. **知识库索引**
   - 位置：`data/knowledge_base/`
   - 文件数：6 个医学题库文档
   - 总大小：约 204KB
   - 向量数据库：ChromaDB（4.1MB）
   - Embedding 模型：`paraphrase-multilingual-MiniLM-L12-v2`

2. **检索策略**（`src/rag_engine.py:322-363`）
   ```python
   # 根据知识点权重（默认 0.7）调整检索策略
   if knowledge_weight >= 0.9:
       query = knowledge_point  # 主要基于知识点
   elif knowledge_weight >= 0.5:
       query = f"{knowledge_point}\n\n相关内容：{question_content[:150]}"  # 知识点优先
   else:
       query = f"{knowledge_point}\n{question_content[:200]}"  # 并重
   ```

3. **使用位置**（`src/workflow.py:273-277`）
   ```python
   rag_context = self.rag_engine.get_context_for_question(
       question.content,
       question.knowledge_point,
       knowledge_weight=settings.rag.knowledge_weight,  # 0.7
   )
   ```

#### RAG 的作用

✅ **提供参考资料**
- 为每道题检索相关的医学知识
- 返回 Top-K（默认 5）个最相关的文档片段
- 包含来源、页码、相似度等元数据

✅ **增强生成质量**
- 生成器在 prompt 中使用 RAG 上下文（`src/agents/generator.py:390-391`）
- 要求在解析中标注引用来源
- 确保生成内容有医学知识支撑

✅ **防止知识偏差**
- 避免 LLM 编造不存在的医学知识
- 提供权威的参考依据

#### 当前配置
```bash
RAG_CHUNK_SIZE=500          # 每个文档块 500 字符
RAG_CHUNK_OVERLAP=50        # 块之间重叠 50 字符
RAG_TOP_K=5                 # 返回前 5 个最相关结果
RAG_KNOWLEDGE_WEIGHT=0.7    # 70% 权重给知识点，30% 给题目内容
```

### 优化建议

#### 如果 RAG 效果不好
1. **增加知识库内容**
   - 添加更多医学教材、指南
   - 添加护理学专业书籍

2. **调整检索参数**
   ```bash
   RAG_TOP_K=10                # 增加到 10 个结果
   RAG_KNOWLEDGE_WEIGHT=0.8    # 提高知识点权重
   ```

3. **更换 Embedding 模型**
   ```bash
   # 使用更强的中文模型
   EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
   ```

#### 如果不需要 RAG
可以禁用，但**不推荐**：
- 生成质量会下降
- 可能出现医学知识错误
- 缺少引用来源

---

## 2. 翻新策略分析

### 策略列表（`src/agents/generator.py:467-493`）

#### 基础策略（所有题型可用）
1. **策略B（干扰项重构）**
   - 重新设计错误选项
   - 改变干扰项的迷惑性

2. **策略C（同义替换）**
   - 用同义词替换关键词
   - 改变表达方式但保持意思

3. **策略D（问法转换）**
   - 改变提问角度
   - 例如："哪项正确" → "哪项错误"

4. **策略E（细节丰满）**
   - 增加背景信息
   - 丰富题干细节

5. **策略F（全面翻新）** ⭐ **新增**
   - ⚠️ **题型绝对不能改变**（单选保持单选，多选保持多选，判断保持判断）
   - ✅ 知识点必须完全相同
   - ✅ 题干、选项、场景可以全面改写
   - ✅ 可以改变具体场景、案例背景、数据参数
   - ✅ 可以重新设计所有选项内容（但保持选项数量）
   - ✅ 可以改变提问角度和表达方式
   - 🎯 目标：生成考察相同知识点、相同题型但表述完全不同的新题

#### 病例题专用策略
6. **策略A（病例参数微调）**
   - 仅用于 A2/A3/A4/病例题
   - 修改患者年龄、性别等非关键信息
   - 保留核心症状和体征

### 随机组合机制

#### 代码实现
```python
# 每道题随机选择 1-3 个策略
combo_size = random.randint(1, min(3, len(strategy_pool)))
selected = random.sample(strategy_pool, combo_size)

# 按优先级排序
preferred_order = {
    "策略A（病例参数微调）": 0,
    "策略B（干扰项重构）": 1,
    "策略C（同义替换）": 2,
    "策略D（问法转换）": 3,
    "策略E（细节丰满）": 4,
}
```

#### 组合示例
- 单选题可能组合：`策略C + 策略D`
- 病例题可能组合：`策略A + 策略B + 策略E`
- 判断题可能组合：`策略C`

### 策略应用流程

1. **选择阶段**（`src/agents/generator.py:363`）
   ```python
   strategy_combo = self._choose_strategy_combo(original_question)
   ```

2. **提示阶段**（`src/agents/generator.py:381-382`）
   ```python
   strategy_hint = self._suggest_strategy(original_question, strategy_combo)
   prompt_parts.append(strategy_hint)
   ```

3. **生成阶段**（`src/agents/generator.py:422`）
   ```python
   "6. **策略随机化**：必须优先采用以下组合策略生成，
    并在 renovation_strategy_detail 中原样写出：{策略组合}"
   ```

4. **记录阶段**
   - LLM 在 `renovation_strategy_detail` 字段中记录使用的策略
   - 用于后续分析和统计

### 策略效果

#### 优点
✅ **多样性**：每道题使用不同的策略组合，避免模式化
✅ **针对性**：根据题型自动选择合适的策略池
✅ **可追溯**：记录使用的策略，便于质量分析
✅ **灵活性**：1-3 个策略的随机组合，适应不同难度

#### 当前问题
⚠️ **随机性过强**：完全随机可能导致某些题目策略不够优化
⚠️ **缺少反馈**：没有根据审核结果调整策略选择

### 优化建议

#### 1. 增加策略权重
```python
# 根据题型和历史成功率调整策略选择概率
strategy_weights = {
    "策略C（同义替换）": 0.4,  # 最常用
    "策略B（干扰项重构）": 0.3,
    "策略D（问法转换）": 0.2,
    "策略E（细节丰满）": 0.1,
}
```

#### 2. 添加策略组合模板
```python
# 预定义高质量组合
PROVEN_COMBOS = {
    QuestionType.SINGLE_CHOICE: [
        ["策略C（同义替换）", "策略B（干扰项重构）"],
        ["策略D（问法转换）", "策略E（细节丰满）"],
    ],
    QuestionType.CLINICAL_CASE: [
        ["策略A（病例参数微调）", "策略C（同义替换）", "策略E（细节丰满）"],
    ],
}
```

#### 3. 根据审核反馈调整
```python
# 如果审核失败，尝试不同的策略组合
if audit_feedback and not audit_feedback.passed:
    # 避免使用上次失败的策略组合
    # 选择新的组合
```

---

## 3. 总结

### RAG 使用情况
- ✅ **已启用且有效**
- ✅ 知识库包含 6 个医学题库文档
- ✅ 使用知识点权重 0.7 的混合检索策略
- ✅ 为每道题提供 Top-5 相关文档片段
- 💡 建议：增加更多医学教材和指南

### 翻新策略
- ✅ **5 种策略**：A（病例微调）、B（干扰项重构）、C（同义替换）、D（问法转换）、E（细节丰满）
- ✅ **随机组合**：每道题随机选择 1-3 个策略
- ✅ **题型适配**：病例题可用策略 A，其他题型不可用
- ✅ **可追溯**：记录使用的策略组合
- 💡 建议：增加策略权重和预定义组合模板

### 配置建议

#### 当前配置（已优化）
```bash
# RAG 配置
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=5
RAG_KNOWLEDGE_WEIGHT=0.7

# 生成配置
MAX_RETRY_ATTEMPTS=5
MAX_CONCURRENCY=3
```

#### 如果想提高质量
```bash
# 增加 RAG 检索结果
RAG_TOP_K=10
RAG_KNOWLEDGE_WEIGHT=0.8

# 增加重试次数
MAX_RETRY_ATTEMPTS=7
```

#### 如果想提高速度
```bash
# 减少 RAG 检索结果
RAG_TOP_K=3

# 增加并发度（注意 API 限流）
MAX_CONCURRENCY=5
```

---

**结论**：RAG 和翻新策略都在正常工作，且设计合理。建议保持当前配置，如需优化可参考上述建议。
