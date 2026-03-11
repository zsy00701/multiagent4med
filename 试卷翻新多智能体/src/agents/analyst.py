"""
医学试卷翻新多智能体系统 - 结构分析智能体

负责解析原始试卷，将非结构化的试卷文本转换为结构化的题目列表。
"""

import logging
import json
import re
import inspect
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field

from ..llm_client import LLMClient, get_llm_client
from ..schemas import Question, QuestionType, ExamPaper

logger = logging.getLogger(__name__)


class ParsedQuestion(BaseModel):
    """LLM 返回的解析题目结构"""
    id: str
    type: str
    content: str
    knowledge_point: str
    options: dict
    answer: str
    explanation: Optional[str] = ""


class ParsedExam(BaseModel):
    """LLM 返回的解析试卷结构"""
    title: str
    questions: List[ParsedQuestion]


ANALYST_SYSTEM_PROMPT = """你是一个专业的医学试卷分析专家。你的任务是解析医学试卷文本，将其转换为结构化的 JSON 格式。

## 一、内容筛选

忽略页眉、页脚、考试说明、注意事项、评分标准等非题目内容。只提取正式题目。

## 二、题型识别（核心能力）

### 题型判定决策树

按以下顺序判断题型：

1. **有"对/错""√/×""正确/错误"选项或明确要求判断真伪** → `判断题`
2. **无选项，要求文字作答（论述、简述、列举）** → `简答题`
3. **有空格/横线要求填写** → `填空题`
4. **有选项且答案含多个字母（如 ABD）或标注"多选"** → `多选题`
5. **题干以病例开头（患者/病人+年龄+症状），含选项** → 按结构细分：
   - 单个病例 + 单个问题 → `A2型题`
   - 单个病例 + 多个子问题 → `A3型题` 或 `A4型题`
   - 标注为"病例分析"的大题 → `病例分析题`
6. **其余有 A/B/C/D/E 选项的单选** → `单选题`（即 A1 型题）

### 易混淆场景处理

| 场景 | 正确判定 | 常见误判 |
|------|---------|---------|
| "下列说法正确/错误的是"，有 A/B/C/D 选项 | 单选题 | ~~判断题~~ |
| "对/错"只有两个选项 | 判断题 | ~~单选题~~ |
| 题干有病例但只有单个问题 | A2型题 | ~~A3型题~~ |
| 一个病例下有 2-3 个子问题 | A3/A4型题 | ~~多道独立A2题~~ |

## 三、考点提取（思维链方法）

提取考点时，请按以下步骤思考（不需要输出思考过程，直接输出结果）：

1. **识别题目考察的核心医学概念**（疾病/症状/操作/药物）
2. **确定考察维度**（病因/发病机制/临床表现/诊断/鉴别诊断/治疗/护理/预后）
3. **组合为"核心概念 + 考察维度"格式**

### 考点质量标准

| 质量等级 | 示例 | 问题 |
|---------|------|------|
| ✅ 优秀 | "嗜铬细胞瘤术前α受体阻滞剂的应用" | 精确到操作层面 |
| ✅ 良好 | "前列腺增生的临床表现与分级" | 具体且有层次 |
| ⚠️ 一般 | "前列腺增生" | 缺少考察维度 |
| ❌ 过宽 | "泌尿系统疾病" | 完全无法定位考点 |

**要求**：考点长度 10-30 字，必须达到"良好"以上水平。

## 四、选项与答案处理

| 题型 | options 格式 | answer 格式 |
|------|-------------|------------|
| 单选题/A1/A2 | {"A":"..","B":"..","C":"..","D":".."} | 单字母如 "B" |
| 多选题/X型 | {"A":"..","B":"..","C":"..","D":".."} | 多字母如 "ABD" |
| 判断题 | {"A":"正确","B":"错误"} | "A" 或 "B" |
| 简答题 | {} | 完整答案文本 |
| 填空题 | {} | 填写内容 |

- 如果选项排列在表格中，也要正确识别并提取
- 如果答案未给出，answer 填写 "未知"

## 五、特殊处理

### A3/A4 型题（共用病例）
每个子问题**单独**作为一道题，但每道题的 content 中**必须包含共用的病例信息**（复制到每道子题中），确保每道题能独立理解。

### 答案解析
如果试卷中包含答案解析，提取到 explanation 字段中；如果没有，填写空字符串 ""。

## 六、输出格式

```json
{
  "title": "试卷标题",
  "questions": [
    {
      "id": "1",
      "type": "单选题",
      "content": "题干内容",
      "knowledge_point": "核心概念+考察维度",
      "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
      "answer": "A",
      "explanation": ""
    }
  ]
}
```

**关键要求**：
- 题目 ID 从 "1" 开始递增
- **零遗漏**：试卷中的每一道题都必须提取，不论题型
- type 字段使用中文（单选题/多选题/判断题/简答题/填空题/病例分析题/A1型题/A2型题/A3型题/A4型题/B1型题/X型题）
"""


_QUESTION_TYPE_MAP = {
    "单选题": QuestionType.SINGLE_CHOICE,
    "单选": QuestionType.SINGLE_CHOICE,
    "多选题": QuestionType.MULTIPLE_CHOICE,
    "多选": QuestionType.MULTIPLE_CHOICE,
    "判断题": QuestionType.TRUE_FALSE,
    "判断": QuestionType.TRUE_FALSE,
    "简答题": QuestionType.SHORT_ANSWER,
    "简答": QuestionType.SHORT_ANSWER,
    "填空题": QuestionType.FILL_IN_BLANK,
    "填空": QuestionType.FILL_IN_BLANK,
    "病例分析题": QuestionType.CASE_ANALYSIS,
    "病例分析": QuestionType.CASE_ANALYSIS,
    "A1型题": QuestionType.A1, "A1": QuestionType.A1,
    "A2型题": QuestionType.A2, "A2": QuestionType.A2,
    "A3型题": QuestionType.A3, "A3": QuestionType.A3,
    "A4型题": QuestionType.A4, "A4": QuestionType.A4,
    "B1型题": QuestionType.B1, "B1": QuestionType.B1,
    "X型题": QuestionType.X, "X": QuestionType.X,
}

_TITLE_PATTERNS = [
    re.compile(r'^(.+(?:试卷|考试|测验|练习).*)$'),
    re.compile(r'^(.+(?:期末|期中|模拟|真题).*)$'),
]

_LONG_EXAM_THRESHOLD = 5000  # 降低阈值，更多试卷使用分批处理（更快更稳定）
_BATCH_SIZE_LIMIT = 8000  # 每批处理的字符数上限
_SECTION_TYPE_PATTERNS: List[Tuple[re.Pattern, QuestionType]] = [
    (re.compile(r'是非题|判断题'), QuestionType.TRUE_FALSE),
    (re.compile(r'单项选择题|单选题|A1型题'), QuestionType.A1),
    (re.compile(r'多项选择题|多选题|X型题'), QuestionType.X),
    (re.compile(r'简答题|论述题'), QuestionType.SHORT_ANSWER),
    (re.compile(r'案例分析题|病例分析题'), QuestionType.CASE_ANALYSIS),
    (re.compile(r'填空题'), QuestionType.FILL_IN_BLANK),
    (re.compile(r'配对题|匹配题'), QuestionType.FILL_IN_BLANK),
]
_SECTION_HEADING_PATTERNS = [pattern for pattern, _ in _SECTION_TYPE_PATTERNS]
_QUESTION_START_PATTERN = re.compile(r'^(\d+)[、.，]\s*(.*)')
_OPTION_SPLIT_PATTERN = re.compile(r'(?=\b[A-H][、.．])')
_OPTION_MATCH_PATTERN = re.compile(r'^([A-H])[、.．]\s*(.*)$')


class AnalystAgent:
    """结构分析智能体"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_llm_client()
    
    # ── 公共入口 ──────────────────────────────────────────────
    
    def analyze(self, exam_text: str, source_file: Optional[str] = None) -> ExamPaper:
        """同步分析试卷文本"""
        return self._do_analyze(exam_text, source_file, use_async=False)
    
    async def analyze_async(self, exam_text: str, source_file: Optional[str] = None) -> ExamPaper:
        """异步分析试卷文本"""
        result = self._do_analyze(exam_text, source_file, use_async=True)
        if inspect.isawaitable(result):
            return await result
        return result
    
    # ── 核心分析逻辑 ─────────────────────────────────────────
    
    def _do_analyze(self, exam_text: str, source_file: Optional[str], use_async: bool):
        """统一的分析入口，根据 use_async 切换调用方式"""
        logger.info("开始分析试卷结构...")
        exam_text = self._preprocess_text(exam_text)

        structured_exam = self._try_parse_structured_exam(exam_text, source_file)
        if structured_exam is not None:
            logger.info(f"采用结构化规则解析，共提取 {len(structured_exam.questions)} 道顶层题目")
            return structured_exam
        
        if len(exam_text) > _LONG_EXAM_THRESHOLD:
            return self._analyze_long_exam(exam_text, source_file, use_async)
        
        prompt = self._build_analyze_prompt(exam_text)
        
        if use_async:
            async def _async_impl():
                return await self._acall_llm_and_parse(prompt, source_file)
            return _async_impl()
        return self._call_llm_and_parse(prompt, source_file)
    
    def _call_llm_and_parse(self, prompt: str, source_file: Optional[str]) -> ExamPaper:
        """同步调用 LLM 并解析结果"""
        try:
            response = self.llm.chat(prompt=prompt, system_prompt=ANALYST_SYSTEM_PROMPT, temperature=0.1, max_tokens=12288, json_mode=True)
            return self._parse_response(response, source_file)
        except Exception as e:
            logger.error(f"试卷分析失败: {e}")
            raise
    
    async def _acall_llm_and_parse(self, prompt: str, source_file: Optional[str]) -> ExamPaper:
        """异步调用 LLM 并解析结果"""
        try:
            response = await self.llm.achat(prompt=prompt, system_prompt=ANALYST_SYSTEM_PROMPT, temperature=0.1, max_tokens=12288, json_mode=True)
            return self._parse_response(response, source_file)
        except Exception as e:
            logger.error(f"异步试卷分析失败: {e}")
            raise
    
    def _parse_response(self, response: str, source_file: Optional[str]) -> ExamPaper:
        """解析 LLM 响应为 ExamPaper"""
        data = self.llm.extract_json_from_response(response)
        parsed = ParsedExam.model_validate(data)
        
        questions = [
            Question(
                id=pq.id,
                type=_QUESTION_TYPE_MAP.get(pq.type, QuestionType.SINGLE_CHOICE),
                content=pq.content,
                knowledge_point=pq.knowledge_point,
                options=pq.options,
                answer=pq.answer,
                explanation=pq.explanation or "",
                source_exam=source_file,
            )
            for pq in parsed.questions
        ]
        
        logger.info(f"试卷分析完成，共提取 {len(questions)} 道题目")
        return ExamPaper(title=parsed.title, questions=questions, source_file=source_file)
    
    # ── 长文档分段处理 ────────────────────────────────────────
    
    def _analyze_long_exam(self, exam_text: str, source_file: Optional[str], use_async: bool):
        """处理长试卷，分段分析（同步/异步统一）"""
        logger.info("试卷过长，采用分段分析...")
        batches = self._split_into_batches(exam_text)
        
        async def _async_impl():
            all_questions: List[Question] = []
            for batch_id, batch_text in enumerate(batches):
                questions = await self._analyze_batch_async(batch_text, batch_id)
                all_questions.extend(questions)
            return self._assemble_exam(exam_text, all_questions, source_file)
        
        if use_async:
            return _async_impl()
        
        all_questions: List[Question] = []
        for batch_id, batch_text in enumerate(batches):
            all_questions.extend(self._analyze_batch(batch_text, batch_id))
        return self._assemble_exam(exam_text, all_questions, source_file)
    
    def _split_into_batches(self, exam_text: str) -> List[str]:
        parts = re.split(r'(?=(?:^|\n)\s*(?:\d+[.、]|[一二三四五六七八九十]+[、.]))', exam_text)
        batches = []
        current_batch = ""
        for part in parts:
            if len(current_batch) + len(part) < _BATCH_SIZE_LIMIT:
                current_batch += part
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = part
        if current_batch:
            batches.append(current_batch)
        return batches
    
    def _assemble_exam(self, exam_text: str, questions: List[Question], source_file: Optional[str]) -> ExamPaper:
        for i, q in enumerate(questions, 1):
            q.id = str(i)
        title = self._extract_title(exam_text) or "医学试卷"
        return ExamPaper(title=title, questions=questions, source_file=source_file)
    
    def _analyze_batch(self, text: str, batch_id: int) -> List[Question]:
        """同步分析一批题目"""
        prompt = self._build_batch_prompt(text, batch_id)
        try:
            response = self.llm.chat(prompt=prompt, system_prompt=ANALYST_SYSTEM_PROMPT, temperature=0.1, max_tokens=12288, json_mode=True)
            return self._parse_batch_response(response)
        except Exception as e:
            logger.error(f"分析批次 {batch_id} 失败: {e}")
            return []
    
    async def _analyze_batch_async(self, text: str, batch_id: int) -> List[Question]:
        """异步分析一批题目"""
        prompt = self._build_batch_prompt(text, batch_id)
        try:
            response = await self.llm.achat(prompt=prompt, system_prompt=ANALYST_SYSTEM_PROMPT, temperature=0.1, max_tokens=12288, json_mode=True)
            return self._parse_batch_response(response)
        except Exception as e:
            logger.error(f"异步分析批次 {batch_id} 失败: {e}")
            return []
    
    def _parse_batch_response(self, response: str) -> List[Question]:
        data = self.llm.extract_json_from_response(response)
        parsed = ParsedExam.model_validate(data)
        return [
            Question(
                id=pq.id,
                type=_QUESTION_TYPE_MAP.get(pq.type, QuestionType.SINGLE_CHOICE),
                content=pq.content,
                knowledge_point=pq.knowledge_point,
                options=pq.options,
                answer=pq.answer,
                explanation=pq.explanation or "",
            )
            for pq in parsed.questions
        ]
    
    # ── 辅助方法 ──────────────────────────────────────────────
    
    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return '\n'.join(line.strip() for line in text.split('\n'))

    def _try_parse_structured_exam(self, exam_text: str, source_file: Optional[str]) -> Optional[ExamPaper]:
        """优先使用确定性规则解析标准护理试卷，保留章节、题数和大小题结构。"""
        lines = [line.strip() for line in exam_text.split('\n') if line.strip()]
        if len(lines) < 5:
            return None

        title = lines[0]
        first_section_index = next((i for i, line in enumerate(lines) if self._is_section_heading(line)), None)
        if first_section_index is None:
            return None

        header_lines = lines[1:first_section_index]
        section_title = None
        blocks: List[Tuple[str, List[str]]] = []
        current_block: List[str] = []

        for line in lines[first_section_index:]:
            if self._is_section_heading(line):
                if current_block:
                    blocks.append((section_title, current_block))
                    current_block = []
                section_title = line
                continue

            if section_title is None:
                continue

            if self._is_question_start(line, section_title, current_block):
                if current_block:
                    blocks.append((section_title, current_block))
                current_block = [line]
            else:
                if not current_block:
                    current_block = [line]
                else:
                    current_block.append(line)

        if current_block:
            blocks.append((section_title, current_block))

        questions: List[Question] = []
        section_counts: Dict[str, int] = {}
        for index, (block_section_title, block_lines) in enumerate(blocks, 1):
            section_counts[block_section_title] = section_counts.get(block_section_title, 0) + 1
            question = self._build_question_from_block(
                index,
                block_section_title,
                block_lines,
                source_file,
                section_counts[block_section_title],
            )
            if question is not None:
                questions.append(question)

        if len(questions) < 5:
            return None

        return ExamPaper(
            title=title,
            questions=questions,
            header_lines=header_lines,
            source_file=source_file,
        )

    @staticmethod
    def _is_section_heading(line: str) -> bool:
        return any(pattern.search(line) for pattern in _SECTION_HEADING_PATTERNS)

    @staticmethod
    def _resolve_section_type(section_title: str) -> QuestionType:
        for pattern, question_type in _SECTION_TYPE_PATTERNS:
            if pattern.search(section_title):
                return question_type
        return QuestionType.SINGLE_CHOICE

    @staticmethod
    def _looks_like_option_line(line: str) -> bool:
        stripped = line.strip()
        if _OPTION_MATCH_PATTERN.match(stripped):
            return True
        parts = _OPTION_SPLIT_PATTERN.split(stripped)
        return any(_OPTION_MATCH_PATTERN.match(p.strip()) for p in parts if p.strip())

    @staticmethod
    def _looks_like_question_stem(line: str) -> bool:
        stripped = line.strip()
        if re.search(r'[（(]\s*[）)]\s*$', stripped):
            return True
        if re.search(r'(的是|哪项|哪些|为何|包括|有哪些)\s*[：:（(]?\s*$', stripped):
            return True
        if re.search(r'[？?]\s*$', stripped):
            return True
        if re.match(r'^(下列|以下|关于|对于)', stripped):
            return True
        return False

    def _is_question_start(self, line: str, section_title: str, current_block: List[str] = None) -> bool:
        section_type = self._resolve_section_type(section_title)
        if section_type == QuestionType.TRUE_FALSE:
            return True
        if section_type == QuestionType.FILL_IN_BLANK:
            return True
        if _QUESTION_START_PATTERN.match(line):
            return True
        if section_type in {QuestionType.A1, QuestionType.SINGLE_CHOICE,
                            QuestionType.X, QuestionType.MULTIPLE_CHOICE}:
            if current_block and not self._looks_like_option_line(line):
                has_options = any(self._looks_like_option_line(bl) for bl in current_block)
                if has_options and self._looks_like_question_stem(line):
                    return True
        return False

    def _build_question_from_block(
        self,
        index: int,
        section_title: str,
        block_lines: List[str],
        source_file: Optional[str],
        section_question_index: int,
    ) -> Optional[Question]:
        if not block_lines:
            return None

        question_type = self._resolve_section_type(section_title)
        number_match = _QUESTION_START_PATTERN.match(block_lines[0])
        display_number = number_match.group(1) if number_match else str(section_question_index)
        first_line_content = number_match.group(2).strip() if number_match else block_lines[0].strip()
        remaining_lines = block_lines[1:] if number_match else block_lines[1:]

        if question_type == QuestionType.TRUE_FALSE:
            content = self._strip_answer_placeholder(first_line_content)
            options = {"A": "正确", "B": "错误"}
            answer = "未知"
            subquestion_count = 1
        elif question_type in {QuestionType.A1, QuestionType.SINGLE_CHOICE, QuestionType.X, QuestionType.MULTIPLE_CHOICE}:
            content = self._strip_answer_placeholder(first_line_content)
            options = self._parse_choice_options(remaining_lines)
            answer = "未知"
            subquestion_count = 1
        elif question_type == QuestionType.SHORT_ANSWER:
            content = "\n".join([first_line_content, *remaining_lines]).strip()
            options = {}
            answer = "未知"
            subquestion_count = self._count_subquestions(content)
        else:
            content = "\n".join(([first_line_content] if first_line_content else []) + remaining_lines).strip()
            options = {}
            answer = "未知"
            subquestion_count = self._count_subquestions(content)

        return Question(
            id=str(index),
            type=question_type,
            content=content,
            knowledge_point=self._infer_knowledge_point(section_title, content),
            options=options,
            answer=answer,
            explanation="",
            source_exam=source_file,
            section_title=section_title,
            display_number=display_number,
            subquestion_count=max(subquestion_count, 1),
        )

    @staticmethod
    def _strip_answer_placeholder(text: str) -> str:
        text = re.sub(r'[（(]\s*[TtFf√×✓✗对错]?\s*[）)]\s*$', '', text)
        text = re.sub(r'[（(]\s*[_＿ ]+[）)]\s*$', '', text)
        return text.strip()

    @staticmethod
    def _parse_choice_options(lines: List[str]) -> Dict[str, str]:
        options: Dict[str, str] = {}
        unlabeled_lines: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            chunks = [chunk.strip() for chunk in _OPTION_SPLIT_PATTERN.split(line) if chunk.strip()]
            matched_any = False
            for chunk in chunks:
                match = _OPTION_MATCH_PATTERN.match(chunk)
                if match:
                    matched_any = True
                    key, value = match.groups()
                    options[key] = value.strip()
            if not matched_any:
                unlabeled_lines.append(line)

        if unlabeled_lines:
            next_key_ord = ord('A') + len(options)
            for line in unlabeled_lines:
                key = chr(next_key_ord)
                options[key] = line
                next_key_ord += 1

        return options

    @staticmethod
    def _count_subquestions(content: str) -> int:
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        scored_markers = sum(bool(re.search(r'[（(]\s*\d+\s*分\s*[）)]', line)) for line in lines)
        if scored_markers > 0:
            return scored_markers
        markers = 0
        for line in lines:
            if any(tag in line for tag in ["第一问", "第二问", "第三问", "第四问", "第五问"]):
                markers += 1
            elif re.match(r'^[（(]?\s*[1-9]\s*[）)]\s*', line):
                markers += 1
            elif re.match(r'^\d+\s*[、.．]\s*\S', line) and '？' in line:
                markers += 1
            elif line.endswith("？") or line.endswith("?"):
                markers += 1
        return markers if markers > 0 else 1

    @staticmethod
    def _infer_knowledge_point(section_title: str, content: str) -> str:
        base = re.sub(r'\s+', '', content)
        if len(base) <= 24:
            return base
        if "护理" in base:
            start = max(base.find("护理") - 8, 0)
            return base[start:start + 18]
        return base[:18]
    
    @staticmethod
    def _build_analyze_prompt(exam_text: str) -> str:
        return f"请分析以下医学试卷，提取所有题目并转换为结构化格式：\n\n---试卷内容开始---\n{exam_text}\n---试卷内容结束---\n\n请输出 JSON 格式的解析结果。"
    
    @staticmethod
    def _build_batch_prompt(text: str, batch_id: int) -> str:
        return f"请分析以下医学试卷片段，提取所有题目：\n\n---试卷片段---\n{text}\n---片段结束---\n\n请输出 JSON 格式的解析结果。注意：这只是试卷的一部分，title 可以写\"片段{batch_id}\"。"
    
    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.split('\n')[:10]:
            line = line.strip()
            if 5 < len(line) < 50:
                if any(p.match(line) for p in _TITLE_PATTERNS):
                    return line
        return None
