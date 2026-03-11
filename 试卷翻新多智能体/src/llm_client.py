"""
医学试卷翻新多智能体系统 - LLM 客户端封装

提供统一的 LLM 调用接口，支持 OpenAI 和 Google Gemini API，支持 JSON Mode。
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, BadRequestError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import warnings
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from .config import settings

logger = logging.getLogger(__name__)
_CLIENT_CACHE: Dict[tuple, "LLMClient"] = {}

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)


class LLMClient:
    """LLM 客户端封装类，支持 OpenAI 和 Google Gemini"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        provider: Optional[str] = None,
        proxy: Optional[str] = None,
    ):
        self.provider = provider or settings.llm.provider
        self.api_key = api_key or settings.llm.api_key
        self.base_url = base_url or settings.llm.base_url
        self.model = model or settings.llm.model
        self.temperature = temperature if temperature is not None else settings.llm.temperature
        self.max_tokens = max_tokens
        self.proxy = proxy or settings.llm.proxy
        
        if self.proxy:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ[key] = self.proxy
            logger.info(f"已配置代理: {self.proxy}")
        
        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_openai()
    
    def _init_gemini(self) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini 支持需要安装 google-generativeai 库。\n"
                "请运行: pip install google-generativeai"
            )
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
        self.async_client = None
        logger.info(f"初始化 Gemini 客户端，模型: {self.model}")
    
    def _init_openai(self) -> None:
        client_kwargs = dict(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=settings.llm.timeout,
        )
        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)
        logger.info(f"初始化 OpenAI 客户端，模型: {self.model}")
    
    # ── 消息构建 ──────────────────────────────────────────────
    
    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages
    
    @staticmethod
    def _build_gemini_prompt(
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        role_map = {"user": "User", "assistant": "Assistant", "system": "System"}
        parts: list[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        if history:
            for msg in history:
                label = role_map.get(msg.get("role", "user"), "User")
                parts.append(f"{label}: {msg.get('content', '')}\n")
        parts.append(f"User: {prompt}")
        return "\n".join(parts)
    
    # ── 核心参数构建（消除同步/异步重复） ────────────────────
    
    _FIXED_TEMPERATURE_MODELS = {"kimi-k2.5", "kimi-k2.0"}
    
    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        model_lower = self.model.lower()
        if any(m in model_lower for m in self._FIXED_TEMPERATURE_MODELS):
            if temperature is not None and temperature != 1.0:
                logger.debug(f"模型 {self.model} 要求 temperature=1，忽略请求的 temperature={temperature}")
            return 1.0
        return temperature if temperature is not None else self.temperature
    
    def _resolve_max_tokens(self, max_tokens: Optional[int]) -> int:
        return max_tokens if max_tokens is not None else self.max_tokens
    
    def _build_openai_kwargs(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt, history),
            "temperature": self._resolve_temperature(temperature),
            "max_completion_tokens": self._resolve_max_tokens(max_tokens),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs
    
    def _build_gemini_config(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> tuple:
        """返回 (full_prompt, generation_config)"""
        full_prompt = self._build_gemini_prompt(prompt, system_prompt, history)
        if json_mode:
            full_prompt += "\n\nPlease respond with valid JSON only."
        
        config_kwargs = {
            "temperature": self._resolve_temperature(temperature),
            "max_output_tokens": self._resolve_max_tokens(max_tokens),
        }
        generation_config = genai.GenerationConfig(**config_kwargs)
        if json_mode:
            generation_config.response_mime_type = "application/json"
        
        return full_prompt, generation_config
    
    # ── OpenAI temperature 容错 ───────────────────────────────
    
    @staticmethod
    def _handle_temperature_error(e: BadRequestError, kwargs: Dict[str, Any]) -> bool:
        """处理某些模型强制 temperature=1 的情况，返回是否应重试"""
        err_msg = str(e)
        if "invalid temperature" in err_msg and "only 1 is allowed" in err_msg:
            logger.warning(f"模型要求 temperature=1，正在重试... 原错误: {e}")
            kwargs["temperature"] = 1.0
            return True
        return False
    
    # ── 同步接口 ──────────────────────────────────────────────
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """同步聊天接口"""
        if self.provider == "gemini":
            return self._chat_gemini(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        return self._chat_openai(prompt, system_prompt, history, temperature, max_tokens, json_mode)
    
    def _chat_openai(self, prompt, system_prompt, history, temperature, max_tokens, json_mode) -> str:
        kwargs = self._build_openai_kwargs(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        try:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                if self._handle_temperature_error(e, kwargs):
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM 返回空响应")
            logger.debug(f"OpenAI Response: {content[:200]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI 调用失败: {e}")
            raise
    
    def _chat_gemini(self, prompt, system_prompt, history, temperature, max_tokens, json_mode) -> str:
        full_prompt, gen_config = self._build_gemini_config(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        try:
            response = self.client.generate_content(full_prompt, generation_config=gen_config)
            if not response.text:
                raise ValueError("Gemini 返回空响应")
            logger.debug(f"Gemini Response: {response.text[:200]}...")
            return response.text
        except Exception as e:
            logger.error(f"Gemini 调用失败: {e}")
            raise
    
    # ── 异步接口 ──────────────────────────────────────────────
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    async def achat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """异步聊天接口"""
        if self.provider == "gemini":
            return await self._achat_gemini(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        return await self._achat_openai(prompt, system_prompt, history, temperature, max_tokens, json_mode)
    
    async def _achat_openai(self, prompt, system_prompt, history, temperature, max_tokens, json_mode) -> str:
        kwargs = self._build_openai_kwargs(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        try:
            try:
                response = await self.async_client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                if self._handle_temperature_error(e, kwargs):
                    response = await self.async_client.chat.completions.create(**kwargs)
                else:
                    raise
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM 返回空响应")
            logger.debug(f"OpenAI Async Response: {content[:200]}...")
            return content
        except Exception as e:
            logger.error(f"OpenAI 异步调用失败: {e}")
            raise
    
    async def _achat_gemini(self, prompt, system_prompt, history, temperature, max_tokens, json_mode) -> str:
        full_prompt, gen_config = self._build_gemini_config(prompt, system_prompt, history, temperature, max_tokens, json_mode)
        try:
            response = await self.client.generate_content_async(full_prompt, generation_config=gen_config)
            if not response.text:
                raise ValueError("Gemini 返回空响应")
            logger.debug(f"Gemini Async Response: {response.text[:200]}...")
            return response.text
        except Exception as e:
            logger.error(f"Gemini 异步调用失败: {e}")
            raise
    
    # ── 结构化输出（统一同步/异步） ──────────────────────────
    
    def _build_schema_system_prompt(self, response_schema: Type[T], system_prompt: Optional[str]) -> str:
        schema_info = response_schema.model_json_schema()
        schema_prompt = f"\n请以 JSON 格式返回响应，严格遵循以下 schema:\n```json\n{json.dumps(schema_info, ensure_ascii=False, indent=2)}\n```\n"
        return (system_prompt or "") + "\n\n" + schema_prompt
    
    @staticmethod
    def _parse_schema_response(response: str, response_schema: Type[T]) -> T:
        try:
            data = json.loads(response)
            return response_schema.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\nResponse: {response}")
            raise ValueError(f"LLM 返回的内容不是有效的 JSON: {e}")
        except Exception as e:
            logger.error(f"Schema 验证失败: {e}")
            raise
    
    def chat_with_schema(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """带结构化输出的同步聊天接口"""
        full_system_prompt = self._build_schema_system_prompt(response_schema, system_prompt)
        response = self.chat(prompt=prompt, system_prompt=full_system_prompt, history=history, temperature=temperature, json_mode=True)
        return self._parse_schema_response(response, response_schema)
    
    async def achat_with_schema(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
    ) -> T:
        """带结构化输出的异步聊天接口"""
        full_system_prompt = self._build_schema_system_prompt(response_schema, system_prompt)
        response = await self.achat(prompt=prompt, system_prompt=full_system_prompt, history=history, temperature=temperature, json_mode=True)
        return self._parse_schema_response(response, response_schema)
    
    # ── JSON 提取 ─────────────────────────────────────────────
    
    @staticmethod
    def _try_repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
        """尝试修复被 max_tokens 截断的不完整 JSON"""
        stack = []
        in_string = False
        escape_next = False

        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        if not stack:
            return None

        if in_string:
            text += '"'

        for bracket in reversed(stack):
            if bracket == '{':
                text += '}'
            else:
                text += ']'

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        truncated = text
        for _ in range(10):
            last_comma = max(truncated.rfind(','), truncated.rfind('"'))
            if last_comma <= 0:
                break
            truncated = truncated[:last_comma]
            suffix = ""
            for bracket in reversed(stack):
                suffix += '}' if bracket == '{' else ']'
            try:
                return json.loads(truncated + suffix)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def extract_json_from_response(response: str) -> Dict[str, Any]:
        """从 LLM 响应中提取 JSON，处理可能的 markdown 代码块和截断"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        for match in _JSON_BLOCK_RE.findall(response):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end + 1])
            except json.JSONDecodeError:
                pass

        if start != -1:
            candidate = response[start:]
            result = LLMClient._try_repair_truncated_json(candidate)
            if result is not None:
                logger.warning("JSON 响应被截断，已自动修复（部分字段可能丢失）")
                return result
        
        raise ValueError(f"无法从响应中提取有效的 JSON: {response[:500]}...")


def get_llm_client(role: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """获取 LLM 客户端实例，支持按角色选择模型并复用实例。"""
    resolved_model = model or settings.llm.resolve_model(role)
    cache_key = (
        settings.llm.provider,
        settings.llm.base_url,
        resolved_model,
        settings.llm.proxy,
    )
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        client = LLMClient(model=resolved_model)
        _CLIENT_CACHE[cache_key] = client
    return client
