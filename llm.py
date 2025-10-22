from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
from typing import List, Dict, Optional, Any

GLOBAL_LLM = None

class LLM:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        lang: str = "English",
        default_temperature: float = 0.0,
        default_top_p: float = 1.0
    ):
        """
        若提供 api_key -> 使用 OpenAI 相容 API；否則退回本地 llama_cpp。
        default_temperature 與 default_top_p 作為每次 generate 的預設值，
        但每次呼叫都可以被覆蓋。
        """
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang
        self.default_temperature = float(default_temperature)
        self.default_top_p = float(default_top_p)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        統一的聊天生成介面。
        - temperature/top_p：控制取樣多樣性；若為 None 則使用物件預設值。
        - max_tokens/stop/seed/extra：按底層支援情況傳入；不支援時自動忽略。
        回傳：文字（第一個候選的 content）
        """
        t = self.default_temperature if temperature is None else float(temperature)
        p = self.default_top_p if top_p is None else float(top_p)
        extra = extra or {}

        if isinstance(self.llm, OpenAI):
            # Chat Completions（相容 openai-compatible 伺服器）
            max_retries = 3
            last_err = None
            for attempt in range(max_retries):
                try:
                    kwargs = dict(
                        messages=messages,
                        model=self.model,
                        temperature=t,
                        top_p=p,
                    )
                    if max_tokens is not None:
                        kwargs["max_tokens"] = int(max_tokens)
                    if stop is not None:
                        kwargs["stop"] = stop
                    if seed is not None:
                        # 新版 OpenAI Chat Completions 支援 seed；若伺服器不支援會丟錯
                        kwargs["seed"] = int(seed)
                    # 允許傳入額外低頻參數
                    kwargs.update(extra)

                    response = self.llm.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                except Exception as e:
                    last_err = e
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            # normally unreachable
            raise last_err
        else:
            # llama_cpp 路徑
            # create_chat_completion 支援 temperature/top_p/max_tokens/stop/seed
            kwargs = dict(
                messages=messages,
                temperature=t,
                top_p=p,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            if stop is not None:
                kwargs["stop"] = stop
            if seed is not None:
                kwargs["seed"] = int(seed)
            kwargs.update(extra)

            response = self.llm.create_chat_completion(**kwargs)
            return response["choices"][0]["message"]["content"]

def set_global_llm(
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    lang: str = "English",
    default_temperature: float = 0.0,
    default_top_p: float = 1.0
):
    """
    你可以在專案啟動時設定全域 LLM 與預設溫度/Top-p。
    例如：set_global_llm(api_key=..., model="gpt-5-mini", lang="Chinese Traditional", default_temperature=0.2)
    """
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        lang=lang,
        default_temperature=default_temperature,
        default_top_p=default_top_p
    )

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
