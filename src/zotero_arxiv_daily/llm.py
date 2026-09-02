"""Single entry point for LLM calls.

Handles the ``llm.api_mode`` (``chat_completion`` or ``response``) and the
optional streaming flag, so callers only pass messages. ``response`` mode maps
``max_tokens`` to the Responses API's ``max_output_tokens``; streaming is only
supported for chat completions (the flag is ignored otherwise).
"""

from typing import Any

from openai import OpenAI
from loguru import logger


def request_llm(openai_client: OpenAI, llm_params: Any, messages: list[dict]) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        if generation_kwargs.get("stream"):
            chunks = []
            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue
                if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content is not None:
                    chunks.append(content)
            return "".join(chunks)
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in generation_kwargs:
            generation_kwargs["max_output_tokens"] = max_tokens
        if generation_kwargs.pop("stream", None):
            logger.debug("Streaming is not supported in response api_mode; ignoring")
        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )
