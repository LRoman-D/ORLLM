from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ZhipuChatConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 90
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


class ZhipuChatClient:
    def __init__(self, config: ZhipuChatConfig) -> None:
        self.config = config

    @classmethod
    def from_env(
        cls,
        model_env_var: str = "ZHIPUAI_MODEL",
        timeout_seconds: int = 90,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        fallback_model: str | None = None,
    ) -> "ZhipuChatClient | None":
        api_key = os.getenv("ZHIPUAI_API_KEY")
        if not api_key:
            return None
        base_url = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        model = os.getenv(model_env_var) or fallback_model or os.getenv("ZHIPUAI_MODEL", "glm-4.5")
        return cls(
            ZhipuChatConfig(
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
        raise RuntimeError(f"Zhipu API request failed after {self.config.max_retries} attempts: {last_error}") from last_error

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        disable_thinking: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        body = self._post(payload)
        message = body["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if content:
            return content
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            return reasoning
        raise RuntimeError(f"Zhipu API returned empty content: {json.dumps(body, ensure_ascii=False)[:500]}")

    def json_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        content = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(self._strip_json_fences(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Zhipu API did not return valid JSON: {content[:500]}") from exc

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
        return stripped.strip()
