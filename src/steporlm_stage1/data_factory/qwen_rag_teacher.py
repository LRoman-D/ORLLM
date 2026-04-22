from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from steporlm_stage1.data_factory.zhipu_teacher import (
    QUESTION_REWRITE_SYSTEM_PROMPT,
    TRAJECTORY_SYSTEM_PROMPT,
    build_question_rewrite_prompt,
    build_solver_hint,
)
from steporlm_stage1.rag.index import KeywordRagRetriever
from steporlm_stage1.utils.modeling import load_causal_lm, load_tokenizer


RAG_TRAJECTORY_SYSTEM_PROMPT = (
    TRAJECTORY_SYSTEM_PROMPT
    + "\nUse the supplied reference excerpts as modeling guidance, but preserve the user's numeric instance exactly. "
    "If an excerpt conflicts with the problem statement, the problem statement wins."
)


class QwenRagTeacherGenerator:
    backend_name = "qwen_rag_teacher"

    def __init__(
        self,
        model_path: str | Path,
        rag_index_dir: str | Path,
        *,
        load_in_4bit: bool = True,
        max_context_chars: int = 5000,
        top_k: int = 4,
        use_bf16_if_available: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.model_name = str(model_path)
        self.max_context_chars = max_context_chars
        self.top_k = top_k
        self.retriever = KeywordRagRetriever(rag_index_dir)
        self.tokenizer = load_tokenizer(model_path)
        self.model = load_causal_lm(
            model_path,
            load_in_4bit=load_in_4bit,
            use_bf16_if_available=use_bf16_if_available,
            is_adapter=False,
        )
        self.model.eval()
        self.last_retrieval_contexts: list[dict] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QwenRagTeacherGenerator":
        return cls(
            model_path=config.get("teacher_model_path") or config.get("model_path") or config.get("model_name_or_path"),
            rag_index_dir=config.get("rag_index_dir", "data/rag/or_books"),
            load_in_4bit=bool(config.get("teacher_load_in_4bit", config.get("load_in_4bit", True))),
            max_context_chars=int(config.get("rag_max_context_chars", 5000)),
            top_k=int(config.get("rag_top_k", 4)),
            use_bf16_if_available=bool(config.get("use_bf16_if_available", True)),
        )

    def rewrite_question_variants(
        self,
        template_name: str,
        canonical_question: str,
        instance: dict[str, Any],
        num_variants: int,
        rewrite_styles: list[str],
        temperature: float = 0.6,
        max_tokens: int = 1200,
    ) -> list[str]:
        context = self._retrieve_context(f"{template_name} operations research modeling question variants")
        user_prompt = (
            build_question_rewrite_prompt(template_name, canonical_question, instance, num_variants, rewrite_styles)
            + "\n\nReference excerpts for more varied but valid wording:\n"
            + context
            + "\n\nReturn JSON only."
        )
        content = self._chat(
            [
                {"role": "system", "content": QUESTION_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        payload = self._parse_json_object(content)
        raw_variants = payload.get("variants", []) if isinstance(payload, dict) else []
        variants = []
        for item in raw_variants:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in variants:
                    variants.append(cleaned)
        return variants[:num_variants]

    def generate_trajectories(
        self,
        question: str,
        template_name: str,
        temperatures: list[float],
        max_tokens: int = 2000,
    ) -> list[str]:
        context = self._retrieve_context(f"{template_name}\n{question}")
        user_prompt = self._build_rag_trajectory_prompt(question=question, template_name=template_name, context=context)
        trajectories = []
        for temperature in temperatures:
            content = self._chat(
                [
                    {"role": "system", "content": RAG_TRAJECTORY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            cleaned = content.strip()
            if cleaned:
                trajectories.append(cleaned)
        return trajectories

    def _retrieve_context(self, query: str) -> str:
        results = self.retriever.search(query, top_k=self.top_k)
        self.last_retrieval_contexts = [
            {
                "chunk_id": item.get("chunk_id"),
                "source_name": item.get("source_name"),
                "page_start": item.get("page_start"),
                "score": item.get("score"),
            }
            for item in results
        ]
        return self.retriever.format_context(results, max_chars=self.max_context_chars)

    @staticmethod
    def _build_rag_trajectory_prompt(question: str, template_name: str, context: str) -> str:
        solver_hint = build_solver_hint(template_name)
        return f"""Template family: {template_name}
Preferred solver guidance:
{solver_hint}

Retrieved reference excerpts:
{context}

Optimization question:
{question}

Produce the full 8-step trajectory and executable OR-Tools Python solver now.
"""

    def _chat(self, messages: list[dict[str, str]], *, temperature: float, max_new_tokens: int) -> str:
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        do_sample = temperature > 0.0
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.9
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        return {}
