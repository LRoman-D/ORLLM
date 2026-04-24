from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from steporlm_stage1.teachers.zhipu import (
    QUESTION_REWRITE_SYSTEM_PROMPT,
    TRAJECTORY_SYSTEM_PROMPT,
    build_question_rewrite_prompt,
    build_solver_hint,
)
from steporlm_stage1.prompts import ORTOOLS_STYLE_GUIDE
from steporlm_stage1.quality.genprm import GENPRM_SYSTEM_PROMPT, build_genprm_prompt, parse_genprm_output
from steporlm_stage1.rag.index import HybridRagRetriever
from steporlm_stage1.utils.modeling import load_causal_lm, load_tokenizer


RAG_TRAJECTORY_SYSTEM_PROMPT = (
    TRAJECTORY_SYSTEM_PROMPT
    + "\nUse the supplied reference excerpts as modeling guidance, but preserve the user's numeric instance exactly. "
    "If an excerpt conflicts with the problem statement, the problem statement wins."
)


class QwenRagTeacherGenerator:
    backend_name = "qwen_rag_teacher"
    _TSP_REQUIRED_TERMS = (
        "traveling salesman",
        "travelling salesman",
        "tsp",
        "tour",
        "route",
        "routing",
        "distance matrix",
        "city",
    )
    _TSP_BANNED_TERMS = (
        "subtour elimination inequalities",
        "cut-set inequality",
        "facet",
        "polytope",
        "stsp",
        "held-karp",
        "branch-and-cut",
        "miller-tucker-zemlin",
        "mtz",
        "x_ij",
        "xij",
    )

    def __init__(
        self,
        model_path: str | Path,
        rag_index_dir: str | Path,
        *,
        load_in_4bit: bool = True,
        max_context_chars: int = 5000,
        top_k: int = 5,
        candidate_top_k: int = 25,
        use_semantic_rerank: bool = True,
        reranker_model_name_or_path: str = "BAAI/bge-reranker-base",
        reranker_batch_size: int = 8,
        reranker_max_length: int = 512,
        rerank_keyword_blend: float = 0.75,
        use_query_rewrite: bool = False,
        fail_on_reranker_error: bool = False,
        context_max_chunk_chars: int = 1200,
        context_include_scores: bool = False,
        modeling_boost: float = 0.08,
        trajectory_top_p: float = 0.9,
        trajectory_repetition_penalty: float = 1.0,
        use_bf16_if_available: bool = True,
        enable_thinking: bool = False,
        template_constraints_enabled: bool = True,
        tsp_filtered_top_k: int = 2,
        tsp_min_alpha_token_ratio: float = 0.52,
        tsp_max_digit_token_ratio: float = 0.36,
        tsp_require_keyword_match: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.model_name = str(model_path)
        self.max_context_chars = max_context_chars
        self.context_max_chunk_chars = max(200, int(context_max_chunk_chars))
        self.context_include_scores = bool(context_include_scores)
        self.modeling_boost = max(0.0, float(modeling_boost))
        self.top_k = top_k
        self.candidate_top_k = candidate_top_k
        self.trajectory_top_p = float(trajectory_top_p)
        self.trajectory_repetition_penalty = float(trajectory_repetition_penalty)
        self.enable_thinking = bool(enable_thinking)
        self.template_constraints_enabled = bool(template_constraints_enabled)
        self.tsp_filtered_top_k = max(1, int(tsp_filtered_top_k))
        self.tsp_min_alpha_token_ratio = max(0.1, min(0.9, float(tsp_min_alpha_token_ratio)))
        self.tsp_max_digit_token_ratio = max(0.05, min(0.95, float(tsp_max_digit_token_ratio)))
        self.tsp_require_keyword_match = bool(tsp_require_keyword_match)
        self.retriever = HybridRagRetriever(
            rag_index_dir,
            use_semantic_rerank=use_semantic_rerank,
            reranker_model_name_or_path=reranker_model_name_or_path,
            reranker_batch_size=reranker_batch_size,
            reranker_max_length=reranker_max_length,
            rerank_keyword_blend=rerank_keyword_blend,
            use_query_rewrite=use_query_rewrite,
            fail_on_reranker_error=fail_on_reranker_error,
        )
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
            top_k=int(config.get("rag_top_k", 5)),
            candidate_top_k=int(config.get("rag_candidate_top_k", 25)),
            use_semantic_rerank=bool(config.get("rag_use_semantic_rerank", True)),
            reranker_model_name_or_path=str(config.get("rag_reranker_model", "BAAI/bge-reranker-base")),
            reranker_batch_size=int(config.get("rag_reranker_batch_size", 8)),
            reranker_max_length=int(config.get("rag_reranker_max_length", 512)),
            rerank_keyword_blend=float(config.get("rag_rerank_keyword_blend", 0.75)),
            use_query_rewrite=bool(config.get("rag_use_query_rewrite", False)),
            fail_on_reranker_error=bool(config.get("rag_fail_on_reranker_error", False)),
            context_max_chunk_chars=int(config.get("rag_context_max_chunk_chars", 1200)),
            context_include_scores=bool(config.get("rag_context_include_scores", False)),
            modeling_boost=float(config.get("rag_modeling_boost", 0.08)),
            trajectory_top_p=float(config.get("trajectory_top_p", 0.9)),
            trajectory_repetition_penalty=float(config.get("trajectory_repetition_penalty", 1.0)),
            use_bf16_if_available=bool(config.get("use_bf16_if_available", True)),
            enable_thinking=bool(config.get("teacher_enable_thinking", config.get("qwen_enable_thinking", False))),
            template_constraints_enabled=bool(config.get("rag_template_constraints_enabled", True)),
            tsp_filtered_top_k=int(config.get("rag_tsp_filtered_top_k", 2)),
            tsp_min_alpha_token_ratio=float(config.get("rag_tsp_min_alpha_token_ratio", 0.52)),
            tsp_max_digit_token_ratio=float(config.get("rag_tsp_max_digit_token_ratio", 0.36)),
            tsp_require_keyword_match=bool(config.get("rag_tsp_require_keyword_match", True)),
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
        context = self._retrieve_context(
            f"{template_name} operations research modeling question variants",
            template_name=template_name,
        )
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
        context = self._retrieve_context(f"{template_name}\n{question}", template_name=template_name)
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
                top_p=self.trajectory_top_p,
                repetition_penalty=self.trajectory_repetition_penalty,
            )
            cleaned = content.strip()
            if cleaned:
                trajectories.append(cleaned)
        return trajectories

    def audit_trajectory(
        self,
        *,
        question: str,
        response: str,
        verification: dict[str, Any],
        template_name: str | None = None,
        max_tokens: int = 4200,
    ) -> dict[str, Any]:
        context = self._retrieve_context(
            f"GenPRM audit OR modeling semantics {template_name or ''}\n{question}",
            template_name=template_name,
        )
        prompt = build_genprm_prompt(
            question=question,
            trajectory=response,
            verification=verification,
            template_name=template_name,
            reference_context=context,
        )
        content = self._chat(
            [
                {"role": "system", "content": GENPRM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_new_tokens=max_tokens,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        return parse_genprm_output(content, model=self.model_name).to_dict()

    def _retrieve_context(self, query: str, *, template_name: str | None = None) -> str:
        results = self.retriever.search(
            query,
            candidate_top_k=self.candidate_top_k,
            top_k=self.top_k,
        )
        results = self._apply_modeling_boost(results)
        results = self._apply_template_retrieval_constraints(template_name, results)
        if not results:
            self.last_retrieval_contexts = []
            return "No trusted reference snippets passed template-aware filtering. Solve from the question and solver guidance only."
        self.last_retrieval_contexts = [
            {
                "chunk_id": item.get("chunk_id"),
                "source_name": item.get("source_name"),
                "page_start": item.get("page_start"),
                "keyword_score": item.get("keyword_score"),
                "rerank_score": item.get("rerank_score"),
                "score": item.get("score"),
                "query_used": item.get("query_used"),
            }
            for item in results
        ]
        return self.retriever.format_context(
            results,
            max_chars=self.max_context_chars,
            max_chunk_chars=self.context_max_chunk_chars,
            include_scores=self.context_include_scores,
            include_chunk_id=True,
        )

    def _apply_template_retrieval_constraints(self, template_name: str | None, results: list[dict]) -> list[dict]:
        if not results:
            return results
        if not self.template_constraints_enabled or not template_name:
            return results
        if str(template_name).strip().lower() != "tsp":
            return results

        filtered: list[dict] = []
        for row in results:
            text = str(row.get("text", ""))
            lowered = text.lower()
            if self.tsp_require_keyword_match and not any(term in lowered for term in self._TSP_REQUIRED_TERMS):
                continue
            if any(term in lowered for term in self._TSP_BANNED_TERMS):
                continue
            if self._is_low_quality_tsp_chunk(text):
                continue
            filtered.append(row)

        if not filtered:
            return []
        limit = min(max(1, self.top_k), self.tsp_filtered_top_k)
        return filtered[:limit]

    def _is_low_quality_tsp_chunk(self, text: str) -> bool:
        # Heuristic: reject OCR table-like chunks with heavy numeric density.
        alpha_tokens = re.findall(r"[A-Za-z]+", text)
        digit_tokens = re.findall(r"\d+(?:\.\d+)?", text)
        total = len(alpha_tokens) + len(digit_tokens)
        if total <= 0:
            return True
        alpha_ratio = len(alpha_tokens) / total
        digit_ratio = len(digit_tokens) / total
        if alpha_ratio < self.tsp_min_alpha_token_ratio:
            return True
        if digit_ratio > self.tsp_max_digit_token_ratio:
            return True
        if re.search(r"\{\s*\d+(?:\s+\d+){4,}\s*\}", text):
            return True
        return False

    def _apply_modeling_boost(self, results: list[dict]) -> list[dict]:
        if not results or self.modeling_boost <= 0:
            return results
        boosted = []
        markers = (
            "decision variable",
            "decision variables",
            "objective",
            "constraint",
            "constraints",
            "subject to",
            "maximize",
            "minimize",
            "变量",
            "目标函数",
            "约束",
        )
        for row in results:
            item = dict(row)
            text = str(item.get("text", "")).lower()
            hit_count = sum(1 for marker in markers if marker in text)
            boost = self.modeling_boost * min(3, hit_count)
            item["score"] = round(float(item.get("score", 0.0)) + boost, 6)
            item["modeling_hit_count"] = hit_count
            boosted.append(item)
        boosted.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return boosted

    @staticmethod
    def _build_rag_trajectory_prompt(question: str, template_name: str, context: str) -> str:
        solver_hint = build_solver_hint(template_name)
        tsp_api_guard = ""
        if str(template_name).strip().lower() == "tsp":
            tsp_api_guard = """
- TSP代码稳定性约束（必须遵守）：
  * 仅使用这些 OR-Tools Routing API：RoutingIndexManager, RoutingModel, RegisterTransitCallback, SetArcCostEvaluatorOfAllVehicles, DefaultRoutingSearchParameters, SolveWithParameters, NextVar, IsEnd, ObjectiveValue
  * 禁止调用不存在或未文档化的方法（例如 AddDimensionWithZeroToAllTransitsAndUIConstraint、GetRouteForVehicle）
  * 提取路径时必须写成 `next_index = solution.Value(routing.NextVar(index))`，禁止写 `solution.NextVar()` 或 `solution.NextVar().Value()`
  * 禁止硬编码目标值，必须使用 `solution.ObjectiveValue()` 读取
  * 你必须显式构造并打印 `result = {"status": "...", "objective_value": ...}`；不要依赖外部辅助函数推断状态
  * 若 `solution is None`，输出 status=`INFEASIBLE` 且 objective_value=`None`
"""
        return f"""Template family: {template_name}
Preferred solver guidance:
{solver_hint}

【问题】
{question}

【参考资料】
{context}

【要求】
基于参考资料回答问题，如涉及建模请给出：
* 决策变量
* 目标函数
* 约束条件
- 代码必须包含 import json，并输出唯一结果标记：__STEPORLM_RESULT__=
- 严禁改动题目中的任何数字、上/下界、容量、成本、需求等参数
- 参考资料仅用于建模启发，不要照抄其中变量名/索引集合；以当前题目定义为准
- 必须输出 9 个 `<step>...</step>` 推理步骤，最后仅输出一个完整 Python 代码块
- 代码必须遵守以下 OR-Tools 规范：
{ORTOOLS_STYLE_GUIDE}
{tsp_api_guard}

Produce the full 9-step trajectory and executable OR-Tools Python solver now.
"""

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_new_tokens: int,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> str:
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
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
        if repetition_penalty and repetition_penalty > 0:
            generation_kwargs["repetition_penalty"] = float(repetition_penalty)
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = float(top_p)
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
