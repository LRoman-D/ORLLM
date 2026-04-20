from __future__ import annotations

import json
from dataclasses import dataclass

from steporlm_stage1.clients import ZhipuChatClient


TEACHER_SYSTEM_PROMPT = """You are an operations research trajectory critic.
Evaluate a candidate solution to an OR modeling problem holistically.
Focus on:
1. Whether the reasoning steps are logically coherent.
2. Whether the mathematical formulation matches the problem.
3. Whether the generated Python/OR-Tools code appears executable and semantically aligned.
4. Whether the final result is likely correct.

Return only valid JSON with keys:
- overall_score: float in [0, 1]
- step_score: float in [0, 1]
- code_score: float in [0, 1]
- verdict: one of ["excellent", "good", "weak", "bad"]
- strengths: array of short strings
- issues: array of short strings
- rationale: short paragraph
"""


def build_teacher_prompt(question: str, response: str, verification: dict) -> str:
    verification_text = json.dumps(
        {
            "execution_ok": verification.get("execution_ok"),
            "status": verification.get("status"),
            "objective_match": verification.get("objective_match"),
            "error_message": verification.get("error_message"),
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""Question:
{question}

Candidate trajectory:
{response}

Solver verification:
{verification_text}

Assess the full trajectory and code quality. Return JSON only.
"""


@dataclass
class TeacherConfig:
    model: str
    timeout_seconds: int = 60


class ZhipuTeacherEvaluator:
    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self.client = ZhipuChatClient.from_env(model_env_var="ZHIPUAI_MODEL", timeout_seconds=config.timeout_seconds)
        if self.client is None:
            raise RuntimeError("ZHIPUAI_API_KEY is not set.")

    @classmethod
    def from_env(cls) -> "ZhipuTeacherEvaluator | None":
        client = ZhipuChatClient.from_env()
        if client is None:
            return None
        return cls(TeacherConfig(model=client.config.model))

    def evaluate(self, question: str, response: str, verification: dict) -> dict:
        parsed = self.client.json_chat(
            [
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": build_teacher_prompt(question, response, verification)},
            ],
            temperature=0.0,
        )
        parsed["model"] = self.config.model
        return parsed
