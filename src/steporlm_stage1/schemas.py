from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReferenceSolution:
    status: str
    objective_value: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    success: bool
    execution_ok: bool
    status: str
    objective_value: float | None
    objective_match: bool
    tolerance: float
    stdout: str
    stderr: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessVerification:
    score: float
    correct_count: int
    total_steps: int
    all_correct: bool
    step_correct: list[bool]
    explanations: list[str] = field(default_factory=list)
    model: str = "unknown"
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRecord:
    problem_id: str
    split: str
    template_name: str
    question: str
    response: str
    code: str
    instance: dict[str, Any]
    reference_solution: dict[str, Any]
    verification: dict[str, Any]
    process_verification: dict[str, Any] = field(default_factory=dict)
    generation_notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreferenceTrajectory:
    trajectory_id: str
    response: str
    code: str
    verification: dict[str, Any]
    process_score: float
    source: str
    process_verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreferencePair:
    problem_id: str
    template_name: str
    question: str
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    weight: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
