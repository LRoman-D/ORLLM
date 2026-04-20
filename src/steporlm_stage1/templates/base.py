from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from steporlm_stage1.schemas import ReferenceSolution


class ProblemTemplate(ABC):
    name: str
    optimization_sense: str

    @abstractmethod
    def sample_instance(self, rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def render_question(self, instance: dict[str, Any], rng: random.Random) -> str:
        raise NotImplementedError

    @abstractmethod
    def solve_reference(self, instance: dict[str, Any]) -> ReferenceSolution:
        raise NotImplementedError
