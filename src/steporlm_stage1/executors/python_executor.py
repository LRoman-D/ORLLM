from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from pathlib import Path

from steporlm_stage1.schemas import ReferenceSolution, VerificationResult


class PythonCodeExecutor:
    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def verify(self, code: str, reference: ReferenceSolution, tolerance: float = 1e-4) -> VerificationResult:
        marker = "__STEPORLM_RESULT__="
        with tempfile.TemporaryDirectory(prefix="steporlm_stage1_") as tmp_dir:
            script_path = Path(tmp_dir) / "candidate.py"
            script_path.write_text(code, encoding="utf-8")
            try:
                completed = subprocess.run(
                    ["python", str(script_path)],
                    cwd=tmp_dir,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return VerificationResult(
                    success=False,
                    execution_ok=False,
                    status="TIMEOUT",
                    objective_value=None,
                    objective_match=False,
                    tolerance=tolerance,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "",
                    error_message=f"Execution timed out after {self.timeout_seconds} seconds.",
                )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            return VerificationResult(
                success=False,
                execution_ok=False,
                status="RUNTIME_ERROR",
                objective_value=None,
                objective_match=False,
                tolerance=tolerance,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Python exited with code {completed.returncode}.",
            )
        result_line = None
        for line in stdout.splitlines()[::-1]:
            if line.startswith(marker):
                result_line = line[len(marker) :]
                break
        if result_line is None:
            return VerificationResult(
                success=False,
                execution_ok=False,
                status="MISSING_RESULT",
                objective_value=None,
                objective_match=False,
                tolerance=tolerance,
                stdout=stdout,
                stderr=stderr,
                error_message="Result marker not found in stdout.",
            )
        try:
            payload = self._parse_result_payload(result_line)
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(
                success=False,
                execution_ok=True,
                status="INVALID_RESULT",
                objective_value=None,
                objective_match=False,
                tolerance=tolerance,
                stdout=stdout,
                stderr=stderr,
                error_message=f"Failed to parse result marker payload: {exc}",
            )
        objective_value = payload.get("objective_value")
        if objective_value is not None:
            try:
                objective_value = float(objective_value)
            except (TypeError, ValueError):
                objective_value = None
        status_value = payload.get("status", "UNKNOWN")
        if isinstance(status_value, (int, float)):
            status_value = "OPTIMAL" if int(status_value) == 0 else str(int(status_value))
        else:
            raw_status = str(status_value).strip()
            upper_status = raw_status.upper()
            if upper_status == "OPTIMAL":
                status_value = "OPTIMAL"
            elif upper_status == "FEASIBLE":
                status_value = "FEASIBLE"
            else:
                status_value = raw_status
        objective_match = False
        if objective_value is not None and reference.objective_value is not None:
            objective_match = abs(objective_value - reference.objective_value) <= tolerance
        success = status_value == "OPTIMAL" and objective_match
        return VerificationResult(
            success=success,
            execution_ok=True,
            status=status_value,
            objective_value=objective_value,
            objective_match=objective_match,
            tolerance=tolerance,
            stdout=stdout,
            stderr=stderr,
            error_message=None if success else "Objective mismatch or non-optimal solver status.",
        )

    @staticmethod
    def _parse_result_payload(result_line: str) -> dict:
        text = result_line.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Unsupported result payload: {text[:200]}")
