from __future__ import annotations

import ast
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from steporlm_stage1.data_factory.teacher_factory import build_teacher_generator, teacher_backend_name, teacher_model_name
from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.quality.genprm import audit_passes_threshold
from steporlm_stage1.schemas import DatasetRecord
from steporlm_stage1.templates.registry import TEMPLATE_REGISTRY
from steporlm_stage1.utils.io import ensure_dir, load_yaml_config, read_jsonl, write_json, write_jsonl
from steporlm_stage1.utils.text import extract_python_code


class Stage1DataFactory:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.rng = random.Random(config["seed"])
        self.executor = PythonCodeExecutor(timeout_seconds=config.get("timeout_seconds", 20))
        self.teacher_generator = build_teacher_generator(config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage1DataFactory":
        return cls(load_yaml_config(path))

    def _sample_template_name(self) -> str:
        weights = self.config["template_weights"]
        names = list(weights.keys())
        probs = [weights[name] for name in names]
        return self.rng.choices(names, weights=probs, k=1)[0]

    def _split_group_names(self, total_groups: int) -> list[str]:
        splits = self.config["splits"]
        names = []
        assigned = 0
        items = list(splits.items())
        for idx, (name, ratio) in enumerate(items):
            if idx == len(items) - 1:
                count = total_groups - assigned
            else:
                count = int(total_groups * ratio)
                assigned += count
            names.extend([name] * count)
        self.rng.shuffle(names)
        return names

    def _build_teacher_variants(self, template_name: str, canonical_question: str, instance: dict[str, Any]) -> list[str]:
        total_variants = int(self.config.get("question_variants_per_seed", 2))
        include_canonical = bool(self.config.get("include_canonical_question", True))
        if total_variants <= 0:
            return [canonical_question]
        if self.teacher_generator is None:
            return [canonical_question]

        desired_rewrites = max(0, total_variants - (1 if include_canonical else 0))
        variants: list[str] = [canonical_question] if include_canonical else []
        if desired_rewrites > 0:
            rewrite_styles = list(self.config.get("question_rewrite_styles", []))
            generated = self.teacher_generator.rewrite_question_variants(
                template_name=template_name,
                canonical_question=canonical_question,
                instance=instance,
                num_variants=desired_rewrites,
                rewrite_styles=rewrite_styles,
                temperature=float(self.config.get("question_rewrite_temperature", 0.6)),
                max_tokens=int(self.config.get("question_rewrite_max_tokens", 1200)),
            )
            for item in generated:
                if item not in variants:
                    variants.append(item)
        return variants[:total_variants] if variants else [canonical_question]

    def _temperature_schedule(self) -> list[float]:
        explicit = self.config.get("trajectory_temperatures")
        if explicit:
            return [float(value) for value in explicit]
        count = int(self.config.get("trajectories_per_variant", 3))
        if count <= 1:
            return [0.3]
        start = float(self.config.get("trajectory_temperature_min", 0.25))
        stop = float(self.config.get("trajectory_temperature_max", 0.8))
        return [round(start + idx * (stop - start) / (count - 1), 2) for idx in range(count)]

    @staticmethod
    def _group_id_from_problem_id(problem_id: str) -> str:
        if "-t" not in problem_id:
            return problem_id
        return problem_id.rsplit("-t", 1)[0]

    @staticmethod
    def _append_jsonl(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_resume_snapshot(self, output_dir: Path, pending_path: Path) -> tuple[dict[str, list[dict]], set[str], Counter[str]]:
        groups_by_id: dict[str, list[dict]] = {}
        problem_ids: set[str] = set()
        template_counter: Counter[str] = Counter()
        rows: list[dict] = []
        if pending_path.exists():
            rows = read_jsonl(pending_path)
        else:
            # Backward compatibility: bootstrap resume cache from split files created by older logic.
            for split_name in self.config["splits"].keys():
                split_path = output_dir / f"{split_name}.jsonl"
                if split_path.exists():
                    rows.extend(read_jsonl(split_path))
            if rows:
                write_jsonl(pending_path, rows)

        if not rows:
            return groups_by_id, problem_ids, template_counter

        for row in rows:
            problem_id = str(row.get("problem_id", "")).strip()
            if not problem_id or problem_id in problem_ids:
                continue
            group_id = self._group_id_from_problem_id(problem_id)
            groups_by_id.setdefault(group_id, []).append(row)
            problem_ids.add(problem_id)
            template_name = str(row.get("template_name", "")).strip()
            if template_name:
                template_counter[template_name] += 1
        return groups_by_id, problem_ids, template_counter

    @staticmethod
    def _load_state(state_path: Path) -> dict[str, Any]:
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _write_generation_state(
        self,
        state_path: Path,
        *,
        target_verified: int,
        accepted_counter: int,
        attempted_trajectories: int,
        api_request_errors: int,
        seed_problem_count: int,
        variant_count: int,
        next_seed_index: int,
        max_seed_problems: int,
        template_counter: Counter[str],
        teacher_model: str,
        genprm_attempted: int = 0,
        genprm_rejected: int = 0,
        interrupted: bool,
        completed: bool,
    ) -> None:
        write_json(
            state_path,
            {
                "version": 1,
                "target_verified_samples": target_verified,
                "accepted_samples": accepted_counter,
                "attempted_trajectories": attempted_trajectories,
                "api_request_errors": api_request_errors,
                "seed_problem_count": seed_problem_count,
                "question_variant_count": variant_count,
                "next_seed_index": next_seed_index,
                "max_seed_problems": max_seed_problems,
                "template_distribution": dict(template_counter),
                "teacher_model": teacher_model,
                "genprm_attempted": genprm_attempted,
                "genprm_rejected": genprm_rejected,
                "interrupted": interrupted,
                "completed": completed,
                "rng_state": repr(self.rng.getstate()),
            },
        )

    def _records_by_split(self, groups_by_id: dict[str, list[dict]]) -> dict[str, list[dict]]:
        records_by_split: dict[str, list[dict]] = {name: [] for name in self.config["splits"].keys()}
        group_rows_list = [rows for rows in groups_by_id.values() if rows]
        if not group_rows_list:
            return records_by_split

        split_names = self._split_group_names(len(group_rows_list))
        for split, group_rows in zip(split_names, group_rows_list):
            for row in group_rows:
                row["split"] = split
            records_by_split[split].extend(group_rows)
        return records_by_split

    def _genprm_enabled(self) -> bool:
        return bool(self.config.get("genprm_evaluation", self.config.get("genprm_enabled", False)))

    def _audit_candidate_process(
        self,
        *,
        question: str,
        response: str,
        verification: dict[str, Any],
        template_name: str,
    ) -> dict[str, Any]:
        if not self._genprm_enabled():
            return {}
        if self.teacher_generator is None or not hasattr(self.teacher_generator, "audit_trajectory"):
            return {
                "score": 0.0,
                "correct_count": 0,
                "total_steps": 9,
                "all_correct": False,
                "step_correct": [],
                "explanations": ["Teacher backend does not implement audit_trajectory."],
                "model": teacher_model_name(self.teacher_generator) if self.teacher_generator is not None else "unknown",
                "raw_output": "",
                "error": "audit_not_supported",
            }
        try:
            return self.teacher_generator.audit_trajectory(
                question=question,
                response=response,
                verification=verification,
                template_name=template_name,
                max_tokens=int(self.config.get("genprm_max_tokens", 4200)),
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "score": 0.0,
                "correct_count": 0,
                "total_steps": 9,
                "all_correct": False,
                "step_correct": [],
                "explanations": [str(exc)],
                "model": teacher_model_name(self.teacher_generator),
                "raw_output": "",
                "error": "audit_failed",
            }

    def _process_audit_passes(self, audit: dict[str, Any]) -> bool:
        if not self._genprm_enabled():
            return True
        return audit_passes_threshold(
            audit,
            min_correct_steps=int(self.config.get("genprm_min_correct_steps", 8)),
            require_all_correct=bool(self.config.get("genprm_require_all_correct", False)),
        )

    def _generate_teacher_dataset(self) -> dict[str, int | float | dict | str]:
        if self.teacher_generator is None:
            raise RuntimeError("Teacher generator is not available. Check teacher_backend, model path, RAG index, or API env vars.")

        output_dir = ensure_dir(self.config["output_dir"])
        target_verified = int(self.config.get("target_verified_samples", self.config.get("num_samples", 120)))
        max_seed_problems = int(
            self.config.get("max_seed_problems", max(40, math.ceil(target_verified / max(1, self.config.get("question_variants_per_seed", 2)))))
        )

        pending_file = str(self.config.get("pending_records_file", "accepted_pending.jsonl"))
        state_file = str(self.config.get("state_file", "generation_state.json"))
        trace_file = str(self.config.get("trace_file", "generation_trace.jsonl"))
        pending_path = output_dir / pending_file
        state_path = output_dir / state_file
        trace_path = output_dir / trace_file

        resume_from_checkpoint = bool(self.config.get("resume_from_checkpoint", True))
        if not resume_from_checkpoint:
            for reset_path in [pending_path, state_path, trace_path]:
                if reset_path.exists():
                    reset_path.unlink()

        groups_by_id, problem_ids, template_counter = self._load_resume_snapshot(output_dir, pending_path)
        state = self._load_state(state_path) if resume_from_checkpoint else {}

        rng_state = state.get("rng_state")
        if resume_from_checkpoint and isinstance(rng_state, str) and rng_state:
            try:
                self.rng.setstate(ast.literal_eval(rng_state))
            except Exception:  # noqa: BLE001
                pass

        accepted_counter = len(problem_ids)
        attempted_trajectories = int(state.get("attempted_trajectories", 0))
        api_request_errors = int(state.get("api_request_errors", 0))
        genprm_attempted = int(state.get("genprm_attempted", 0))
        genprm_rejected = int(state.get("genprm_rejected", 0))
        seed_problem_count = int(state.get("seed_problem_count", 0))
        variant_count = int(state.get("question_variant_count", 0))
        next_seed_index = int(state.get("next_seed_index", 0))

        if accepted_counter > 0 and next_seed_index == 0:
            max_seed_seen = -1
            for group_rows in groups_by_id.values():
                for row in group_rows:
                    seed_index = row.get("generation_notes", {}).get("seed_problem_index")
                    if isinstance(seed_index, int):
                        max_seed_seen = max(max_seed_seen, seed_index)
            if max_seed_seen >= 0:
                next_seed_index = max_seed_seen + 1

        next_seed_index = min(max(next_seed_index, 0), max_seed_problems)
        temperatures = self._temperature_schedule()
        progress = tqdm(
            total=target_verified,
            initial=min(accepted_counter, target_verified),
            desc="Generating teacher-verified dataset",
        )

        current_next_seed_index = next_seed_index
        interrupted = False

        try:
            for seed_idx in range(next_seed_index, max_seed_problems):
                current_next_seed_index = seed_idx
                if accepted_counter >= target_verified:
                    break

                template_name = self._sample_template_name()
                template = TEMPLATE_REGISTRY[template_name]
                instance = template.sample_instance(self.rng)
                canonical_question = template.render_question(instance, self.rng)
                reference = template.solve_reference(instance)
                seed_problem_count += 1
                progress.set_postfix(
                    seed=seed_problem_count,
                    attempted=attempted_trajectories,
                    accepted=accepted_counter,
                    api_errors=api_request_errors,
                )
                if self.config.get("verbose_generation", True):
                    tqdm.write(f"[seed {seed_problem_count}/{max_seed_problems}] template={template_name} generating question variants")

                try:
                    question_variants = self._build_teacher_variants(template_name, canonical_question, instance)
                except Exception as exc:  # noqa: BLE001
                    api_request_errors += 1
                    if self.config.get("verbose_generation", True):
                        tqdm.write(f"  rewrite failed: {exc}")
                    question_variants = [canonical_question]

                for variant_idx, question in enumerate(question_variants):
                    if accepted_counter >= target_verified:
                        break
                    variant_count += 1
                    group_id = f"{template_name}-seed{seed_idx:04d}-q{variant_idx:02d}"
                    group_success_count = 0
                    if self.config.get("verbose_generation", True):
                        tqdm.write(
                            f"  variant {variant_idx + 1}/{len(question_variants)}: sampling {len(temperatures)} trajectories"
                        )
                    try:
                        responses = self.teacher_generator.generate_trajectories(
                            question=question,
                            template_name=template_name,
                            temperatures=temperatures,
                            max_tokens=int(self.config.get("trajectory_max_tokens", 2200)),
                        )
                    except Exception as exc:  # noqa: BLE001
                        api_request_errors += 1
                        if self.config.get("verbose_generation", True):
                            tqdm.write(f"  trajectory generation failed: {exc}")
                        responses = []

                    for traj_idx, response in enumerate(responses):
                        attempted_trajectories += 1
                        code = extract_python_code(response)
                        verification = self.executor.verify(code, reference)
                        process_verification: dict[str, Any] = {}
                        if verification.success and self._genprm_enabled():
                            genprm_attempted += 1
                            process_verification = self._audit_candidate_process(
                                question=question,
                                response=response,
                                verification=verification.to_dict(),
                                template_name=template_name,
                            )
                        self._append_jsonl(
                            trace_path,
                            [
                                {
                                    "seed_problem_index": seed_idx,
                                    "question_variant_index": variant_idx,
                                    "trajectory_index": traj_idx,
                                    "template_name": template_name,
                                    "question": question,
                                    "response_preview": response[:1200],
                                    "code_preview": code[:1200],
                                    "verification": verification.to_dict(),
                                    "process_verification": process_verification,
                                }
                            ],
                        )
                        progress.set_postfix(
                            seed=seed_problem_count,
                            attempted=attempted_trajectories,
                            accepted=accepted_counter,
                            api_errors=api_request_errors,
                        )
                        if not verification.success:
                            if self.config.get("verbose_generation", True):
                                tqdm.write(
                                    f"    traj {traj_idx + 1}/{len(responses)} rejected: "
                                    f"status={verification.status}, objective_match={verification.objective_match}"
                                )
                            continue
                        if self._genprm_enabled() and not self._process_audit_passes(process_verification):
                            genprm_rejected += 1
                            if self.config.get("verbose_generation", True):
                                tqdm.write(
                                    f"    traj {traj_idx + 1}/{len(responses)} rejected by GenPRM: "
                                    f"{process_verification.get('correct_count', 0)}/"
                                    f"{process_verification.get('total_steps', 9)} correct steps"
                                )
                            continue

                        problem_id = f"{group_id}-t{traj_idx:02d}"
                        if problem_id in problem_ids:
                            continue

                        record = DatasetRecord(
                            problem_id=problem_id,
                            split="pending",
                            template_name=template_name,
                            question=question,
                            response=response,
                            code=code,
                            instance=instance,
                            reference_solution=reference.to_dict(),
                            verification=verification.to_dict(),
                            process_verification=process_verification,
                            generation_notes={
                                "generator_backend": teacher_backend_name(self.teacher_generator),
                                "teacher_model": teacher_model_name(self.teacher_generator),
                                "seed_problem_index": seed_idx,
                                "question_variant_index": variant_idx,
                                "temperature": temperatures[min(traj_idx, len(temperatures) - 1)],
                                "canonical_question": canonical_question,
                                "rag_retrieval": getattr(self.teacher_generator, "last_retrieval_contexts", []),
                            },
                        ).to_dict()
                        groups_by_id.setdefault(group_id, []).append(record)
                        self._append_jsonl(pending_path, [record])
                        template_counter[template_name] += 1
                        problem_ids.add(problem_id)
                        group_success_count += 1
                        accepted_counter += 1
                        progress.update(1)
                        progress.set_postfix(
                            seed=seed_problem_count,
                            attempted=attempted_trajectories,
                            accepted=accepted_counter,
                            api_errors=api_request_errors,
                        )
                        if self.config.get("verbose_generation", True):
                            tqdm.write(f"    traj {traj_idx + 1}/{len(responses)} accepted")
                        if accepted_counter >= target_verified:
                            break

                    if group_success_count <= 0 and self.config.get("verbose_generation", True):
                        tqdm.write("  no verified trajectories accepted for this variant")

                    self._write_generation_state(
                        state_path,
                        target_verified=target_verified,
                        accepted_counter=accepted_counter,
                        attempted_trajectories=attempted_trajectories,
                        api_request_errors=api_request_errors,
                        seed_problem_count=seed_problem_count,
                        variant_count=variant_count,
                        next_seed_index=seed_idx,
                        max_seed_problems=max_seed_problems,
                        template_counter=template_counter,
                        teacher_model=teacher_model_name(self.teacher_generator),
                        genprm_attempted=genprm_attempted,
                        genprm_rejected=genprm_rejected,
                        interrupted=False,
                        completed=accepted_counter >= target_verified,
                    )

                current_next_seed_index = seed_idx + 1
                self._write_generation_state(
                    state_path,
                    target_verified=target_verified,
                    accepted_counter=accepted_counter,
                    attempted_trajectories=attempted_trajectories,
                    api_request_errors=api_request_errors,
                    seed_problem_count=seed_problem_count,
                    variant_count=variant_count,
                    next_seed_index=current_next_seed_index,
                    max_seed_problems=max_seed_problems,
                    template_counter=template_counter,
                    teacher_model=teacher_model_name(self.teacher_generator),
                    genprm_attempted=genprm_attempted,
                    genprm_rejected=genprm_rejected,
                    interrupted=False,
                    completed=accepted_counter >= target_verified,
                )

        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            progress.close()
            self._write_generation_state(
                state_path,
                target_verified=target_verified,
                accepted_counter=accepted_counter,
                attempted_trajectories=attempted_trajectories,
                api_request_errors=api_request_errors,
                seed_problem_count=seed_problem_count,
                variant_count=variant_count,
                next_seed_index=current_next_seed_index,
                max_seed_problems=max_seed_problems,
                template_counter=template_counter,
                teacher_model=teacher_model_name(self.teacher_generator),
                genprm_attempted=genprm_attempted,
                genprm_rejected=genprm_rejected,
                interrupted=interrupted,
                completed=accepted_counter >= target_verified,
            )

            records_by_split = self._records_by_split(groups_by_id)
            for split, rows in records_by_split.items():
                write_jsonl(output_dir / f"{split}.jsonl", rows)

            summary = {
                "generator_backend": teacher_backend_name(self.teacher_generator),
                "target_verified_samples": target_verified,
                "accepted_samples": sum(len(rows) for rows in records_by_split.values()),
                "verification_rate": round(accepted_counter / attempted_trajectories, 4) if attempted_trajectories else 0.0,
                "seed_problem_count": seed_problem_count,
                "question_variant_count": variant_count,
                "attempted_trajectories": attempted_trajectories,
                "api_request_errors": api_request_errors,
                "genprm_evaluation": self._genprm_enabled(),
                "genprm_attempted": genprm_attempted,
                "genprm_rejected": genprm_rejected,
                "genprm_acceptance_rate": round((genprm_attempted - genprm_rejected) / genprm_attempted, 4)
                if genprm_attempted
                else 0.0,
                "trajectories_per_variant": len(temperatures),
                "question_variants_per_seed": int(self.config.get("question_variants_per_seed", 2)),
                "template_distribution": dict(template_counter),
                "splits": {split: len(rows) for split, rows in records_by_split.items()},
                "teacher_model": teacher_model_name(self.teacher_generator),
                "resume_from_checkpoint": resume_from_checkpoint,
                "pending_records_file": str(pending_path),
                "state_file": str(state_path),
                "interrupted": interrupted,
                "completed": accepted_counter >= target_verified,
                "recommended_range_for_7b_rag": "Start with 80-150 verified samples on an 8GB GPU, then scale after feasible/optimal rates hold up.",
            }
            write_json(output_dir / "summary.json", summary)

        return summary

    def generate(self) -> dict[str, int | float | dict | str]:
        return self._generate_teacher_dataset()
