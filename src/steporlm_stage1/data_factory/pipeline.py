from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from steporlm_stage1.data_factory.zhipu_teacher import ZhipuTeacherGenerator
from steporlm_stage1.executors.python_executor import PythonCodeExecutor
from steporlm_stage1.schemas import DatasetRecord
from steporlm_stage1.templates.registry import TEMPLATE_REGISTRY
from steporlm_stage1.utils.io import ensure_dir, load_yaml_config, write_json, write_jsonl
from steporlm_stage1.utils.text import extract_python_code


class Stage1DataFactory:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.rng = random.Random(config["seed"])
        self.executor = PythonCodeExecutor(timeout_seconds=config.get("timeout_seconds", 20))
        self.teacher_generator = ZhipuTeacherGenerator.from_env(
            model_env_var=str(config.get("data_model_env_var", "ZHIPUAI_DATA_MODEL")),
            timeout_seconds=int(config.get("request_timeout_seconds", 60)),
            max_retries=int(config.get("request_max_retries", 2)),
            retry_backoff_seconds=float(config.get("request_retry_backoff_seconds", 1.5)),
            fallback_model=str(config.get("data_model_fallback", "glm-4.5-air")),
        )

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

    def _generate_teacher_dataset(self) -> dict[str, int | float | dict | str]:
        if self.teacher_generator is None:
            raise RuntimeError("ZHIPUAI_API_KEY is not set. Set the environment variable before running generate-dataset.")

        output_dir = ensure_dir(self.config["output_dir"])
        target_verified = int(self.config.get("target_verified_samples", self.config.get("num_samples", 120)))
        max_seed_problems = int(
            self.config.get("max_seed_problems", max(40, math.ceil(target_verified / max(1, self.config.get("question_variants_per_seed", 2)))))
        )
        records_by_group: list[list[dict]] = []
        trace_rows: list[dict] = []
        template_counter: Counter[str] = Counter()
        accepted_counter = 0
        attempted_trajectories = 0
        api_request_errors = 0
        seed_problem_count = 0
        variant_count = 0
        temperatures = self._temperature_schedule()
        progress = tqdm(total=target_verified, desc="Generating teacher-verified dataset")

        for seed_idx in range(max_seed_problems):
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
                group_rows: list[dict] = []
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
                    trace_rows.append(
                        {
                            "seed_problem_index": seed_idx,
                            "question_variant_index": variant_idx,
                            "trajectory_index": traj_idx,
                            "template_name": template_name,
                            "question": question,
                            "response_preview": response[:1200],
                            "code_preview": code[:1200],
                            "verification": verification.to_dict(),
                        }
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

                    record = DatasetRecord(
                        problem_id=f"{group_id}-t{traj_idx:02d}",
                        split="pending",
                        template_name=template_name,
                        question=question,
                        response=response,
                        code=code,
                        instance=instance,
                        reference_solution=reference.to_dict(),
                        verification=verification.to_dict(),
                        generation_notes={
                            "generator_backend": "zhipu_teacher",
                            "teacher_model": self.teacher_generator.client.config.model,
                            "seed_problem_index": seed_idx,
                            "question_variant_index": variant_idx,
                            "temperature": temperatures[min(traj_idx, len(temperatures) - 1)],
                            "canonical_question": canonical_question,
                        },
                    ).to_dict()
                    group_rows.append(record)
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

                if group_rows:
                    records_by_group.append(group_rows)
                    template_counter[template_name] += len(group_rows)
                elif self.config.get("verbose_generation", True):
                    tqdm.write("  no verified trajectories accepted for this variant")

        progress.close()

        split_names = self._split_group_names(len(records_by_group))
        records_by_split: dict[str, list[dict]] = {name: [] for name in self.config["splits"].keys()}
        for split, group_rows in zip(split_names, records_by_group):
            for row in group_rows:
                row["split"] = split
            records_by_split[split].extend(group_rows)

        for split, rows in records_by_split.items():
            write_jsonl(output_dir / f"{split}.jsonl", rows)
        if trace_rows:
            write_jsonl(output_dir / "generation_trace.jsonl", trace_rows)

        summary = {
            "generator_backend": "zhipu_teacher",
            "target_verified_samples": target_verified,
            "accepted_samples": sum(len(rows) for rows in records_by_split.values()),
            "verification_rate": round(accepted_counter / attempted_trajectories, 4) if attempted_trajectories else 0.0,
            "seed_problem_count": seed_problem_count,
            "question_variant_count": variant_count,
            "attempted_trajectories": attempted_trajectories,
            "api_request_errors": api_request_errors,
            "trajectories_per_variant": len(temperatures),
            "question_variants_per_seed": int(self.config.get("question_variants_per_seed", 2)),
            "template_distribution": dict(template_counter),
            "splits": {split: len(rows) for split, rows in records_by_split.items()},
            "teacher_model": self.teacher_generator.client.config.model,
            "recommended_range_for_1p5b": "80-150 verified samples; default target 120",
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    def generate(self) -> dict[str, int | float | dict | str]:
        return self._generate_teacher_dataset()
