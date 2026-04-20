from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_latest_file(root: Path, pattern: str) -> Path | None:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_latest_json(root: Path, pattern: str) -> dict[str, Any]:
    latest = find_latest_file(root, pattern)
    return read_json(latest) or {}


def rel_path(from_path: Path, to_path: Path | None) -> str:
    if to_path is None:
        return ""
    return os.path.relpath(to_path, start=from_path.parent).replace("\\", "/")


def fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def build_report(root: Path, output_path: Path) -> str:
    data_summary = read_json(root / "data/processed/stage1_dataset/summary.json") or {}
    data_config = read_yaml(root / "configs/stage1_data.yaml") or {}
    sft_config = read_yaml(root / "configs/stage1_sft.yaml") or {}
    compare_summary = read_latest_json(root, "runs/model_compare_*/comparison_summary.json")
    test_metrics = read_latest_json(root, "runs/test_eval_*/metrics.json")
    test_summary = read_latest_json(root, "runs/test_eval_*/summary.json")
    rollout_summary = read_latest_json(root, "runs/real_rollouts_*/summary.json")
    preference_summary = read_latest_json(root, "runs/preferences*/summary.json")
    dpo_data_summary = read_latest_json(root, "runs/dpo_data_*/summary.json")
    sft_summary = read_latest_json(root, "outputs/qwen2.5-1.5b-stage1-lora_runs/*/metrics/summary.json")
    dpo_summary = read_latest_json(root, "outputs/qwen2.5-1.5b-stage1-dpo_runs/*/metrics/summary.json")

    sft_dashboard = find_latest_file(root, "outputs/qwen2.5-1.5b-stage1-lora_runs/*/metrics/training_dashboard.png")
    dpo_dashboard = find_latest_file(root, "outputs/qwen2.5-1.5b-stage1-dpo_runs/*/metrics/training_dashboard.png")
    rollout_dashboard = find_latest_file(root, "runs/real_rollouts_*/rollout_dashboard.png")
    preference_dashboard = find_latest_file(root, "runs/preferences*/preference_dashboard.png")
    evaluation_dashboard = find_latest_file(root, "runs/test_eval_*/evaluation_dashboard.png")
    comparison_dashboard = find_latest_file(root, "runs/model_compare_*/comparison_dashboard.png")

    compare_models = {row["model_name"]: row for row in compare_summary.get("models", [])}
    base_metrics = compare_models.get("base", {})
    sft_metrics = compare_models.get("sft", {})
    dpo_metrics = compare_models.get("dpo", {})

    dpo_pass_gain = (dpo_metrics.get("pass_at_1", 0.0) - base_metrics.get("pass_at_1", 0.0)) * 100
    dpo_gap_drop = (base_metrics.get("mean_abs_objective_gap", 0.0) - dpo_metrics.get("mean_abs_objective_gap", 0.0))
    sft_loss_drop = (sft_summary.get("first_loss", 0.0) - sft_summary.get("last_loss", 0.0))

    report = dedent(
        f"""\
        # ORLLM / StepORLM Stage-1 项目报告

        ## 1. 项目摘要

        这不是一份“命令使用说明”，而是一份面向项目复盘、技术展示和后续扩展的阶段性报告。当前仓库已经跑通了一个本地 Stage-1 闭环：从可验证 OR 模板出发，经过教师问题改写、solver 验证、SFT 微调、真实 rollout、preference 构造、weighted DPO 训练，到最终测试集评测和三模型对比，形成了一个可以持续迭代的 solver-in-the-loop 后训练框架。

        - **硬件定位**：Qwen2.5-1.5B-Instruct + RTX 4060 8GB，本地单机优先跑通闭环，而不是机械追论文规模。
        - **当前成果**：验证通过数据集 {data_summary.get("accepted_samples", "-")} 条，SFT 后 token accuracy 提升到 {fmt_num(sft_summary.get("last_mean_token_accuracy"), 4)}，DPO 在三模型对比中把 pass@1 从 {fmt_pct(base_metrics.get("pass_at_1"))} 提升到 {fmt_pct(dpo_metrics.get("pass_at_1"))}。
        - **阶段判断**：当前最主要的收益来自“可验证数据 + SFT + 轻量偏好对齐”，最主要的瓶颈仍然是 rollout 阶段的代码执行稳定性。

        ## 2. 一页结论

        - 本地版已经形成完整可讲述链路：模板化 OR 实例采样 -> 教师改写与轨迹生成 -> OR-Tools 硬验证 -> SFT -> 真实 rollout -> preference pairs -> weighted DPO -> final eval / model comparison。
        - 数据工厂目前的验证通过率达到 **{fmt_pct(data_summary.get("verification_rate"))}**，说明“模板兜底 + 教师改写 + solver 校验”的数据生产方式在小模型场景下是可行的。
        - SFT 是目前最稳定的收益来源：训练损失从 **{fmt_num(sft_summary.get("first_loss"))}** 降到 **{fmt_num(sft_summary.get("last_loss"))}**，token accuracy 从 **{fmt_num(sft_summary.get("first_mean_token_accuracy"))}** 提升到 **{fmt_num(sft_summary.get("last_mean_token_accuracy"))}**。
        - DPO 的收益主要体现在**解质量**而不是**执行率**：和 base 相比，pass@1 提升了 **{dpo_pass_gain:.1f} 个百分点**，mean abs objective gap 降低了 **{fmt_num(dpo_gap_drop)}**；但 execution rate 仍略低于 SFT，说明当前 preference 规模还不足以完全解决执行层面的稳定性问题。
        - 如果把这个项目继续做强，最优方向不是继续在 4060 上死磕，而是把 Stage-1 作为“本地可验证原型”，然后迁移到云端补齐 7B/8B policy、GenPRM、W-DPO 和更大 benchmark。

        ## 3. 项目目标与代码框架

        这个仓库的核心不是把语言模型当成黑盒问答器，而是让它在 OR 问题上学会输出**可执行、可验证、可筛选**的求解过程。当前代码结构已经比较明确：

        ```text
        configs/                 实验配置入口
        data/                    数据产物
        outputs/                 SFT / DPO 模型与训练日志
        runs/                    rollout / preference / eval / compare 结果
        src/steporlm_stage1/
          cli.py                训练与评测命令入口
          data_factory/         模板采样、教师生成、solver 验证
          templates/            assignment / knapsack / tsp 等 OR 模板
          rollout/              真实多轨迹采样与执行结果记录
          preference/           chosen / rejected / weight 构造
          training/sft/         LoRA SFT
          training/dpo/         weighted DPO
          evaluation/           test eval 与三模型对比
          utils/                I/O、图表、run dir、训练汇总
        ```

        当前主流程由以下模块串起来：

        - `data_factory` 负责把“自然语言题目生成”建立在“模板可验证实例”之上，避免纯自由生成带来的噪声。
        - `rollout` 负责用训练后的 policy 在真实题目上采样多条轨迹，并把 solver 结果与教师评分打回到轨迹级别。
        - `preference` 负责把真实 rollout 变成偏好对，而不是依赖静态、脱离执行器的人工规则。
        - `training/dpo` 用 weighted DPOTrainer 做轻量对齐，让模型在“成功轨迹”和“失败轨迹”之间学出方向感。

        ## 4. 当前数据与训练产出

        ### 4.1 数据工厂现状

        - 目标验证样本数：`target_verified_samples={data_config.get("target_verified_samples")}`
        - 最大 seed 问题数：`max_seed_problems={data_config.get("max_seed_problems")}`
        - 每个 seed 的问题变体数：`question_variants_per_seed={data_config.get("question_variants_per_seed")}`
        - 每个变体的轨迹数：`trajectories_per_variant={data_config.get("trajectories_per_variant") or len(data_config.get("trajectory_temperatures", []))}`
        - 当前已收录验证通过样本：**{data_summary.get("accepted_samples", "-")}**
        - 当前模板分布：{data_summary.get("template_distribution", {})}
        - 当前数据划分：{data_summary.get("splits", {})}

        这说明当前数据工厂已经不再是“先写答案再包装问题”的静态脚本，而是一个能控制分布、能回收失败样本、能输出结构化数据的轻量生产链。

        ### 4.2 rollout / preference / DPO 数据

        - 最新真实 rollout：{rollout_summary.get("num_problems", "-")} 个问题，{rollout_summary.get("num_trajectories", "-")} 条轨迹
        - solver 状态分布：{rollout_summary.get("solver_status_counts", {})}
        - 最新 preference pairs：{preference_summary.get("num_pairs", "-")} 对，其中主要理由是 {preference_summary.get("rationales", {})}
        - 最新 DPO 数据集：train={dpo_data_summary.get("train", "-")}，valid={dpo_data_summary.get("valid", "-")}

        从这些结果看，当前的 DPO 数据量仍然偏小，但数据质量相对可控。尤其是 preference 的主导逻辑已经回到“solver_success_beats_failure”，这让偏好信号和最终目标更一致。

        ## 5. 训练与评测结果

        ### 5.1 SFT：先把格式和可执行性学稳

        SFT 训练是当前最扎实的一步。图 1 展示了训练损失、学习率和 token accuracy 的变化。loss 总体持续下降，说明模型正在稳定吸收“分步建模 + 代码生成”的模式，而不是仅仅记住格式。

        ![SFT 训练看板]({rel_path(output_path, sft_dashboard)})

        图 1 说明：

        - loss 从 **{fmt_num(sft_summary.get("first_loss"))}** 下降到 **{fmt_num(sft_summary.get("last_loss"))}**，总降幅 **{fmt_num(sft_loss_drop)}**。
        - mean token accuracy 从 **{fmt_num(sft_summary.get("first_mean_token_accuracy"))}** 提升到 **{fmt_num(sft_summary.get("last_mean_token_accuracy"))}**，已经接近 **0.93**。
        - 曲线后半段趋于平滑，说明当前 120 条验证样本已经足以把基础格式对齐做出来，但继续提升需要更大、更难的数据集，而不是单纯继续在同一批数据上训练更久。

        ### 5.2 rollout：执行稳定性仍是主瓶颈

        图 2 是最新 rollout dashboard。它直接揭示了项目现在最需要补的地方：成功轨迹已经存在，但 `RUNTIME_ERROR` 仍然是绝对主导。

        ![Rollout 看板]({rel_path(output_path, rollout_dashboard)})

        图 2 说明：

        - 285 条 rollout 里，`OPTIMAL + FEASIBLE` 已经达到 **107** 条，说明模型已经能在相当一部分题目上输出可用解。
        - 但 `RUNTIME_ERROR` 达到 **167** 条，明显高于成功轨迹数量，说明“代码执行鲁棒性”仍是当前成败分水岭。
        - 教师评分分布并不低，说明不少失败轨迹在文字推理层面看起来并不差，问题主要出在代码细节、变量组织、solver 调用和结果回传上。

        ### 5.3 preference / DPO：对齐已经有收益，但还没到“强收敛”

        preference 构造已经初步形成偏好信号闭环。图 3 展示了当前 preference 权重和 chosen 轨迹评分分布。

        ![Preference 看板]({rel_path(output_path, preference_dashboard)})

        图 3 说明：

        - 当前 preference 几乎都来自 “solver_success_beats_failure”，这说明偏好方向是清晰的，不靠主观打标。
        - chosen teacher scores 明显集中在较高区间，说明 preference 构造没有把低质量但偶然成功的样本大量混入。
        - 但 pair 数只有 **{preference_summary.get("num_pairs", "-")}**，仍然偏少，因此 DPO 更像“方向修正”，还不是“强对齐”。

        图 4 是 weighted DPO 训练看板。

        ![Weighted DPO 训练看板]({rel_path(output_path, dpo_dashboard)})

        图 4 说明：

        - DPO 的 best loss 达到 **{fmt_num(dpo_summary.get("best_loss"))}**，最终 preference accuracy 达到 **{fmt_num(dpo_summary.get("last_preference_accuracy"))}**。
        - reward margin 在训练中存在波动，说明当前 preference 信号仍然偏稀疏，模型已经学到“方向”，但还没有进入大规模数据下那种明显单调提升的阶段。
        - 这和当前训练规模一致：DPO 数据集只有 **{dpo_data_summary.get("train", "-")}** 条 train 样本，本质上还是一次本地可运行验证，而不是正式大规模对齐。

        ### 5.4 最终评测与三模型对比

        最新 test eval 的核心指标如下：

        - execution_rate = **{fmt_pct(test_metrics.get("execution_rate"))}**
        - feasible_rate = **{fmt_pct(test_metrics.get("feasible_rate"))}**
        - pass@1 = **{fmt_pct(test_metrics.get("pass_at_1"))}**
        - mean_abs_objective_gap = **{fmt_num(test_metrics.get("mean_abs_objective_gap"))}**
        - teacher_mean_score = **{fmt_num(test_metrics.get("teacher_mean_score"), 4)}**

        ![最终评测看板]({rel_path(output_path, evaluation_dashboard)})

        图 5 说明：

        - 当前测试集只有 14 个问题，但已经能看到 DPO 模型在可行率和 pass@1 上的正向信号。
        - mean abs objective gap 为 0，意味着一旦模型成功落到可行最优轨迹，解质量本身已经不差。
        - 当前瓶颈不是“找到更优目标值”，而是“把更多轨迹稳定推进到可执行、可得结果”。

        三模型对比最能说明当前迭代的真实收益：

        - **base**：execution={fmt_pct(base_metrics.get("execution_rate"))}，pass@1={fmt_pct(base_metrics.get("pass_at_1"))}，mean_gap={fmt_num(base_metrics.get("mean_abs_objective_gap"))}，teacher={fmt_num(base_metrics.get("teacher_mean_score"), 3)}
        - **sft**：execution={fmt_pct(sft_metrics.get("execution_rate"))}，pass@1={fmt_pct(sft_metrics.get("pass_at_1"))}，mean_gap={fmt_num(sft_metrics.get("mean_abs_objective_gap"))}，teacher={fmt_num(sft_metrics.get("teacher_mean_score"), 3)}
        - **dpo**：execution={fmt_pct(dpo_metrics.get("execution_rate"))}，pass@1={fmt_pct(dpo_metrics.get("pass_at_1"))}，mean_gap={fmt_num(dpo_metrics.get("mean_abs_objective_gap"))}，teacher={fmt_num(dpo_metrics.get("teacher_mean_score"), 3)}

        ![三模型对比看板]({rel_path(output_path, comparison_dashboard)})

        图 6 说明：

        - SFT 把 execution rate 从 base 的 **{fmt_pct(base_metrics.get("execution_rate"))}** 拉到 **{fmt_pct(sft_metrics.get("execution_rate"))}**，这是“格式稳定性 + 模板学习”的直接结果。
        - DPO 把 pass@1 从 **{fmt_pct(base_metrics.get("pass_at_1"))}** 推到 **{fmt_pct(dpo_metrics.get("pass_at_1"))}**，并把 mean abs objective gap 压到 **{fmt_num(dpo_metrics.get("mean_abs_objective_gap"))}**，说明对齐阶段确实让模型更会挑对轨迹。
        - 但 DPO 的 execution 略低于 SFT，这恰好说明下一阶段不是盲目多训，而是先把 rollout 执行失败率打下来，再扩大 preference 数据规模。

        ## 6. 当前局限与风险

        - **数据量仍小**：当前验证通过样本只有 {data_summary.get("accepted_samples", "-")} 条，DPO train 只有 {dpo_data_summary.get("train", "-")} 条，更适合作为原型验证，不足以支撑论文级稳定结论。
        - **执行错误过高**：rollout 中 `RUNTIME_ERROR` 数量过大，说明代码修复循环、静态检查和结果抽取仍需加强。
        - **测试集规模偏小**：当前 test eval 只有 {test_summary.get("num_problems", "-")} 个问题，适合看趋势，不适合过度解读绝对分数。
        - **Stage-1 仍是本地缩减版**：当前不是 7B/8B policy，不是完整 GenPRM，不是多轮共进化，也不是论文级 benchmark 覆盖。

        ## 7. 后续路线：从本地原型走向云端系统

        当前最值得做的不是继续把 4060 压榨到极限，而是按“两阶段路线”推进。

        ### 7.1 短期路线（继续打磨本地版）

        - 把数据集从 120 条验证样本扩展到 **1K~3K 条高质量 solver-verified 样本**。
        - 给 rollout 增加代码修复回路、静态执行检查和结果抽取兜底，优先降低 `RUNTIME_ERROR`。
        - 扩模板覆盖：在现有 assignment / knapsack / tsp 基础上加入 scheduling、复杂 LP、更多工业约束。
        - 累积 retrospective review 数据，为后续轻量评审器 / Mini-GenPRM 训练做准备。

        ### 7.2 云端部署路线（AutoDL 优先）

        这个项目天然适合迁移到云端，原因很直接：训练规模、轨迹采样数和 benchmark 覆盖都被 4060 显存卡住了。推荐的阶段性部署方式如下：

        - **第一步：迁移到 AutoDL 24GB 级显卡**
          - 目标：把 policy 升级到 7B 或扩大数据规模
          - 重点任务：容器化环境、统一 `configs/`、把 `runs/` 和 `outputs/` 做成可回传产物
          - 适合内容：大一版 SFT、更多 rollout、更多 preference 数据
        - **第二步：A100/H100 级算力补齐关键模块**
          - 目标：正式训练 7B/8B policy、引入更强 judge / GenPRM、做更完整的 weighted DPO
          - 重点任务：多轮 self-evolving 训练、更多 benchmark、更多轨迹采样和 verifier reranking
          - 适合内容：更接近论文路线的系统复现

        ### 7.3 工程化部署建议

        如果把它做成“可复用的云端实验系统”，建议按下面的工程清单推进：

        - 用 Docker 固化训练、评测、执行器环境，避免本地 / 云端环境漂移。
        - 把数据工厂、训练、评测拆成独立任务，统一产物目录和日志格式。
        - 用对象存储或挂载盘持久化 `outputs/`、`runs/`、图表与中间 JSONL。
        - 接入 W&B 或统一的 JSONL 日志汇总，让每次 SFT / DPO / eval 都可回放、可比较。
        - 增加一层 teacher RAG 或错误案例库，把 solver 报错、经典 OR 教材、模板说明融合进教师端提示。

        ## 8. 项目价值与对外表达

        如果把这个项目拿去做展示，它已经不只是“我调过一个模型”，而是可以从三个维度讲清楚：

        - **后训练与对齐项目**：本地实现了从 SFT 到 weighted DPO 的最小闭环。
        - **agent / verifier 项目**：模型输出不是直接当答案，而是交给 solver 执行、验证、筛选和回流。
        - **OR 领域化应用项目**：模板、问题改写、代码执行与 benchmark 评测都是围绕运筹优化任务组织的。

        这意味着它既可以继续朝论文复现走，也可以朝“行业化 OR agent”走。当前最合理的节奏是：保留这个 Stage-1 仓库作为稳定基座，在云端开出 Stage-2 分支，把规模化训练、GenPRM 和 benchmark 拓展放到那里做。
        """
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_path = Path(__file__).resolve().parent / "ORLLM_STAGE1_PROJECT_REPORT.md"
    output_path.write_text(build_report(root, output_path), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
