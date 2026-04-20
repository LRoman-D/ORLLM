# StepORLM Stage-1: Local & Remote Engineering Scaffold

这个仓库是面向 StepORLM 一阶段的工程化实验框架。当前版本已针对“本地开发 + 远程 Linux 服务器训练”场景完成工程化整理。

## 1. 项目结构说明

Git 仅管理代码和元数据，大文件通过本地持久化目录管理。

- `src/` & `tools/`: 源代码
- `configs/`: 训练与运行配置文件
- `scripts/`: 环境初始化与示例运行脚本
- `requirements.txt` & `environment.yml`: 环境依赖
- `.gitignore`: 核心配置，确保 `data/`, `models/`, `outputs/`, `logs/`, `.cache/` 等不进入 Git。

## 2. 目录职责划分

| 目录 | 内容 | Git 管理 | 说明 |
| :--- | :--- | :--- | :--- |
| `src/` | 业务逻辑、算法实现 | 是 | 核心代码 |
| `configs/` | YAML 配置文件 | 是 | 实验参数 |
| `data/` | 原始数据、生成的数据集 | **否** | 持久化存储 |
| `models/` | 基座模型、Adapter 权重 | **否** | 持久化存储 |
| `outputs/` | 训练产物、Runs、Checkpoints | **否** | 持久化存储 |
| `logs/` | 运行日志、nohup 输出 | **否** | 持久化存储 |
| `.cache/` | HF 缓存、Torch 缓存 | **否** | 临时/持久缓存 |

## 3. 快速上手 (Linux 服务器)

### 3.1 环境安装

**方法 A: Conda (推荐)**
```bash
conda env create -f environment.yml
conda activate orllm
```

**方法 B: Pip + Venv**
```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

### 3.2 环境变量配置

在根目录创建 `.env` 文件：
```env
ZHIPUAI_API_KEY=你的智谱Key
ZHIPUAI_MODEL=glm-4.5
ZHIPUAI_DATA_MODEL=glm-4.5-air
```

### 3.3 运行示例

仓库内置了统一路径映射逻辑。你可以通过环境变量指定持久化目录位置（默认在项目根目录下）：

```bash
# 使用脚本运行示例评测
bash scripts/run_example.sh

# 或者手动指定目录挂载点（适用于多云/多盘环境）
export PROJECT_DATA_DIR="/mnt/data/orllm_data"
export PROJECT_OUTPUT_DIR="/mnt/output/orllm_results"
python3 -m steporlm_stage1.cli evaluate-model --config-path configs/stage1_test_eval.yaml
```

## 4. 核心工作流命令

按以下顺序运行以完成 Stage-1 全流程：

1. **生成数据集**: `steporlm-local generate-dataset --config-path configs/stage1_data.yaml`
2. **准备 SFT**: `steporlm-local prepare-sft`
3. **SFT 训练**: `python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml`
4. **生成 Rollout**: `python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout.yaml`
5. **构造偏好**: `steporlm-local build-preferences`
6. **准备 DPO**: `steporlm-local prepare-dpo --config-path configs/stage1_dpo_data.yaml`
7. **DPO 训练**: `python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train.yaml`
8. **模型评测**: `python -m steporlm_stage1.cli evaluate-model --config-path configs/stage1_test_eval.yaml`
9. **三模型对比**: `python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare.yaml`

## 5. 远程运行建议

- **后台运行**: 建议使用 `tmux` 或 `screen` 开启会话。
- **防止中断**: 使用 `nohup` 配合输出重定向，例如：
  `nohup bash scripts/run_example.sh > logs/run_$(date +%Y%m%d).log 2>&1 &`
- **Git 使用**: 本地修改代码/配置后，`git push`，服务器 `git pull` 即可同步逻辑，数据和模型保持在服务器持久化盘。

## 6. 代码修改原则

项目内部路径通过 `steporlm_stage1.utils.paths` 统一管理。代码和 YAML 配置中使用的 `data/`, `models/`, `outputs/` 等相对路径会自动映射到对应的持久化目录。
