# StepORLM Stage-1: Cloud branch for deployment & training

这个仓库当前 `cloud` 分支用于“新服务器快速部署 + 直接复现当前版本训练结果”。

当前分支已一并纳入：

- `data/processed/stage1_dataset/train.jsonl`：当前版本生成的 Stage-1 数据集
- `data/processed/stage1_sft/train.jsonl`：当前版本使用的 SFT 训练数据
- `outputs/qwen2.5-7b-stage1-lora/`：当前版本训练好的 7B LoRA 适配器
- `outputs/qwen2.5-7b-stage1-lora_runs/sft_train_20260421_004434/`：对应训练运行目录与指标产物

未纳入 Git 的内容：

- 基座模型 `models/Qwen/Qwen2.5-7B-Instruct` 本体
- HuggingFace / Torch 缓存
- 其余中间输出、日志、检查点

这样做的目的是让你在新服务器只需要补齐基座模型与环境，就能直接继续 rollout / DPO / evaluation，或者基于现成 SFT adapter 继续训练。

## 1. 项目结构说明

Git 默认管理代码与配置；本次 `cloud` 分支额外跟踪了指定的数据集与 LoRA 成品权重，便于跨服务器迁移。

- `src/` & `tools/`: 源代码
- `configs/`: 训练与运行配置文件
- `scripts/`: 环境初始化与示例运行脚本
- `requirements.txt` / `requirements-train.txt` / `environment.yml`: 环境依赖
- `.gitignore`: 默认忽略大文件，仅对白名单数据和当前 LoRA 权重放开

## 2. 目录职责划分


| 目录         | 内容             | Git 管理 | 说明                       |
| ---------- | -------------- | ------ | ------------------------ |
| `src/`     | 业务逻辑、算法实现      | 是      | 核心代码                     |
| `configs/` | YAML 配置文件      | 是      | 实验参数                     |
| `data/`    | 数据集与中间数据       | **部分** | 当前分支已纳入 SFT/Stage-1 关键数据 |
| `models/`  | 基座模型           | **否**  | 新服务器需自行下载或挂载             |
| `outputs/` | 训练产物           | **部分** | 当前分支已纳入 7B SFT LoRA 成品   |
| `logs/`    | 运行日志、nohup 输出  | **否**  | 持久化存储                    |
| `.cache/`  | HF 缓存、Torch 缓存 | **否**  | 临时/持久缓存                  |


## 3. 新服务器快速部署

### 3.1 拉取 cloud 分支

```bash
git clone -b cloud https://github.com/LRoman-D/ORLLM.git
cd ORLLM
```

### 3.2 安装环境

**方法 A: Conda**

```bash
conda env create -f environment.yml
conda activate orllm
pip install -r requirements-train.txt
pip install -e .
```

**方法 B: Pip + Venv**

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e .
```

### 3.3 配置环境变量

在根目录创建 `.env`：

```env
ZHIPUAI_API_KEY=你的智谱Key
ZHIPUAI_MODEL=glm-4.5
ZHIPUAI_DATA_MODEL=glm-4.5-air
```

如需把数据、输出、缓存放到独立挂载盘，可额外设置：

```bash
export PROJECT_DATA_DIR="/mnt/data/orllm_data"
export PROJECT_MODEL_DIR="/mnt/models/orllm_models"
export PROJECT_OUTPUT_DIR="/mnt/output/orllm_outputs"
export PROJECT_LOG_DIR="/mnt/output/orllm_logs"
export PROJECT_CACHE_DIR="/mnt/cache/orllm_cache"
```

### 3.4 准备基座模型

当前配置依赖：

- `models/Qwen/Qwen2.5-7B-Instruct`

你只需在新服务器保证该路径存在即可；当前仓库中的该目录可以是实际目录，也可以是指向真实模型目录的符号链接。

## 4. 当前版本可直接复现/继续训练的内容

### 4.1 直接使用已有 SFT LoRA

当前现成适配器路径：

- `outputs/qwen2.5-7b-stage1-lora`

这意味着你在新服务器拉取 `cloud` 分支后，只要补齐基座模型，就可以直接：

- 执行 rollout
- 构造 preference pairs
- 生成 DPO 数据
- 继续 DPO 训练
- 做 base / SFT / DPO 对比评测

### 4.2 重新执行当前版本 SFT

```bash
python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml
```

SFT 配置默认使用：

- 数据：`data/processed/stage1_sft/train.jsonl`
- 基座：`models/Qwen/Qwen2.5-7B-Instruct`
- 输出：`outputs/qwen2.5-7b-stage1-lora`

### 4.3 一键继续 7B DPO 流程

```bash
bash scripts/run_7b_dpo_pipeline.sh
```

该脚本会：

1. 从 `data/processed/stage1_dataset/train.jsonl` 切出 `train_500.jsonl`
2. 基于当前 SFT adapter 执行 rollout
3. 构造 preference pairs
4. 生成 DPO 数据
5. 启动 DPO 训练
6. 执行 base / SFT / DPO 三模型对比

## 5. 核心工作流命令

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

## 6. 远程运行建议

- 建议使用 `tmux` 或 `screen` 长时运行训练。
- 可使用 `nohup` 后台运行，例如：
`nohup bash scripts/run_7b_dpo_pipeline.sh > logs/run_$(date +%Y%m%d).log 2>&1 &`
- `data/`、`outputs/`、`models/`、`.cache/` 都支持通过环境变量映射到持久化目录。

## 7. 代码与路径约定

项目内部路径通过 `steporlm_stage1.utils.paths` 统一管理。代码和 YAML 配置中使用的 `data/`, `models/`, `outputs/`, `runs/`, `logs/` 等相对路径会自动映射到对应目录。