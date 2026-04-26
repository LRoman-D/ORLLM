# ORLLM Qwen3-8B Stage-1 实验仓库

这个仓库保存了一条精简的 StepORLM 风格运筹优化建模闭环，用 Qwen3-8B 作为基础模型，通过 RAG teacher 生成可验证轨迹，再依次完成 SFT、rollout、preference pair 构造、DPO 和三模型对比评测。

当前仓库已经包含本次实验的核心代码、SFT/DPO LoRA 权重、RAG 相关数据、外部 OR 问题数据、SFT 训练数据轨迹、rollout 轨迹、preference 数据和评测结果。Qwen3-8B 原始 8B 权重没有上传，需要单独放到 `models/Qwen3-8B`。

## 主链路

1. 从外部数据源整理 OR 问题和参考答案。
2. 用冻结的 Qwen3-8B RAG teacher 生成 SFT 轨迹，并用 OR-Tools 执行结果过滤。
3. 将 teacher 轨迹转换为 chat SFT 数据，训练 SFT LoRA。
4. 使用 SFT policy 在 rollout split 上生成多条候选解。
5. 根据 solver 成功率、目标值匹配和过程评分构造 preference pairs。
6. 在 SFT adapter 基础上训练 DPO adapter。
7. 在 held-out eval split 上比较 base、SFT、DPO 三个模型。

## 目录说明

| 路径 | 内容 |
| --- | --- |
| `src/steporlm_stage1/` | 数据准备、RAG teacher、rollout、preference、SFT、DPO、评测主代码 |
| `src/steporlm_stage1/rag/` | RAG 索引构建与检索逻辑 |
| `src/steporlm_stage1/quality/` | GenPRM 风格过程评分 prompt 和解析逻辑 |
| `src/steporlm_stage1/solvers/` | OR-Tools 执行与状态辅助函数 |
| `configs/` | 当前主链路配置，包括 SFT、rollout、DPO、hard DPO 和三模型对比 |
| `data/external_or/` | 原始外部问题、划分后的 SFT/rollout/eval split、SFT teacher 轨迹 |
| `data/processed/` | 转换后的 SFT/DPO 训练数据 |
| `data/rag/or_books/` | 已构建的 RAG chunk、manifest 和索引 |
| `referencebooks/` | RAG 使用的运筹优化参考书 |
| `artifacts/qwen3-8b/` | 已保存的 SFT/DPO LoRA adapter |
| `runs/` | rollout、preference、训练指标和三模型对比结果 |
| `docs/model_comparison_summary.md` | 三模型对比结果解读 |

## 环境准备

```bash
cd /root/shared-nvme/ORLLM
bash scripts/setup_env.sh
source .venv/bin/activate
pip install -e .
```

基础模型需要单独准备：

```text
models/Qwen3-8B/
```

仓库里只保留 LoRA adapter，不包含 `models/Qwen3-8B/model-*.safetensors` 这些原始 8B 权重。

如果需要重建 RAG 索引：

```bash
python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml
```

## 复现实验

完整主链路可以逐步运行：

```bash
python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data.yaml
python -m steporlm_stage1.cli prepare-sft --input-dir data/external_or/sft_teacher --output-dir data/processed/qwen3_sft
python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml
python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout.yaml
python -m steporlm_stage1.cli build-preferences \
  --rollout-path runs/qwen3_8b/rollouts/real_rollouts.jsonl \
  --output-path runs/qwen3_8b/preferences/preferences.jsonl \
  --run-root runs/qwen3_8b \
  --run-prefix preferences \
  --require-chosen-success \
  --require-chosen-process-pass
python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data.yaml
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train.yaml
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare.yaml
```

也可以用脚本运行标准闭环：

```bash
bash scripts/run_qwen3_8b_pipeline.sh
```

hard DPO 与当前保存的 hard eval 对比链路：

```bash
bash scripts/run_hard_dpo_and_eval.sh
```

查看 SFT 数据生成进度和质量统计：

```bash
python scripts/report_generation_stats.py --config configs/stage1_data.yaml
```

## 当前实验结果

保存的三模型对比目录：

```text
runs/compare_qwen3_8b_hard_eval_20260426_112720/
```

100 道 held-out 外部 OR 题、每题 2 条采样轨迹下：

| 模型 | Execution@k | Optimal execution@k | Pass@k |
| --- | ---: | ---: | ---: |
| `qwen3_base` | 1% | 0% | 0% |
| `qwen3_sft` | 82% | 66% | 47% |
| `qwen3_dpo_hard` | 60% | 47% | 31% |

简要结论：SFT 是主要增益来源，它显著提升了模型输出可执行 OR-Tools 程序的能力；当前 hard DPO 明显强于 base，但低于 SFT，主要因为 hard preference 数据只有 74 对，筛选较严，容易过拟合到较窄的偏好分布。更详细解释见 `docs/model_comparison_summary.md`。

## `/root/clash` 代理逻辑

服务器上 `/root/clash` 是一个 mihomo/Clash 运行目录，用来给 GitHub、OpenAI/Codex、apt 等网络请求走代理。

核心文件：

| 路径 | 作用 |
| --- | --- |
| `/root/clash/mihomo` | mihomo 可执行文件 |
| `/root/clash/config.yaml` | 当前订阅转换后的 Clash 配置 |
| `/root/clash/config.yaml.bak` | 配置备份 |
| `/root/clash/geoip.metadb` | GeoIP 数据库 |
| `/root/clash/mihomo.log` | 运行日志 |
| `/root/clash/cache.db` | mihomo 缓存 |

当前配置关键点：

| 项 | 当前值/含义 |
| --- | --- |
| `mixed-port` | `7890`，HTTP/SOCKS 混合代理端口 |
| `external-controller` | `127.0.0.1:9090`，本机控制 API |
| `secret` | 空字符串，本机访问控制 API 不需要 token |
| `allow-lan` | `false`，只允许本机使用 |
| `mode` | `rule`，按规则分流 |
| `ChatGPT` 分组 | `chatgpt.com`、`openai.com` 等规则会走这个分组 |
| `节点选择` 分组 | GitHub/LFS、apt 等未特殊命中的流量通常会走这里 |

启动 mihomo：

```bash
cd /root/clash
nohup ./mihomo -d /root/clash > /root/clash/mihomo.log 2>&1 &
```

设置当前 shell 走代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

这些环境变量目前也写在 `/root/.bashrc` 里，新 shell 通常会自动生效。如果发现网络不通，先检查进程和日志：

```bash
ps aux | grep '[m]ihomo'
tail -f /root/clash/mihomo.log
curl -I https://github.com
curl -I https://chatgpt.com
```

下次根据新的梯子订阅连接 Codex/OpenAI 节点，可以按这个流程：

```bash
cd /root/clash
cp config.yaml config.yaml.bak.$(date +%Y%m%d_%H%M%S)
curl -L '你的 Clash 订阅链接' -o config.yaml
pkill -f '/root/clash/mihomo' || true
nohup ./mihomo -d /root/clash > /root/clash/mihomo.log 2>&1 &
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

如果新订阅里有专门标注 GPT、OpenAI、Codex、Gemini 等节点，建议优先把 `ChatGPT` 分组切到这类节点。控制 API 示例：

```bash
curl -X PUT 'http://127.0.0.1:9090/proxies/ChatGPT' \
  -H 'Content-Type: application/json' \
  -d '{"name":"你的 GPT/Codex 节点名"}'
```

节点名可以从配置里查：

```bash
grep -n 'ChatGPT' /root/clash/config.yaml
grep -n 'GPT\|OpenAI\|Codex\|Gemini' /root/clash/config.yaml
```

注意不要把订阅链接、节点密码、GitHub token 等敏感信息提交到仓库。`/root/clash/config.yaml` 只应该留在服务器本地。
