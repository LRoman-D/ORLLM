# StepORLM Stage-1

This repo builds stage-1 OR modeling data for Qwen2.5-7B on a single 8 GB GPU. The default teacher path is now local Qwen2.5-7B-Instruct with a lightweight RAG index over `referencebooks/`. The previous Zhipu API teacher is still preserved and can be selected with `configs/stage1_data_zhipu_api.yaml`.

## Layout

| Path | Purpose |
| --- | --- |
| `src/steporlm_stage1/` | Data generation, rollout, evaluation, SFT and DPO code |
| `src/steporlm_stage1/rag/` | Reference-book extraction, hybrid retrieval (BM25 candidate recall + semantic rerank), retrieval/SFT quality summaries |
| `configs/` | RAG, data generation, SFT, rollout, DPO and comparison configs |
| `referencebooks/` | Local OR/optimization books used to build the RAG index |
| `models/Qwen2.5-7B-Instruct/` | Local base model path expected by the default configs |
| `data/processed/` | Generated stage-1 datasets |
| `outputs/` | LoRA adapters and training outputs |

## Environment

```bash
conda env create -f environment.yml
conda activate orllm
pip install -r requirements.txt
pip install -e .
```

On Linux, `bitsandbytes` enables 4-bit loading for Qwen2.5-7B. On native Windows, 4-bit `bitsandbytes` is usually unavailable; use WSL/Linux for full 7B generation on a 4060 8 GB card, or provide a GPTQ/AWQ model and adjust `teacher_model_path`.

Optional API teacher credentials can live in `.env`:

```env
ZHIPUAI_API_KEY=...
ZHIPUAI_MODEL=glm-4.5
ZHIPUAI_DATA_MODEL=glm-4.5-air
```

## RAG Teacher Workflow

Build the local reference index:

```bash
python -m steporlm_stage1.cli build-rag-index --config-path configs/rag_index.yaml
python -m steporlm_stage1.cli evaluate-rag --config-path configs/rag_eval.yaml
```

Generate SFT teacher trajectories with local Qwen+RAG:

```bash
python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data.yaml
python -m steporlm_stage1.cli summarize-sft-quality --dataset-dir data/processed/stage1_dataset
python -m steporlm_stage1.cli prepare-sft
```

Keep using the API teacher when needed:

```bash
python -m steporlm_stage1.cli generate-dataset --config-path configs/stage1_data_zhipu_api.yaml
```

## Training And Evaluation

```bash
python -m steporlm_stage1.cli train-lora --config-path configs/stage1_sft.yaml
python -m steporlm_stage1.cli generate-real-rollouts --config-path configs/stage1_real_rollout.yaml
python -m steporlm_stage1.cli build-preferences --rollout-path outputs/runs/real_rollouts_7b/real_rollouts.jsonl --require-chosen-success
python -m steporlm_stage1.cli prepare-dpo --config-path configs/stage1_dpo_data.yaml
python -m steporlm_stage1.cli train-dpo --config-path configs/stage1_dpo_train.yaml
python -m steporlm_stage1.cli compare-models --config-path configs/stage1_compare.yaml
python -m steporlm_stage1.cli rag-ablation --config-path configs/rag_ablation_20_semantic_v2.yaml
```

The comparison command generates a RAG-backed benchmark and reports execution, feasible/optimal, pass@1, objective-gap and model comparison metrics. Use this to compare the RAG teacher data against the base model and later SFT/DPO adapters.

## Config Notes

- Redundant `*_7b*` config copies were removed; keep using the single canonical files under `configs/`.
- `rag_ablation_20_semantic_v2.yaml` is the default semantic rerank A/B test config.

## Current Local Index Check

The current `referencebooks/` index build extracted 5 PDF files into 6164 chunks. The `.chm` book is listed as unsupported by the lightweight extractor. The default retrieval probes reached hit rate `1.0` and mean expected-term coverage `1.0`; full quality still needs the slower Qwen+RAG data-generation run and downstream model comparison.
