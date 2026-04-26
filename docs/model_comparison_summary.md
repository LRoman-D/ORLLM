# Qwen3-8B Stage-1 Model Comparison

Evaluation run: `runs/compare_qwen3_8b_hard_eval_20260426_112720`

Dataset: first 100 held-out external OR questions from `data/external_or/splits/eval_questions.jsonl`.

Sampling: 2 trajectories per question, `temperature=0.45`, `top_p=0.9`, `max_new_tokens=1200`.

## Overall Results

| Model | Execution@k | Optimal execution@k | Pass@k | Avg successful trajectories | Avg optimal trajectories |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3_base` | 1% | 0% | 0% | 0.00 | 0.00 |
| `qwen3_sft` | 82% | 66% | 47% | 0.81 | 1.17 |
| `qwen3_dpo_hard` | 60% | 47% | 31% | 0.49 | 0.70 |

`qwen3_sft` is the best model in this run. It converts the base model's natural-language reasoning into executable OR-Tools-style programs much more reliably, raising execution from 1% to 82% and pass@k from 0% to 47%.

`qwen3_dpo_hard` improves strongly over the base model but regresses from SFT. The likely reason is that the hard DPO set is small and narrow: 74 selected pairs versus 671 accepted SFT samples, with 252 process-score-gap pairs filtered out by `min_weight=0.9`. Training accuracy reached 1.0 on that small preference set, but the DPO objective appears to have over-specialized to the hard pair distribution instead of improving held-out generation.

## By Source

| Model | IndustryOR | MAMO Complex | MAMO Easy | NL4Opt | ReSocratic |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qwen3_base` pass@k | 0% | 0% | 0% | 0% | 0% |
| `qwen3_sft` pass@k | 9% | 31% | 76% | 78% | 21% |
| `qwen3_dpo_hard` pass@k | 9% | 8% | 50% | 67% | 14% |

The easier structured sources, especially MAMO Easy and NL4Opt, benefit most from SFT because the learned format is enough to produce executable code and objective values close to the references. IndustryOR, MAMO Complex, and ReSocratic remain harder because they require more robust formulation choices and are less forgiving of small modeling mistakes.

## Interpretation

The base model usually mixes explanation with code or emits code that is not directly executable, so solver-based evaluation almost always fails before optimization quality can be measured.

SFT is doing the heavy lifting: teacher-verified trajectories teach the policy the full pattern of parsing the problem, constructing OR-Tools variables and constraints, solving, and printing a checkable objective.

The current hard DPO pass is not yet additive. Its preference data is useful signal, but this run used too few pairs and a stricter pair filter, so it likely reduced response diversity and shifted away from the broader SFT behavior. The next DPO attempt should use more preference pairs, mix process-score and solver-success rationales, and evaluate both standard and hard DPO adapters before replacing the SFT baseline.
