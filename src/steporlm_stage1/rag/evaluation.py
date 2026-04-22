from __future__ import annotations

from pathlib import Path

from steporlm_stage1.rag.index import HybridRagRetriever, tokenize
from steporlm_stage1.utils.io import load_yaml_config, read_jsonl, write_json


DEFAULT_PROBES = [
    {
        "name": "linear_programming",
        "query": "linear programming objective function constraints decision variables resource allocation production planning",
        "expected_terms": ["linear", "programming", "objective", "constraints"],
    },
    {
        "name": "integer_programming",
        "query": "integer programming binary variables assignment knapsack combinatorial optimization",
        "expected_terms": ["integer", "binary", "assignment", "knapsack"],
    },
    {
        "name": "network_tsp",
        "query": "traveling salesman problem routing tour subtour elimination shortest route",
        "expected_terms": ["traveling", "salesman", "tour", "subtour"],
    },
    {
        "name": "convex_optimization",
        "query": "convex optimization feasible set optimal solution duality constraints",
        "expected_terms": ["convex", "feasible", "optimal", "duality"],
    },
]


def evaluate_rag_retrieval(config_path: str | Path = "configs/rag_eval.yaml") -> dict:
    config = load_yaml_config(config_path)
    retriever = HybridRagRetriever(
        config.get("index_dir", "data/rag/or_books"),
        use_semantic_rerank=bool(config.get("use_semantic_rerank", True)),
        reranker_model_name_or_path=str(config.get("reranker_model", "BAAI/bge-reranker-base")),
        reranker_batch_size=int(config.get("reranker_batch_size", 8)),
        reranker_max_length=int(config.get("reranker_max_length", 512)),
        use_query_rewrite=bool(config.get("use_query_rewrite", False)),
        fail_on_reranker_error=bool(config.get("fail_on_reranker_error", False)),
    )
    top_k = int(config.get("top_k", 5))
    candidate_top_k = int(config.get("candidate_top_k", 25))
    probes = list(config.get("probes") or DEFAULT_PROBES)

    rows = []
    hit_count = 0
    coverage_values = []
    source_names: set[str] = set()
    for probe in probes:
        results = retriever.search(str(probe["query"]), candidate_top_k=candidate_top_k, top_k=top_k)
        combined = " ".join(str(item.get("text", "")) for item in results).lower()
        combined_tokens = set(tokenize(combined))
        expected_terms = [str(item).lower() for item in probe.get("expected_terms", [])]
        matched = [term for term in expected_terms if term in combined_tokens or term in combined]
        coverage = len(matched) / len(expected_terms) if expected_terms else 0.0
        coverage_values.append(coverage)
        hit = coverage > 0
        hit_count += 1 if hit else 0
        for item in results:
            if item.get("source_name"):
                source_names.add(str(item["source_name"]))
        rows.append(
            {
                "name": probe.get("name"),
                "query": probe.get("query"),
                "expected_terms": expected_terms,
                "matched_terms": matched,
                "coverage": round(coverage, 4),
                "top_chunks": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "source_name": item.get("source_name"),
                        "page_start": item.get("page_start"),
                        "keyword_score": item.get("keyword_score"),
                        "rerank_score": item.get("rerank_score"),
                        "score": item.get("score"),
                    }
                    for item in results
                ],
            }
        )

    summary = {
        "num_probes": len(probes),
        "candidate_top_k": candidate_top_k,
        "top_k": top_k,
        "semantic_rerank_enabled": bool(retriever.semantic_rerank_enabled),
        "hit_rate": round(hit_count / len(probes), 4) if probes else 0.0,
        "mean_expected_term_coverage": round(sum(coverage_values) / len(coverage_values), 4) if coverage_values else 0.0,
        "source_diversity": len(source_names),
        "probe_results": rows,
    }
    output_path = Path(config.get("output_path", "data/rag/or_books/retrieval_eval.json"))
    write_json(output_path, summary)
    summary["output_path"] = str(output_path)
    return summary


def summarize_sft_quality(dataset_dir: str | Path = "data/processed/stage1_dataset", output_path: str | Path | None = None) -> dict:
    root = Path(dataset_dir)
    dataset_rows = []
    for split in ["train", "valid", "test"]:
        path = root / f"{split}.jsonl"
        if path.exists():
            dataset_rows.extend(read_jsonl(path))

    trace_path = root / "generation_trace.jsonl"
    trace_rows = read_jsonl(trace_path) if trace_path.exists() else []

    def _is_optimal(row: dict) -> bool:
        verification = row.get("verification") or {}
        return verification.get("status") == "OPTIMAL"

    def _is_success(row: dict) -> bool:
        verification = row.get("verification") or {}
        return bool(verification.get("success"))

    accepted = len(dataset_rows)
    accepted_success = sum(1 for row in dataset_rows if _is_success(row))
    accepted_optimal = sum(1 for row in dataset_rows if _is_optimal(row))
    attempted = len(trace_rows)
    attempted_success = sum(1 for row in trace_rows if _is_success(row))
    attempted_optimal = sum(1 for row in trace_rows if _is_optimal(row))

    summary = {
        "dataset_dir": str(root),
        "accepted_samples": accepted,
        "accepted_success_rate": round(accepted_success / accepted, 4) if accepted else 0.0,
        "accepted_optimal_rate": round(accepted_optimal / accepted, 4) if accepted else 0.0,
        "attempted_trajectories": attempted,
        "attempted_success_rate": round(attempted_success / attempted, 4) if attempted else 0.0,
        "attempted_optimal_rate": round(attempted_optimal / attempted, 4) if attempted else 0.0,
        "verified_yield": round(accepted / attempted, 4) if attempted else 0.0,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary
