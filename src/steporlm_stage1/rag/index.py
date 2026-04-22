from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from steporlm_stage1.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z0-9_+\-]+|[\u4e00-\u9fff]")
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
SUPPORTED_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf"}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "are",
    "is",
    "to",
    "of",
    "in",
    "on",
    "by",
    "as",
    "be",
    "or",
    "an",
    "a",
}
STRUCTURAL_MARKERS = (
    "definition",
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "proof",
    "decision variable",
    "decision variables",
    "objective",
    "constraint",
    "constraints",
    "subject to",
    "maximize",
    "minimize",
    "模型",
    "定义",
    "定理",
    "约束",
    "目标函数",
)


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    source_path: str
    source_name: str
    page_start: int | None
    page_end: int | None
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS and len(token.strip()) > 1]


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = []
    for block in re.split(r"\n{2,}", text):
        cleaned = re.sub(r"\s+", " ", block).strip()
        if cleaned:
            blocks.append(cleaned)
    return "\n\n".join(blocks)


def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF extraction. Install with `pip install pypdf`.") from exc

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        cleaned = _clean_text(text)
        if cleaned:
            pages.append((idx, cleaned))
    return pages


def _extract_text_file(path: Path) -> list[tuple[int | None, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = _clean_text(raw)
    return [(None, cleaned)] if cleaned else []


def _count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _is_structural_paragraph(paragraph: str) -> bool:
    lowered = paragraph.lower()
    return any(marker in lowered for marker in STRUCTURAL_MARKERS)


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    words = _count_words(paragraph)
    if words <= max_words:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?。！？])\s+", paragraph)
    if len(sentences) <= 1:
        sentence = paragraph
        word_tokens = WORD_RE.findall(sentence)
        if not word_tokens:
            return []
        chunks = []
        for idx in range(0, len(word_tokens), max_words):
            chunks.append(" ".join(word_tokens[idx : idx + max_words]))
        return chunks

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = _count_words(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_words = sentence_words
        else:
            current.append(sentence)
            current_words += sentence_words
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_paragraphs(text: str, max_paragraph_words: int) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\n+", text):
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        paragraphs.extend(_split_long_paragraph(cleaned, max_words=max_paragraph_words))
    return paragraphs


def _merge_structural_headers(paragraphs: list[str]) -> list[str]:
    merged: list[str] = []
    idx = 0
    while idx < len(paragraphs):
        current = paragraphs[idx]
        current_words = _count_words(current)
        if _is_structural_paragraph(current) and current_words <= 40 and idx + 1 < len(paragraphs):
            combined = f"{current}\n{paragraphs[idx + 1]}".strip()
            merged.append(combined)
            idx += 2
        else:
            merged.append(current)
            idx += 1
    return merged


def _is_safe_boundary(paragraphs: list[str], end_idx: int) -> bool:
    if end_idx <= 0 or end_idx >= len(paragraphs):
        return True
    prev_para = paragraphs[end_idx - 1]
    next_para = paragraphs[end_idx]
    if _is_structural_paragraph(prev_para) or _is_structural_paragraph(next_para):
        return False
    return True


def _choose_chunk_end(
    paragraphs: list[str],
    counts: list[int],
    start_idx: int,
    *,
    target_words: int,
    min_words: int,
    max_words: int,
) -> tuple[int, int]:
    end_idx = start_idx
    total_words = 0

    best_safe_end = None
    best_safe_total = 0
    best_safe_delta = float("inf")

    best_fallback_end = None
    best_fallback_total = 0
    best_fallback_delta = float("inf")

    while end_idx < len(paragraphs):
        total_words += counts[end_idx]
        end_idx += 1

        if min_words <= total_words <= max_words:
            delta = abs(total_words - target_words)
            if delta < best_fallback_delta:
                best_fallback_end = end_idx
                best_fallback_total = total_words
                best_fallback_delta = delta
            if _is_safe_boundary(paragraphs, end_idx) and delta < best_safe_delta:
                best_safe_end = end_idx
                best_safe_total = total_words
                best_safe_delta = delta

        if total_words > max_words:
            break

    if best_safe_end is not None:
        return best_safe_end, best_safe_total
    if best_fallback_end is not None:
        return best_fallback_end, best_fallback_total

    if end_idx >= len(paragraphs):
        tail_total = sum(counts[start_idx:])
        return len(paragraphs), tail_total

    force_end = min(start_idx + 1, len(paragraphs))
    return force_end, counts[start_idx]


def _window_text_by_words(
    text: str,
    *,
    chunk_target_words: int,
    chunk_min_words: int,
    chunk_max_words: int,
    overlap_ratio: float,
) -> list[str]:
    paragraphs = _merge_structural_headers(_split_paragraphs(text, max_paragraph_words=chunk_max_words))
    if not paragraphs:
        return []

    counts = [_count_words(paragraph) for paragraph in paragraphs]
    chunks: list[str] = []
    start_idx = 0

    while start_idx < len(paragraphs):
        end_idx, total_words = _choose_chunk_end(
            paragraphs,
            counts,
            start_idx,
            target_words=chunk_target_words,
            min_words=chunk_min_words,
            max_words=chunk_max_words,
        )

        chunk_text = "\n\n".join(paragraphs[start_idx:end_idx]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end_idx >= len(paragraphs):
            break

        overlap_words = max(1, int(total_words * overlap_ratio))
        next_start = end_idx
        covered_words = 0
        while next_start > start_idx + 1 and covered_words < overlap_words:
            next_start -= 1
            covered_words += counts[next_start]

        if next_start <= start_idx:
            start_idx = end_idx
        else:
            start_idx = next_start

    return chunks


def _normalize_chunking_config(
    *,
    chunk_target_words: int,
    chunk_min_words: int,
    chunk_max_words: int,
    overlap_ratio: float,
    legacy_chunk_size: int | None,
    legacy_overlap: int | None,
) -> tuple[int, int, int, float]:
    target = int(chunk_target_words)
    min_words = int(chunk_min_words)
    max_words = int(chunk_max_words)
    ratio = float(overlap_ratio)

    if legacy_chunk_size:
        estimated_words = max(60, int(legacy_chunk_size / 4))
        target = estimated_words
        min_words = max(60, int(estimated_words * 0.7))
        max_words = max(min_words + 20, int(estimated_words * 1.35))

    if legacy_overlap and legacy_chunk_size and legacy_chunk_size > 0:
        ratio = float(legacy_overlap) / float(legacy_chunk_size)

    ratio = max(0.1, min(0.2, ratio))

    if min_words > target:
        min_words = target
    if target > max_words:
        max_words = target
    if min_words < 60:
        min_words = 60
    if max_words < min_words:
        max_words = min_words

    return target, min_words, max_words, ratio


def build_rag_index(
    source_dir: str | Path = "referencebooks",
    output_dir: str | Path = "data/rag/or_books",
    *,
    chunk_target_words: int = 450,
    chunk_min_words: int = 300,
    chunk_max_words: int = 600,
    overlap_ratio: float = 0.15,
    min_chunk_chars: int = 200,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict:
    source_root = Path(source_dir)
    output_root = ensure_dir(output_dir)
    chunk_rows: list[dict] = []
    skipped_files = []
    chunk_counter = 0

    chunk_target_words, chunk_min_words, chunk_max_words, overlap_ratio = _normalize_chunking_config(
        chunk_target_words=chunk_target_words,
        chunk_min_words=chunk_min_words,
        chunk_max_words=chunk_max_words,
        overlap_ratio=overlap_ratio,
        legacy_chunk_size=chunk_size,
        legacy_overlap=overlap,
    )

    source_files = [path for path in sorted(source_root.rglob("*")) if path.is_file()]
    supported_files = [path for path in source_files if path.suffix.lower() in SUPPORTED_SUFFIXES]
    unsupported_files = [str(path) for path in source_files if path.suffix.lower() not in SUPPORTED_SUFFIXES]

    pages_by_file: defaultdict[Path, list[tuple[int | None, str]]] = defaultdict(list)
    for path in tqdm(supported_files, desc="Extracting reference text"):
        try:
            if path.suffix.lower() == ".pdf":
                pages = _extract_pdf_pages(path)
            else:
                pages = _extract_text_file(path)
            pages_by_file[path].extend(pages)
        except Exception as exc:  # noqa: BLE001
            skipped_files.append({"path": str(path), "reason": str(exc)})

    for path, pages in pages_by_file.items():
        if not pages:
            continue
        for page_num, text in pages:
            chunks = _window_text_by_words(
                text,
                chunk_target_words=chunk_target_words,
                chunk_min_words=chunk_min_words,
                chunk_max_words=chunk_max_words,
                overlap_ratio=overlap_ratio,
            )
            for chunk_text in chunks:
                if len(chunk_text) < min_chunk_chars:
                    continue
                chunk_id = f"chunk-{chunk_counter:07d}"
                chunk_counter += 1
                chunk_rows.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        source_path=str(path),
                        source_name=path.stem,
                        page_start=page_num,
                        page_end=page_num,
                        text=chunk_text,
                    ).to_dict()
                )

    doc_freqs: Counter[str] = Counter()
    lengths = []
    for row in chunk_rows:
        tokens = tokenize(row["text"])
        lengths.append(len(tokens))
        doc_freqs.update(set(tokens))

    manifest = {
        "version": 2,
        "source_dir": str(source_root),
        "output_dir": str(output_root),
        "num_source_files": len(source_files),
        "num_supported_files": len(supported_files),
        "num_chunks": len(chunk_rows),
        "avg_chunk_tokens": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        "chunk_target_words": chunk_target_words,
        "chunk_min_words": chunk_min_words,
        "chunk_max_words": chunk_max_words,
        "overlap_ratio": overlap_ratio,
        "min_chunk_chars": min_chunk_chars,
        "unsupported_files": unsupported_files,
        "skipped_files": skipped_files,
    }
    write_jsonl(output_root / "chunks.jsonl", chunk_rows)
    write_json(output_root / "index.json", {"doc_freqs": dict(doc_freqs), "num_docs": len(chunk_rows), "avgdl": manifest["avg_chunk_tokens"]})
    write_json(output_root / "manifest.json", manifest)
    return manifest


class KeywordRagRetriever:
    """Stage-1 keyword retriever (BM25) for candidate recall."""

    def __init__(self, index_dir: str | Path = "data/rag/or_books") -> None:
        self.index_dir = Path(index_dir)
        chunks_path = self.index_dir / "chunks.jsonl"
        index_path = self.index_dir / "index.json"
        if not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"RAG index not found at {self.index_dir}. Run build-rag-index first.")
        self.chunks = read_jsonl(chunks_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.doc_freqs = {str(k): int(v) for k, v in index.get("doc_freqs", {}).items()}
        self.num_docs = max(1, int(index.get("num_docs", len(self.chunks))))
        self.avgdl = max(1.0, float(index.get("avgdl", 1.0)))
        self._chunk_tokens: list[list[str]] = [tokenize(row.get("text", "")) for row in self.chunks]
        self._chunk_term_counts: list[Counter[str]] = [Counter(tokens) for tokens in self._chunk_tokens]

    def search(self, query: str, *, top_k: int = 25, min_score: float = 0.0) -> list[dict]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        scores = []
        for idx, counts in enumerate(self._chunk_term_counts):
            score = self._bm25_score(query_terms, counts, len(self._chunk_tokens[idx]))
            if score > min_score:
                scores.append((score, idx))
        scores.sort(reverse=True, key=lambda item: item[0])
        results = []
        for rank, (score, idx) in enumerate(scores[:top_k], start=1):
            row = dict(self.chunks[idx])
            row["keyword_score"] = round(score, 6)
            row["keyword_rank"] = rank
            row["score"] = row["keyword_score"]
            results.append(row)
        return results

    @staticmethod
    def format_context(
        results: list[dict],
        *,
        max_chars: int = 5000,
        max_chunk_chars: int = 1400,
        include_scores: bool = False,
        include_chunk_id: bool = True,
    ) -> str:
        parts = []
        used = 0
        for idx, item in enumerate(results, start=1):
            page = item.get("page_start")
            location = f"{item.get('source_name')}"
            if page is not None:
                location += f", page {page}"
            id_part = f"{item.get('chunk_id')} | " if include_chunk_id else ""
            if include_scores:
                header = (
                    f"chunk{idx} [{id_part}{location} | "
                    f"keyword={item.get('keyword_score', item.get('score'))} | rerank={item.get('rerank_score', item.get('score'))}]"
                )
            else:
                header = f"chunk{idx} [{id_part}{location}]"
            text = str(item.get("text", "")).strip()
            if max_chunk_chars > 0 and len(text) > max_chunk_chars:
                text = text[:max_chunk_chars].rstrip()
            block = f"{header}\n{text}"
            if used + len(block) > max_chars:
                remaining = max_chars - used
                if remaining <= 200:
                    break
                block = block[:remaining].rstrip()
            parts.append(block)
            used += len(block)
            if used >= max_chars:
                break
        return "\n\n".join(parts)

    def _bm25_score(self, query_terms: list[str], counts: Counter[str], doc_len: int) -> float:
        k1 = 1.5
        b = 0.75
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq <= 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / self.avgdl)
            score += idf * freq * (k1 + 1) / denom
        return score


class QueryRewriter(Protocol):
    def rewrite(self, query: str) -> str:
        ...


class HeuristicOrQueryRewriter:
    """Lightweight query rewrite interface (optional, off by default)."""

    _OR_TERMS = (
        "linear programming",
        "integer programming",
        "mixed integer programming",
        "objective function",
        "decision variables",
        "constraints",
        "operations research",
        "optimization model",
        "库存",
        "选址",
        "运输",
        "约束",
    )

    _EXPANSION = "operations research optimization model linear programming integer programming objective function decision variables constraints"

    def rewrite(self, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", query).strip()
        lowered = cleaned.lower()
        if not cleaned:
            return cleaned
        if any(term in lowered for term in self._OR_TERMS):
            return cleaned
        return f"{cleaned} {self._EXPANSION}"


class HfCrossEncoderReranker:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        *,
        batch_size: int = 8,
        max_length: int = 512,
        use_fp16_on_cuda: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.batch_size = max(1, int(batch_size))
        self.max_length = max(128, int(max_length))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        if self.device.type == "cuda" and use_fp16_on_cuda:
            self.model.half()
        self.model.eval()

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        import torch

        if not texts:
            return []
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits

            if logits.ndim == 1:
                batch_scores = logits
            elif logits.shape[-1] == 1:
                batch_scores = logits[:, 0]
            else:
                batch_scores = logits[:, -1]
            scores.extend(float(item) for item in batch_scores.detach().cpu())
        return scores


class HybridRagRetriever:
    """Hybrid retrieval: keyword candidate recall -> semantic rerank -> top-k context."""

    def __init__(
        self,
        index_dir: str | Path = "data/rag/or_books",
        *,
        use_semantic_rerank: bool = True,
        reranker_model_name_or_path: str = "BAAI/bge-reranker-base",
        reranker_batch_size: int = 8,
        reranker_max_length: int = 512,
        rerank_keyword_blend: float = 0.75,
        use_query_rewrite: bool = False,
        query_rewriter: QueryRewriter | None = None,
        fail_on_reranker_error: bool = False,
    ) -> None:
        self.keyword_retriever = KeywordRagRetriever(index_dir)
        self.index_dir = self.keyword_retriever.index_dir
        self.chunks = self.keyword_retriever.chunks

        self.query_rewriter = query_rewriter
        if self.query_rewriter is None and use_query_rewrite:
            self.query_rewriter = HeuristicOrQueryRewriter()

        self.reranker = None
        self.semantic_rerank_enabled = bool(use_semantic_rerank)
        self.rerank_keyword_blend = min(1.0, max(0.0, float(rerank_keyword_blend)))
        self.last_query = ""
        self.last_query_used = ""

        if self.semantic_rerank_enabled:
            try:
                self.reranker = HfCrossEncoderReranker(
                    model_name_or_path=reranker_model_name_or_path,
                    batch_size=reranker_batch_size,
                    max_length=reranker_max_length,
                )
            except Exception as exc:  # noqa: BLE001
                if fail_on_reranker_error:
                    raise RuntimeError(f"Failed to initialize reranker `{reranker_model_name_or_path}`: {exc}") from exc
                self.semantic_rerank_enabled = False
                self.reranker = None
                warnings.warn(
                    f"Semantic reranker init failed, fallback to keyword-only retrieval: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def rewrite_query(self, query: str) -> str:
        if self.query_rewriter is None:
            return query
        rewritten = self.query_rewriter.rewrite(query)
        return rewritten if rewritten and rewritten.strip() else query

    def search(
        self,
        query: str,
        *,
        candidate_top_k: int = 25,
        top_k: int = 5,
        min_keyword_score: float = 0.0,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        self.last_query = query
        query_used = self.rewrite_query(query)
        self.last_query_used = query_used

        candidates = self.keyword_retriever.search(query_used, top_k=max(top_k, int(candidate_top_k)), min_score=min_keyword_score)
        if not candidates:
            return []

        if not self.semantic_rerank_enabled or self.reranker is None:
            results = []
            for item in candidates[:top_k]:
                row = dict(item)
                row["rerank_score"] = row.get("keyword_score", row.get("score", 0.0))
                row["final_score"] = row["rerank_score"]
                row["score"] = round(float(row["final_score"]), 6)
                row["query_used"] = query_used
                results.append(row)
            return results

        rerank_scores = self.reranker.score_pairs(query_used, [str(item.get("text", "")) for item in candidates])
        keyword_values = [float(item.get("keyword_score", 0.0)) for item in candidates]
        rerank_values = [float(rerank_scores[idx]) if idx < len(rerank_scores) else 0.0 for idx in range(len(candidates))]

        def _minmax_norm(values: list[float]) -> list[float]:
            if not values:
                return []
            v_min = min(values)
            v_max = max(values)
            if v_max <= v_min:
                return [0.5 for _ in values]
            return [(val - v_min) / (v_max - v_min) for val in values]

        keyword_norm = _minmax_norm(keyword_values)
        rerank_norm = _minmax_norm(rerank_values)
        scored: list[dict] = []
        for idx, item in enumerate(candidates):
            row = dict(item)
            row["rerank_score"] = rerank_values[idx]
            row["keyword_score_norm"] = keyword_norm[idx]
            row["rerank_score_norm"] = rerank_norm[idx]
            row["final_score"] = (
                self.rerank_keyword_blend * row["rerank_score_norm"]
                + (1.0 - self.rerank_keyword_blend) * row["keyword_score_norm"]
            )
            row["query_used"] = query_used
            scored.append(row)

        scored.sort(key=lambda row: (row.get("final_score", 0.0), row.get("rerank_score", 0.0), row.get("keyword_score", 0.0)), reverse=True)
        final_rows = scored[:top_k]
        for rank, row in enumerate(final_rows, start=1):
            row["rerank_rank"] = rank
            row["score"] = round(float(row.get("final_score", 0.0)), 6)
        return final_rows

    def format_context(
        self,
        results: list[dict],
        *,
        max_chars: int = 5000,
        max_chunk_chars: int = 1400,
        include_scores: bool = False,
        include_chunk_id: bool = True,
    ) -> str:
        return self.keyword_retriever.format_context(
            results,
            max_chars=max_chars,
            max_chunk_chars=max_chunk_chars,
            include_scores=include_scores,
            include_chunk_id=include_chunk_id,
        )
