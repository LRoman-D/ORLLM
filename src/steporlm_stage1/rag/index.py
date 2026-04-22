from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from steporlm_stage1.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def _iter_source_pages(source_dir: Path) -> Iterable[tuple[Path, int | None, str]]:
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == ".pdf":
            for page_num, text in _extract_pdf_pages(path):
                yield path, page_num, text
        else:
            for page_num, text in _extract_text_file(path):
                yield path, page_num, text


def _window_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_rag_index(
    source_dir: str | Path = "referencebooks",
    output_dir: str | Path = "data/rag/or_books",
    *,
    chunk_size: int = 1800,
    overlap: int = 220,
    min_chunk_chars: int = 200,
) -> dict:
    source_root = Path(source_dir)
    output_root = ensure_dir(output_dir)
    chunk_rows: list[dict] = []
    skipped_files = []
    chunk_counter = 0

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
            for part_idx, chunk_text in enumerate(_window_text(text, chunk_size=chunk_size, overlap=overlap)):
                if len(chunk_text) < min_chunk_chars:
                    continue
                chunk_id = f"chunk-{chunk_counter:07d}"
                chunk_counter += 1
                chunk_rows.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        source_path=str(path),
                        source_name=path.name,
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
        "version": 1,
        "source_dir": str(source_root),
        "output_dir": str(output_root),
        "num_source_files": len(source_files),
        "num_supported_files": len(supported_files),
        "num_chunks": len(chunk_rows),
        "avg_chunk_tokens": round(sum(lengths) / len(lengths), 4) if lengths else 0.0,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "min_chunk_chars": min_chunk_chars,
        "unsupported_files": unsupported_files,
        "skipped_files": skipped_files,
    }
    write_jsonl(output_root / "chunks.jsonl", chunk_rows)
    write_json(output_root / "index.json", {"doc_freqs": dict(doc_freqs), "num_docs": len(chunk_rows), "avgdl": manifest["avg_chunk_tokens"]})
    write_json(output_root / "manifest.json", manifest)
    return manifest


class KeywordRagRetriever:
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

    def search(self, query: str, *, top_k: int = 4, min_score: float = 0.0) -> list[dict]:
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
        for score, idx in scores[:top_k]:
            row = dict(self.chunks[idx])
            row["score"] = round(score, 6)
            results.append(row)
        return results

    def format_context(self, results: list[dict], *, max_chars: int = 5000) -> str:
        parts = []
        used = 0
        for item in results:
            page = item.get("page_start")
            location = f"{item.get('source_name')}"
            if page is not None:
                location += f", page {page}"
            header = f"[{item.get('chunk_id')} | {location} | score={item.get('score')}]"
            text = str(item.get("text", "")).strip()
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
