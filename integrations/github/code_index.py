"""Semantic code search over repo contents.

Chunks each source file into overlapping line windows, embeds them via the
same nomic-embed-text model used for chat memory, and stores them in a
dedicated ChromaDB collection called `code_chunks`. Queries hit ChromaDB
directly and return the top-K chunks with metadata so the LLM can jump
straight to the right file range.

Freshness is tracked by storing the file's blob SHA in the chunk metadata
(`sha`). Before a search we consult the repo-map sidecar for the current
head SHA and drop / re-index any file whose stored SHA no longer matches.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Reuse the ChromaDB client + embedding function already initialised for chat memory
from vector_store import _client as _chroma_client, _chroma_embeddings

from . import repo_map as _repo_map

# ── Config ───────────────────────────────────────────────────────────────────

COLLECTION_NAME = "code_chunks"

# nomic-embed-text has an ~8 K context window. Real-world limit is around
# 4-6 KB of input before it 500s with "input length exceeds context length".
# We chunk small enough to stay well inside that budget.
CHUNK_LINES = int(os.environ.get("CODE_INDEX_CHUNK_LINES", "80"))
CHUNK_OVERLAP = int(os.environ.get("CODE_INDEX_CHUNK_OVERLAP", "15"))
INDEX_WORKERS = int(os.environ.get("CODE_INDEX_WORKERS", "4"))
MAX_CHUNK_BYTES = int(os.environ.get("CODE_INDEX_MAX_CHUNK_BYTES", "6000"))
EMBED_MAX_RETRIES = int(os.environ.get("CODE_INDEX_EMBED_RETRIES", "2"))

_collection_lock = threading.Lock()
_collection = None


def _get_collection():
    """Get-or-create the code_chunks collection (lazy, thread-safe)."""
    global _collection
    with _collection_lock:
        if _collection is None:
            _collection = _chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=_chroma_embeddings,
                metadata={"hnsw:space": "cosine"},
            )
        return _collection


# ── Chunker ─────────────────────────────────────────────────────────────────

def _chunk_content(content: str, path: str, lang: str) -> list[dict]:
    """Split file content into overlapping line windows."""
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return []

    chunks: list[dict] = []
    step = max(CHUNK_LINES - CHUNK_OVERLAP, 1)
    for start in range(0, total, step):
        end = min(start + CHUNK_LINES, total)
        block = "\n".join(lines[start:end])
        if len(block) > MAX_CHUNK_BYTES:
            block = block[:MAX_CHUNK_BYTES]
        chunks.append({
            "start_line": start + 1,
            "end_line": end,
            "text": block,
        })
        if end == total:
            break
    return chunks


def _chunk_id(repo: str, branch: str, path: str, start: int, end: int) -> str:
    safe_repo = repo.replace("/", "__")
    safe_path = path.replace("/", "__").replace(".", "_")
    safe_branch = branch.replace("/", "_")
    return f"{safe_repo}__{safe_branch}__{safe_path}__L{start}-L{end}"


def _file_key(repo: str, branch: str, path: str) -> dict:
    """Metadata filter to select all chunks for one file."""
    return {
        "$and": [
            {"repo": {"$eq": repo}},
            {"branch": {"$eq": branch}},
            {"path": {"$eq": path}},
        ]
    }


# ── Indexer ─────────────────────────────────────────────────────────────────

def _delete_file_chunks(repo: str, branch: str, path: str) -> int:
    coll = _get_collection()
    try:
        coll.delete(where=_file_key(repo, branch, path))
    except Exception:
        pass
    return 0


def _index_single_file(repo: str, branch: str, path: str, lang: str, sha: str,
                       content: str, log=print) -> int:
    """Chunk + embed + upsert one file. Returns number of chunks written."""
    _delete_file_chunks(repo, branch, path)
    chunks = _chunk_content(content, path, lang)
    if not chunks:
        return 0

    ids = [_chunk_id(repo, branch, path, c["start_line"], c["end_line"]) for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "repo": repo,
            "branch": branch,
            "path": path,
            "lang": lang,
            "sha": sha,
            "start_line": c["start_line"],
            "end_line": c["end_line"],
        }
        for c in chunks
    ]
    coll = _get_collection()
    last_exc: Optional[Exception] = None
    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)
            return len(chunks)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            # Ollama's "context length" error is a permanent per-chunk failure —
            # drop the offending chunks (any > MAX_CHUNK_BYTES) and retry.
            if "context length" in msg.lower():
                keep = [i for i, d in enumerate(docs) if len(d) <= MAX_CHUNK_BYTES]
                if len(keep) < len(docs):
                    ids = [ids[i] for i in keep]
                    docs = [docs[i] for i in keep]
                    metas = [metas[i] for i in keep]
                    log(f"[code_index] dropped {len(chunks) - len(keep)} oversized chunks from {path}")
                    if not docs:
                        return 0
                    continue
            if attempt < EMBED_MAX_RETRIES:
                import time as _t
                _t.sleep(0.5 * (attempt + 1))
                continue
            break
    raise last_exc if last_exc else RuntimeError("upsert failed")


def index_repo(repo: str, branch: str, force: bool = False, log=print,
               limit: Optional[int] = None) -> dict:
    """Build (or refresh) the semantic index for a repo+branch.

    Uses the repo-map sidecar as the source of truth for which files to index.
    Skips files whose SHA is already stored (unless force=True).
    """
    from .github_tools import _clean_repo
    clean = _clean_repo(repo)

    # Always refresh the sidecar first — cheap if head SHA hasn't moved
    sidecar = _repo_map.build_repo_map(repo, branch, force=False, log=log)
    head_sha = sidecar["head_sha"]

    files = [f for f in sidecar["files"] if f.get("sha")]
    if limit is not None:
        files = files[:limit]

    log(f"[code_index] indexing {len(files)} files for {clean}@{branch} (head={head_sha[:8]})")

    coll = _get_collection()

    # Discover which files already have chunks with the right SHA
    already_indexed: set[tuple[str, str]] = set()
    if not force:
        try:
            existing = coll.get(
                where={
                    "$and": [
                        {"repo": {"$eq": clean}},
                        {"branch": {"$eq": branch}},
                    ]
                },
                include=["metadatas"],
            )
            for meta in existing.get("metadatas") or []:
                if meta and meta.get("sha") and meta.get("path"):
                    already_indexed.add((meta["path"], meta["sha"]))
        except Exception as exc:
            log(f"[code_index] warning while reading existing index: {exc}")

    to_process = []
    skipped = 0
    for f in files:
        if not force and (f["path"], f["sha"]) in already_indexed:
            skipped += 1
            continue
        to_process.append(f)

    log(f"[code_index] {skipped} files already up to date, {len(to_process)} to (re)index")

    if not to_process:
        return {
            "repo": clean,
            "branch": branch,
            "head_sha": head_sha,
            "files_indexed": 0,
            "files_skipped": skipped,
            "chunks_written": 0,
        }

    # Phase A: fetch file contents in parallel (network I/O).
    fetched: list[tuple[dict, str]] = []
    fetched_lock = threading.Lock()

    def _fetch_worker(f):
        content = _repo_map._fetch_content(clean, f["path"], head_sha)
        if content is not None:
            with fetched_lock:
                fetched.append((f, content))

    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as pool:
        futures = [pool.submit(_fetch_worker, f) for f in to_process]
        for i, _ in enumerate(as_completed(futures), start=1):
            if i % 50 == 0:
                log(f"[code_index]   fetched {i}/{len(to_process)} file bodies")

    log(f"[code_index] fetched {len(fetched)}/{len(to_process)} file bodies; embedding serially")

    # Phase B: chunk + embed + upsert one file at a time (Ollama is single-writer friendly).
    total_chunks = 0
    processed = 0
    for f, content in fetched:
        try:
            n = _index_single_file(
                clean, branch, f["path"], f.get("lang", "text"),
                f["sha"], content, log=log,
            )
            total_chunks += n
            processed += 1
            if processed % 25 == 0:
                log(f"[code_index]   {processed}/{len(fetched)} files embedded, {total_chunks} chunks written")
        except Exception as exc:
            log(f"[code_index] embed error on {f['path']}: {exc}")

    return {
        "repo": clean,
        "branch": branch,
        "head_sha": head_sha,
        "files_indexed": processed,
        "files_skipped": skipped,
        "chunks_written": total_chunks,
    }


# ── Search ──────────────────────────────────────────────────────────────────

def search(query: str, repo: Optional[str] = None, branch: Optional[str] = None,
           top_k: int = 8, path_filter: Optional[str] = None) -> list[dict]:
    """Semantic search across indexed code chunks."""
    coll = _get_collection()

    where_terms: list[dict] = []
    if repo:
        from .github_tools import _clean_repo
        where_terms.append({"repo": {"$eq": _clean_repo(repo)}})
    if branch:
        where_terms.append({"branch": {"$eq": branch}})

    where = None
    if len(where_terms) == 1:
        where = where_terms[0]
    elif len(where_terms) > 1:
        where = {"$and": where_terms}

    res = coll.query(query_texts=[query], n_results=top_k, where=where)

    hits: list[dict] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for idx, (mid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        if path_filter and meta and path_filter.lower() not in meta.get("path", "").lower():
            continue
        hits.append({
            "rank": len(hits) + 1,
            "id": mid,
            "score": round(1.0 - float(dist), 4) if dist is not None else None,
            "repo": meta.get("repo"),
            "branch": meta.get("branch"),
            "path": meta.get("path"),
            "lang": meta.get("lang"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "sha": meta.get("sha"),
            "preview": (text or "")[:800],
        })
    return hits


def stats(repo: Optional[str] = None, branch: Optional[str] = None) -> dict:
    coll = _get_collection()
    where = None
    if repo or branch:
        from .github_tools import _clean_repo
        terms = []
        if repo:
            terms.append({"repo": {"$eq": _clean_repo(repo)}})
        if branch:
            terms.append({"branch": {"$eq": branch}})
        where = terms[0] if len(terms) == 1 else {"$and": terms}
    try:
        total = coll.count() if not where else len(coll.get(where=where, include=[]).get("ids") or [])
    except Exception as exc:
        return {"error": str(exc), "collection": COLLECTION_NAME}
    return {
        "collection": COLLECTION_NAME,
        "repo": repo,
        "branch": branch,
        "chunk_count": total,
    }
