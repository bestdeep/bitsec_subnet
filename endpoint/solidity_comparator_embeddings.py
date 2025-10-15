#!/usr/bin/env python3
"""
solidity_comparator_embeddings.py

Comparator that uses embedding-based semantic retrieval as the first step, then
runs exact normalized diff with original-line mapping on a shortlist.

Now uses MongoDB to store embeddings and metadata instead of pickle files.

Usage (example):
    sc = SolidityComparatorEmbeddings(
        index_method="faiss",
        shortlist_k=200,
        mongo_uri="mongodb://localhost:27017",
        mongo_db="solidity_embeddings"
    )
    sc.load_candidates("/path/to/candidates", recursive=True)
    results = sc.compare_with_base("/path/to/base.sol", top=3, workers=4)
"""
from pathlib import Path
import os
import pickle
import tempfile
import re
import difflib
import json
import math
import concurrent.futures
from typing import List, Tuple, Dict, Optional
import hashlib
import time
from bitsec.utils.chutes_llm import chutes_client
from endpoint.logger import logger

# tqdm (progress bars) with safe fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, *args, **kwargs):
        return it

# MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("Warning: pymongo not available. Install with: pip install pymongo")

# ANN index: prefer faiss, fallback to sklearn
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

SKLEARN_AVAILABLE = False
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ----------------------------
# Reuse parsing & normalization that preserve original line numbers
# ----------------------------

def strip_comments_preserve_lines_with_orig(source: str) -> List[Tuple[List[int], str]]:
    """
    Remove comments while preserving original line numbers.
    Returns list of (original_line_numbers_list, line_text_without_comments)
    """
    result_lines = []
    i = 0
    n = len(source)
    lineno = 1
    cur_chars = []
    cur_orig_lines = []
    in_block = False
    in_string = False
    string_delim = None
    while i < n:
        ch = source[i]
        cur_orig_lines.append(lineno)
        if in_block:
            if ch == '*' and i + 1 < n and source[i+1] == '/':
                in_block = False
                i += 2
                continue
            if ch == '\n':
                result_lines.append((cur_orig_lines.copy(), "".join(cur_chars)))
                cur_chars = []
                cur_orig_lines = []
                lineno += 1
            i += 1
            continue
        if in_string:
            if ch == '\\' and i + 1 < n:
                cur_chars.append(ch); cur_chars.append(source[i+1]); i += 2; continue
            if ch == string_delim:
                cur_chars.append(ch); in_string = False; string_delim = None; i += 1; continue
            if ch == '\n':
                result_lines.append((cur_orig_lines.copy(), "".join(cur_chars)))
                cur_chars = []
                cur_orig_lines = []
                lineno += 1
                i += 1
                continue
            cur_chars.append(ch); i += 1; continue
        if ch == '"' or ch == "'":
            in_string = True; string_delim = ch; cur_chars.append(ch); i += 1; continue
        if ch == '/' and i + 1 < n and source[i+1] == '*':
            in_block = True; i += 2; continue
        if ch == '/' and i + 1 < n and source[i+1] == '/':
            i += 2
            while i < n and source[i] not in '\n\r':
                i += 1
            continue
        if ch == '\r':
            i += 1; continue
        if ch == '\n':
            result_lines.append((cur_orig_lines.copy(), "".join(cur_chars)))
            cur_chars = []
            cur_orig_lines = []
            lineno += 1
            i += 1
            continue
        cur_chars.append(ch)
        i += 1
    if cur_chars or lineno == 1:
        result_lines.append((cur_orig_lines.copy(), "".join(cur_chars)))
    return result_lines

_PUNCT = r"\{\}$$$$\[\]\.,;:+\-*/%=\<\>!&\|^~\?:"
_PUNCT_RE_BEFORE = re.compile(rf"\s+([{re.escape(_PUNCT)}])")
_PUNCT_RE_AFTER = re.compile(rf"([{re.escape(_PUNCT)}])\s+")
_MULTI_WHITESPACE = re.compile(r"\s+")

def normalize_line(line: str) -> str:
    s = line.strip()
    s = _MULTI_WHITESPACE.sub(" ", s)
    s = _PUNCT_RE_BEFORE.sub(r"\1", s)
    s = _PUNCT_RE_AFTER.sub(r"\1", s)
    return s.strip()

def build_normalized_sequence_with_orig(lines_with_orig: List[Tuple[List[int], str]]) -> Tuple[List[str], List[Tuple[int,int]]]:
    norm_lines = []
    orig_ranges = []
    for orig_lines, raw in lines_with_orig:
        n = normalize_line(raw)
        if n == "":
            continue
        norm_lines.append(n)
        orig_ranges.append((orig_lines[0], orig_lines[-1]))
    return norm_lines, orig_ranges

def diff_normalized_sequences_with_orig(base_lines, base_orig_ranges, other_lines, other_orig_ranges):
    """
    Compare normalized line sequences, returning detailed diff info.

    Always returns both:
      - base_source_line_range
      - other_source_line_range
      - base_snippet
      - other_snippet

    For pure insert/delete operations, we infer single-line synthetic ranges
    so that both sides are non-null.
    """

    sm = difflib.SequenceMatcher(a=base_lines, b=other_lines, autojunk=False)
    results = []
    total_changed = 0

    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == "equal":
            continue

        base_range = base_orig_ranges[a1:a2]
        other_range = other_orig_ranges[b1:b2]

        # Compute actual line ranges or synthetic fallback
        if base_range:
            base_ln = (base_range[0][0], base_range[-1][1])
        else:
            # synthetic single-line fallback
            base_ln = (base_orig_ranges[a1 - 1][1] if a1 > 0 else 1,
                       base_orig_ranges[a1 - 1][1] if a1 > 0 else 1)

        if other_range:
            other_ln = (other_range[0][0], other_range[-1][1])
        else:
            # synthetic single-line fallback
            other_ln = (other_orig_ranges[b1 - 1][1] if b1 > 0 else 1,
                        other_orig_ranges[b1 - 1][1] if b1 > 0 else 1)

        total_changed += (a2 - a1) + (b2 - b1)

        # keep human-friendly flip for insert/delete
        out_tag = tag
        if tag == "delete":
            out_tag = "insert"
        elif tag == "insert":
            out_tag = "delete"

        # normalized snippets for both sides
        base_snip = "\n".join(base_lines[a1:a2]) if base_lines[a1:a2] else ""
        other_snip = "\n".join(other_lines[b1:b2]) if other_lines[b1:b2] else ""

        results.append({
            "tag": out_tag,
            "changed_source_line_range": base_ln,     # always tuple (start, end)
            "original_source_line_range": other_ln,   # always tuple (start, end)
            "changed_source_code_snippet": base_snip,             # may be empty string
            "original_source_code_snippet": other_snip,           # may be empty string
        })

    return {
        "ranges": results,
        "total_changed_lines": total_changed
    }

# ----------------------------
# Embedding utilities
# ----------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts (strings). Returns list of vectors."""
    out = []
    for text in texts:
        max_attempts = 10
        attempt = 0
        base_delay = 1.0
        while attempt < max_attempts:
            try:
                emb = chutes_client.embed(text)
                break
            except Exception as e:
                attempt += 1
                logger.warning(f"Embedding failed (attempt {attempt}/{max_attempts}): {e}")
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay)
                if attempt == max_attempts:
                    raise
        out.append(emb)
    return out

# ----------------------------
# ANN Index wrapper (FAISS preferred)
# ----------------------------
class ANNIndex:
    def __init__(self, dim: int, method="faiss"):
        self.dim = dim
        self.method = method
        self.index = None
        self.ids = None
        if method == "faiss":
            if not FAISS_AVAILABLE:
                raise RuntimeError("faiss not available; install faiss-cpu or use method='sklearn'")
        elif method == "sklearn":
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("sklearn not available; pip install scikit-learn")
            self.nn = None
        else:
            raise ValueError("method must be 'faiss' or 'sklearn'")

    def build(self, vectors: List[List[float]], ids: List[int]):
        import numpy as np
        arr = np.array(vectors, dtype='float32')
        self.ids = list(ids)
        if self.method == "faiss":
            index = faiss.IndexFlatIP(self.dim)  # inner-product on normalized vectors -> cosine if normalized
            # normalize arr to unit length
            faiss.normalize_L2(arr)
            index.add(arr)
            self.index = index
        else:
            # sklearn NearestNeighbors (cosine metric)
            self.nn = NearestNeighbors(metric="cosine", algorithm="auto").fit(arr)
            self.index = self.nn

    def query(self, vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        import numpy as np
        v = np.array(vector, dtype='float32').reshape(1, -1)
        if self.method == "faiss":
            faiss.normalize_L2(v)
            D, I = self.index.search(v, k)
            res = []
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                cand_id = self.ids[idx]
                # faiss IndexFlatIP returns inner product between normalized vectors -> cosine similarity
                score = float(dist)
                res.append((cand_id, score))
            return res
        else:
            distances, indices = self.nn.kneighbors(v, n_neighbors=k)
            res = []
            for d, idx in zip(distances[0], indices[0]):
                cand_id = self.ids[idx]
                # sklearn returns cosine distance in [0,2]; convert to similarity
                score = 1.0 - float(d)
                res.append((cand_id, score))
            return res

# ----------------------------
# MongoDB Storage Manager (updated for incremental upserts & resume)
# ----------------------------
class MongoDBStorage:
    """Handles MongoDB operations for storing embeddings and metadata."""

    def __init__(self, mongo_uri: str, db_name: str, collection_name: str = "candidates"):
        if not MONGO_AVAILABLE:
            raise RuntimeError("pymongo not available; pip install pymongo")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Create indexes for efficient queries
        # path unique to allow per-file upsert/replace
        self.collection.create_index("path", unique=True)
        self.collection.create_index("snapshot_hash")
        self.collection.create_index([("mtime", 1), ("size", 1)])

    def get_snapshot_hash(self, snapshot: List[Tuple[str, int, int]]) -> str:
        """Generate hash from file snapshot."""
        return hashlib.sha1(json.dumps(snapshot).encode()).hexdigest()

    def check_cache_valid(self, snapshot_hash: str) -> bool:
        """Check if cached data exists for this snapshot (at least one doc)."""
        count = self.collection.count_documents({"snapshot_hash": snapshot_hash})
        return count > 0

    def load_candidates(self, snapshot_hash: str) -> List[Dict]:
        """Load candidates from MongoDB."""
        cursor = self.collection.find({"snapshot_hash": snapshot_hash})
        candidates = []
        for doc in cursor:
            candidate = {
                "path": doc["path"],
                "norm_lines": doc["norm_lines"],
                "orig_ranges": [tuple(r) for r in doc["orig_ranges"]],
                "embedding": doc.get("embedding")
            }
            candidates.append(candidate)
        return candidates

    def get_saved_paths_for_snapshot(self, snapshot_hash: str) -> set:
        """Return set of file paths already stored for this snapshot."""
        cursor = self.collection.find({"snapshot_hash": snapshot_hash}, {"path": 1})
        return {doc["path"] for doc in cursor}

    def save_candidates_batch_upsert(self, candidates: List[Dict], snapshot_hash: str):
        """Upsert a batch of candidate documents into MongoDB (idempotent)."""
        if not candidates:
            return
        # Use bulk write with upsert for idempotence
        from pymongo import UpdateOne
        ops = []
        for candidate in candidates:
            doc = {
                "path": candidate["path"],
                "snapshot_hash": snapshot_hash,
                "norm_lines": candidate["norm_lines"],
                "orig_ranges": candidate["orig_ranges"],
                "embedding": candidate.get("embedding"),
                # optionally store mtime/size if present
            }
            ops.append(UpdateOne({"path": doc["path"]}, {"$set": doc}, upsert=True))
        # perform in reasonable chunk sizes to avoid huge operation lists
        if ops:
            B = 500
            for i in range(0, len(ops), B):
                batch_ops = ops[i:i+B]
                self.collection.bulk_write(batch_ops, ordered=False)

    def close(self):
        """Close MongoDB connection."""
        self.client.close()

    def count_documents_for_snapshot(self, snapshot_hash: str) -> int:
        """Return total documents for snapshot (any doc)."""
        return self.collection.count_documents({"snapshot_hash": snapshot_hash})

    def count_embedded_for_snapshot(self, snapshot_hash: str) -> int:
        """Return number of documents that have a non-null embedding for this snapshot."""
        return self.collection.count_documents({"snapshot_hash": snapshot_hash, "embedding": {"$ne": None}})

    def get_paths_without_embedding(self, snapshot_hash: str, limit: Optional[int] = None) -> List[str]:
        """Return list of paths that lack an embedding (useful for re-embedding)."""
        q = {"snapshot_hash": snapshot_hash, "$or": [{"embedding": {"$exists": False}}, {"embedding": None}]}
        cursor = self.collection.find(q, {"path": 1})
        if limit:
            cursor = cursor.limit(limit)
        return [d["path"] for d in cursor]

    def get_sample_missing_paths(self, snapshot_hash: str, sample_n: int = 20) -> List[str]:
        """Return a small sample of missing paths for quick inspection."""
        return self.get_paths_without_embedding(snapshot_hash, limit=sample_n)

# ----------------------------
# Worker functions for parallel processing
# ----------------------------

# Global variables for worker processes
_WORKER_CANDIDATES = None
_WORKER_BASE = None

def _worker_initializer(candidates_pickle_path: str, base_pickle_path: str):
    """Initialize worker process with candidate and base data."""
    global _WORKER_CANDIDATES, _WORKER_BASE
    with open(candidates_pickle_path, "rb") as f:
        _WORKER_CANDIDATES = pickle.load(f)
    with open(base_pickle_path, "rb") as f:
        _WORKER_BASE = pickle.load(f)

def _compare_candidate_worker(args: Tuple[int, bool]) -> Dict:
    """Worker function to compare a single candidate with base."""
    idx, include_diff = args
    global _WORKER_CANDIDATES, _WORKER_BASE

    try:
        cand = _WORKER_CANDIDATES[idx]
        base_norm_lines = _WORKER_BASE["norm_lines"]
        base_orig_ranges = _WORKER_BASE["orig_ranges"]
        base_raw_lines = _WORKER_BASE.get("raw_lines", [])

        other_norm_lines = cand["norm_lines"]
        other_orig_ranges = cand["orig_ranges"]

        diff_info = diff_normalized_sequences_with_orig(
            base_norm_lines, base_orig_ranges,
            other_norm_lines, other_orig_ranges
        )

        # Attempt to read the candidate file raw lines (so we can show original source)
        try:
            with open(cand["path"], "r", encoding="utf-8") as f:
                cand_raw_text_lines = f.read().splitlines()
        except Exception:
            cand_raw_text_lines = []

        # Replace normalized snippets with original-file snippets (prefer base snippet when available)
        for r in diff_info["ranges"]:
            raw_snip = ""
            # Prefer the base original snippet (since tag denotes relation to base)
            if r.get("changed_source_line_range") and base_raw_lines:
                s, e = r["changed_source_line_range"]
                try:
                    raw_snip = "\n".join(base_raw_lines[s-1:e])
                except Exception:
                    raw_snip = ""
            # otherwise fall back to other (candidate) file raw snippet
            if not raw_snip and r.get("original_source_line_range") and cand_raw_text_lines:
                s, e = r["original_source_line_range"]
                try:
                    raw_snip = "\n".join(cand_raw_text_lines[s-1:e])
                except Exception:
                    raw_snip = ""
            # final fallback: use normalized snippet (may be empty)
            # if not raw_snip:
            #     raw_snip = r.get("normalized_changed_snippet", "")

            # r["changed_snippet"] = raw_snip
            # remove normalized_snippet key to produce cleaner output
            # r.pop("normalized_changed_snippet", None)

        res = {
            "candidate": cand["path"],
            "total_changed_lines": diff_info["total_changed_lines"],
            "ranges": diff_info["ranges"],
        }

        if include_diff:
            a_text = "\n".join(base_norm_lines)
            b_text = "\n".join(other_norm_lines)
            ud = list(difflib.unified_diff(
                a_text.splitlines(), b_text.splitlines(),
                fromfile="base (normalized)",
                tofile=Path(cand["path"]).name + " (normalized)",
                lineterm=""
            ))
            res["unified_diff_normalized"] = "\n".join(ud)

        return res
    except Exception as e:
        return {"error": str(e), "candidate": _WORKER_CANDIDATES[idx]["path"] if _WORKER_CANDIDATES else "unknown"}

# ----------------------------
# Comparator class (embeddings shortlist + accurate diff)
# ----------------------------
class SolidityComparatorEmbeddings:
    def __init__(self,
                 index_method: str = "faiss",
                 shortlist_k: int = 200,
                 embed_batch_size: int = 64,
                 mongo_uri: str = "mongodb://localhost:27017",
                 mongo_db: str = "solidity_embeddings",
                 mongo_collection: str = "candidates"):
        """
        index_method: 'faiss' or 'sklearn'
        shortlist_k: how many nearest neighbors to retrieve from embeddings before diffs
        mongo_uri: MongoDB connection URI
        mongo_db: MongoDB database name
        mongo_collection: MongoDB collection name for storing candidates
        """
        self.index_method = index_method
        self.shortlist_k = shortlist_k
        self.embed_batch_size = embed_batch_size

        # Initialize MongoDB storage
        self.mongo_storage = MongoDBStorage(mongo_uri, mongo_db, mongo_collection)

        self._candidates: List[Dict] = []
        self._embeddings_index: Optional[ANNIndex] = None
        self._embeddings_vectors: Optional[List[List[float]]] = None

    # ----------------------------
    # load_candidates (modified to persist per-batch and support resume)
    # ----------------------------
    def load_candidates(self, directory: str, recursive: bool = False, use_cache: bool = True, workers: int = 8):
        """
        Load candidates + compute & cache per-file embeddings (document-level) using MongoDB.
        Now writes embeddings incrementally and supports resuming.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)

        # build snapshot
        files = list(directory.rglob("*.sol")) if recursive else list(directory.glob("*.sol"))
        files = [p for p in files if p.is_file()]
        files.sort()
        snapshot = []
        for p in files:
            st = p.stat()
            snapshot.append((str(p.resolve()), int(st.st_mtime), st.st_size))
        snapshot_hash = self.mongo_storage.get_snapshot_hash(snapshot)

        # If fully cached and use_cache -> load from DB and build index
        if use_cache and self.mongo_storage.check_cache_valid(snapshot_hash):
            logger.info(f"Loading candidates from MongoDB cache (snapshot: {snapshot_hash[:8]}...)")
            self._candidates = self.mongo_storage.load_candidates(snapshot_hash)
            self._embeddings_vectors = [c["embedding"] for c in self._candidates if c.get("embedding") is not None]
            ids = [i for i, c in enumerate(self._candidates) if c.get("embedding") is not None]
            if self._embeddings_vectors:
                dim = len(self._embeddings_vectors[0])
                self._embeddings_index = ANNIndex(dim, method=self.index_method)
                self._embeddings_index.build(self._embeddings_vectors, ids)
            logger.info(f"Loaded {len(self._candidates)} candidates from MongoDB")

        logger.info(f"Processing {len(files)} Solidity files (snapshot {snapshot_hash[:8]}...), resume enabled")

        # Determine which paths are already saved (resume)
        saved_paths = self.mongo_storage.get_saved_paths_for_snapshot(snapshot_hash) if use_cache else set()
        if saved_paths:
            logger.info(f"Resuming: {len(saved_paths)} files already in DB for this snapshot")

        # Prepare list of to-do files (full resolved path strings)
        todo_paths = [p for p in files if str(p.resolve()) not in saved_paths]
        logger.info(f"{len(todo_paths)} files to process (skipping {len(files) - len(todo_paths)} saved)")

        # helper to read+normalize a single file (threadsafe)
        def process_read(path: Path):
            try:
                src = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                return None
            lines_with_orig = strip_comments_preserve_lines_with_orig(src)
            norm_lines, orig_ranges = build_normalized_sequence_with_orig(lines_with_orig)
            doc_text = "\n".join(norm_lines)
            return {
                "path": str(path.resolve()),
                "norm_lines": norm_lines,
                "orig_ranges": orig_ranges,
                "doc_text": doc_text
            }

        # Read and normalize all todo files using a thread pool (we keep entries in memory per batch)
        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, workers*2)) as ex:
            for res in tqdm(ex.map(process_read, [Path(p) for p in todo_paths], chunksize=32),
                            total=len(todo_paths), desc="Reading files"):
                if res:
                    entries.append(res)

        logger.info(f"Computed normalized text for {len(entries)} files; now embedding & upserting in batches")

        # Embed and upsert incrementally in batches
        B = self.embed_batch_size or 64
        for i in range(0, len(entries), B):
            batch = entries[i:i+B]
            docs_texts = [e["doc_text"] for e in batch]
            # compute embeddings (with retries inside embed_texts)
            embs = embed_texts(docs_texts)
            # attach embeddings and prepare docs for upsert
            docs_for_db = []
            for e, emb in zip(batch, embs):
                e["embedding"] = emb
                # remove doc_text payload to avoid storing it
                e.pop("doc_text", None)
                # ensure orig_ranges are serializable lists (they already are)
                docs_for_db.append({
                    "path": e["path"],
                    "norm_lines": e["norm_lines"],
                    "orig_ranges": e["orig_ranges"],
                    "embedding": e["embedding"]
                })
            # upsert the batch to MongoDB (idempotent)
            try:
                self.mongo_storage.save_candidates_batch_upsert(docs_for_db, snapshot_hash)
                logger.info(f"  Upserted {min(i+B, len(entries))}/{len(entries)} embeddings to MongoDB")
            except Exception as ex:
                logger.exception(f"Failed to upsert batch starting at {i}: {ex}")
                # on failure we continue (so partial progress remains)

        # Now load the full set of candidates for this snapshot from DB (this includes previously saved + newly saved)
        self._candidates = self.mongo_storage.load_candidates(snapshot_hash)
        # Build ANN index from embeddings present (skip docs without embeddings)
        self._embeddings_vectors = [c["embedding"] for c in self._candidates if c.get("embedding") is not None]
        ids = [i for i, c in enumerate(self._candidates) if c.get("embedding") is not None]
        if self._embeddings_vectors:
            dim = len(self._embeddings_vectors[0])
            self._embeddings_index = ANNIndex(dim, method=self.index_method)
            self._embeddings_index.build(self._embeddings_vectors, ids)

        logger.info(f"Finished loading candidates: total documents for snapshot = {len(self._candidates)}")

    def reembed_failed_docs(self, snapshot_hash: str, reembed_batch_size: int = None, workers: int = 4, embed_batch_size: Optional[int] = None):
        """
        Re-embed documents for a given snapshot that are missing embeddings, and upsert results.
        - snapshot_hash: snapshot id to target
        - reembed_batch_size: number of file-documents to batch read+embed per DB upsert step
        - workers: threadpool workers for reading files
        - embed_batch_size: embedding batch size (defaults to self.embed_batch_size)
        Returns a dict with counts and errors (if any).
        """
        if reembed_batch_size is None:
            reembed_batch_size = embed_batch_size or self.embed_batch_size or 64
        if embed_batch_size is None:
            embed_batch_size = self.embed_batch_size or 64

        missing_paths = self.mongo_storage.get_paths_without_embedding(snapshot_hash)
        if not missing_paths:
            return {"skipped": 0, "reembedded": 0, "errors": []}

        logger.info(f"Re-embedding {len(missing_paths)} documents for snapshot {snapshot_hash[:8]}...")

        # Helper to read & normalize a list of paths (threadpool friendly)
        def read_and_normalize(path_str: str):
            p = Path(path_str)
            try:
                src = p.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading {p}: {e}")
                return None
            lines_with_orig = strip_comments_preserve_lines_with_orig(src)
            norm_lines, orig_ranges = build_normalized_sequence_with_orig(lines_with_orig)
            doc_text = "\n".join(norm_lines)
            return {"path": str(p.resolve()), "norm_lines": norm_lines, "orig_ranges": orig_ranges, "doc_text": doc_text}

        reembedded = 0
        errors = []

        # read files in parallel in chunks
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, workers*2)) as ex:
            it = ex.map(read_and_normalize, missing_paths, chunksize=16)
            pending_entries = []
            for entry in tqdm(it, total=len(missing_paths), desc="Reading missing files"):
                if entry is None:
                    errors.append("read_error")
                    continue
                pending_entries.append(entry)
                # when pending reaches reembed_batch_size, compute embeddings and upsert
                if len(pending_entries) >= reembed_batch_size:
                    docs_for_db, re_count, errs = self._embed_and_upsert_entries(pending_entries, embed_batch_size, snapshot_hash)
                    reembedded += re_count
                    errors.extend(errs)
                    pending_entries = []

            # handle any remaining
            if pending_entries:
                docs_for_db, re_count, errs = self._embed_and_upsert_entries(pending_entries, embed_batch_size, snapshot_hash)
                reembedded += re_count
                errors.extend(errs)

        # After re-embedding, optionally refresh in-memory candidates/index
        # Re-load candidates for this snapshot (so in-memory view is consistent)
        self._candidates = self.mongo_storage.load_candidates(snapshot_hash)
        self._embeddings_vectors = [c["embedding"] for c in self._candidates if c.get("embedding") is not None]
        ids = [i for i, c in enumerate(self._candidates) if c.get("embedding") is not None]
        if self._embeddings_vectors:
            dim = len(self._embeddings_vectors[0])
            self._embeddings_index = ANNIndex(dim, method=self.index_method)
            self._embeddings_index.build(self._embeddings_vectors, ids)

        return {"skipped": 0, "reembedded": reembedded, "errors": errors}

    def _embed_and_upsert_entries(self, entries: List[Dict], embed_batch_size: int, snapshot_hash: str):
        """
        Helper that takes a list of entries (each has doc_text) -> computes embeddings in batches,
        upserts them using mongo_storage.save_candidates_batch_upsert, and returns summary.
        Returns (docs_for_db, reembedded_count, errors_list)
        """
        docs_for_db = []
        errors = []
        reembedded = 0
        B = embed_batch_size or self.embed_batch_size or 64
        for i in range(0, len(entries), B):
            batch = entries[i:i+B]
            texts = [e["doc_text"] for e in batch]
            try:
                embs = embed_texts(texts)
            except Exception as ex:
                logger.exception(f"Embedding batch failed: {ex}")
                errors.append(str(ex))
                continue
            for e, emb in zip(batch, embs):
                e["embedding"] = emb
                e.pop("doc_text", None)
                docs_for_db.append({
                    "path": e["path"],
                    "norm_lines": e["norm_lines"],
                    "orig_ranges": e["orig_ranges"],
                    "embedding": e["embedding"]
                })
            # upsert this batch
            try:
                # pass snapshot_hash so DB entries are consistent
                self.mongo_storage.save_candidates_batch_upsert(docs_for_db[-len(batch):], snapshot_hash)
                reembedded += len(batch)
            except Exception as ex:
                logger.exception(f"Failed to upsert re-embedded batch: {ex}")
                errors.append(str(ex))
        return docs_for_db, reembedded, errors

    def compare_with_base(self, base_path: str, top: int = 1, workers: Optional[int] = None,
                         include_diff: bool = False, shortlist_k: Optional[int] = None) -> List[Dict]:
        """Compare base file with candidates using embedding-based retrieval."""
        base_src = Path(base_path).read_text(encoding="utf-8")
        return self.compare_with_base_src(base_src, top=top, workers=workers,
                                         include_diff=include_diff, shortlist_k=shortlist_k)

    def compare_with_base_src(self, base_src: str, top: int = 1, workers: Optional[int] = None,
                             include_diff: bool = False, shortlist_k: Optional[int] = None) -> List[Dict]:
        """Compare base source code with candidates using embedding-based retrieval."""
        if not self._candidates:
            raise RuntimeError("candidates not loaded; call load_candidates() first")
        if shortlist_k is None:
            shortlist_k = self.shortlist_k

        # Compute base normalized doc string
        base_lines_with_numbers = strip_comments_preserve_lines_with_orig(base_src)
        base_norm_lines, base_orig_ranges = build_normalized_sequence_with_orig(base_lines_with_numbers)
        base_doc = "\n".join(base_norm_lines)
        base_emb = embed_texts([base_doc])[0]

        # Query ANN index
        if not self._embeddings_index:
            raise RuntimeError("ANN index not built; ensure embeddings exist")

        requested_k = min(len(self._candidates), max(1, shortlist_k * 2))
        neighbors = self._embeddings_index.query(base_emb, k=requested_k)

        # Select top shortlist_k unique candidates
        seen = set()
        selected = []
        for cid, score in neighbors:
            if cid in seen:
                continue
            seen.add(cid)
            selected.append(cid)
            if len(selected) >= shortlist_k:
                break

        logger.info(f"Selected {len(selected)} candidates from embedding search, running detailed diffs...")

        # Create temporary pickle files for worker processes, include base raw lines
        fd_base, base_pickle_path = tempfile.mkstemp(prefix="solcmp_base_", suffix=".pkl")
        os.close(fd_base)
        base_raw_lines = base_src.splitlines()
        with open(base_pickle_path, "wb") as f:
            pickle.dump({
                "norm_lines": base_norm_lines,
                "orig_ranges": base_orig_ranges,
                "raw_lines": base_raw_lines
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        fd_cand, candidates_pickle_path = tempfile.mkstemp(prefix="solcmp_candidates_", suffix=".pkl")
        os.close(fd_cand)
        with open(candidates_pickle_path, "wb") as f:
            pickle.dump(self._candidates, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Run detailed diffs in parallel
        if workers is None:
            workers = min(os.cpu_count() or 4, max(1, len(selected)))

        job_args = [(i, include_diff) for i in selected]
        results = []

        # Use tqdm to show progress for detailed diffs
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initializer,
            initargs=(candidates_pickle_path, base_pickle_path)
        ) as ex:
            for res in tqdm(ex.map(_compare_candidate_worker, job_args, chunksize=4), total=len(job_args), desc="Detailed diffs"):
                if res and "error" not in res:
                    results.append(res)

        # Cleanup temporary files
        try:
            Path(base_pickle_path).unlink()
            Path(candidates_pickle_path).unlink()
        except Exception as e:
            logger.warning(f"Warning: Could not delete temp files: {e}")

        # Sort by similarity (fewer changed lines = more similar)
        results.sort(key=lambda r: r["total_changed_lines"])
        return results[:top]

    def close(self):
        """Close MongoDB connection."""
        self.mongo_storage.close()

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import argparse
    from fastapi import FastAPI
    import uvicorn

    parser = argparse.ArgumentParser(description="SolidityComparatorEmbeddings CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # load subcommand
    p_load = sub.add_parser("load", help="Load candidates (with incremental upsert/resume).")
    p_load.add_argument("--dir", required=True, help="Directory containing .sol files")
    p_load.add_argument("--recursive", action="store_true")
    p_load.add_argument("--workers", type=int, default=8)
    p_load.add_argument("--embed-batch-size", type=int, default=64)
    p_load.add_argument("--index-method", choices=["faiss", "sklearn"], default="faiss")
    p_load.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p_load.add_argument("--mongo-db", default="solidity_embeddings")
    p_load.add_argument("--mongo-collection", default="candidates")
    p_load.add_argument("--resume", action="store_true", help="Resume using existing DB snapshot if present")

    # status subcommand (CLI status print)
    p_status = sub.add_parser("status", help="Print snapshot status for a directory or a snapshot_hash")
    p_status.add_argument("--dir", help="Directory to compute snapshot hash and show progress")
    p_status.add_argument("--snapshot-hash", help="Snapshot hash to inspect (if given, --dir is ignored)")
    p_status.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p_status.add_argument("--mongo-db", default="solidity_embeddings")
    p_status.add_argument("--mongo-collection", default="candidates")
    p_status.add_argument("--sample-missing", type=int, default=10, help="Return sample missing paths")

    # reembed-failed subcommand
    p_reembed = sub.add_parser("reembed-failed", help="Re-embed documents that lack embeddings for a snapshot")
    p_reembed.add_argument("--snapshot-hash", required=True)
    p_reembed.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p_reembed.add_argument("--mongo-db", default="solidity_embeddings")
    p_reembed.add_argument("--mongo-collection", default="candidates")
    p_reembed.add_argument("--workers", type=int, default=4)
    p_reembed.add_argument("--embed-batch-size", type=int, default=64)

    # status server (FastAPI)
    p_server = sub.add_parser("serve-status", help="Run a status HTTP server (FastAPI) to inspect snapshots")
    p_server.add_argument("--host", default="0.0.0.0")
    p_server.add_argument("--port", type=int, default=5000)
    p_server.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    p_server.add_argument("--mongo-db", default="solidity_embeddings")
    p_server.add_argument("--mongo-collection", default="candidates")

    args = parser.parse_args()

    if args.cmd == "load":
        comp = SolidityComparatorEmbeddings(
            index_method=args.index_method,
            shortlist_k=200,
            embed_batch_size=args.embed_batch_size,
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            mongo_collection=args.mongo_collection
        )
        start_time = time.time()
        logger.info("Starting load_candidates...")
        # call load_candidates; use_cache True will resume using DB entries for the snapshot
        comp.load_candidates(args.dir, recursive=args.recursive, use_cache=args.resume, workers=args.workers)
        end_time = time.time()
        logger.info(f"load_candidates finished in {end_time - start_time:.2f} seconds")
        logger.info("Load finished.")
        start_time = time.time()
        # example compare - ensure you pass a real path
        try:
            result = comp.compare_with_base("code1.sol", top=1, shortlist_k=3)  # example call to ensure it works
            end_time = time.time()
            logger.info(f"Example comparison took {end_time - start_time:.2f} seconds")
            if result:
                logger.info(json.dumps(result[0], indent=2))
        except Exception as e:
            logger.exception(f"Example comparison failed: {e}")
        comp.close()

    elif args.cmd == "status":
        storage = MongoDBStorage(args.mongo_uri, args.mongo_db, args.mongo_collection)
        if args.dir:
            # compute snapshot for directory
            d = Path(args.dir)
            files = list(d.rglob("*.sol")) if args.dir and d.exists() and d.is_dir() else []
            files = [p for p in files if p.is_file()]
            snapshot = []
            for p in files:
                st = p.stat()
                snapshot.append((str(p.resolve()), int(st.st_mtime), st.st_size))
            snapshot_hash = storage.get_snapshot_hash(snapshot)
            logger.info(f"Snapshot hash: {snapshot_hash}")
        else:
            if not args.snapshot_hash:
                logger.error("Provide --dir or --snapshot-hash")
                raise SystemExit(1)
            snapshot_hash = args.snapshot_hash

        total = storage.count_documents_for_snapshot(snapshot_hash)
        embedded = storage.count_embedded_for_snapshot(snapshot_hash)
        missing = total - embedded
        sample_missing = storage.get_sample_missing_paths(snapshot_hash, sample_n=args.sample_missing)
        out = {"snapshot_hash": snapshot_hash, "total_docs": total, "embedded": embedded, "missing": missing, "sample_missing": sample_missing}
        logger.info(json.dumps(out, indent=2))

    elif args.cmd == "reembed-failed":
        comp = SolidityComparatorEmbeddings(
            index_method="faiss",
            shortlist_k=200,
            embed_batch_size=args.embed_batch_size,
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            mongo_collection=args.mongo_collection
        )
        res = comp.reembed_failed_docs(args.snapshot_hash, reembed_batch_size=args.embed_batch_size, workers=args.workers, embed_batch_size=args.embed_batch_size)
        logger.info(json.dumps(res, indent=2))
        comp.close()

    elif args.cmd == "serve-status":
        storage = MongoDBStorage(args.mongo_uri, args.mongo_db, args.mongo_collection)
        status_app = FastAPI()

        @status_app.get("/status")
        def status(snapshot_hash: str, sample_missing: int = 10):
            total = storage.count_documents_for_snapshot(snapshot_hash)
            embedded = storage.count_embedded_for_snapshot(snapshot_hash)
            missing = total - embedded
            sample = storage.get_sample_missing_paths(snapshot_hash, sample_n=sample_missing)
            return {"snapshot_hash": snapshot_hash, "total_docs": total, "embedded": embedded, "missing": missing, "sample_missing": sample}

        @status_app.get("/status_by_dir")
        def status_by_dir(dir: str, sample_missing: int = 10):
            d = Path(dir)
            if not d.exists() or not d.is_dir():
                return {"error": "dir not found", "dir": dir}
            files = list(d.rglob("*.sol"))
            files = [p for p in files if p.is_file()]
            snapshot = []
            for p in files:
                st = p.stat()
                snapshot.append((str(p.resolve()), int(st.st_mtime), st.st_size))
            snapshot_hash = storage.get_snapshot_hash(snapshot)
            total = storage.count_documents_for_snapshot(snapshot_hash)
            embedded = storage.count_embedded_for_snapshot(snapshot_hash)
            missing = total - embedded
            sample = storage.get_sample_missing_paths(snapshot_hash, sample_n=sample_missing)
            return {"snapshot_hash": snapshot_hash, "total_docs": total, "embedded": embedded, "missing": missing, "sample_missing": sample}

        logger.info(f"Starting status server on {args.host}:{args.port} ...")
        uvicorn.run(status_app, host=args.host, port=args.port)

    else:
        parser.print_help()
