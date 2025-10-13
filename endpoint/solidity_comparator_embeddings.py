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
from time import time
import hashlib
from bitsec.utils.chutes_llm import chutes_client

# MongoDB
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("Warning: pymongo not available. Install with: pip install pymongo")

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
    sm = difflib.SequenceMatcher(a=base_lines, b=other_lines, autojunk=False)
    ranges = []
    total_changed = 0
    for tag, a1, a2, b1, b2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        base_range = base_orig_ranges[a1:a2]
        other_range = other_orig_ranges[b1:b2]
        base_ln = (base_range[0][0], base_range[-1][1]) if base_range else None
        other_ln = (other_range[0][0], other_range[-1][1]) if other_range else None
        total_changed += (a2 - a1) + (b2 - b1)
        ranges.append({
            "tag": tag,
            "base_line_range": base_ln,
            "other_line_range": other_ln,
            "base_snippet": "\n".join(base_lines[a1:a2]),
            "other_snippet": "\n".join(other_lines[b1:b2])
        })
    return {"ranges": ranges, "total_changed_lines": total_changed}

# ----------------------------
# Embedding utilities
# ----------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts (strings). Returns list of vectors."""
    out = []
    for text in texts:
        emb = chutes_client.embed(text)
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
# MongoDB Storage Manager
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
        self.collection.create_index("path", unique=True)
        self.collection.create_index("snapshot_hash")
        self.collection.create_index([("mtime", 1), ("size", 1)])
    
    def get_snapshot_hash(self, snapshot: List[Tuple[str, int, int]]) -> str:
        """Generate hash from file snapshot."""
        return hashlib.sha1(json.dumps(snapshot).encode()).hexdigest()
    
    def check_cache_valid(self, snapshot_hash: str) -> bool:
        """Check if cached data exists for this snapshot."""
        count = self.collection.count_documents({"snapshot_hash": snapshot_hash})
        return count > 0
    
    def load_candidates(self, snapshot_hash: str) -> List[Dict]:
        """Load candidates from MongoDB."""
        cursor = self.collection.find({"snapshot_hash": snapshot_hash})
        candidates = []
        for doc in cursor:
            # Convert MongoDB document to candidate format
            candidate = {
                "path": doc["path"],
                "norm_lines": doc["norm_lines"],
                "orig_ranges": [tuple(r) for r in doc["orig_ranges"]],  # Convert back to tuples
                "embedding": doc["embedding"]
            }
            candidates.append(candidate)
        return candidates
    
    def save_candidates(self, candidates: List[Dict], snapshot_hash: str):
        """Save candidates to MongoDB."""
        # Clear old entries with this snapshot hash
        self.collection.delete_many({"snapshot_hash": snapshot_hash})
        
        # Prepare documents for insertion
        docs = []
        for candidate in candidates:
            doc = {
                "path": candidate["path"],
                "snapshot_hash": snapshot_hash,
                "norm_lines": candidate["norm_lines"],
                "orig_ranges": candidate["orig_ranges"],  # Store as list of lists
                "embedding": candidate["embedding"]
            }
            docs.append(doc)
        
        # Bulk insert
        if docs:
            self.collection.insert_many(docs)
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()

# ----------------------------
# Worker functions for parallel processing
# ----------------------------
# <CHANGE> Fixed worker functions to properly handle data in separate processes

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
        
        other_norm_lines = cand["norm_lines"]
        other_orig_ranges = cand["orig_ranges"]
        
        diff_info = diff_normalized_sequences_with_orig(
            base_norm_lines, base_orig_ranges, 
            other_norm_lines, other_orig_ranges
        )
        
        res = {
            "candidate": cand["path"],
            "total_changed_lines": diff_info["total_changed_lines"],
            "ranges": diff_info["ranges"],
            "num_norm_base_lines": len(base_norm_lines),
            "num_norm_other_lines": len(other_norm_lines)
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
        
        # <CHANGE> Initialize MongoDB storage instead of pickle cache
        self.mongo_storage = MongoDBStorage(mongo_uri, mongo_db, mongo_collection)

        self._candidates: List[Dict] = []
        self._embeddings_index: Optional[ANNIndex] = None
        self._embeddings_vectors: Optional[List[List[float]]] = None

    def load_candidates(self, directory: str, recursive: bool = False, use_cache: bool = True, workers: int = 8):
        """
        Load candidates + compute & cache per-file embeddings (document-level) using MongoDB.
        Each candidate entry will be: {path, norm_lines, orig_ranges, embedding}
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(directory)

        # <CHANGE> Build snapshot and check MongoDB cache
        snapshot = []
        files = list(directory.rglob("*.sol")) if recursive else list(directory.glob("*.sol"))
        files = [p for p in files if p.is_file()]
        files.sort()
        for p in files:
            st = p.stat()
            snapshot.append((str(p.resolve()), int(st.st_mtime), st.st_size))
        
        snapshot_hash = self.mongo_storage.get_snapshot_hash(snapshot)

        # Check MongoDB cache
        if use_cache and self.mongo_storage.check_cache_valid(snapshot_hash):
            print(f"Loading candidates from MongoDB cache (snapshot: {snapshot_hash[:8]}...)")
            self._candidates = self.mongo_storage.load_candidates(snapshot_hash)
            
            # Build index from cached embeddings
            self._embeddings_vectors = [c["embedding"] for c in self._candidates if "embedding" in c]
            ids = [i for i, c in enumerate(self._candidates) if "embedding" in c]
            if self._embeddings_vectors:
                dim = len(self._embeddings_vectors[0])
                self._embeddings_index = ANNIndex(dim, method=self.index_method)
                self._embeddings_index.build(self._embeddings_vectors, ids)
            print(f"Loaded {len(self._candidates)} candidates from MongoDB")
            return

        # <CHANGE> Process files and compute embeddings
        print(f"Processing {len(files)} Solidity files...")
        
        def process_read(path: Path):
            try:
                src = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Error reading {path}: {e}")
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

        # Run reads in parallel
        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, workers*2)) as ex:
            for res in ex.map(process_read, files, chunksize=32):
                if res:
                    entries.append(res)

        print(f"Computing embeddings for {len(entries)} files...")
        
        # Compute embeddings in batches
        doc_texts = [e["doc_text"] for e in entries]
        embeddings = []
        B = self.embed_batch_size
        for i in range(0, len(doc_texts), B):
            batch = doc_texts[i:i+B]
            embs = embed_texts(batch)
            embeddings.extend(embs)
            if (i + B) % (B * 10) == 0:
                print(f"  Embedded {min(i + B, len(doc_texts))}/{len(doc_texts)} files")
        
        # Attach embeddings and clean up
        for e, emb in zip(entries, embeddings):
            e["embedding"] = emb
            e.pop("doc_text", None)
        
        self._candidates = entries

        # Build ANN index
        if entries and entries[0].get("embedding") is not None:
            dim = len(entries[0]["embedding"])
            self._embeddings_vectors = [c["embedding"] for c in self._candidates]
            ids = list(range(len(self._candidates)))
            self._embeddings_index = ANNIndex(dim, method=self.index_method)
            self._embeddings_index.build(self._embeddings_vectors, ids)

        # <CHANGE> Save to MongoDB instead of pickle
        print(f"Saving {len(entries)} candidates to MongoDB...")
        self.mongo_storage.save_candidates(self._candidates, snapshot_hash)
        print("Candidates saved to MongoDB")

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

        print(f"Selected {len(selected)} candidates from embedding search, running detailed diffs...")

        # <CHANGE> Fixed parallel processing with proper worker initialization
        # Create temporary pickle files for worker processes
        fd_base, base_pickle_path = tempfile.mkstemp(prefix="solcmp_base_", suffix=".pkl")
        os.close(fd_base)
        with open(base_pickle_path, "wb") as f:
            pickle.dump({
                "norm_lines": base_norm_lines, 
                "orig_ranges": base_orig_ranges
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
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_initializer,
            initargs=(candidates_pickle_path, base_pickle_path)
        ) as ex:
            for res in ex.map(_compare_candidate_worker, job_args, chunksize=4):
                if res and "error" not in res:
                    results.append(res)

        # Cleanup temporary files
        try:
            Path(base_pickle_path).unlink()
            Path(candidates_pickle_path).unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp files: {e}")

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
    # Example: Create comparator with MongoDB backend
    comparator = SolidityComparatorEmbeddings(
        index_method="faiss",  # or "sklearn"
        shortlist_k=200,
        mongo_uri="mongodb://localhost:27017",
        mongo_db="solidity_embeddings",
        mongo_collection="candidates"
    )
    
    # Load candidates from directory
    comparator.load_candidates(
        "samples/clean-codebases",
        recursive=True,
        use_cache=True,
        workers=8
    )
    
    # Compare with base file
    results = comparator.compare_with_base(
        "code1.sol",
        top=1,
        workers=4,
        include_diff=True
    )
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['candidate']}")
        print(f"   Changed lines: {result['total_changed_lines']}")
        print(f"   Base lines: {result['num_norm_base_lines']}, Other lines: {result['num_norm_other_lines']}")
    
    # Close MongoDB connection
    comparator.close()