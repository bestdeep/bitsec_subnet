#!/usr/bin/env python3
"""
solidity_comparator_fast_loadopt.py

Fast comparator with optimized loading for very large candidate sets (~10k-50k).
Features:
 - Persistent cache of preprocessed candidates (validated by file mtimes & sizes).
 - Parallel preprocessing with ProcessPoolExecutor (faster for CPU-heavy normalization + hashing).
 - Optional smaller "worker" pickle to reduce worker memory footprint.
 - MinHash-based shortlist and accurate difflib SequenceMatcher on shortlist.

Usage:
    sc = SolidityComparatorFastLoadOpt(minhash_k=64, shortlist_k=200)
    sc.load_candidates("/path/to/candidates", recursive=True)
    results = sc.compare_with_base("/path/to/base.sol", top=3, workers=8)
"""

from pathlib import Path
import re
import difflib
import pickle
import tempfile
import os
import hashlib
import random
import json
from typing import List, Tuple, Dict, Optional
import concurrent.futures
import multiprocessing
from functools import partial
from time import time
from itertools import repeat
from tqdm import tqdm
try:
    import xxhash
    _USE_XXHASH = True
except Exception:
    _USE_XXHASH = False

def _process_file_for_load(p_path_str: str, exclude_set: set, a_list: List[int], b_list: List[int], PRIME: int) -> Optional[Dict]:
    """
    Top-level helper used by ProcessPoolExecutor.map to process a single file.
    Returns the candidate dict or None on failure / excluded path.
    """
    try:
        p = Path(p_path_str)
        pr = str(p.resolve())
        if pr in exclude_set:
            return None
        src = p.read_text(encoding="utf-8")
    except Exception:
        return None

    try:
        lines_with_nums = strip_comments_preserve_lines_with_orig(src)
        norm_lines, orig_ranges = build_normalized_sequence_with_orig(lines_with_nums)
        line_hashes = [_hash_line_to_int(ln) for ln in norm_lines]
        minhash_sig = compute_minhash_signature(line_hashes, a_list, b_list, PRIME)
        return {
            "path": pr,
            "norm_lines": norm_lines,
            "orig_ranges": orig_ranges,
            "line_hashes": line_hashes,
            "minhash_sig": minhash_sig
        }
    except Exception:
        return None

# ----------------------------
# Parsing & normalization (line-aware, preserve original line numbers)
# ----------------------------
def strip_comments_preserve_lines_with_orig(source: str) -> List[Tuple[List[int], str]]:
    result_lines: List[Tuple[List[int], str]] = []
    i = 0
    n = len(source)
    lineno = 1
    cur_chars: List[str] = []
    cur_orig_lines: List[int] = []

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

_PUNCT = r"\{\}\(\)\[\]\.,;:+\-*/%=\<\>!&\|^~\?:"
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
    norm_lines: List[str] = []
    orig_ranges: List[Tuple[int,int]] = []
    for orig_lines, raw in lines_with_orig:
        n = normalize_line(raw)
        if n == "":
            continue
        norm_lines.append(n)
        orig_ranges.append((orig_lines[0], orig_lines[-1]))
    return norm_lines, orig_ranges

# ----------------------------
# MinHash helpers
# ----------------------------
def _hash_line_to_int(line: str) -> int:
    # prefer xxhash for speed if available
    if _USE_XXHASH:
        # xxh64 returns 64-bit int
        return xxhash.xxh64(line.encode("utf-8")).intdigest()
    else:
        h = hashlib.sha1(line.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big", signed=False)

def make_minhash_funcs(k: int, seed: int = 12345):
    rnd = random.Random(seed)
    PRIME = 18446744073709551557
    a = [rnd.randrange(1, PRIME - 1) for _ in range(k)]
    b = [rnd.randrange(0, PRIME - 1) for _ in range(k)]
    return a, b, PRIME

def compute_minhash_signature(line_hashes: List[int], a_list, b_list, PRIME) -> List[int]:
    if not line_hashes:
        return [2**63 - 1] * len(a_list)
    sig = []
    for a, b in zip(a_list, b_list):
        minv = None
        for x in line_hashes:
            val = ((a * (x % PRIME)) + b) % PRIME
            if minv is None or val < minv:
                minv = val
        sig.append(minv if minv is not None else 2**63 - 1)
    return sig

def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    assert len(sig_a) == len(sig_b)
    eq = sum(1 for x,y in zip(sig_a, sig_b) if x == y)
    return eq / len(sig_a)

# ----------------------------
# Diff using original ranges
# ----------------------------
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
# Worker-global storage for initializer
# ----------------------------
_WORKER_CANDIDATES = None
_WORKER_BASE = None

def _worker_initializer(cands_pickle_path: str, base_pickle_path: Optional[str]):
    global _WORKER_CANDIDATES, _WORKER_BASE
    try:
        with open(cands_pickle_path, "rb") as f:
            _WORKER_CANDIDATES = pickle.load(f)
    except Exception as e:
        _WORKER_CANDIDATES = []
        raise RuntimeError(f"Worker failed to load candidates: {e}")

    if base_pickle_path:
        try:
            with open(base_pickle_path, "rb") as f:
                _WORKER_BASE = pickle.load(f)
        except Exception as e:
            _WORKER_BASE = None
            raise RuntimeError(f"Worker failed to load base: {e}")

def _compare_candidate_worker(arg):
    candidate_index, include_diff = arg
    global _WORKER_CANDIDATES, _WORKER_BASE
    try:
        cand = _WORKER_CANDIDATES[candidate_index]
    except Exception as e:
        return {"candidate": None, "error": f"candidate_lookup_error: {e}"}

    if not _WORKER_BASE:
        return {"candidate": cand.get("path"), "error": "base_not_loaded_in_worker"}

    cand_path = cand["path"]
    other_norm_lines = cand["norm_lines"]
    other_orig_ranges = cand["orig_ranges"]

    base_norm_lines = _WORKER_BASE["norm_lines"]
    base_orig_ranges = _WORKER_BASE["orig_ranges"]

    diff_info = diff_normalized_sequences_with_orig(base_norm_lines, base_orig_ranges,
                                                    other_norm_lines, other_orig_ranges)
    result = {
        "candidate": cand_path,
        "total_changed_lines": diff_info["total_changed_lines"],
        "ranges": diff_info["ranges"],
        "num_norm_base_lines": len(base_norm_lines),
        "num_norm_other_lines": len(other_norm_lines)
    }

    if include_diff:
        a_text = "\n".join(base_norm_lines)
        b_text = "\n".join(other_norm_lines)
        ud = list(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(),
                                       fromfile="base (normalized)",
                                       tofile=Path(cand_path).name + " (normalized)",
                                       lineterm=""))
        result["unified_diff_normalized"] = "\n".join(ud)
    return result

# ----------------------------
# Cache helpers
# ----------------------------
def _make_dir_snapshot(directory: Path, recursive: bool) -> List[Tuple[str, float, int]]:
    files = list(directory.rglob("*.sol")) if recursive else list(directory.glob("*.sol"))
    entries = []
    for p in files:
        if p.is_file():
            st = p.stat()
            entries.append((str(p.resolve()), st.st_mtime, st.st_size))
    entries.sort()
    return entries

def _snapshot_hash(entries: List[Tuple[str, float, int]]) -> str:
    m = hashlib.sha1()
    for path, mtime, size in entries:
        m.update(path.encode("utf-8"))
        m.update(str(int(mtime)).encode("utf-8"))
        m.update(str(size).encode("utf-8"))
    return m.hexdigest()

# ----------------------------
# Main class
# ----------------------------
class SolidityComparatorFastLoadOpt:
    def __init__(self, minhash_k: int = 64, minhash_seed: int = 12345, shortlist_k: int = 200,
                 chunksize: int = 8, cache_dir: Optional[str] = None):
        self._candidates: List[Dict] = []
        self._pickle_path: Optional[str] = None
        self.minhash_k = minhash_k
        self.minhash_seed = minhash_seed
        self.shortlist_k = shortlist_k
        self.chunksize = chunksize
        self._a_list, self._b_list, self._PRIME = make_minhash_funcs(self.minhash_k, seed=self.minhash_seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_paths(self, base_dir: Path) -> Tuple[Path, Path]:
        # cache files: full and worker-minimal
        base_hash_file = base_dir / ".solcmp_cache_meta.json"
        # use snapshot hash for file set
        return (base_hash_file.with_suffix(".full.pkl"), base_hash_file.with_suffix(".worker.pkl"))

    def load_candidates(self, directory: str, recursive: bool = False, exclude_paths: Optional[List[str]] = None,
                        use_cache: bool = True, max_workers: Optional[int] = None, build_cache_only: bool = False):
        """
        Load candidate .sol files, with progress bar and optional build-cache-only mode.
        If build_cache_only is True, this builds the caches then returns (useful for precomputing).
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        exclude_set = set(str(Path(p).resolve()) for p in exclude_paths) if exclude_paths else set()

        # snapshot and cache paths
        entries = _make_dir_snapshot(directory, recursive)
        snap_hash = _snapshot_hash(entries)
        cache_dir = Path(self.cache_dir) if self.cache_dir else directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        full_cache_path = cache_dir / f"solcmp_full_{snap_hash}.pkl"
        worker_cache_path = cache_dir / f"solcmp_worker_{snap_hash}.pkl"

        # quick load if caches exist and not forced to rebuild
        if use_cache and full_cache_path.exists() and worker_cache_path.exists() and not build_cache_only:
            try:
                with open(worker_cache_path, "rb") as f:
                    self._candidates = pickle.load(f)
                    self._pickle_path = str(worker_cache_path)
                    return
            except Exception:
                pass

        files = [Path(p) for p,_,__ in entries if Path(p).is_file()]
        n_files = len(files)
        if max_workers is None:
            max_workers = max(1, min(64, (os.cpu_count() or 4) * 2))

        # use top-level helper _process_file_for_load (must be defined at module level)
        from itertools import repeat
        cand_list = []
        # Use ProcessPoolExecutor and as_completed to get progress updates
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for p in files:
                futures.append(ex.submit(_process_file_for_load,
                                        str(p),
                                        exclude_set,
                                        self._a_list,
                                        self._b_list,
                                        self._PRIME))
            for fut in tqdm(concurrent.futures.as_completed(futures), total=n_files, desc="Preprocessing candidates"):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    cand_list.append(res)

        # Save full cache and worker-minimal cache
        try:
            with open(full_cache_path, "wb") as f:
                pickle.dump(cand_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            worker_list = [{"path": c["path"], "norm_lines": c["norm_lines"], "orig_ranges": c["orig_ranges"]} for c in cand_list]
            with open(worker_cache_path, "wb") as f:
                pickle.dump(worker_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._candidates = worker_list
            self._pickle_path = str(worker_cache_path)
        except Exception as e:
            # fallback to in-memory only
            self._candidates = cand_list
            self._pickle_path = None

        if build_cache_only:
            print(f"Built cache: full={full_cache_path}, worker={worker_cache_path}")
            return


    # compare base convenience
    def compare_with_base(self, base_path: str, top: int = 1, workers: Optional[int] = None,
                          include_diff: bool = False, timeout: Optional[float] = None,
                          shortlist_k: Optional[int] = None) -> List[Dict]:
        base_path = Path(base_path)
        if not base_path.is_file():
            raise FileNotFoundError(f"Base file not found: {base_path}")
        base_src = base_path.read_text(encoding="utf-8")
        return self.compare_with_base_src(base_src, top=top, workers=workers, include_diff=include_diff, timeout=timeout, shortlist_k=shortlist_k)

    def compare_with_base_src(self, base_src: str, top: int = 1, workers: Optional[int] = None,
                          include_diff: bool = False, timeout: Optional[float] = None,
                          shortlist_k: Optional[int] = None) -> List[Dict]:
        """
        Faster compare_with_base_src:
        - Precomputes missing candidate minhash signatures in parallel (once).
        - Then computes minhash similarity cheaply in-process.
        - Shortlists and runs expensive diffs on shortlist.
        """
        start_time = time()
        if not self._candidates:
            raise RuntimeError("Candidates not loaded. Call load_candidates() first.")
        if shortlist_k is None:
            shortlist_k = self.shortlist_k

        # Normalize base
        t0 = time()
        base_lines_with_numbers = strip_comments_preserve_lines_with_orig(base_src)
        base_norm_lines, base_orig_ranges = build_normalized_sequence_with_orig(base_lines_with_numbers)
        base_line_hashes = [_hash_line_to_int(ln) for ln in base_norm_lines]
        base_sig = compute_minhash_signature(base_line_hashes, self._a_list, self._b_list, self._PRIME)
        t_base = time() - t0

        # Determine which candidates are missing minhash_sig (or line_hashes)
        missing_indices = []
        for idx, cand in enumerate(self._candidates):
            if "minhash_sig" not in cand or cand["minhash_sig"] is None:
                # if we do have line_hashes we can compute minhash quickly; otherwise compute line_hashes then sig
                missing_indices.append(idx)

        if missing_indices:
            # parallel compute missing signatures
            t1 = time()
            if workers is None:
                workers = max(1, min(os.cpu_count() or 4, len(missing_indices)))
            # worker helper (module-level) to compute minhash signature for a single candidate index
            # We'll create an argument list (index, a_list, b_list, PRIME) and map to a module-level helper
            def _compute_sig_for_index(idx):
                cand = self._candidates[idx]
                # try using cached line_hashes
                if "line_hashes" in cand and cand["line_hashes"]:
                    lh = cand["line_hashes"]
                else:
                    # compute line hashes from norm_lines
                    lh = [_hash_line_to_int(ln) for ln in cand["norm_lines"]]
                    cand["line_hashes"] = lh
                sig = compute_minhash_signature(lh, self._a_list, self._b_list, self._PRIME)
                # store back in the candidate dict for reuse
                cand["minhash_sig"] = sig
                return idx  # return index to indicate completion

            # Run in a process pool (CPU bound). Use map with chunksize to be efficient.
            # Use a ProcessPoolExecutor because compute_minhash_signature may be CPU intensive.
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                # Submit tasks for missing indices
                # We use submit+as_completed so we can update progress / measure time
                futures = {ex.submit(_compute_sig_for_index, idx): idx for idx in missing_indices}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        # if a worker fails, print and continue: we remove that candidate from selection
                        print(f"Warning: failed to compute minhash for candidate idx {futures[fut]}: {e}")

            t_sig = time() - t1
        else:
            t_sig = 0.0

        # Now compute minhash similarity cheaply
        t2 = time()
        scores = []
        # we assume now every candidate has 'minhash_sig'
        for idx, cand in enumerate(self._candidates):
            sig = cand.get("minhash_sig")
            if sig is None:
                # fallback: extremely rare — treat as 0 similarity
                sim = 0.0
            else:
                sim = minhash_similarity(base_sig, sig)
            scores.append((idx, sim))
        scores.sort(key=lambda t: t[1], reverse=True)
        selected = [i for i, _ in scores[:max(1, shortlist_k)]]
        t_score = time() - t2

        # create base pickle for workers (as before)
        fd, base_pth = tempfile.mkstemp(prefix="solcmp_base_", suffix=".pkl")
        os.close(fd)
        with open(base_pth, "wb") as f:
            pickle.dump({"norm_lines": base_norm_lines, "orig_ranges": base_orig_ranges}, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Base processed in {t_base:.2f}s, computed missing {len(missing_indices)} candidate sigs in {t_sig:.2f}s, scoring in {t_score:.2f}s, {len(selected)} candidates selected for detailed comparison")

        # proceed to shortlist expensive diffs (same as before)
        if workers is None:
            workers = min(os.cpu_count() or 4, max(1, len(selected)))

        job_args = [(i, include_diff) for i in selected]
        results = []

        if self._pickle_path:
            init_args = (self._pickle_path, base_pth)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer, initargs=init_args) as ex:
                for res in ex.map(_compare_candidate_worker, job_args, chunksize=self.chunksize):
                    if res:
                        results.append(res)
        else:
            # fallback local worker_local (as before)
            local_candidates = self._candidates

            def worker_local(arg):
                idx, include_diff_flag = arg
                cand = local_candidates[idx]
                other_norm_lines = cand["norm_lines"]
                other_orig_ranges = cand["orig_ranges"]
                diff_info = diff_normalized_sequences_with_orig(base_norm_lines, base_orig_ranges, other_norm_lines, other_orig_ranges)
                result = {
                    "candidate": cand["path"],
                    "total_changed_lines": diff_info["total_changed_lines"],
                    "ranges": diff_info["ranges"],
                    "num_norm_base_lines": len(base_norm_lines),
                    "num_norm_other_lines": len(other_norm_lines)
                }
                if include_diff_flag:
                    a_text = "\n".join(base_norm_lines)
                    b_text = "\n".join(other_norm_lines)
                    ud = list(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(),
                                                fromfile="base (normalized)",
                                                tofile=Path(cand["path"]).name + " (normalized)",
                                                lineterm=""))
                    result["unified_diff_normalized"] = "\n".join(ud)
                return result

            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                for res in ex.map(worker_local, job_args, chunksize=self.chunksize):
                    if res:
                        results.append(res)

        # cleanup base pickle
        try:
            Path(base_pth).unlink()
        except Exception:
            pass

        results = [r for r in results if r.get("candidate") is not None and "error" not in r]
        results.sort(key=lambda r: r["total_changed_lines"])
        total_time = time() - start_time
        print(f"Total compare_with_base_src time: {total_time:.2f}s")
        return results[:top]

    def cleanup(self):
        # no automatic removal of cache (cache preserved)
        self._candidates = []
        self._pickle_path = None

    def __del__(self):
        self.cleanup()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse, pprint
    ap = argparse.ArgumentParser(description="SolidityComparatorFastLoadOpt CLI")
    ap.add_argument("--base", "-b", help="base solidity file", required=True)
    ap.add_argument("--dir", "-d", help="directory with candidate .sol files", required=True)
    ap.add_argument("--recursive", "-r", action="store_true", help="search recursively")
    ap.add_argument("--top", type=int, default=1, help="top N matches")
    ap.add_argument("--workers", type=int, default=None, help="number of worker processes")
    ap.add_argument("--shortlist", type=int, default=None, help="override shortlist_k")
    ap.add_argument("--minhash", type=int, default=64, help="minhash signature size")
    ap.add_argument("--chunksize", type=int, default=8, help="executor.map chunksize")
    ap.add_argument("--no-cache", action="store_true", help="do not use on-disk cache (always rebuild)")
    ap.add_argument("--build-cache", action="store_true", help="Only build candidate cache and exit")
    args = ap.parse_args()

    sc = SolidityComparatorFastLoadOpt(minhash_k=args.minhash, shortlist_k=(args.shortlist or 200), chunksize=args.chunksize)
    t0 = time()
    sc.load_candidates(args.dir, recursive=args.recursive, max_workers=args.workers, use_cache=(not args.no_cache), build_cache_only=args.build_cache)
    if args.build_cache:
        print("Cache build complete.")
        exit(0)

    t1 = time()
    print(f"Loaded {len(sc._candidates)} candidates in {t1 - t0:.2f}s")
    t2 = time()
    res = sc.compare_with_base(args.base, top=args.top, workers=args.workers, shortlist_k=args.shortlist)
    t3 = time()
    print(f"Compared in {t3 - t2:.2f}s")
    pprint.pprint(res)
