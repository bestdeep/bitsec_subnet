#!/usr/bin/env python3
"""
solidity_comparator.py

A class-based module to compare a base Solidity file to many .sol candidates in a directory,
ignoring comments and formatting differences, and running comparisons in parallel.

Key features:
 - Loads candidate files once (in main process) and serializes them to a temporary pickle.
 - Worker processes use an initializer to load candidates from that pickle once per worker.
 - Uses difflib.SequenceMatcher on normalized lines for differences and maps them back to original line numbers.
 - Returns top-N matches with difference ranges and optional unified diff of normalized content.

Usage example:
    from solidity_comparator import SolidityComparator

    sc = SolidityComparator()
    sc.load_candidates("/path/to/candidates", recursive=True)
    results = sc.compare_with_base("/path/to/base.sol", top=3, workers=8, include_diff=True)
    for r in results:
        print(r['candidate'], r['total_changed_lines'])
"""

from pathlib import Path
import re
import difflib
import json
import tempfile
import pickle
import os
from typing import List, Tuple, Dict, Optional
import concurrent.futures
import multiprocessing
import functools

# ---------------------------------------------------------------------
# Low-level parsing and normalization (same semantics as previous scripts)
# ---------------------------------------------------------------------
def strip_comments_preserve_lines(source: str) -> List[Tuple[int, str]]:
    result_lines: List[Tuple[int, str]] = []
    i = 0
    n = len(source)
    lineno = 1
    cur_chars: List[str] = []

    in_block = False
    in_string = False
    string_delim = None
    while i < n:
        ch = source[i]

        if in_block:
            if ch == '*' and i + 1 < n and source[i+1] == '/':
                in_block = False
                i += 2
                continue
            if ch == '\n':
                result_lines.append((lineno, "".join(cur_chars)))
                cur_chars = []
                lineno += 1
            i += 1
            continue

        if in_string:
            if ch == '\\' and i + 1 < n:
                cur_chars.append(ch); cur_chars.append(source[i+1]); i += 2; continue
            if ch == string_delim:
                cur_chars.append(ch); in_string = False; string_delim = None; i += 1; continue
            if ch == '\n':
                cur_chars.append(ch); result_lines.append((lineno, "".join(cur_chars))); cur_chars = []; lineno += 1; i += 1; continue
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
            result_lines.append((lineno, "".join(cur_chars)))
            cur_chars = []
            lineno += 1
            i += 1
            continue

        cur_chars.append(ch)
        i += 1

    if cur_chars or lineno == 1:
        result_lines.append((lineno, "".join(cur_chars)))
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

def build_normalized_sequence(lines_with_numbers: List[Tuple[int, str]]) -> Tuple[List[str], List[int]]:
    norm_lines: List[str] = []
    orig_nums: List[int] = []
    for lineno, raw in lines_with_numbers:
        n = normalize_line(raw)
        if n == "":
            continue
        norm_lines.append(n)
        orig_nums.append(lineno)
    return norm_lines, orig_nums

def diff_normalized_sequences(base_lines: List[str], base_nums: List[int],
                              other_lines: List[str], other_nums: List[int]) -> Dict:
    sm = difflib.SequenceMatcher(a=base_lines, b=other_lines, autojunk=False)
    opcodes = sm.get_opcodes()
    ranges = []
    total_changed = 0

    for tag, a1, a2, b1, b2 in opcodes:
        if tag == 'equal':
            continue
        base_range = None
        other_range = None
        base_count = 0
        other_count = 0

        if a1 < a2:
            start_ln = base_nums[a1]
            end_ln = base_nums[a2-1]
            base_range = (start_ln, end_ln)
            base_count = a2 - a1
        if b1 < b2:
            start_ln2 = other_nums[b1]
            end_ln2 = other_nums[b2-1]
            other_range = (start_ln2, end_ln2)
            other_count = b2 - b1

        total_changed += base_count + other_count

        ranges.append({
            "tag": tag,
            "base_index_range": (a1, a2),
            "other_index_range": (b1, b2),
            "base_line_range": base_range,
            "other_line_range": other_range,
            "base_snippet": "\n".join(base_lines[a1:a2]),
            "other_snippet": "\n".join(other_lines[b1:b2])
        })

    return {
        "opcodes": opcodes,
        "ranges": ranges,
        "total_changed_lines": total_changed
    }

# ---------------------------------------------------------------------
# Worker-global variable(s) loaded by initializer
# ---------------------------------------------------------------------
_WORKER_CANDIDATES = None  # will be List[dict] where each dict has keys: path, norm_lines, orig_nums

def _worker_initializer(pickle_path: str):
    """
    ProcessPoolExecutor initializer: load the pickled candidates into a process-global variable.
    This is run once per worker process.
    """
    global _WORKER_CANDIDATES
    try:
        with open(pickle_path, "rb") as f:
            _WORKER_CANDIDATES = pickle.load(f)
    except Exception as e:
        # If loading fails, set to empty list and we will report errors in compare.
        _WORKER_CANDIDATES = []
        # avoid printing in workers in normal runs; raise to make failures visible
        raise RuntimeError(f"Worker failed to load candidates from {pickle_path}: {e}")

def _compare_candidate_worker(idx_and_base, include_diff: bool=False):
    """
    Worker function that expects a tuple (candidate_index, base_norm_lines, base_nums, pickle_path)
    but because we use initializer, we only need candidate_index, base_norm_lines, base_nums, include_diff.
    To keep pickling small, we pass base_norm_lines/base_nums as arguments and only candidate index is used to pick candidate from _WORKER_CANDIDATES.
    """
    candidate_index, base_norm_lines, base_nums, include_diff = idx_and_base
    global _WORKER_CANDIDATES
    try:
        cand = _WORKER_CANDIDATES[candidate_index]
    except Exception as e:
        return {"candidate": None, "error": f"candidate_lookup_error: {e}"}

    cand_path = cand["path"]
    other_norm_lines = cand["norm_lines"]
    other_nums = cand["orig_nums"]

    diff_info = diff_normalized_sequences(base_norm_lines, base_nums, other_norm_lines, other_nums)
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

# ---------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------
class SolidityComparator:
    """
    Class-based comparator.

    Typical usage:
        sc = SolidityComparator()
        sc.load_candidates("/path/to/dir", recursive=True)
        results = sc.compare_with_base("/path/to/base.sol", top=1, workers=4, include_diff=True)

    Methods:
      - load_candidates(directory, recursive=False): load and normalize all candidate .sol files (and serialize them to a temp pickle).
      - compare_with_base(base_path, top=1, workers=None, include_diff=False, timeout=None): run parallel comparisons and return top matches.
      - cleanup(): remove internal temp pickle file (optional; done automatically on object delete).
    """
    def __init__(self):
        self._candidates: List[Dict] = []
        self._pickle_path: Optional[str] = None

    def load_candidates(self, directory: str, recursive: bool=False, exclude_paths: Optional[List[str]]=None):
        """
        Load candidate .sol files from `directory` (string path). This reads the files into memory,
        strips comments, normalizes per-line, and stores: path, norm_lines, orig_nums.
        Then writes the entire structure to a temporary pickle file for worker initializer use.

        - exclude_paths: optional list of file paths to exclude (absolute or relative).
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        exclude_set = set()
        if exclude_paths:
            exclude_set = set(str(Path(p).resolve()) for p in exclude_paths)

        files = list(directory.rglob("*.sol")) if recursive else list(directory.glob("*.sol"))
        files = [p for p in files if p.is_file()]
        cand_list = []
        for p in files:
            pr = str(p.resolve())
            if pr in exclude_set:
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except Exception as e:
                # skip unreadable file but continue
                continue
            lines_with_nums = strip_comments_preserve_lines(src)
            norm_lines, orig_nums = build_normalized_sequence(lines_with_nums)
            cand_list.append({
                "path": pr,
                "norm_lines": norm_lines,
                "orig_nums": orig_nums
            })

        self._candidates = cand_list

        # serialize to temp pickle for worker initializer
        fd, ppath = tempfile.mkstemp(prefix="sol_cmp_cands_", suffix=".pkl")
        os.close(fd)
        with open(ppath, "wb") as f:
            pickle.dump(self._candidates, f)
        self._pickle_path = ppath

    def compare_with_base(self, base_path: str, top: int=1, workers: Optional[int]=None,
                          include_diff: bool=False, timeout: Optional[float]=None) -> List[Dict]:
        """
        Compare base_path to loaded candidates in parallel.

        - top: how many top matches to return (sorted by least total_changed_lines).
        - workers: number of worker processes (default = cpu_count()).
        - include_diff: whether to include unified diff of normalized content.
        - timeout: optional overall timeout in seconds for the entire operation (None means no timeout).
        """
        if not self._candidates or not self._pickle_path:
            raise RuntimeError("Candidates not loaded. Call load_candidates() first.")

        base_path = Path(base_path)
        if not base_path.is_file():
            raise FileNotFoundError(f"Base file not found: {base_path}")

        # parse base once in main process
        base_src = base_path.read_text(encoding="utf-8")
        base_lines_with_numbers = strip_comments_preserve_lines(base_src)
        base_norm_lines, base_nums = build_normalized_sequence(base_lines_with_numbers)

        # Prepare worker pool
        if workers is None:
            workers = os.cpu_count() or 4

        # Create a ProcessPoolExecutor with initializer that loads the candidate pickle once per worker
        # We'll pass to each worker only the candidate index and the base_norm/base_nums (to avoid pickling full candidates for each job)
        candidate_count = len(self._candidates)
        indices = list(range(candidate_count))

        # Prepare inputs as tuples: (candidate_index, base_norm_lines, base_nums, include_diff)
        # Note: base_norm_lines/base_nums are pickled and sent to each worker per job; to avoid repeated pickling,
        # we submit one job per candidate, which still requires pickling base_norm_lines once per job.
        # For a very large base, we could also pickle base into a temp file and have worker initializer load it once.
        # For most practical bases this cost is acceptable. If you want base loaded once per worker too, tell me and I will adjust.
        job_args = [(idx, base_norm_lines, base_nums, include_diff) for idx in indices]

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer, initargs=(self._pickle_path,)) as ex:
            # submit all jobs
            futures = [ex.submit(_compare_candidate_worker, arg) for arg in job_args]
            # collect as they finish
            for fut in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"candidate": None, "error": f"worker_exception: {e}"}
                results.append(res)

        # filter out errors
        results = [r for r in results if r.get("candidate") is not None and "error" not in r]
        # sort by least changed lines
        results.sort(key=lambda r: r["total_changed_lines"])
        return results[:top]

    def cleanup(self):
        """Remove temporary pickle file (if any)."""
        if self._pickle_path and Path(self._pickle_path).exists():
            try:
                Path(self._pickle_path).unlink()
            except Exception:
                pass
        self._pickle_path = None

    def __del__(self):
        self.cleanup()

# ---------------------------------------------------------------------
# If run as script: simple CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pprint
    from time import time
    ap = argparse.ArgumentParser(description="SolidityComparator CLI")
    ap.add_argument("--base", help="base solidity file")
    ap.add_argument("--dir", help="directory with candidate .sol files")
    ap.add_argument("--recursive", "-r", action="store_true", help="search recursively")
    ap.add_argument("--top", type=int, default=1, help="top N matches")
    ap.add_argument("--workers", type=int, default=None, help="number of worker processes")
    ap.add_argument("--diff", action="store_true", help="include normalized unified diff")
    args = ap.parse_args()

    sc = SolidityComparator()
    start_time = time()
    sc.load_candidates(args.dir, recursive=args.recursive, exclude_paths=[args.base])
    load_time = time() - start_time
    print(f"Loaded {len(sc._candidates)} candidates in {load_time:.2f} seconds.")
    start_time = time()
    res = sc.compare_with_base(args.base, top=args.top, workers=args.workers, include_diff=args.diff)
    comp_time = time() - start_time
    print(f"Compared with base in {comp_time:.2f} seconds.")
    pprint.pprint(res)

    start_time = time()
    res = sc.compare_with_base("2196.sol", top=args.top, workers=args.workers, include_diff=args.diff)
    comp_time = time() - start_time
    print(f"Compared with base in {comp_time:.2f} seconds.")
    pprint.pprint(res)
    # sc.cleanup()
