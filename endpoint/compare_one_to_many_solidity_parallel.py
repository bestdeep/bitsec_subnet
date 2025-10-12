#!/usr/bin/env python3
"""
Parallel compare: Compare a base Solidity file against many .sol files in a directory,
ignoring comments, using multiple CPU cores.

Usage:
    python compare_one_to_many_solidity_parallel.py base.sol /path/to/dir \
        --workers 8 --top 3 --recursive --diff --json out.json

Notes:
 - Uses ProcessPoolExecutor to parallelize per-candidate comparisons.
 - Keeps same normalization and line-mapping semantics as the single-threaded version.
"""
from pathlib import Path
import re
import sys
import argparse
import json
from typing import List, Tuple, Dict
import difflib
import concurrent.futures
import os
from time import time

# -------------------------
# Comment-removal parser (line-aware)
# -------------------------
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


# -------------------------
# Normalization utilities
# -------------------------
_PUNCT = r"\{\}\(\)\[\]\.,;:+\-*/%=\<\>!&\|^~\?:"
PUNCT_RE_BEFORE = re.compile(rf"\s+([{re.escape(_PUNCT)}])")
PUNCT_RE_AFTER = re.compile(rf"([{re.escape(_PUNCT)}])\s+")
MULTI_WHITESPACE = re.compile(r"\s+")

def normalize_line(line: str) -> str:
    s = line.strip()
    s = MULTI_WHITESPACE.sub(" ", s)
    s = PUNCT_RE_BEFORE.sub(r"\1", s)
    s = PUNCT_RE_AFTER.sub(r"\1", s)
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


# -------------------------
# Diff and range extraction
# -------------------------
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


# -------------------------
# Candidate comparison (worker-friendly)
# -------------------------
def compare_candidate(candidate_path: str,
                      base_norm_lines: List[str],
                      base_nums: List[int],
                      show_diff: bool = False) -> Dict:
    """
    Compare one candidate file (path string) to the provided base normalized
    sequence. Returns a result dict (serializable).
    Designed to be run in a worker process.
    """
    cand_path = Path(candidate_path)
    try:
        other_src = cand_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"candidate": candidate_path, "error": f"read_error: {e}"}

    other_lines_with_numbers = strip_comments_preserve_lines(other_src)
    other_norm_lines, other_nums = build_normalized_sequence(other_lines_with_numbers)

    diff_info = diff_normalized_sequences(base_norm_lines, base_nums, other_norm_lines, other_nums)
    result = {
        "candidate": candidate_path,
        "total_changed_lines": diff_info["total_changed_lines"],
        "ranges": diff_info["ranges"],
        "num_norm_base_lines": len(base_norm_lines),
        "num_norm_other_lines": len(other_norm_lines)
    }
    if show_diff:
        a_text = "\n".join(base_norm_lines)
        b_text = "\n".join(other_norm_lines)
        ud = list(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(),
                                       fromfile="base (normalized)",
                                       tofile=cand_path.name + " (normalized)",
                                       lineterm=""))
        result["unified_diff_normalized"] = "\n".join(ud)
    return result


# -------------------------
# Directory walk & parallel execution
# -------------------------
def find_sol_files(directory: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in directory.rglob("*.sol") if p.is_file()]
    else:
        return [p for p in directory.glob("*.sol") if p.is_file()]

def compare_base_to_directory_parallel(base_path: Path, directory: Path,
                                       recursive: bool, top_n: int,
                                       show_diff: bool, workers: int,
                                       chunk_size: int = 16) -> List[Dict]:
    base_src = base_path.read_text(encoding="utf-8")
    base_lines_with_numbers = strip_comments_preserve_lines(base_src)
    base_norm_lines, base_nums = build_normalized_sequence(base_lines_with_numbers)

    candidates = find_sol_files(directory, recursive)
    candidates = [str(p) for p in candidates if p.resolve() != base_path.resolve()]
    if not candidates:
        return []

    # Use ProcessPoolExecutor to parallelize compare_candidate
    results: List[Dict] = []
    # Prepare arguments to pass (we use map with tuples)
    # We'll use executor.submit in chunks to report progress
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        # Worker function partial via map: pass base_norm_lines and base_nums as part of params
        futures = []
        for cand in candidates:
            futures.append(ex.submit(compare_candidate, cand, base_norm_lines, base_nums, show_diff))
        # Collect results as they complete
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                res = {"candidate": "unknown", "error": f"worker_exception: {e}"}
            results.append(res)

    # Sort by least differences
    results = [r for r in results if "error" not in r]
    results.sort(key=lambda r: r["total_changed_lines"])
    return results[:top_n]


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Parallel compare base Solidity file against many .sol files (ignore comments).")
    ap.add_argument("--base", type=str, help="base solidity file path")
    ap.add_argument("--dir", type=str, help="directory with candidate .sol files")
    ap.add_argument("--recursive", "-r", action="store_true", help="search directory recursively")
    ap.add_argument("--top", type=int, default=1, help="how many top matches to return (default 1)")
    ap.add_argument("--diff", action="store_true", help="include unified diff of normalized content in output")
    ap.add_argument("--json", type=str, default=None, help="write full results to JSON file")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="number of worker processes (default: CPU count)")
    args = ap.parse_args()

    base_path = Path(args.base)
    directory = Path(args.dir)

    start_time = time()
    if not base_path.is_file():
        print("Base file not found:", base_path)
        sys.exit(2)
    if not directory.is_dir():
        print("Directory not found:", directory)
        sys.exit(2)

    print(f"Using {args.workers} worker(s). Scanning directory {directory} (recursive={args.recursive})...")
    results = compare_base_to_directory_parallel(base_path, directory, recursive=args.recursive,
                                                top_n=args.top, show_diff=args.diff, workers=args.workers)

    if not results:
        print("No candidate .sol files found or all candidates failed.")
        return

    print(f"\nBase file: {base_path}")
    print(f"Top {len(results)} matches (sorted by least differing lines):\n")
    for idx, r in enumerate(results, start=1):
        print(f"=== Match #{idx}: {r['candidate']}")
        print(f"Total changed lines (base + candidate): {r['total_changed_lines']}")
        print(f"Normalized lengths: base={r['num_norm_base_lines']}, candidate={r['num_norm_other_lines']}")
        if r.get("ranges"):
            print("Difference ranges (original line numbers):")
            for rng in r["ranges"]:
                tag = rng["tag"]
                b_range = rng["base_line_range"]
                o_range = rng["other_line_range"]
                parts = []
                if b_range:
                    parts.append(f"base lines {b_range[0]}-{b_range[1]}")
                if o_range:
                    parts.append(f"other lines {o_range[0]}-{o_range[1]}")
                print(f"  - {tag}: " + "; ".join(parts))
        else:
            print("  (no differences)")

        if args.diff and "unified_diff_normalized" in r:
            print("\nUnified diff (normalized):\n")
            print(r["unified_diff_normalized"])
            print("\n--- end diff ---\n")

    if args.json:
        outp = {"base": str(base_path), "results": results}
        Path(args.json).write_text(json.dumps(outp, indent=2), encoding="utf-8")
        print(f"Wrote JSON results to {args.json}")
    end_time = time()
    print(f"\nCompleted in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
