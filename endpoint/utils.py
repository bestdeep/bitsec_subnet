import re
from typing import Dict, List, Any
from enum import Enum
from bitsec.base.vulnerability_category import VulnerabilityCategory

def extract_keyword_context_with_bounds(solidity_code: str, keyword: str) -> Dict:
    """
    General-purpose function to find all lines containing `keyword` in Solidity code
    and map them to the function/constructor they belong to, including start/end lines.
    """
    lines = solidity_code.splitlines()
    func_pattern = re.compile(r'\b(function|constructor)\s*([A-Za-z0-9_]*)?\s*\(')
    
    matches = []
    functions = {}

    # Stack to track current function and its braces
    func_stack = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Detect function or constructor
        func_match = func_pattern.search(stripped)
        if func_match:
            func_type, func_name = func_match.groups()
            if func_type == "constructor":
                func_name = "constructor"
            elif not func_name:
                func_name = "<anonymous>"

            # Push new function context on stack
            func_stack.append({
                "name": func_name,
                "start_line": lineno,
                "end_line": None,
                "brace_count": 0,
                "matches": []
            })

        # Count braces to detect scope
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        if func_stack:
            func_stack[-1]["brace_count"] += open_braces - close_braces

            # If function scope ends
            while func_stack and func_stack[-1]["brace_count"] <= 0:
                func_stack[-1]["end_line"] = lineno
                finished_func = func_stack.pop()
                if finished_func["matches"]:
                    functions[finished_func["name"]] = {
                        "start_line": finished_func["start_line"],
                        "end_line": finished_func["end_line"],
                        "matches": finished_func["matches"]
                    }

        # Check for keyword in line
        if keyword.lower() in stripped.lower():
            matches.append({"line": lineno, "content": stripped})
            if func_stack:
                func_stack[-1]["matches"].append({"line": lineno, "content": stripped})

    return {"matches": matches, "functions": functions}

# Map category to indicative keywords in comments
VULN_COMMENT_KEYWORDS = {
    VulnerabilityCategory.WEAK_ACCESS_CONTROL: ["owner only", "weak", "access"],
    VulnerabilityCategory.GOVERNANCE_ATTACKS: ["vote", "manipulation", "voting", "governance"],
    VulnerabilityCategory.REENTRANCY: ["reentrancy"],
    VulnerabilityCategory.FRONT_RUNNING: ["front running", "frontrunning", "nonce"],
    VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW: ["overflow", "underflow", "arithmetic error"],
    VulnerabilityCategory.SELF_DESTRUCT: ["selfdestruct", "self destruct", "suicide"],
    VulnerabilityCategory.UNINITIALIZED_PROXY: ["uninitialized proxy", "initialize", "constructor"],
    VulnerabilityCategory.INCORRECT_CALCULATION: ["incorrect calculation", "wrong math", "incorrect"],
    VulnerabilityCategory.ROUNDING_ERROR: ["rounding error", "precision loss"],
    VulnerabilityCategory.IMPROPER_INPUT_VALIDATION: ["input validation", "improper"],
    VulnerabilityCategory.BAD_RANDOMNESS: ["bad randomness", "predictable random", "pseudo-random", "pseudorandom"],
    VulnerabilityCategory.REPLAY_SIGNATURE_MALLEABILITY: ["replay", "malleability", "signature replay", "signature", "replay attack", "nonce"],
    VulnerabilityCategory.ORACLE_PRICE_MANIPULATION: ["manipulation", "price", "oracle",],
}

# Optional: override weights for particular keywords
# keys should match the literal keyword strings used in VULN_COMMENT_KEYWORDS (case-insensitive)
KEYWORD_WEIGHTS: Dict[str, float] = {
    # Example: give more importance to explicit "uninitialized proxy" and "reentrancy"
    "uninitialized proxy": 2.0,
    "reentrancy": 2.0,
    "selfdestruct": 3.0,
    "self destruct": 3.0,
    "overflow": 1.5,
    "underflow": 1.5,
    "front running": 1.5,
    "frontrunning": 1.5,
    "weak access control": 1.2,
    "owner only": 1.2,
    "governance attack": 1.2,
    "vote manipulation": 1.2,
    "input validation": 1.0,
    "bad randomness": 1.0,
    "replay attack": 1.0,
    "signature malleability": 1.0,
    "arithmetic error": 1.0,
    "incorrect calculation": 1.0,
    "rounding error": 1.0,
    "price manipulation": 1.0,
    "oracle": 1.0,
    "malleability": 1.0,
    # lower weight for generic hints
    "constructor": 0.5,
    "initialize": 0.8,
    "require": 0.3,
}

def score_match_result(match_result: Dict[str, Any], keyword_weights: Dict[str, float] | None = None) -> Dict[VulnerabilityCategory, float]:
    """
    Score each vulnerability category using match_result and optional per-keyword weights.

    Rules:
      - For each keyword of a vulnerability:
          * If keyword appears in any function-level match -> contribution = weight * 1.0
          * Else if keyword appears in any global match -> contribution = weight * 0.5
          * Else contribution = 0.0
      - Vulnerability score = sum(contributions) / sum(weights for that vuln's keywords)
    """
    if keyword_weights is None:
        keyword_weights = KEYWORD_WEIGHTS

    scores: Dict[VulnerabilityCategory, float] = {v: 0.0 for v in VulnerabilityCategory}

    # Prepare lists of matched contents (lowercased) for quick searching
    global_matches = [m["content"].lower() for m in match_result.get("matches", [])]

    # Flatten function-level matches into a list of contents for stronger signals
    func_matches = []
    for func_data in match_result.get("functions", {}).values():
        for m in func_data.get("matches", []):
            func_matches.append(m["content"].lower())

    for vuln, keywords in VULN_COMMENT_KEYWORDS.items():
        if not keywords:
            scores[vuln] = 0.0
            continue

        total_weight = 0.0
        accumulated = 0.0

        for kw in keywords:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue

            # determine weight for this keyword (default 1.0)
            w = float(keyword_weights.get(kw_l, 1.0))
            total_weight += w

            # strong match if keyword appears in any function-level matched line
            strong = any(kw_l in fm for fm in func_matches)
            if strong:
                accumulated += w * 1.0
                continue

            # weak match if keyword appears in any global matched line
            weak = any(kw_l in gm for gm in global_matches)
            if weak:
                accumulated += w * 0.5
                continue

            # no match -> add 0

        # avoid division by zero
        score = (accumulated / total_weight) if total_weight > 0 else 0.0
        # clamp to [0,1]
        score = max(0.0, min(1.0, score * 2))
        scores[vuln] = score

    return scores

def aggregate_keyword_matches(code: str, keywords: list[str]) -> dict:
    """
    Runs extract_keyword_context_with_bounds() for each keyword, then merges results
    into a single unified match_result dictionary for scoring.
    """
    merged = {"matches": [], "functions": {}}

    for kw in keywords:
        result = extract_keyword_context_with_bounds(code, kw)
        if not result:
            continue

        # Merge top-level matches
        merged["matches"].extend(result.get("matches", []))

        # Merge function-level matches
        for func_name, func_data in result.get("functions", {}).items():
            if func_name not in merged["functions"]:
                merged["functions"][func_name] = func_data
            else:
                # merge matches if already exists
                merged["functions"][func_name]["matches"].extend(func_data.get("matches", []))

    # Deduplicate matches by line content
    seen = set()
    unique_matches = []
    for m in merged["matches"]:
        key = (m.get("line"), m.get("content"))
        if key not in seen:
            unique_matches.append(m)
            seen.add(key)
    merged["matches"] = unique_matches

    return merged

if __name__ == "__main__":
    with open("code2.sol", "r") as f:
        code = f.read()

    import json
    print(f"code: {json.dumps(code, indent=2)}")
    # result = aggregate_keyword_matches(code, ["vulnerability"])
    # import pprint
    # pprint.pprint(result)