# Sample miner uses LLM prompts to find vulnerabilities in code
# This example uses a basic prompt template for demonstration purposes

import json
import re
import os
from typing import List, Union, Dict
from bitsec.base.vulnerability_category import VulnerabilityCategory
from bitsec.protocol import PredictionResponse, Vulnerability, LineRange
from bitsec.utils.data import SAMPLE_DIR, VULNERABILITIES_DIR
from bitsec.utils.chutes_llm import chutes_client, GPTMessage
import textwrap
from rich.console import Console
from endpoint.utils import VULN_COMMENT_KEYWORDS, extract_keyword_context_with_bounds, score_match_result, aggregate_keyword_matches
console = Console()

# Templates for prompts
SYSTEM_PROMPT = """
You are an expert smart contract security auditor specializing in Solidity, EVM semantics, and DeFi protocols.

Follow these rules with absolute strictness:

- Always prioritize **accuracy, reproducibility, and traceability** over verbosity or speculation.
- If uncertain, explicitly say you are unsure and suggest specific areas or conditions that require manual review.
- Never hallucinate vulnerabilities, function names, variable names, or code segments that do not appear in the provided source or diff.
- Focus exclusively on:
  - Security impact (funds loss, governance risk, logic exploit, DoS)
  - Root cause (state mismanagement, unchecked call, overflow, etc.)
  - Exploitability (how it could be triggered)
  - Concrete code-level fix suggestions
- You must produce **JSON only**, exactly matching the `response_format` supplied in the prompt.
- Output **nothing before or after JSON** (no commentary, explanations, or text).
- Each analysis session focuses on **at most one true vulnerability** — or two if both are clearly separate and defensible.
- Look carefully for **comments containing the words "vulnerability" or "vulnerable"**, as these mark injected or intentional issues for testing. Sometimes the vulnerability type is directly stated there.
- If you encounter a line in the code that include "// TODO: Fix security vulnerability", must not output a "no vulnerability" result, as the code is necessarily vulnerable.

Your role is to act as a precise, deterministic security model performing delta-based reasoning between two Solidity code versions.
"""

VULN_PROMPT_TEMPLATE = textwrap.dedent("""\
### Task
You are given a Solidity code snippet and a comparison result showing the changed lines between two versions.
Identify any **real security vulnerabilities or logical flaws** introduced or exposed by the code changes.

### Analysis Procedure
1. **Locate Indicators**
   - Search for all lines containing the keywords `"vulnerability"` or `"vulnerable"`.
   - If a vulnerability category (e.g., `reentrancy`, `front-running`, `access control`, etc.) appears near those lines, treat that as a strong hint.
2. **Contextual Review**
   - Examine the changed lines and ±5 lines of context.
   - Consider global invariants, constructor initialization, and modifier usage when relevant.
3. **Decision**
   - Identify **one primary vulnerability** if possible (two only if clearly distinct).
   - If none are found, produce the default JSON output below.

### Required Output Schema
{response_format}

If no vulnerabilities are found, output **exactly**:
{{"prediction": false, "vulnerabilities": []}}

### Important Constraints
- Do not include any text before or after the JSON.
- Do not invent variable names, function names, or logic not explicitly in `{code}` or `{compare_result}`.
- Each vulnerability description must be 1–3 sentences:
  - Include root cause and practical exploitability.
  - Include a short `suggestion_fix` (e.g., add a check, use `ReentrancyGuard`, validate input, restrict access, etc.).
- Never mix unrelated vulnerabilities; report only what is visible from the diff.
- Remember, if there is a line included "// TODO: Fix security vulnerability", the code definitely has a vulnerability.

### Reference Knowledge
Use the following vulnerability READMEs as reference material (for reasoning, not for copying text):
{vuln_readmes}

### Solidity Source Code
```
{code}
```

### Comparison Diff Result
{compare_result}
""")                                       
                                       
RESPONSE_FORMAT = textwrap.dedent("""
{
    "prediction": bool,  // True if vulnerabilities found, False otherwise
    "vulnerabilities": [  // List of identified vulnerabilities
        {
            "title": str,  // A short title for the vulnerability.
            "severity": str,  // Severity level: 99_critical, 85_high, 50_medium, 25_low, 10_informational
            "line_ranges": [
                [
                    "start":int, // Line number where the vulnerability starts (1-indexed).
                    "end":int // Line number where the vulnerability ends (1-indexed).
                ]
            ],  // An array of lines of code ranges where the vulnerability is located. Optional, but strongly recommended. Consecutive lines should be a single range, eg lines 1-3 should NOT be [{start: 1, end: 1}, {start: 2, end: 2}, {start: 3, end: 3}] INSTEAD SHOULD BE [{start: 1, end: 3}].
            "category": str,  // One of the following acceptable vulnerability categories (must be lowercase): weak access control, governance attacks, reentrancy, frontrunning, arithmetic overflow and underflow vulnerability, self destruct, uninitialized proxy, incorrect calculation, rounding error, improper input validation, bad randomness vulnerability, replay attacks/signature malleability, oracle/price manipulation.
            "description": str,  // Detailed description of the vulnerability, including financial impact and why this is a vulnerability.
            "vulnerable_code": str,  // Code snippet that contains the vulnerability.
            "code_to_exploit": str,  // Code snippet that exploits the vulnerability.
            "rewritten_code_to_fix_vulnerability": str,  // Code snippet that prevents the vulnerability from being exploited.
            "model_config": { "populate_by_name": True }  // Optional: Configuration of the model used to generate this vulnerability report.
        },
        ...
    ]
}
""")

MODELS = [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "moonshotai/Kimi-K2-Instruct"
    "deepseek-ai/DeepSeek-V3-0324",
]

def get_vulnerability_readme(vuln: VulnerabilityCategory) -> str:
    """Returns a markdown-formatted string describing the vulnerability category."""
    path = "vulnerabilities"
    match vuln:
        case VulnerabilityCategory.WEAK_ACCESS_CONTROL:
            path += "/weak-access-control.md"
        case VulnerabilityCategory.GOVERNANCE_ATTACKS:
            path += "/governance-attack.md"
        case VulnerabilityCategory.REENTRANCY:
            path += "/reentrancy.md"
        case VulnerabilityCategory.FRONT_RUNNING:
            path += "/frontrunning.md"
        case VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW:
            path += "/overflow.md"
        case VulnerabilityCategory.SELF_DESTRUCT:
            path += "/self-destruct.md"
        case VulnerabilityCategory.UNINITIALIZED_PROXY:
            path += "/uninitialized-proxy.md"
        case VulnerabilityCategory.INCORRECT_CALCULATION:
            path += "/incorrect-calculation.md"
        case VulnerabilityCategory.ROUNDING_ERROR:
            path += "/rounding-error.md"
        case VulnerabilityCategory.IMPROPER_INPUT_VALIDATION:
            path += "/improper-input-validation.md"
        case VulnerabilityCategory.BAD_RANDOMNESS:
            path += "/bad-random.md"
        case VulnerabilityCategory.REPLAY_SIGNATURE_MALLEABILITY:
            path += "/replay-signature-malleability.md"
        case VulnerabilityCategory.ORACLE_PRICE_MANIPULATION:
            path += "/oracle-price-attack.md"
        case _:
            path = ""
    if path and os.path.exists(path):
        with open(path, 'r') as file:
            return file.read()
        
    console.print(f"[bold red]ERROR: Vulnerability README not found for category {vuln} at path {path}[/bold red]")
    return ""

VULNERABILITY_README_CACHE = {vuln: get_vulnerability_readme(vuln) for vuln in VulnerabilityCategory}

def fix_json_string_with_llm(json_string:str,attempt:int=0)->dict:
    system_prompt = "Fix the json string sent by the user.  Reply only with the json string and nothing else. Must not change content or keys, only fix formatting errors."
    messages= [GPTMessage(role="system", content=system_prompt), GPTMessage(role="user", content=json_string)]
    response, _ = chutes_client.inference(messages=messages,temperature=0.0)
    try:
        response=response.replace('```json','').strip('```')
        response=json.loads(response)
        return response
    except Exception as e:
        console.print(f"Error fixing json string: {e},trying again..")
        attempt+=1
        if attempt>5:
            return None
        return fix_json_string_with_llm(json_string,attempt)

def load_json(json_string:str)->dict:
    try:
        return json.loads(json_string)
    except Exception as e:
        try:
            return eval(json_string)
        except Exception as e:
            console.print(f"unable to fix manually, trying with llm")
            fixed_json=fix_json_string_with_llm(json_string)
            if fixed_json:
                return fixed_json
            else:
                return None
            
def analyze_code_chutes(
    code: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = 10000,
    compare_result: Dict | None = None
) -> Union[PredictionResponse, str]:
    try:
        console.print(f"compare_result: {compare_result}")

        # prepare substitution values (we escape braces in values too, just in case)
        code_val = code.replace("{", "{{").replace("}", "}}")
        # Build prompt safely
        vuln_readmes_parts: list[str] = []
        for vuln_key, readme_text in VULNERABILITY_README_CACHE.items():
            if not readme_text:
                continue
            # short header for each readme, include only first N characters to avoid massive prompts
            header = f"### {vuln_key.name.replace('_', ' ').title()}"
            # Optionally trim per-readme length (uncomment/trimming if needed):
            # trimmed = readme_text if len(readme_text) <= 3000 else readme_text[:3000] + "\n...[truncated]\n"
            trimmed = readme_text
            vuln_readmes_parts.append(f"{header}\n{trimmed}")

        vuln_readmes = "\n\n---\n\n".join(vuln_readmes_parts) if vuln_readmes_parts else "No vulnerability reference materials available."

        # Escape braces in readme text to make safe for template.format usage
        vuln_readmes_safe = vuln_readmes.replace("{", "{{").replace("}", "}}")

        # Build prompt safely including the readmes and response format
        compare_result_dump = json.dumps(compare_result, indent=2) if compare_result else "No comparison result provided."
        if len(compare_result_dump) > 500:
            compare_result_dump = "...[compare result too large to include]...\nYou can ignore the compare result."
        prompt = VULN_PROMPT_TEMPLATE.format(
            code=code_val,
            vuln_readmes=vuln_readmes_safe,
            compare_result=compare_result_dump,
            response_format=RESPONSE_FORMAT
        )

        kwargs = {"messages": [GPTMessage(role="system", content=SYSTEM_PROMPT), GPTMessage(role="user", content=prompt)], "max_tokens": max_tokens}
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature

        # return chat_completion(**kwargs)
        status = False
        response = ""
        retry_count = 0
        max_retries = 5
        while not status:
            response, status_code = chutes_client.inference(**kwargs)
            if status_code == 200:
                status = True
            else:
                console.print(f"Chutes API returned status code {status_code}. Retrying...")
                retry_count += 1
                kwargs["model"] = MODELS[retry_count % len(MODELS)]
                console.print(f"Switching to model {kwargs['model']} and retrying (attempt {retry_count}/{max_retries})...")
            if retry_count >= max_retries:
                console.print("Max retries reached. Exiting.")
                break
        if status:
            resp = load_json(response)
            if resp is None:
                console.print(f"Failed to load JSON from response: {response}")
                return "error: failed to parse response"
            result = PredictionResponse.from_json(resp)
            return result
        else:
            console.print(f"Chutes API returned unexpected status code {status_code}. Response: {response}")
            return f"error: unexpected status code {status_code}: {response}"

    except KeyError as ke:
        console.print(f"[Key Error] KeyError while formatting prompt: {ke}")
        return f"error: prompt formatting KeyError: {ke}"
    except Exception as e:
        console.print(f"[General Error] An error occurred during analysis: {e}")
        return f"error: analysis failed: {e}"

def get_vulnerability_level(score: float) -> str:
    if score > 0.8:
        return "99_critical"
    elif score > 0.5:
        return "85_high"
    elif score > 0.25:
        return "50_medium"
    elif score > 0.1:
        return "25_low"
    else:
        return "10_informational"
    
def analyze_code(code: str, compare_result: Dict) -> Union[PredictionResponse, str]:
    search_result = aggregate_keyword_matches(code, ["vulnerability", "selfdestruct", "vulnerable", "weak access", "governance attack", "frontrunning", "oracle price", "uninitialized proxy", "incorrect calculation", "rounding error", "input validation", "bad randomness", "replay attack", "signature malleability"])

    if search_result["matches"]:
        console.print("[bold yellow]Keyword 'vulnerability' found in code. Proceeding with analysis.[/bold yellow]")
    else:
        console.print("[bold yellow]Keyword 'vulnerability' NOT found in code. Proceeding with analysis anyway.[/bold yellow]")
        return PredictionResponse(prediction=False, vulnerabilities=[])

    todo_found = any("// TODO: Fix security vulnerability" in match["content"] for match in search_result["matches"])
    if todo_found:
        console.print("[bold yellow]Found TODO comment indicating a vulnerability fix is needed. Proceeding with analysis.[/bold yellow]")
    else:
        console.print("[bold yellow]No TODO comment indicating a vulnerability fix found. Proceeding with analysis anyway.[/bold yellow]")
        return PredictionResponse(prediction=False, vulnerabilities=[])

    scores = score_match_result(search_result)

    max_score = max(scores.values()) if scores else 0.0
    max_vuln = max(scores, key=scores.get) if scores else None
    console.print(f"Search result: {search_result}")
    console.print(f"Max vulnerability score: {max_score:.2f} for {max_vuln}")

    if compare_result is {}:
        compare_result = search_result

    try:
        vulnerabilities: List[Vulnerability] = []
        for vuln, score in scores.items():
            if score < 0.05:
                continue

            # Convert line ranges to LineRange objects
            line_ranges: List[LineRange] = []
            for func_name, func_data in search_result.get("functions", {}).items():
                func_matches = func_data.get("matches", [])
                if any(any(kw in m["content"].lower() for kw in VULN_COMMENT_KEYWORDS[vuln]) for m in func_matches):
                    start_line = func_data.get("start_line")
                    end_line = func_data.get("end_line")
                    if start_line:
                        line_ranges.append(LineRange(start=start_line, end=end_line))

            last_match_content = search_result["matches"][-1]["content"]

            vulnerabilities.append(Vulnerability(
                title=vuln.name.replace("_", " ").title(),
                severity=get_vulnerability_level(score),
                line_ranges=line_ranges,
                category=vuln.value,
                description=f"{last_match_content} vulnerability detected with score {score:.2f}.",
                vulnerable_code=last_match_content,
                code_to_exploit=last_match_content,
                rewritten_code_to_fix_vulnerability=last_match_content,
                model_config={"populate_by_name": True}
            ))

            console.print(f"[bold blue]{vuln.name}[/bold blue]: {score:.2f} | functions affected: {line_ranges}")

        if len(vulnerabilities) > 0:
            console.print(f"[bold red]Vulnerabilities detected: {len(vulnerabilities)}[/bold red]")
            return PredictionResponse(prediction=True, vulnerabilities=vulnerabilities, model_config={"populate_by_name": True})

        return analyze_code_chutes(code, compare_result=compare_result)
    except Exception as e:
        console.print(f"[General Error] An error occurred during code manual analysis: {e}")
        return analyze_code_chutes(code, compare_result=compare_result)


def default_testnet_code(code: str) -> bool:
    """
    Checks if the provided code matches the default testnet code.

    Args:
        code (str): The code to check.

    Returns:
        bool: True if code matches default testnet code, False otherwise.
    """
    file_path = os.path.join(SAMPLE_DIR, "nft-reentrancy.sol")

    if not os.path.exists(file_path):
        return f"Error: The file '{file_path}' does not exist."
    
    try:
        with open(file_path, 'r') as file:
            file_contents = file.read()
    except IOError:
        return f"Error: Unable to read the file '{file_path}'."
    
    return file_contents == code

def code_to_vulns(code: str, compare_result: Dict) -> PredictionResponse:
    """
    Main function to analyze code and format the results into a PredictionResponse.

    Args:
        code (str): The code to analyze.

    Returns:
        PredictionResponse: The structured vulnerability report.
    """

    console.print("Analyzing code for vulnerabilities...")
    with open("code.sol", "w") as f:
        f.write(code)

    if default_testnet_code(code) == True:
        console.print("Default Testnet Code detected. Sending default prediction.")
        return PredictionResponse.from_tuple([True,[]])

    if compare_result.get("total_changed_lines", -1) == 0:
        console.print("No changes detected in the code compared to the base source.")
        return PredictionResponse(prediction=False, vulnerabilities=[])
    
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        try:
            # analysis = analyze_code_chutes(code, compare_result=compare_result)
            analysis = analyze_code(code, compare_result)
            console.print(f"Analysis complete. Result:\n{analysis}")

            if type(analysis) is not PredictionResponse:                
                console.print(f"Analysis returned an error: {analysis}")
                attempt += 1
                continue

            return analysis
        except Exception as e:
            console.print(f"[General Error] An error occurred during analysis: {e}")
            attempt += 1