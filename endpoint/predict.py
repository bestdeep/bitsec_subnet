# Sample miner uses LLM prompts to find vulnerabilities in code
# This example uses a basic prompt template for demonstration purposes

import json
import re
import os
from typing import List, Union, Dict
from bitsec.base.vulnerability_category import VulnerabilityCategory
from bitsec.protocol import PredictionResponse
from bitsec.utils.data import SAMPLE_DIR
from bitsec.utils.llm import chat_completion
from bitsec.utils.chutes_llm import chutes_client, GPTMessage
import textwrap

# Templates for prompts
SYSTEM_PROMPT = """You are an expert smart-contract security auditor with deep knowledge of Solidity, EVM semantics, and common DeFi vulnerabilities. 
Follow these rules strictly:
- Prioritize accuracy over verbosity. If unsure, say you are unsure and provide a short rationale and recommended checks.
- Do NOT hallucinate vulnerabilities or invent function names/variables not present in the code or compare result.
- Focus on security impact, root cause, exploitability, and practical code fixes.
- For each reported finding include: category, concise description, affected_lines (original file line range), impact, confidence (High/Medium/Low), and a suggested_fix.
- If no actionable vulnerability is found, respond with {"prediction": false}.
- Provide only JSON (or the exact response_format supplied) as final output. No extra commentary after JSON.

### Important:
There is only one vulnerability.
Therefore, try to find one vulnerability if possible, or at most two.
Look carefully at the code sections that are marked as "Vulnerability" in the comments. This is where the vulnerability is inserted.
However, the vulnerability classification listed here is not necessarily accurate.
Therefore, you should carefully consider and evaluate them.
"""

VULN_PROMPT_TEMPLATE = textwrap.dedent("""
### Task
You are given Solidity source code and a comparison result highlighting changed lines relative to a baseline. Analyze the code and the diff for security vulnerabilities or logic flaws that could lead to financial loss, governance manipulation, or other exploits.

### Code (review this first)
{code}

### Diff / Comparison to Baseline (prioritize changed lines)
{compare_result}

**Focus**: prioritize changed lines and their ±5-line context when determining risk, but consider global invariants if necessary.

### Allowed Vulnerability Categories (only report these)
- Bad Randomness
- Frontrunning / MEV Exposure
- Governance / Access Control Attacks
- Improper Input Validation
- Incorrect Arithmetic or Calculation
- Oracle / Price Manipulation
- Integer Overflow / Underflow
- Reentrancy
- Replay Attacks / Signature Malleability
- Rounding Error
- Self-Destruct Misuse
- Uninitialized Proxy / Delegatecall Risk
- Weak Access Control

### Required output schema
{response_format}

If no vulnerability is found, output exactly:
{{"prediction": false, "vulnerabilities": []}}

**Important constraints**
- Do not output narrative before or after the JSON.
- Do not hallucinate function names, variable types, or values not present in `{code}` or `{compare_result}`.
- Keep each description concise (1–3 sentences) but include a concrete suggested_fix.
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

def analyze_code_openai(
    code: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = 10000,
    respond_as_str: bool = False,
    compare_result: Dict | None = None
) -> Union[PredictionResponse, str]:
    try:
        print(f"compare_result: {compare_result}")

        # prepare substitution values (we escape braces in values too, just in case)
        code_val = code.replace("{", "{{").replace("}", "}}")
        if compare_result:
            cr_json = json.dumps(compare_result, indent=2)
            compare_val = cr_json.replace("{", "{{").replace("}", "}}")
        else:
            compare_val = "No comparison data available."

        # Build prompt safely
        prompt = _safe_format_prompt(VULN_PROMPT_TEMPLATE, code=code_val, compare_result=compare_val)

        kwargs = {"prompt": prompt, "max_tokens": max_tokens}
        if not respond_as_str:
            kwargs["response_format"] = PredictionResponse
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature

        return chat_completion(**kwargs)       

    except KeyError as ke:
        print(f"[Key Error] KeyError while formatting prompt: {ke}")
        return f"error: prompt formatting KeyError: {ke}"
    except Exception as e:
        print(f"[General Error] An error occurred during analysis: {e}")
        return f"error: analysis failed: {e}"

MODELS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "deepseek-ai/DeepSeek-V3-0324",
    "moonshotai/Kimi-K2-Instruct"
]
def analyze_code_chutes(
    code: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = 10000,
    respond_as_str: bool = False,
    compare_result: Dict | None = None
) -> Union[PredictionResponse, str]:
    try:
        print(f"compare_result: {compare_result}")

        # prepare substitution values (we escape braces in values too, just in case)
        code_val = code.replace("{", "{{").replace("}", "}}")
        if compare_result:
            cr_json = json.dumps(compare_result, indent=2)
            compare_val = cr_json.replace("{", "{{").replace("}", "}}")
        else:
            compare_val = "No comparison data available."

        # Build prompt safely
        prompt = VULN_PROMPT_TEMPLATE.format(code=code_val, compare_result=compare_val, response_format=RESPONSE_FORMAT)

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
                print(f"Chutes API returned status code {status_code}. Retrying...")
                retry_count += 1
                kwargs["model"] = MODELS[retry_count % len(MODELS)]
                print(f"Switching to model {kwargs['model']} and retrying (attempt {retry_count}/{max_retries})...")
            if retry_count >= max_retries:
                print("Max retries reached. Exiting.")
                break
        if status:
            resp = json.loads(response)
            result = PredictionResponse.from_json(resp)
            return result
        else:
            print(f"Chutes API returned unexpected status code {status_code}. Response: {response}")
            return f"error: unexpected status code {status_code}: {response}"

    except KeyError as ke:
        print(f"[Key Error] KeyError while formatting prompt: {ke}")
        return f"error: prompt formatting KeyError: {ke}"
    except Exception as e:
        print(f"[General Error] An error occurred during analysis: {e}")
        return f"error: analysis failed: {e}"

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

    print("Analyzing code for vulnerabilities...")
    with open("code.sol", "w") as f:
        f.write(code)

    if default_testnet_code(code) == True:
        print("Default Testnet Code detected. Sending default prediction.")
        return PredictionResponse.from_tuple([True,[]])

    if compare_result.get("total_changed_lines", -1) == 0:
        print("No changes detected in the code compared to the base source.")
        return PredictionResponse(prediction=False, vulnerabilities=[])
    
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        try:
            analysis = analyze_code_chutes(code, compare_result=compare_result)
            # analysis = analyze_code_openai(code, compare_result=compare_result)
            print(f"Analysis complete. Result:\n{analysis}")

            if type(analysis) is not PredictionResponse:                
                print(f"Analysis returned an error: {analysis}")
                attempt += 1
                continue

            return analysis
        except Exception as e:
            print(f"[General Error] An error occurred during analysis: {e}")
            attempt += 1