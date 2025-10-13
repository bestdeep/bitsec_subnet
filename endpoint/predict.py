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
import bittensor as bt

# Templates for prompts
VULN_PROMPT_TEMPLATE = """
### Instructions:
Analyze the provided code carefully, line by line, for any security vulnerabilities or logic flaws that could result in financial loss or exploitation.

Ignore privacy concerns since this code is deployed on a public blockchain.

### Code to Analyze:
{code}

### Comparison result with Base Source:
{compare_result}

### Acceptable Vulnerability Categories:
- Bad Randomness Vulnerability
- Frontrunning
- Governance Attacks
- Improper Input Validation
- Incorrect Calculation
- Oracle/Price Manipulation
- Arithmetic Overflow and Underflow Vulnerability
- Reentrancy
- Replay Attacks/Signature Malleability
- Rounding Error
- Self Destruct
- Uninitialized Proxy
- Weak Access Control

Focus on **changed lines and their context** when identifying vulnerabilities.

### Output Requirements:
1. Return JSON that matches the `PredictionResponse` model:
   - `prediction`: boolean, `true` if any vulnerabilities are found, `false` otherwise.
   - `vulnerabilities`: list of objects, each containing:
       - `category`: one of the acceptable vulnerability categories
       - `location`: line number(s) or snippet where the vulnerability occurs
       - `description`: short explanation of the vulnerability
       - `mitigation`: suggested fix or recommendation
2. Ensure the JSON is **valid and directly parseable** by the `PredictionResponse` model.
3. Sort vulnerabilities by severity if possible.

### Example Output:
{
  "prediction": true,
  "vulnerabilities": [
    {
      "category": "Reentrancy",
      "location": "lines 42-45",
      "description": "Function withdraw() can be re-entered before balance is updated.",
      "mitigation": "Update state before calling external contracts or use ReentrancyGuard."
    },
    {
      "category": "Arithmetic Overflow and Underflow Vulnerability",
      "location": "line 10",
      "description": "Addition of user input can overflow.",
      "mitigation": "Use safe math operations or Solidity 0.8+ built-in checks."
    }
  ]
}
"""

def analyze_code(
    code: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 10000,
    respond_as_str: bool = False,
    compare_result: Dict | None = None
) -> Union[PredictionResponse, str]:
    """
    Calls OpenAI API to analyze provided code for vulnerabilities.

    Args:
        code (str): The code to analyze.
        acceptable_vulnerability_categories (List[VulnerabilityCategory]): List of acceptable vulnerability categories.
        model (str, optional): The model to use for analysis.
        temperature (float, optional): Sampling temperature.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 4000.

    Returns:
        Union[PredictionResponse, str]: The analysis result from the model.
    """
    bt.logging.info(f"compare_result: {compare_result}")
    prompt = VULN_PROMPT_TEMPLATE.format(code=code, compare_result=json.dumps(compare_result, indent=2) if compare_result else "N/A")
    kwargs = {
        'prompt': prompt,
        'max_tokens': max_tokens
    }

    # If the model can return structured output, return the PredictionResponse object
    if not respond_as_str:
        kwargs['response_format'] = PredictionResponse

    if model is not None:
        kwargs['model'] = model
    if temperature is not None:
        kwargs['temperature'] = temperature
    
    return chat_completion(**kwargs)

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

    bt.logging.info("Analyzing code for vulnerabilities...")
    with open("code.sol", "w") as f:
        f.write(code)

    if default_testnet_code(code) == True:
        bt.logging.info("Default Testnet Code detected. Sending default prediction.")
        return PredictionResponse.from_tuple([True,[]])

    if compare_result.get("total_changed_lines", -1) == 0:
        bt.logging.info("No changes detected in the code compared to the base source.")
        return PredictionResponse(prediction=False, vulnerabilities=[])
    
    try:
        bt.logging.info(f"analyzing code:\n{code}")
        analysis = analyze_code(code, compare_result)
        bt.logging.info(f"Analysis complete. Result:\n{analysis}")

        if type(analysis) is not PredictionResponse:
            raise ValueError("Analysis did not return a PredictionResponse object.")

        return analysis
    except Exception as e:
        bt.logging.error(f"An error occurred during analysis: {e}")
        raise