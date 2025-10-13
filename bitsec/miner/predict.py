from bitsec.protocol import PredictionResponse
from bitsec.miner.prompt import code_to_vulns
import httpx

## Predicting vulnerabilities is a multifaceted task. Here are some example improvements:
# - train custom model
# - use a more powerful foundational model
# - improve prompt
# - increase inference time compute
# - augmented static analysis output
def predict(code: str) -> PredictionResponse:
    """
    Perform prediction. You may need to modify this if you train a custom model.

    Args:
        code (str): The input str is a challenge that either has a severe code vulnerability or does not.

    Returns:
        PredictionResponse: The predicted output value and list of vulnerabilities.
    """
    
    with httpx.Client(timeout=None) as client:
        response = client.post("http://localhost:8000/predict", json={"code": code})

        if response.status_code != 200:
            return response.text

        data = response.json()
        vulnerabilities = data.get("vulnerabilities", {})
        result = PredictionResponse.from_json(vulnerabilities)
        print(f"Predicted vulnerabilities: {result}")
        return result