# Utility function for interacting with LLMs
from typing import Type, Optional, TypeVar, Union, Dict, List, Any
import httpx
import os
from rich.console import Console
import random
console = Console()
from pydantic import BaseModel, Field
from time import time
import json
from bitsec.protocol import PredictionResponse

class GPTMessage(BaseModel):
    """Model for GPT message structure"""
    role: str = Field(..., description="Role of the message (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


# Define generic type T
T = TypeVar('T')

# Chutes API key config
if not os.getenv("CHUTES_API_KEY"):
    console.print("Chutes API key is not set. Please set the 'CHUTES_API_KEY' environment variable.")
    raise ValueError("Chutes API key is not set.")

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
CHUTES_EMBEDDING_URL = "https://chutes-sentence-transformers-all-minilm-l6-v2.chutes.ai/v1/embeddings"
CHUTES_INFERENCE_URL = "https://llm.chutes.ai/v1/chat/completions"

MODEL_PRICING: Dict[str, float] = {
    "deepseek-ai/DeepSeek-V3-0324": 0.2722,
    "agentica-org/DeepCoder-14B-Preview": 0.02,
    "deepseek-ai/DeepSeek-V3": 0.2722,
    "deepseek-ai/DeepSeek-R1": 0.2722,
    "deepseek-ai/DeepSeek-R1-0528": 0.2722,
    "NousResearch/DeepHermes-3-Mistral-24B-Preview": 0.1411,
    "NousResearch/DeepHermes-3-Llama-3-8B-Preview": 0.224,
    "chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8": 0.2722,
    "Qwen/Qwen3-32B": 0.0272,
    "Qwen/Qwen3-235B-A22B-Instruct-2507": 0.0000,
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": 0.1999,
    "Qwen/QwQ-32B": 0.0151,
    "chutesai/Mistral-Small-3.2-24B-Instruct-2506": 0.0302,
    "unsloth/gemma-3-27b-it": 0.1568,
    "agentica-org/DeepCoder-14B-Preview": 0.0151,
    "THUDM/GLM-Z1-32B-0414": 0.0302,
    "ArliAI/QwQ-32B-ArliAI-RpR-v1": 0.0151,
    "Qwen/Qwen3-30B-A3B": 0.0302,
    "hutesai/Devstral-Small-2505": 0.0302,
    "chutesai/Mistral-Small-3.1-24B-Instruct-2503": 0.0272,
    "chutesai/Llama-4-Scout-17B-16E-Instruct": 0.0302,
    "shisa-ai/shisa-v2-llama3.3-70b": 0.0302,
    "moonshotai/Kimi-Dev-72B": 0.1008,
    "moonshotai/Kimi-K2-Instruct": 0.5292,
    "all-hands/openhands-lm-32b-v0.1": 0.0246,
    "sarvamai/sarvam-m": 0.0224,
    "zai-org/GLM-4.5-FP8": 0.2000,
    "zai-org/GLM-4.5-Air": 0.0000,
    "rayonlabs/Gradients-Instruct-8B": 0.02,
    "openai/gpt-oss-120b": 0.2904,
    "rayonlabs/Gradients-Instruct-8B": 0.02
}
EMBEDDING_PRICE_PER_SECOND = 0.0001

# Default parameters
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 10000

# At the top with other globals
TOTAL_SPEND_CENTS = 0.0
TOTAL_SPEND_DESCRIPTION = []

_HTTP_CLIENT: Optional[httpx.Client] = None

def _get_http_client() -> httpx.Client:
    """Get or create a reusable HTTP client for connection pooling"""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.Client(timeout=60.0)
    return _HTTP_CLIENT

class ChutesProvider:
    """Provider for Chutes API inference"""
    
    def __init__(self):
        self.api_key = CHUTES_API_KEY
        
    @property
    def name(self) -> str:
        return "Chutes"
    
    def is_available(self) -> bool:
        """Check if Chutes provider is available"""
        return bool(self.api_key)
    
    def supports_model(self, model: str) -> bool:
        """Check if model is supported by Chutes (supports all models in pricing)"""
        return model in MODEL_PRICING
    
    def get_pricing(self, model: str) -> float:
        """Get Chutes pricing for the model"""
        if not self.supports_model(model):
            raise KeyError(f"Model {model} not supported by Chutes provider")
        return MODEL_PRICING[model]
    
    def inference(
        self,
        messages: List["GPTMessage"] = None,
        temperature: float = None,
        model: str = None,
        max_tokens: int = 2048,
    ) -> tuple[Any, int]:
        """Perform inference using Chutes API (streaming-safe, omits None fields)."""

        if not self.is_available():
            raise RuntimeError("Chutes API key not set")

        if temperature is not None:
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                console.print(f"Invalid temperature provided: {temperature}")
                raise ValueError(f"Invalid temperature provided: {temperature}")

        model = model or DEFAULT_MODEL
        if not self.supports_model(model):
            model = "openai/gpt-oss-120b"

        # Convert messages to dict format
        messages_dict = []
        if messages:
            for message in messages:
                if message:
                    messages_dict.append({"role": message.role, "content": message.content})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build body but only include optional keys when they are not None
        body = {
            "model": model,
            "messages": messages_dict,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature

        response_text = ""

        try:
            client = _get_http_client()
            with client.stream("POST", CHUTES_INFERENCE_URL, headers=headers, json=body) as response:

                if response.status_code != 200:
                    try:
                        error_bytes = response.read()
                        if isinstance(error_bytes, bytes):
                            error_message = error_bytes.decode(errors="replace")
                        else:
                            error_message = str(error_bytes)
                    except Exception:
                        error_message = f"Chutes returned status {response.status_code} and body could not be read."

                    console.print(
                        f"Chutes API request failed (model: {model}): {response.status_code} - {error_message}"
                    )
                    raise RuntimeError(f"Chutes API failed: {response.status_code} - {error_message}")

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    
                    try:
                        if isinstance(raw_line, bytes):
                            line = raw_line.decode(errors="replace").strip()
                        else:
                            line = str(raw_line).strip()
                    except Exception:
                        continue

                    # handle SSE-style "data: " prefix
                    if line.startswith("data:"):
                        chunk_data = line[len("data:"):].strip()
                    else:
                        chunk_data = line

                    if chunk_data == "[DONE]":
                        break

                    # try decode JSON chunk
                    try:
                        chunk_json = json.loads(chunk_data)
                    except json.JSONDecodeError:
                        # Not JSON — append directly
                        response_text += chunk_data
                        continue

                    if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                        choice = chunk_json["choices"][0]
                        if isinstance(choice, dict):
                            # Try delta.content first (streaming format)
                            delta = choice.get("delta", {})
                            if isinstance(delta, dict):
                                content = delta.get("content", "")
                                if content:
                                    response_text += str(content)

        except Exception as e:
            console.print(f"Error in Chutes API request: {e}")
            raise

        console.print(f"Chutes raw response text: {response_text}")
        return response_text, 200

    def embed(self, input_text: str) -> Dict[str, Any]:
        """Synchronous embedding request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # body = {"inputs": input_text}
        body = {
            "input": input_text,
            "model": None
            }

        try:
            client = _get_http_client()
            response = client.post(CHUTES_EMBEDDING_URL, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            data = data.get("data", [])
            if not data or not isinstance(data, list) or "embedding" not in data[0]:
                raise RuntimeError("Invalid embedding response from Chutes")
            embedding = data[0]["embedding"]
            return embedding
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            raise TimeoutError("Embedding request timed out")
        except Exception as e:
            raise RuntimeError(f"Embedding error: {str(e)}")
            
chutes_client = ChutesProvider()
