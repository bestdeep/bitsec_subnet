# Utility function for interacting with LLMs
import bittensor as bt
from typing import Type, Optional, TypeVar, Union
import httpx
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from rich.console import Console
import random
console = Console()

# Define generic type T
T = TypeVar('T')

# Chutes API key config
if not os.getenv("CHUTES_API_KEY"):
    bt.logging.error("Chutes API key is not set. Please set the 'CHUTES_API_KEY' environment variable.")
    raise ValueError("Chutes API key is not set.")

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
CHUTES_EMBEDDING_URL = "https://chutes-baai-bge-large-en-v1-5.chutes.ai/embed"
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

# Default parameters
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "moonshotai/Kimi-K2-Instruct")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 10000

class ChutesProvider(InferenceProvider):
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
    
    async def ainference(
        self,
        messages: List[GPTMessage] = None,
        temperature: float = None,
        model: str = None,
    ) -> tuple[str, int]:
        """Perform inference using Chutes API"""
        
        if not self.is_available():
            raise RuntimeError("Chutes API key not set")
            
        if not self.supports_model(model):
            raise ValueError(f"Model {model} not supported by Chutes provider")
        
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

        body = {
            "model": model,
            "messages": messages_dict,
            "stream": True,
            "max_tokens": 2048,
            "temperature": temperature,
            "seed": random.randint(0, 2**32 - 1),
        }

        response_text = ""
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", CHUTES_INFERENCE_URL, headers=headers, json=body) as response:

                if response.status_code != 200:
                    error_text = await response.aread()
                    if isinstance(error_text, bytes):
                        error_message = error_text.decode()
                    else:
                        error_message = str(error_text)
                    logger.error(
                        f"Chutes API request failed (model: {model}): {response.status_code} - {error_message}"
                    )
                    return error_message, response.status_code

                # Process streaming response
                async for chunk in response.aiter_lines():
                    if chunk:
                        chunk_str = chunk.strip()
                        if chunk_str.startswith("data: "):
                            chunk_data = chunk_str[6:]  # Remove "data: " prefix

                            if chunk_data == "[DONE]":
                                break

                            try:
                                chunk_json = json.loads(chunk_data)
                                if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                    choice = chunk_json["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            response_text += content

                            except json.JSONDecodeError:
                                # Skip malformed JSON chunks
                                continue
        
        # Validate that we received actual content
        if not response_text.strip():
            # Don't care too much about empty responses for now
            error_msg = f"Chutes API returned empty response for model {model}. This may indicate API issues or malformed streaming response."            
            return error_msg, 200  # Status was 200 but response was empty
        
        return response_text, 200

    async def aembed(self, input_text: str = None) -> Dict[str, Any]:
        """Get embedding for text input"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {"inputs": input_text, "seed": random.randint(0, 2**32 - 1)}

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(CHUTES_EMBEDDING_URL, headers=headers, json=body)
                response.raise_for_status()

                total_time_seconds = time.time() - start_time
                cost = total_time_seconds * EMBEDDING_PRICE_PER_SECOND

                response_data = response.json()
                return response_data

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error in embedding request: {e.response.status_code} - {e.response.text}"
            )
            return {"error": f"HTTP error in embedding request: {e.response.status_code} - {e.response.text}"}
        except httpx.TimeoutException:
            logger.error(f"Timeout in embedding request")
            return {"error": "Embedding request timed out. Please try again."}
        except Exception as e:
            logger.error(f"Error in embedding request: {e}")
            return {"error": f"Error in embedding request: {str(e)}"}

    def inference(
        self,
        messages: List[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        model: str = DEFAULT_MODEL,
    ) -> tuple[str, int]:
        """Perform synchronous inference using Chutes API"""
        if not self.is_available():
            raise RuntimeError("Chutes API key not set")

        if not self.supports_model(model):
            raise ValueError(f"Model {model} not supported by Chutes provider")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": model,
            "messages": messages,
            "stream": False,  # Sync version does not stream
            "max_tokens": 2048,
            "temperature": temperature,
            "seed": random.randint(0, 2**32 - 1),
        }

        response_text = ""

        try:
            with httpx.Client(timeout=None) as client:
                response = client.post(CHUTES_INFERENCE_URL, headers=headers, json=body)

                if response.status_code != 200:
                    return response.text, response.status_code

                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    response_text = data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error in Chutes API request: {str(e)}", 500

        return response_text, 200

    def embed(self, input_text: str) -> Dict[str, Any]:
        """Synchronous embedding request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {"inputs": input_text, "seed": random.randint(0, 2**32 - 1)}

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(CHUTES_EMBEDDING_URL, headers=headers, json=body)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except httpx.TimeoutException:
            return {"error": "Embedding request timed out"}
        except Exception as e:
            return {"error": f"Embedding error: {str(e)}"}
            
chutes_client = ChutesProvider()