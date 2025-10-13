import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from endpoint.predict import code_to_vulns
from endpoint.solidity_comparator import SolidityComparator
from time import time, monotonic
import asyncio
import hashlib
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

class CodeRequest(BaseModel):
    code: str

class _LRUCache:
    """Simple LRU cache with TTL for storing results."""
    def __init__(self, maxsize: int = 1024, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            ts, value = item
            if monotonic() - ts > self.ttl:
                # expired
                del self._data[key]
                return None
            # move to end (most recently used)
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (monotonic(), value)
            # evict oldest if over capacity
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

class VulnerabilityAPI:
    def __init__(self, lru_size: int = 1024, lru_ttl: float = 300.0):
        self.app = FastAPI()
        self._setup_routes()
        self.comparator = SolidityComparator()

        # Cache: completed results
        self._result_cache = _LRUCache(maxsize=lru_size, ttl=lru_ttl)
        # In-progress registry: map request_key -> asyncio.Future
        self._in_progress: Dict[str, asyncio.Future] = {}
        # Lock protecting _in_progress dict
        self._in_progress_lock = asyncio.Lock()

        start_time = time()
        self.comparator.load_candidates("samples/clean-codebases")
        end_time = time()
        print(f"Loaded candidates in {end_time - start_time:.2f} seconds")

    def _setup_routes(self):
        @self.app.post("/predict")
        async def predict_vulnerabilities(request: CodeRequest):
            code = request.code
            # create a stable key for the request
            req_key = hashlib.sha256(code.encode("utf-8")).hexdigest()

            # 1) check completed cache
            cached = await self._result_cache.get(req_key)
            if cached is not None:
                # return cached value immediately
                return {"vulnerabilities": cached}

            # 2) check if an identical request is currently being processed
            async with self._in_progress_lock:
                fut = self._in_progress.get(req_key)
                if fut is None:
                    # Create a Future placeholder and register it
                    fut = asyncio.get_running_loop().create_future()
                    self._in_progress[req_key] = fut
                    is_starter = True
                else:
                    is_starter = False

            if not is_starter:
                # Another task is processing the same request; await its result
                try:
                    result = await fut  # will raise if the starter set an exception
                except Exception as e:
                    # If the starter errored, remove the future and propagate
                    async with self._in_progress_lock:
                        self._in_progress.pop(req_key, None)
                    raise
                return {"vulnerabilities": result}

            # If we get here, this coroutine is responsible for computing the result.
            try:
                # comparator.compare_with_base_src may be CPU-bound; run in executor
                start_time = time()
                loop = asyncio.get_running_loop()
                compare_results = None
                compare_results = await loop.run_in_executor(
                    None, self.comparator.compare_with_base_src, code, 1, None, False, None, 100
                )
                # handle compare_results possibly empty
                compare_for_predict = compare_results[0] if compare_results else {}
                # call code_to_vulns (assumed cheap); if expensive, also move to executor
                result = await loop.run_in_executor(
                    None, code_to_vulns, code, compare_for_predict
                )

                # store in cache
                await self._result_cache.set(req_key, result)

                # set the future result for waiters
                fut.set_result(result)
                end_time = time()
                print(f"Processed request in {end_time - start_time:.2f} seconds")
                return {"vulnerabilities": result}
            except Exception as exc:
                # set exception for waiters
                if not fut.done():
                    fut.set_exception(exc)
                raise
            finally:
                # cleanup in_progress entry
                async with self._in_progress_lock:
                    self._in_progress.pop(req_key, None)

        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = VulnerabilityAPI()
    api.run()
