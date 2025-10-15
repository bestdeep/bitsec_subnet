import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from endpoint.predict import code_to_vulns
from endpoint.solidity_comparator_embeddings import SolidityComparatorEmbeddings
from time import monotonic
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
    # Optional timeout for waiters (seconds). Set to None to wait indefinitely.
    WAITER_TIMEOUT: Optional[float] = 30.0

    def __init__(self, lru_size: int = 1024, lru_ttl: float = 300.0):
        self.app = FastAPI()
        self._setup_routes()
        self.comparator = SolidityComparatorEmbeddings(
            index_method="faiss",
            shortlist_k=200,
            embed_batch_size=32,
            mongo_uri="mongodb://localhost:27017",
            mongo_db="solidity_embeddings",
            mongo_collection="candidates"
        )

        # Cache: completed results
        self._result_cache = _LRUCache(maxsize=lru_size, ttl=lru_ttl)
        # In-progress registry: map request_key -> asyncio.Future
        self._in_progress: Dict[str, asyncio.Future] = {}
        # Lock protecting _in_progress dict
        self._in_progress_lock = asyncio.Lock()

        start_time = monotonic()
        self.comparator.load_candidates(directory="samples/clean-codebases", use_cache=True, workers=8)
        end_time = monotonic()
        print(f"Loaded candidates from mongo db within {end_time - start_time:.2f} seconds")

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
                    loop = asyncio.get_running_loop()
                    fut = loop.create_future()
                    self._in_progress[req_key] = fut
                    is_starter = True
                else:
                    is_starter = False

            if not is_starter:
                # Another task is processing the same request; await its result (with optional timeout)
                try:
                    if self.WAITER_TIMEOUT is not None:
                        result = await asyncio.wait_for(fut, timeout=self.WAITER_TIMEOUT)
                    else:
                        result = await fut
                except asyncio.TimeoutError:
                    # Optionally remove stale in_progress entry if still the same future
                    async with self._in_progress_lock:
                        # only remove if the mapping still points to this future
                        if self._in_progress.get(req_key) is fut:
                            self._in_progress.pop(req_key, None)
                    raise HTTPException(status_code=504, detail="Timed out waiting for result")
                except asyncio.CancelledError:
                    # Waiter cancelled — just propagate cancellation
                    raise
                except Exception as e:
                    # The starter set an exception — remove in_progress and propagate
                    async with self._in_progress_lock:
                        self._in_progress.pop(req_key, None)
                    raise
                return {"vulnerabilities": result}

            # If we get here, this coroutine is responsible for computing the result.
            try:
                start_time = monotonic()
                loop = asyncio.get_running_loop()
                compare_results = None
                # If you want to run heavy compare in executor, uncomment:
                compare_results = await loop.run_in_executor(
                    None, self.comparator.compare_with_base_src, code, 1, None, False, 3
                )
                compare_for_predict = compare_results[0] if compare_results else {}
                # call code_to_vulns (assumed cheap); if expensive, also move to executor
                result = await loop.run_in_executor(
                    None, code_to_vulns, code, compare_for_predict
                )

                # store in cache
                await self._result_cache.set(req_key, result)

                # set the future result for waiters (if not already done)
                if not fut.done():
                    fut.set_result(result)

                end_time = monotonic()
                print(f"Processed request in {end_time - start_time:.2f} seconds")
                return {"vulnerabilities": result}
            except Exception as exc:
                # set exception for waiters if not already set
                if not fut.done():
                    fut.set_exception(exc)
                raise
            finally:
                # defensive: ensure waiters don't hang if something weird happened
                if not fut.done():
                    try:
                        fut.set_exception(RuntimeError("Processing did not complete successfully"))
                    except Exception:
                        # ignore if fut was completed concurrently
                        pass
                async with self._in_progress_lock:
                    # only remove if the mapping still points to this future
                    if self._in_progress.get(req_key) is fut:
                        self._in_progress.pop(req_key, None)

        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = VulnerabilityAPI()
    api.run()
