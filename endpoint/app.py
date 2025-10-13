import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from endpoint.predict import code_to_vulns
from endpoint.solidity_comparator import SolidityComparator
from time import time

class CodeRequest(BaseModel):
    code: str

class VulnerabilityAPI:
    def __init__(self):
        self.app = FastAPI()
        self._setup_routes()
        self.comparator = SolidityComparator()
        start_time = time()
        self.comparator.load_candidates("samples/clean-codebases")
        end_time = time()
        print(f"Loaded candidates in {end_time - start_time:.2f} seconds")

    def _setup_routes(self):
        @self.app.post("/predict")
        async def predict_vulnerabilities(request: CodeRequest):
            code = request.code
            compare_results = self.comparator.compare_with_base_src(code)
            result = code_to_vulns(code, compare_results[0] if compare_results else {})
            return {"vulnerabilities": result}

        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    api = VulnerabilityAPI()
    api.run()
