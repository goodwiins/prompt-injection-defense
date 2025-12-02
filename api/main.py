from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import structlog
import yaml

from src.coordination.guard_agent import GuardAgent
from src.response.circuit_breaker import CircuitBreaker

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Load config
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {}

app = FastAPI(title="Prompt Injection Defense API", version="1.0.0")

# Initialize components
guard_agent = GuardAgent(config)
circuit_breaker = CircuitBreaker(
    threshold=config.get("response", {}).get("circuit_breaker_limit", 10),
    time_window=60
)

# Metrics storage (simple in-memory for now)
metrics = {
    "total_requests": 0,
    "injections_detected": 0,
    "avg_latency_ms": 0.0
}

class DetectRequest(BaseModel):
    prompt: str
    metadata: Optional[Dict[str, Any]] = None

class DetectResponse(BaseModel):
    is_safe: bool
    confidence: float
    recommendation: str
    details: Dict[str, Any]

class BatchDetectRequest(BaseModel):
    prompts: List[str]

@app.middleware("http")
async def check_circuit_breaker(request, call_next):
    if circuit_breaker.is_open():
        return HTTPException(status_code=503, detail="Service unavailable due to high attack volume")
    response = await call_next(request)
    return response

@app.post("/detect", response_model=DetectResponse)
async def detect_prompt(request: DetectRequest):
    start_time = time.time()
    metrics["total_requests"] += 1
    
    try:
        result = guard_agent.analyze(request.prompt)
        
        if not result["is_safe"]:
            metrics["injections_detected"] += 1
            circuit_breaker.record_event()
            
        # Update latency metric
        latency = (time.time() - start_time) * 1000
        # Simple moving average for demo
        metrics["avg_latency_ms"] = (metrics["avg_latency_ms"] * 0.9) + (latency * 0.1)
        
        return DetectResponse(
            is_safe=result["is_safe"],
            confidence=result["confidence"],
            recommendation=result["recommendation"],
            details=result["details"]
        )
    except Exception as e:
        logger.error("Detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def batch_detect(request: BatchDetectRequest):
    results = []
    for prompt in request.prompts:
        # Re-use single detect logic or optimize for batch if GuardAgent supports it
        # GuardAgent currently does one by one but uses classifiers that could batch.
        # For now, simple loop.
        res = guard_agent.analyze(prompt)
        results.append(res)
        if not res["is_safe"]:
            circuit_breaker.record_event()
            
    return {"results": results}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "circuit_breaker_open": circuit_breaker.is_open()}

@app.get("/metrics")
async def get_metrics():
    return metrics
