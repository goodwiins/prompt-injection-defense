
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import structlog
import os
from contextlib import asynccontextmanager
from src.detection.embedding_classifier import EmbeddingClassifier

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Global classifier instance
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model on startup.
    """
    global classifier
    logger.info("Server starting up...")
    
    try:
        # Initialize the classifier
        classifier = EmbeddingClassifier(model_name="all-MiniLM-L6-v2")
        
        # Check if a trained model exists and load it
        model_path = "models/bit_xgboost_model.json"
        if os.path.exists(model_path):
            classifier.load_model(model_path)
            logger.info("Loaded trained model", path=model_path)
        else:
            logger.warning("Trained model not found at 'models/bit_xgboost_model.json'. Using untrained/default state.")
            # In a real scenario, you might want to prevent startup or load a fallback
            
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        # Depending on requirements, we might want to raise the exception to fail startup
        # raise e 
    
    yield
    
    logger.info("Server shutting down...")

app = FastAPI(title="Prompt Injection Defense API", lifespan=lifespan)

class DetectionRequest(BaseModel):
    text: str
    threshold: float = 0.5

class DetectionResponse(BaseModel):
    text: str
    is_injection: bool
    confidence: float
    score: float
    latency_ms: float

@app.post("/detect", response_model=DetectionResponse)
async def detect_injection(request: DetectionRequest):
    """
    Detect if the provided text contains a prompt injection.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        import time
        start_time = time.time()
        
        # Override threshold if provided in request, otherwise use model's threshold
        # For this request, we'll temporarily use the request threshold without modifying the global model state persistently if possible,
        # but the predict method uses self.threshold. 
        # Modifying self.threshold is not thread-safe. 
        # Ideally, predict/predict_proba should accept a threshold argument.
        # Given the current implementation of EmbeddingClassifier, we'll calculate based on score manually here.
        
        # Get probability
        probs = classifier.predict_proba([request.text])
        score = float(probs[0][1])
        
        # Determine injection status based on request threshold
        is_injection = score >= request.threshold
        
        latency = (time.time() - start_time) * 1000
        
        logger.info(
            "Request processed", 
            text_length=len(request.text), 
            score=score, 
            is_injection=is_injection,
            latency_ms=latency
        )
        
        return DetectionResponse(
            text=request.text,
            is_injection=is_injection,
            confidence=max(score, 1-score),
            score=score,
            latency_ms=latency
        )

    except Exception as e:
        logger.error("Detection failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None and classifier.is_trained}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
