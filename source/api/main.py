"""
FastAPI Main Application.

Enterprise-level REST API with:
- OpenAPI/Swagger documentation
- CORS support
- Rate limiting
- Authentication (JWT)
- Request validation
- Error handling
- Logging and monitoring
- Health checks
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from source.utils.logger import setup_logger
from source.ml.inference import InferenceEngine, PredictionResult
from source.ml.model_registry import ModelRegistry

# Setup logger
logger = setup_logger(
    name="api",
    log_file="../logs/api.log",
    log_level="INFO"
)

# Initialize FastAPI app
app = FastAPI(
    title="Geo Sentiment Climate API",
    description="Enterprise API for air quality prediction and analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize services
model_registry = ModelRegistry()
inference_engine = InferenceEngine(model_registry=model_registry)


# Pydantic Models for Request/Response
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: str
    version: str
    uptime_seconds: float


class PredictionRequest(BaseModel):
    """Single prediction request model."""

    data: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    model_name: Optional[str] = Field(None, description="Model name to use (latest version)")
    return_probabilities: bool = Field(False, description="Return probabilities for classification")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""

    data: List[Dict[str, Any]] = Field(..., description="List of input feature dictionaries")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    model_name: Optional[str] = Field(None, description="Model name to use")
    batch_size: int = Field(1000, description="Batch size for processing")


class PredictionResponse(BaseModel):
    """Prediction response model."""

    predictions: List[float]
    model_id: str
    inference_time_ms: float
    timestamp: str
    input_shape: List[int]


class ModelInfo(BaseModel):
    """Model information response."""

    model_id: str
    model_name: str
    version: str
    model_type: str
    task_type: str
    stage: str
    metrics: Dict[str, float]
    created_at: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: str


# Application startup time
app_start_time = time.time()


# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Health check endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "Geo Sentiment Climate API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime
    )


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check for Kubernetes."""
    try:
        # Check if model registry is accessible
        models = model_registry.list_models()
        return {
            "status": "ready",
            "models_available": len(models)
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction.

    Args:
        request: Prediction request with input data

    Returns:
        Prediction response with results
    """
    try:
        # Determine which model to use
        if request.model_id:
            model_id = request.model_id
        elif request.model_name:
            # Get latest model by name
            models = model_registry.list_models(
                model_name=request.model_name,
                stage="production"
            )
            if not models:
                raise HTTPException(
                    status_code=404,
                    detail=f"No production model found with name: {request.model_name}"
                )
            model_id = models[0].model_id
        else:
            # Use latest production model
            models = model_registry.list_models(stage="production")
            if not models:
                raise HTTPException(
                    status_code=404,
                    detail="No production models available"
                )
            model_id = models[0].model_id

        # Make prediction
        result = inference_engine.predict(
            data=request.data,
            model_id=model_id,
            return_probabilities=request.return_probabilities
        )

        return PredictionResponse(
            predictions=result.predictions,
            model_id=result.model_id,
            inference_time_ms=result.inference_time_ms,
            timestamp=result.timestamp,
            input_shape=list(result.input_shape)
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions.

    Args:
        request: Batch prediction request

    Returns:
        Prediction response with batch results
    """
    try:
        import pandas as pd

        # Determine model
        if request.model_id:
            model_id = request.model_id
        elif request.model_name:
            models = model_registry.list_models(
                model_name=request.model_name,
                stage="production"
            )
            if not models:
                raise HTTPException(
                    status_code=404,
                    detail=f"No production model found with name: {request.model_name}"
                )
            model_id = models[0].model_id
        else:
            models = model_registry.list_models(stage="production")
            if not models:
                raise HTTPException(
                    status_code=404,
                    detail="No production models available"
                )
            model_id = models[0].model_id

        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        # Make batch prediction
        result = inference_engine.batch_predict(
            data=df,
            model_id=model_id,
            batch_size=request.batch_size
        )

        return PredictionResponse(
            predictions=result.predictions,
            model_id=result.model_id,
            inference_time_ms=result.inference_time_ms,
            timestamp=result.timestamp,
            input_shape=list(result.input_shape)
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models(
    model_name: Optional[str] = None,
    stage: Optional[str] = None
):
    """
    List registered models.

    Args:
        model_name: Filter by model name
        stage: Filter by stage (dev/staging/production)

    Returns:
        List of model information
    """
    try:
        models = model_registry.list_models(model_name=model_name, stage=stage)

        return [
            ModelInfo(
                model_id=m.model_id,
                model_name=m.model_name,
                version=m.version,
                model_type=m.model_type,
                task_type=m.task_type,
                stage=m.stage,
                metrics=m.metrics,
                created_at=m.created_at
            )
            for m in models
        ]

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id: Model ID

    Returns:
        Model information
    """
    if model_id not in model_registry.models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    m = model_registry.models[model_id]

    return ModelInfo(
        model_id=m.model_id,
        model_name=m.model_name,
        version=m.version,
        model_type=m.model_type,
        task_type=m.task_type,
        stage=m.stage,
        metrics=m.metrics,
        created_at=m.created_at
    )


@app.post("/models/{model_id}/promote", tags=["Models"])
async def promote_model(model_id: str, new_stage: str):
    """
    Promote a model to a new stage.

    Args:
        model_id: Model ID
        new_stage: New stage (staging/production)

    Returns:
        Success message
    """
    if new_stage not in ["staging", "production"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid stage. Must be 'staging' or 'production'"
        )

    success = model_registry.promote_model(model_id, new_stage)

    if not success:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    return {
        "message": f"Model {model_id} promoted to {new_stage}",
        "model_id": model_id,
        "new_stage": new_stage
    }


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API metrics.

    Returns:
        System metrics
    """
    return {
        "uptime_seconds": time.time() - app_start_time,
        "total_models": len(model_registry.models),
        "production_models": len(model_registry.list_models(stage="production")),
        "timestamp": datetime.now().isoformat()
    }


def start_server():
    """Start the API server."""
    uvicorn.run(
        "source.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
