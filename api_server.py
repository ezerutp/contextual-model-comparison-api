"""
API REST para comparación de modelos de IA
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from ai_model_comparison import AIModelComparator, CompareModelRequest, CompareModelResponse
import os
import logging

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
DEFAULT_REPO_PATH = os.getenv("REPO_PATH", ".")

# Modelos Pydantic
class CompareModelRequestDTO(BaseModel):
    """DTO de solicitud de comparación"""
    prompt: str = Field(..., min_length=1, description="Prompt a enviar a los modelos")
    system_prompt: Optional[str] = Field(None, description="Instrucciones del sistema")
    include_repo_context: Optional[bool] = Field(True, description="Incluir contexto del repositorio")
    repo_path: Optional[str] = Field(
        None,
        description="Ruta del repositorio. Si no se envía, usa REPO_PATH o el cwd del servidor."
    )


class CompareModelResponseDTO(BaseModel):
    """DTO de respuesta de comparación"""
    provider: str
    model: str
    latency_ms: float
    output: str
    input_length: int
    output_length: int


class ComparisonMetricsDTO(BaseModel):
    """DTO de métricas de comparación"""
    faster_model: str
    latency_difference_ms: float
    longer_response: str
    length_difference: int


class BothModelsComparisonDTO(BaseModel):
    """DTO de comparación de ambos modelos"""
    models: Dict[str, CompareModelResponseDTO]
    comparison: ComparisonMetricsDTO


class ApiResultDTO(BaseModel):
    """DTO de resultado de API"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


# Inicializa FastAPI
app = FastAPI(
    title="AI Model Comparison API",
    description="API para comparar respuestas entre model one y model two",
    version="1.0.0"
)

# Dependencia: inicializa el comparador ijiji
def get_comparator() -> AIModelComparator:
    """Inicializa y retorna el comparador de modelos"""
    try:
        return AIModelComparator(GOOGLE_AI_API_KEY, default_repo_path=DEFAULT_REPO_PATH)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


def validate_repo_path(repo_path: Optional[str]) -> Optional[str]:
    """Valida que la ruta exista y sea un directorio."""
    if repo_path is None:
        return None
    if not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail=f"repo_path invalido: '{repo_path}' no es un directorio")
    return repo_path

# por si es que lo metemos a un dockerfile
@app.get("/health")
async def health():
    """Verifica el estado de la API"""
    return {"status": "healthy"}


@app.post("/api/ai-model-comparison/model-one", response_model=ApiResultDTO)
async def compare_with_model_one(
    request: CompareModelRequestDTO,
    comparator: AIModelComparator = Depends(get_comparator)
):
    """
    Compara el prompt con el modelo model one (ej: Gemini 2.5 Flash)
    
    Similar a: AIModelComparisonController.compareWithModelOne()
    """
    try:
        model_request = CompareModelRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            include_repo_context=request.include_repo_context,
            repo_path=validate_repo_path(request.repo_path)
        )
        response = comparator.compare_with_model_one(model_request)
        
        return ApiResultDTO(
            success=True,
            data=CompareModelResponseDTO(
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency_ms,
                output=response.output,
                input_length=response.input_length,
                output_length=response.output_length
            )
        )
    except Exception as e:
        logger.error(f"Error comparando con model one: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


@app.post("/api/ai-model-comparison/model-two", response_model=ApiResultDTO)
async def compare_with_model_two(
    request: CompareModelRequestDTO,
    comparator: AIModelComparator = Depends(get_comparator)
):
    """
    Compara el prompt con el modelo model two (ej: Gemma 3 27B IT)
    
    Similar a: AIModelComparisonController.compareWithModelTwo()
    """
    try:
        model_request = CompareModelRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            include_repo_context=request.include_repo_context,
            repo_path=validate_repo_path(request.repo_path)
        )
        response = comparator.compare_with_model_two(model_request)
        
        return ApiResultDTO(
            success=True,
            data=CompareModelResponseDTO(
                provider=response.provider,
                model=response.model,
                latency_ms=response.latency_ms,
                output=response.output,
                input_length=response.input_length,
                output_length=response.output_length
            )
        )
    except Exception as e:
        logger.error(f"Error comparando con model two: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


@app.post("/api/ai-model-comparison/compare-both")
async def compare_both_models(
    request: CompareModelRequestDTO,
    comparator: AIModelComparator = Depends(get_comparator)
):
    """
    Compara con ambos modelos simultáneamente y retorna comparación
    """
    try:
        model_request = CompareModelRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            include_repo_context=request.include_repo_context,
            repo_path=validate_repo_path(request.repo_path)
        )
        
        model_one_response = comparator.compare_with_model_one(model_request)
        model_two_response = comparator.compare_with_model_two(model_request)

        models_map: Dict[str, CompareModelResponseDTO] = {}
        for model_response in [model_one_response, model_two_response]:
            key = model_response.model
            if key in models_map:
                suffix = 2
                while f"{key}#{suffix}" in models_map:
                    suffix += 1
                key = f"{key}#{suffix}"
            models_map[key] = CompareModelResponseDTO(
                provider=model_response.provider,
                model=model_response.model,
                latency_ms=model_response.latency_ms,
                output=model_response.output,
                input_length=model_response.input_length,
                output_length=model_response.output_length
            )
        
        comparison_data = BothModelsComparisonDTO(
            models=models_map,
            comparison=ComparisonMetricsDTO(
                faster_model=model_one_response.model if model_one_response.latency_ms < model_two_response.latency_ms else model_two_response.model,
                latency_difference_ms=abs(model_one_response.latency_ms - model_two_response.latency_ms),
                longer_response=model_one_response.model if model_one_response.output_length > model_two_response.output_length else model_two_response.model,
                length_difference=abs(model_one_response.output_length - model_two_response.output_length)
            )
        )
        
        return ApiResultDTO(
            success=True,
            data=comparison_data.model_dump()
        )
    except Exception as e:
        logger.error(f"Error en comparación: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
