"""
API REST para comparación de modelos de IA
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
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


class ChunkedCompareModelRequestDTO(CompareModelRequestDTO):
    """DTO de solicitud de comparación por chunks"""
    chunk_size: int = Field(1200, ge=100, le=20000, description="Tamaño máximo de cada chunk del prompt")
    chunk_overlap: int = Field(200, ge=0, le=5000, description="Solapamiento entre chunks")
    max_chunks: Optional[int] = Field(8, ge=1, le=100, description="Cantidad máxima de chunks a procesar")
    generate_final_analysis: bool = Field(
        False,
        description="Si true, genera una síntesis final con model one usando los resultados por chunk"
    )
    analysis_instructions: Optional[str] = Field(
        None,
        description="Instrucciones para la síntesis final (cuando generate_final_analysis=true)"
    )


class SingleModelChunkedCompareRequestDTO(CompareModelRequestDTO):
    """DTO de solicitud de chunking para un solo modelo"""
    target_model: str = Field(
        "model-one",
        description="Modelo objetivo: 'model-one' o 'model-two'"
    )
    chunk_size: int = Field(1200, ge=100, le=20000, description="Tamaño máximo de cada chunk del prompt")
    chunk_overlap: int = Field(200, ge=0, le=5000, description="Solapamiento entre chunks")
    max_chunks: Optional[int] = Field(8, ge=1, le=100, description="Cantidad máxima de chunks a procesar")


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


class ChunkResultDTO(BaseModel):
    """DTO del resultado de un chunk"""
    chunk_index: int
    chunk_start: int
    chunk_end: int
    chunk_text: str
    models: Dict[str, CompareModelResponseDTO]
    comparison: ComparisonMetricsDTO


class ChunkedBothModelsComparisonDTO(BaseModel):
    """DTO de comparación por chunks"""
    original_prompt_length: int
    processed_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: List[ChunkResultDTO]
    aggregate: ComparisonMetricsDTO
    final_analysis: Optional[str] = None


class SingleModelChunkResultDTO(BaseModel):
    """DTO del resultado de un chunk para un solo modelo"""
    chunk_index: int
    chunk_start: int
    chunk_end: int
    chunk_text: str
    response: CompareModelResponseDTO


class SingleModelChunkedComparisonDTO(BaseModel):
    """DTO agregado de comparación por chunks para un solo modelo"""
    target_model: str
    original_prompt_length: int
    processed_chunks: int
    chunk_size: int
    chunk_overlap: int
    total_latency_ms: float
    average_latency_ms: float
    total_output_length: int
    chunks: List[SingleModelChunkResultDTO]


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


def split_prompt_into_chunks(text: str, chunk_size: int, chunk_overlap: int, max_chunks: Optional[int]) -> List[Dict[str, Any]]:
    """Divide el texto en chunks por longitud de caracteres."""
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap debe ser menor que chunk_size")

    chunks: List[Dict[str, Any]] = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end]
            }
        )
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        start += step

    return chunks


def run_single_model(
    comparator: AIModelComparator,
    request: CompareModelRequest,
    target_model: str
) -> CompareModelResponse:
    """Ejecuta la comparación para un solo modelo según el selector."""
    normalized_model = target_model.strip().lower()
    if normalized_model == "model-one":
        return comparator.compare_with_model_one(request)
    if normalized_model == "model-two":
        return comparator.compare_with_model_two(request)
    raise HTTPException(status_code=400, detail="target_model invalido. Usa 'model-one' o 'model-two'")

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


@app.post("/api/ai-model-comparison/compare-both-chunked", response_model=ApiResultDTO)
async def compare_both_models_chunked(
    request: ChunkedCompareModelRequestDTO,
    comparator: AIModelComparator = Depends(get_comparator)
):
    """
    Divide el prompt en chunks y compara ambos modelos por cada chunk.
    Opcionalmente, genera una síntesis final.
    """
    try:
        validated_repo_path = validate_repo_path(request.repo_path)
        prompt_chunks = split_prompt_into_chunks(
            text=request.prompt,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_chunks=request.max_chunks
        )

        chunk_results: List[ChunkResultDTO] = []
        total_latency_model_one = 0.0
        total_latency_model_two = 0.0
        total_output_len_model_one = 0
        total_output_len_model_two = 0

        for idx, chunk_data in enumerate(prompt_chunks):
            chunk_request = CompareModelRequest(
                prompt=chunk_data["text"],
                system_prompt=request.system_prompt,
                include_repo_context=request.include_repo_context,
                repo_path=validated_repo_path
            )

            model_one_response = comparator.compare_with_model_one(chunk_request)
            model_two_response = comparator.compare_with_model_two(chunk_request)

            total_latency_model_one += model_one_response.latency_ms
            total_latency_model_two += model_two_response.latency_ms
            total_output_len_model_one += model_one_response.output_length
            total_output_len_model_two += model_two_response.output_length

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

            chunk_results.append(
                ChunkResultDTO(
                    chunk_index=idx,
                    chunk_start=chunk_data["start"],
                    chunk_end=chunk_data["end"],
                    chunk_text=chunk_data["text"],
                    models=models_map,
                    comparison=ComparisonMetricsDTO(
                        faster_model=model_one_response.model if model_one_response.latency_ms < model_two_response.latency_ms else model_two_response.model,
                        latency_difference_ms=abs(model_one_response.latency_ms - model_two_response.latency_ms),
                        longer_response=model_one_response.model if model_one_response.output_length > model_two_response.output_length else model_two_response.model,
                        length_difference=abs(model_one_response.output_length - model_two_response.output_length)
                    )
                )
            )

        aggregate = ComparisonMetricsDTO(
            faster_model=comparator.model_name_one if total_latency_model_one < total_latency_model_two else comparator.model_name_two,
            latency_difference_ms=abs(total_latency_model_one - total_latency_model_two),
            longer_response=comparator.model_name_one if total_output_len_model_one > total_output_len_model_two else comparator.model_name_two,
            length_difference=abs(total_output_len_model_one - total_output_len_model_two)
        )

        final_analysis: Optional[str] = None
        if request.generate_final_analysis and chunk_results:
            synthesis_instructions = request.analysis_instructions or (
                "Analiza de forma profunda y comparativa los resultados por chunk. "
                "Incluye patrones, contradicciones, fortalezas/debilidades por modelo y conclusiones accionables."
            )
            synthesis_prompt = (
                f"{synthesis_instructions}\n\n"
                f"Resultados por chunk:\n"
                f"{[chunk.model_dump() for chunk in chunk_results]}"
            )
            synthesis_request = CompareModelRequest(
                prompt=synthesis_prompt,
                system_prompt="Eres un analista técnico riguroso.",
                include_repo_context=False,
                repo_path=validated_repo_path
            )
            synthesis_response = comparator.compare_with_model_one(synthesis_request)
            final_analysis = synthesis_response.output

        payload = ChunkedBothModelsComparisonDTO(
            original_prompt_length=len(request.prompt),
            processed_chunks=len(chunk_results),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            chunks=chunk_results,
            aggregate=aggregate,
            final_analysis=final_analysis
        )

        return ApiResultDTO(success=True, data=payload.model_dump())
    except Exception as e:
        logger.error(f"Error en comparación por chunks: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


@app.post("/api/ai-model-comparison/compare-single-chunked", response_model=ApiResultDTO)
async def compare_single_model_chunked(
    request: SingleModelChunkedCompareRequestDTO,
    comparator: AIModelComparator = Depends(get_comparator)
):
    """
    Divide el prompt en chunks y ejecuta un único modelo por cada chunk.
    """
    try:
        validated_repo_path = validate_repo_path(request.repo_path)
        prompt_chunks = split_prompt_into_chunks(
            text=request.prompt,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_chunks=request.max_chunks
        )

        chunk_results: List[SingleModelChunkResultDTO] = []
        total_latency_ms = 0.0
        total_output_length = 0
        normalized_target_model = request.target_model.strip().lower()

        for idx, chunk_data in enumerate(prompt_chunks):
            chunk_request = CompareModelRequest(
                prompt=chunk_data["text"],
                system_prompt=request.system_prompt,
                include_repo_context=request.include_repo_context,
                repo_path=validated_repo_path
            )

            model_response = run_single_model(
                comparator=comparator,
                request=chunk_request,
                target_model=normalized_target_model
            )
            total_latency_ms += model_response.latency_ms
            total_output_length += model_response.output_length

            chunk_results.append(
                SingleModelChunkResultDTO(
                    chunk_index=idx,
                    chunk_start=chunk_data["start"],
                    chunk_end=chunk_data["end"],
                    chunk_text=chunk_data["text"],
                    response=CompareModelResponseDTO(
                        provider=model_response.provider,
                        model=model_response.model,
                        latency_ms=model_response.latency_ms,
                        output=model_response.output,
                        input_length=model_response.input_length,
                        output_length=model_response.output_length
                    )
                )
            )

        processed_chunks = len(chunk_results)
        payload = SingleModelChunkedComparisonDTO(
            target_model=normalized_target_model,
            original_prompt_length=len(request.prompt),
            processed_chunks=processed_chunks,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            total_latency_ms=total_latency_ms,
            average_latency_ms=(total_latency_ms / processed_chunks) if processed_chunks > 0 else 0.0,
            total_output_length=total_output_length,
            chunks=chunk_results
        )
        return ApiResultDTO(success=True, data=payload.model_dump())
    except Exception as e:
        logger.error(f"Error en comparación single-model por chunks: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
