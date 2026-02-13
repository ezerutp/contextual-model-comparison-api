"""
Endpoints FastAPI para evaluación de repositorios de candidatos.
Integra RepoEvaluator y BatchEvaluator.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import logging

from repo_evaluator import RepoEvaluator, EvaluationResult
from batch_evaluator import BatchEvaluator, BatchEvaluationResult

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/repo-evaluation", tags=["repo-evaluation"])


# DTOs
class EvaluateRequestDTO(BaseModel):
    """DTO de solicitud de evaluación individual"""

    repo_path: str = Field(..., description="Ruta al repositorio del candidato")
    candidate_id: str = Field(..., description="ID único del candidato")


class BatchEvaluateRequestDTO(BaseModel):
    """DTO de solicitud de evaluación batch"""

    dataset_path: str = Field(
        ...,
        description="Ruta a carpeta con subcarpetas de repos "
        "(ej: dataset/candidato_01_OK, dataset/candidato_02_BAD)",
    )


class ApiResultDTO(BaseModel):
    """DTO genérico de resultado"""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


# Dependencias
def get_repo_evaluator() -> RepoEvaluator:
    """Obtiene instancia de RepoEvaluator"""
    try:
        return RepoEvaluator()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_batch_evaluator() -> BatchEvaluator:
    """Obtiene instancia de BatchEvaluator"""
    try:
        return BatchEvaluator()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints
@router.get("/health")
async def health():
    """
    Verifica el estado de los servicios de evaluación.

    Response:
        {
            "status": "ok",
            "model": "gemini-2.5-flash",
            "rpd_warning": "max 20 requests/day on free tier"
        }
    """
    return {
        "status": "ok",
        "model": "gemini-2.5-flash",
        "rpd_warning": "max 20 requests/day on free tier",
    }


@router.post("/evaluate", response_model=ApiResultDTO)
async def evaluate(
    request: EvaluateRequestDTO,
    evaluator: RepoEvaluator = Depends(get_repo_evaluator),
):
    """
    Evalúa un repositorio individual de candidato.

    Usa la rúbrica oficial de HireLens, retorna EvaluationResult completo
    con breakdown por categoría, scores, decision y métricas de alucinación.

    Args:
        repo_path: Ruta al repositorio
        candidate_id: ID del candidato

    Returns:
        ApiResultDTO con EvaluationResult en data
    """
    try:
        logger.info(f"Evaluando repositorio: {request.candidate_id}")
        result = evaluator.evaluate(request.repo_path, request.candidate_id)
        return ApiResultDTO(success=True, data=result.to_dict())
    except Exception as e:
        logger.error(f"Error evaluando repositorio: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))


@router.post("/batch", response_model=ApiResultDTO)
async def evaluate_batch(
    request: BatchEvaluateRequestDTO,
    evaluator: BatchEvaluator = Depends(get_batch_evaluator),
):
    """
    Evalúa un dataset completo de repositorios etiquetados OK/BAD.

    Este endpoint puede tardar varios minutos dependiendo del tamaño del
    dataset. Loguea progreso por cada repositorio procesado.

    Estructura esperada del dataset:
        dataset/
            candidato_01_OK/
            candidato_02_BAD/
            candidato_03_OK/
            ...

    Retorna:
        - evaluations: lista de EvaluationResult por cada repo
        - metrics: métricas agregadas del batch:
            - discrimination: Man-Whitney U, score gap, etc.
            - classification: accuracy, precision, recall, f1
            - hallucinations: score y criterios más alucinados
            - distribution: std, overlap coefficient, etc.
        - total_tokens_used: tokens consumidos en toda la batch
        - total_latency_ms: tiempo total en ms
        - average_latency_ms: promedio de latencia por repo

    Rate limits:
        Gemini 2.5 Flash free tier: 250K TPM, 20 RPD
        El endpoint respeta espera de 3s entre requests.
    """
    try:
        logger.info(f"Iniciando evaluación batch: {request.dataset_path}")
        result = evaluator.evaluate_dataset(request.dataset_path)
        return ApiResultDTO(success=True, data=result.to_dict())
    except Exception as e:
        logger.error(f"Error en evaluación batch: {str(e)}")
        return ApiResultDTO(success=False, error=str(e))
