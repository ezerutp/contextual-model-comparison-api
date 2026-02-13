"""
Evaluador en batch de repositorios de candidatos con métricas LLM.
Mide capacidad discriminativa del modelo entre repos OK/BAD.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import logging
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from repo_evaluator import RepoEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class DiscriminationMetrics:
    """Métricas de discriminación entre OK/BAD"""

    mean_score_ok: float
    mean_score_bad: float
    score_gap: float
    mann_whitney_u: float
    p_value: float
    discriminates_well: bool


@dataclass
class ClassificationMetrics:
    """Métricas de clasificación binaria"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int] = field(
        default_factory=lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )


@dataclass
class HallucinationMetrics:
    """Métricas de alucinaciones en el LLM"""

    mean_hallucination_score: float
    max_hallucination_score: float
    repos_with_hallucinations: int
    most_hallucinated_criteria: List[Dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class DistributionMetrics:
    """Métricas de distribución de scores"""

    scores_ok: List[float] = field(default_factory=list)
    scores_bad: List[float] = field(default_factory=list)
    std_ok: float = 0.0
    std_bad: float = 0.0
    overlap_coefficient: float = 0.0


@dataclass
class AllMetrics:
    """Todas las métricas agregadas"""

    discrimination: DiscriminationMetrics
    classification: ClassificationMetrics
    hallucinations: HallucinationMetrics
    distribution: DistributionMetrics


@dataclass
class BatchEvaluationResult:
    """Resultado de evaluación en batch"""

    dataset_path: str
    total_repos: int
    ok_repos: int
    bad_repos: int
    evaluations: List[EvaluationResult]
    metrics: AllMetrics
    total_tokens_used: int
    total_latency_ms: float
    average_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario JSON-serializable"""
        return {
            "dataset_path": self.dataset_path,
            "total_repos": self.total_repos,
            "ok_repos": self.ok_repos,
            "bad_repos": self.bad_repos,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "metrics": {
                "discrimination": asdict(self.metrics.discrimination),
                "classification": asdict(self.metrics.classification),
                "hallucinations": asdict(self.metrics.hallucinations),
                "distribution": asdict(self.metrics.distribution),
            },
            "total_tokens_used": self.total_tokens_used,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
        }


class BatchEvaluator:
    """Evaluador de datasets en batch"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Inicializa el evaluador batch.

        Args:
            api_key: Google AI API key (por defecto desde env)
            model_name: Nombre del modelo (por defecto desde env)
        """
        self.evaluator = RepoEvaluator(api_key=api_key, model_name=model_name)
        logger.info("BatchEvaluator inicializado")

    def evaluate_dataset(self, dataset_path: str) -> BatchEvaluationResult:
        """
        Evalúa un dataset de repositorios etiquetados OK/BAD.

        Args:
            dataset_path: Ruta a carpeta con subcarpetas de repos
                         (ej: dataset/candidato_01_OK, dataset/candidato_02_BAD)

        Returns:
            BatchEvaluationResult con resultados y métricas
        """
        logger.info(f"Iniciando evaluación batch: {dataset_path}")

        # 1. Descubrir repositorios
        repos = self._discover_repos(dataset_path)
        logger.info(f"Descubiertos {len(repos)} repositorios")

        # 2. Evaluar cada repo
        evaluations = []
        total_tokens = 0
        total_latency = 0.0

        for idx, (repo_name, repo_path, label) in enumerate(repos, 1):
            try:
                logger.info(f"[{idx}/{len(repos)}] Evaluando: {repo_name}")
                result = self.evaluator.evaluate(repo_path, repo_name)
                evaluations.append(result)
                total_tokens += result.input_tokens_used
                total_latency += result.latency_ms
                
                # Log del resultado
                logger.info(
                    f"  ✓ Score: {result.scores.total_score:.2f} | "
                    f"Decision: {result.decision} | "
                    f"Halluc: {result.llm_metrics.hallucination_score:.2f}"
                )

                # Esperar entre requests para no saturar API
                if idx < len(repos):
                    time.sleep(3)

            except Exception as e:
                logger.error(f"Error evaluando {repo_name}: {e}")
                # Continuar con siguiente repo

        logger.info(f"Evaluaciones completadas: {len(evaluations)}/{len(repos)}")

        # 3. Calcular métricas
        metrics = self._calculate_metrics(evaluations, repos)

        # 4. Construir resultado
        ok_count = sum(1 for _, _, label in repos if label == 1)
        bad_count = len(repos) - ok_count

        avg_latency = total_latency / len(evaluations) if evaluations else 0.0

        return BatchEvaluationResult(
            dataset_path=dataset_path,
            total_repos=len(repos),
            ok_repos=ok_count,
            bad_repos=bad_count,
            evaluations=evaluations,
            metrics=metrics,
            total_tokens_used=total_tokens,
            total_latency_ms=round(total_latency, 2),
            average_latency_ms=round(avg_latency, 2),
        )

    def _discover_repos(
        self, dataset_path: str
    ) -> List[tuple[str, str, int]]:
        """
        Descubre repos en dataset_path.

        Returns:
            Lista de tuples: (repo_name, repo_path, label)
            donde label = 1 si "OK" en nombre, 0 si "BAD"
        """
        repos = []
        dataset_dir = Path(dataset_path)

        if not dataset_dir.exists():
            raise ValueError(f"Dataset path no existe: {dataset_path}")

        for item in sorted(dataset_dir.iterdir()):
            if not item.is_dir():
                continue

            name = item.name.lower()
            if "ok" in name:
                label = 1
            elif "bad" in name:
                label = 0
            else:
                logger.warning(f"Repos sin etiqueta OK/BAD: {item.name}, asumiendo BAD")
                label = 0

            repos.append((item.name, str(item), label))

        return repos

    def _calculate_metrics(
        self, evaluations: List[EvaluationResult], repos: List[tuple[str, str, int]]
    ) -> AllMetrics:
        """Calcula todas las métricas del batch"""

        # Mapeo: eval -> label real
        eval_labels = {}
        for eval_result in evaluations:
            # Buscar el label correspondiente en repos
            for repo_name, repo_path, label in repos:
                if eval_result.candidate_id == repo_name:
                    eval_labels[eval_result.candidate_id] = label
                    break

        # Separar scores por label
        scores_ok = []
        scores_bad = []
        predictions_binary = []
        ground_truth = []

        for eval_result in evaluations:
            true_label = eval_labels.get(eval_result.candidate_id, 0)
            ground_truth.append(true_label)

            # Mapeo decision a predicción binaria
            # PASS → 1, REVIEW → 0.5 (contar como 1 si >= 5.5, sino 0), FAIL → 0
            if eval_result.decision == "PASS":
                pred = 1
            elif eval_result.decision == "REVIEW":
                pred = 1 if eval_result.scores.total_score >= 5.5 else 0
            else:  # FAIL
                pred = 0

            predictions_binary.append(pred)

            # Agrupar scores
            if true_label == 1:
                scores_ok.append(eval_result.scores.total_score)
            else:
                scores_bad.append(eval_result.scores.total_score)

        # A) Métricas de Discriminación
        discrimination = self._calculate_discrimination_metrics(scores_ok, scores_bad)

        # B) Métricas de Clasificación
        classification = self._calculate_classification_metrics(
            ground_truth, predictions_binary
        )

        # C) Métricas de Alucinaciones
        hallucinations = self._calculate_hallucination_metrics(evaluations)

        # D) Métricas de Distribución
        distribution = self._calculate_distribution_metrics(scores_ok, scores_bad)

        return AllMetrics(
            discrimination=discrimination,
            classification=classification,
            hallucinations=hallucinations,
            distribution=distribution,
        )

    def _calculate_discrimination_metrics(
        self, scores_ok: List[float], scores_bad: List[float]
    ) -> DiscriminationMetrics:
        """Calcula capacidad de discriminación con Mann-Whitney U"""

        mean_ok = float(np.mean(scores_ok)) if scores_ok else 0.0
        mean_bad = float(np.mean(scores_bad)) if scores_bad else 0.0
        score_gap = round(mean_ok - mean_bad, 3)

        mann_whitney_u = 0.0
        p_value = 1.0
        discriminates_well = False

        if scores_ok and scores_bad:
            try:
                u_stat, p_val = mannwhitneyu(scores_ok, scores_bad)
                mann_whitney_u = round(float(u_stat), 3)
                p_value = round(float(p_val), 6)
                # Discrimina bien si p < 0.05 y gap > 2.0
                discriminates_well = p_value < 0.05 and score_gap > 2.0
            except Exception as e:
                logger.warning(f"Error en Mann-Whitney U: {e}")

        return DiscriminationMetrics(
            mean_score_ok=round(mean_ok, 3),
            mean_score_bad=round(mean_bad, 3),
            score_gap=score_gap,
            mann_whitney_u=mann_whitney_u,
            p_value=p_value,
            discriminates_well=discriminates_well,
        )

    def _calculate_classification_metrics(
        self, ground_truth: List[int], predictions: List[int]
    ) -> ClassificationMetrics:
        """Calcula métricas de clasificación binaria"""

        if not ground_truth or not predictions:
            return ClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)

        try:
            acc = round(accuracy_score(ground_truth, predictions), 3)
            prec = round(precision_score(ground_truth, predictions, zero_division=0), 3)
            rec = round(recall_score(ground_truth, predictions, zero_division=0), 3)
            f1 = round(f1_score(ground_truth, predictions, zero_division=0), 3)

            # Confusion matrix
            tp = sum(
                1
                for g, p in zip(ground_truth, predictions)
                if g == 1 and p == 1
            )
            fp = sum(
                1
                for g, p in zip(ground_truth, predictions)
                if g == 0 and p == 1
            )
            tn = sum(
                1
                for g, p in zip(ground_truth, predictions)
                if g == 0 and p == 0
            )
            fn = sum(
                1
                for g, p in zip(ground_truth, predictions)
                if g == 1 and p == 0
            )

            return ClassificationMetrics(
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            )
        except Exception as e:
            logger.warning(f"Error calculando métricas de clasificación: {e}")
            return ClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)

    def _calculate_hallucination_metrics(
        self, evaluations: List[EvaluationResult]
    ) -> HallucinationMetrics:
        """Calcula métricas de alucinaciones"""

        if not evaluations:
            return HallucinationMetrics(
                mean_hallucination_score=0.0,
                max_hallucination_score=0.0,
                repos_with_hallucinations=0,
            )

        hallucination_scores = [e.llm_metrics.hallucination_score for e in evaluations]
        mean_halluc = round(np.mean(hallucination_scores), 3)
        max_halluc = round(np.max(hallucination_scores), 3)
        repos_with_halluc = sum(1 for s in hallucination_scores if s > 0.2)

        # Top 3 criterios más frecuentemente alucinados
        all_halluc_flags = []
        for e in evaluations:
            all_halluc_flags.extend(e.llm_metrics.hallucination_flags)

        counter = Counter(all_halluc_flags)
        top_criteria = [
            {"criterio": criterion, "frequency": count}
            for criterion, count in counter.most_common(3)
        ]

        return HallucinationMetrics(
            mean_hallucination_score=mean_halluc,
            max_hallucination_score=max_halluc,
            repos_with_hallucinations=repos_with_halluc,
            most_hallucinated_criteria=top_criteria,
        )

    def _calculate_distribution_metrics(
        self, scores_ok: List[float], scores_bad: List[float]
    ) -> DistributionMetrics:
        """Calcula métricas de distribución de scores"""

        std_ok = round(float(np.std(scores_ok)), 3) if scores_ok else 0.0
        std_bad = round(float(np.std(scores_bad)), 3) if scores_bad else 0.0

        # Overlap coefficient: proporción de solapamiento
        overlap_coeff = 0.0
        if scores_ok and scores_bad:
            min_ok = min(scores_ok)
            max_ok = max(scores_ok)
            min_bad = min(scores_bad)
            max_bad = max(scores_bad)

            overlap_min = max(min_ok, min_bad)
            overlap_max = min(max_ok, max_bad)

            if overlap_max >= overlap_min:
                overlap_range = overlap_max - overlap_min
                total_range = max(max_ok, max_bad) - min(min_ok, min_bad)
                overlap_coeff = round(overlap_range / total_range if total_range > 0 else 0, 3)

        return DistributionMetrics(
            scores_ok=sorted(scores_ok),
            scores_bad=sorted(scores_bad),
            std_ok=std_ok,
            std_bad=std_bad,
            overlap_coefficient=overlap_coeff,
        )
