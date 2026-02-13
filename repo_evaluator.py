"""
Evaluador de repositorios de candidatos usando Gemini 2.5 Flash.
Implementa la rúbrica oficial de HireLens con cálculos en Python.
"""

import os
import json
import re
import time
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging
from dotenv import load_dotenv

load_dotenv()

from prompts.evaluation_prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    RUBRICA,
    MAX_SCORES,
    DECISION_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schema estricto para forzar el formato de respuesta del modelo
# Gemini respeta este schema a nivel de API — más confiable que cualquier
# instrucción en el prompt.
# ─────────────────────────────────────────────────────────────────────────────
CRITERION_BOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "value":    {"type": "boolean"},
        "evidence": {"type": "string"},
    },
    "required": ["value", "evidence"],
}

CRITERION_STRING_SCHEMA = {
    "type": "object",
    "properties": {
        "value":    {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["value", "evidence"],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "backend": {
            "type": "object",
            "properties": {
                "tecnologia_backend":    CRITERION_STRING_SCHEMA,
                "segunda_etapa_folders": CRITERION_BOOL_SCHEMA,
                "arquitectura_capas":    CRITERION_BOOL_SCHEMA,
                "uso_dtos":              CRITERION_BOOL_SCHEMA,
                "implemento_auth":       CRITERION_BOOL_SCHEMA,
                "mappings_orm":          CRITERION_BOOL_SCHEMA,
                "libreria_migraciones":  CRITERION_BOOL_SCHEMA,
            },
            "required": [
                "tecnologia_backend", "segunda_etapa_folders",
                "arquitectura_capas", "uso_dtos", "implemento_auth",
                "mappings_orm", "libreria_migraciones",
            ],
        },
        "frontend": {
            "type": "object",
            "properties": {
                "tecnologia_frontend":     CRITERION_STRING_SCHEMA,
                "requests_clase_separada": CRITERION_BOOL_SCHEMA,
                "minimo_tres_componentes": CRITERION_BOOL_SCHEMA,
                "manejo_asincronia":       CRITERION_BOOL_SCHEMA,
                "sin_dom_manual":          CRITERION_BOOL_SCHEMA,
                "uso_router":              CRITERION_BOOL_SCHEMA,
                "libreria_estilos":        CRITERION_BOOL_SCHEMA,
            },
            "required": [
                "tecnologia_frontend", "requests_clase_separada",
                "minimo_tres_componentes", "manejo_asincronia",
                "sin_dom_manual", "uso_router", "libreria_estilos",
            ],
        },
        "other": {
            "type": "object",
            "properties": {
                "scripts_inicializacion": CRITERION_BOOL_SCHEMA,
                "deploy_plataforma":      CRITERION_BOOL_SCHEMA,
                "nombres_ingles":         CRITERION_BOOL_SCHEMA,
                "uso_correcto_git":       CRITERION_BOOL_SCHEMA,
            },
            "required": [
                "scripts_inicializacion", "deploy_plataforma",
                "nombres_ingles", "uso_correcto_git",
            ],
        },
    },
    "required": ["backend", "frontend", "other"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMMetrics:
    hallucination_flags: List[str] = field(default_factory=list)
    hallucination_score: float = 0.0
    confidence_score: float = 0.0
    low_evidence_criteria: List[str] = field(default_factory=list)


@dataclass
class ScoresDict:
    backend_score: float = 0.0
    frontend_score: float = 0.0
    other_score: float = 0.0
    total_score: float = 0.0
    max_possible: float = MAX_SCORES["total"]


@dataclass
class EvaluationResult:
    candidate_id: str
    model_used: str
    evaluation_timestamp: str
    backend: Dict[str, Dict[str, Any]]
    frontend: Dict[str, Dict[str, Any]]
    other: Dict[str, Dict[str, Any]]
    scores: ScoresDict
    decision: str
    llm_metrics: LLMMetrics
    input_tokens_used: int
    latency_ms: float
    raw_llm_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["llm_metrics"] = asdict(self.llm_metrics)
        result["scores"] = asdict(self.scores)
        result["raw_llm_output"] = self.raw_llm_output
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluador principal
# ─────────────────────────────────────────────────────────────────────────────

class RepoEvaluator:
    """Evaluador de repositorios usando Gemini 2.5 Flash"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY", "")
        if not self.api_key:
            raise ValueError("GOOGLE_AI_API_KEY es requerida")

        genai.configure(api_key=self.api_key)
        self.model_name = model_name or os.getenv(
            "EVALUATOR_MODEL_NAME", "gemini-2.5-flash"
        )
        logger.info(f"RepoEvaluator inicializado con modelo: {self.model_name}")

    def evaluate(self, repo_path: str, candidate_id: str) -> EvaluationResult:
        logger.info(f"Iniciando evaluación: {candidate_id}")
        start_time = time.time()

        repo_context = self._get_repo_context(repo_path)
        logger.info(
            f"Contexto obtenido: {len(repo_context)} chars, "
            f"~{len(repo_context) // 4} tokens estimados"
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(repo_context=repo_context)
        model_response = self._call_gemini(user_prompt)
        latency_ms = (time.time() - start_time) * 1000
        raw_text = model_response.text

        try:
            evaluation_json = json.loads(raw_text)
        except json.JSONDecodeError as e:
            # No debería pasar con response_schema, pero por si acaso
            logger.error(f"JSON inválido en {candidate_id}: {e}")
            logger.error(f"Raw (500 chars): {raw_text[:500]}")
            raise ValueError(f"Respuesta no es JSON válido: {e}")

        result = self._build_evaluation_result(
            candidate_id=candidate_id,
            evaluation_json=evaluation_json,
            input_tokens=self._estimate_tokens(user_prompt),
            latency_ms=latency_ms,
            model_response=model_response,
            raw_llm_output=raw_text,
        )

        logger.info(
            f"Evaluación completada: {candidate_id} | "
            f"Score: {result.scores.total_score:.2f} | "
            f"Decision: {result.decision}"
        )
        return result

    def _call_gemini(self, user_prompt: str):
        """
        Llama a Gemini con response_schema para garantizar estructura fija.
        response_mime_type + response_schema es la combinación correcta:
        el modelo no puede devolver otro formato.
        """
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
            ),
        )
        return model.generate_content(user_prompt)

    # ── Contexto del repo ────────────────────────────────────────────────────

    def _get_repo_context(self, repo_path: str) -> str:
        try:
            subprocess.run(["repomix", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("repomix no instalado, usando fallback")
            return self._get_basic_summary(repo_path)

        try:
            with tempfile.NamedTemporaryFile(
                prefix="repomix-eval-", suffix=".txt", delete=False
            ) as tmp_file:
                output_path = tmp_file.name

            try:
                subprocess.run(
                    [
                        "repomix", repo_path, "--output", output_path,
                        "--include",
                        "**/*.java,**/*.py,**/*.js,**/*.ts,**/*.jsx,**/*.tsx,"
                        "**/*.go,**/*.rb,**/*.kt,**/*.md,**/*.sql,**/*.xml,"
                        "**/*.gradle,**/*.yaml,**/*.yml",
                    ],
                    timeout=30, capture_output=True,
                )
            except Exception:
                logger.info("--include falló, usando --ignore")
                subprocess.run(
                    [
                        "repomix", repo_path, "--output", output_path,
                        "--ignore",
                        "node_modules,dist,build,target,.git,*.lock,"
                        "*.min.js,*.min.css,*.png,*.jpg,*.svg,*.ico,"
                        "*.ttf,*.woff,.idea,__pycache__,.venv,venv",
                    ],
                    timeout=30, capture_output=True,
                )

            with open(output_path, "r", encoding="utf-8") as f:
                context = f.read()

            logger.info(
                f"Repomix output: {len(context)} chars, "
                f"~{len(context) // 4} estimated tokens"
            )
            return context

        except subprocess.TimeoutExpired:
            logger.warning("Repomix timeout, usando fallback")
            return self._get_basic_summary(repo_path)
        except Exception as e:
            logger.warning(f"Error en Repomix: {e}, usando fallback")
            return self._get_basic_summary(repo_path)
        finally:
            if "output_path" in locals() and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

    def _get_basic_summary(self, repo_path: str) -> str:
        try:
            result = subprocess.run(
                [
                    "find", repo_path, "-type", "f",
                    "(", "-name", "*.java", "-o", "-name", "*.py",
                    "-o", "-name", "*.js", "-o", "-name", "*.ts",
                    "-o", "-name", "*.md", ")",
                ],
                capture_output=True, text=True, timeout=10,
            )
            files = result.stdout.strip().split("\n")[:50]
            return "Archivos encontrados en el repositorio:\n" + "\n".join(files)
        except Exception:
            return "Repositorio sin contexto disponible"

    # ── Construcción del resultado ───────────────────────────────────────────

    def _build_evaluation_result(
        self,
        candidate_id: str,
        evaluation_json: Dict[str, Any],
        input_tokens: int,
        latency_ms: float,
        model_response: Any,
        raw_llm_output: str = "",
    ) -> EvaluationResult:

        backend_data  = evaluation_json.get("backend",  {})
        frontend_data = evaluation_json.get("frontend", {})
        other_data    = evaluation_json.get("other",    {})

        backend_score  = self._calculate_category_score(backend_data,  "backend")
        frontend_score = self._calculate_category_score(frontend_data, "frontend")
        other_score    = self._calculate_category_score(other_data,    "other")
        total_score    = round(backend_score + frontend_score + other_score, 2)

        if total_score >= DECISION_THRESHOLDS["PASS"]:
            decision = "PASS"
        elif total_score >= DECISION_THRESHOLDS["REVIEW"]:
            decision = "REVIEW"
        else:
            decision = "FAIL"

        llm_metrics = self._calculate_hallucination_metrics(
            backend_data, frontend_data, other_data
        )

        prompt_tokens = 0
        if hasattr(model_response, "usage_metadata"):
            if hasattr(model_response.usage_metadata, "prompt_token_count"):
                prompt_tokens = model_response.usage_metadata.prompt_token_count

        return EvaluationResult(
            candidate_id=candidate_id,
            model_used=self.model_name,
            evaluation_timestamp=datetime.utcnow().isoformat() + "Z",
            backend=self._format_category(backend_data,  "backend"),
            frontend=self._format_category(frontend_data, "frontend"),
            other=self._format_category(other_data,    "other"),
            scores=ScoresDict(
                backend_score=round(backend_score, 2),
                frontend_score=round(frontend_score, 2),
                other_score=round(other_score, 2),
                total_score=total_score,
                max_possible=MAX_SCORES["total"],
            ),
            decision=decision,
            llm_metrics=llm_metrics,
            input_tokens_used=prompt_tokens or input_tokens,
            latency_ms=round(latency_ms, 2),
            raw_llm_output=raw_llm_output,
        )

    def _calculate_category_score(
        self, category_data: Dict[str, Any], category_name: str
    ) -> float:
        """Pesos siempre desde RUBRICA — el modelo no los conoce."""
        score = 0.0
        rubric_cat = RUBRICA.get(category_name, {})
        for criterion_name, criterion_data in category_data.items():
            if not isinstance(criterion_data, dict):
                continue
            value = criterion_data.get("value")
            peso  = rubric_cat.get(criterion_name, {}).get("peso", 0)
            if value is True and peso > 0:
                score += peso
                logger.debug(f"  +{peso} | {criterion_name}")
        return round(score, 2)

    def _calculate_hallucination_metrics(
        self,
        backend_data: Dict,
        frontend_data: Dict,
        other_data: Dict,
    ) -> LLMMetrics:
        all_data = [
            ("backend",  backend_data),
            ("frontend", frontend_data),
            ("other",    other_data),
        ]
        hallucination_flags   = []
        true_criteria         = []
        low_evidence_criteria = []

        for category_name, category_data in all_data:
            rubric_cat = RUBRICA.get(category_name, {})
            for criterion_name, criterion_data in category_data.items():
                if not isinstance(criterion_data, dict):
                    continue
                value    = criterion_data.get("value")
                peso     = rubric_cat.get(criterion_name, {}).get("peso", 0)
                evidence = criterion_data.get("evidence", "").strip()

                if value is True and peso > 0:
                    true_criteria.append(criterion_name)
                    if (
                        not evidence
                        or len(evidence) < 15
                        or evidence.lower() in [
                            "n/a", "no aplica", "no encontrado", "sí", "si", "yes"
                        ]
                    ):
                        hallucination_flags.append(criterion_name)
                        low_evidence_criteria.append(criterion_name)

        hallucination_score = (
            round(len(hallucination_flags) / len(true_criteria), 3)
            if true_criteria else 0.0
        )
        return LLMMetrics(
            hallucination_flags=hallucination_flags,
            hallucination_score=hallucination_score,
            confidence_score=round(1.0 - hallucination_score, 3),
            low_evidence_criteria=low_evidence_criteria,
        )

    def _format_category(
        self, category_data: Dict[str, Any], category_name: str
    ) -> Dict[str, Any]:
        """Serializa la categoría enriqueciendo con pesos de RUBRICA."""
        rubric_cat = RUBRICA.get(category_name, {})
        return {
            criterion_name: {
                "value":    criterion_data.get("value"),
                "peso":     rubric_cat.get(criterion_name, {}).get("peso", 0),
                "evidence": criterion_data.get("evidence", ""),
            }
            for criterion_name, criterion_data in category_data.items()
            if isinstance(criterion_data, dict)
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4