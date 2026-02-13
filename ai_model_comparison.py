#!/usr/bin/env python3
"""
AI Model Comparison Tool - Compara respuestas entre Gemini y Gemma usando Google AI Studio
Integra repomix para pasar el contexto del repositorio como prompt adicional.
"""

import os
import sys
import time
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional
import google.generativeai as genai


@dataclass
class CompareModelRequest:
    """Solicitud de comparación de modelos"""
    prompt: str
    system_prompt: Optional[str] = None
    include_repo_context: bool = True
    repo_path: Optional[str] = None


@dataclass
class CompareModelResponse:
    """Respuesta de comparación de modelos"""
    provider: str
    model: str
    latency_ms: float
    output: str
    input_length: int
    output_length: int


class RepoContextProvider:
    
    @staticmethod
    def get_repo_context(repo_path: str = ".") -> str:

        try:
            # Verifica si repomix está instalado
            subprocess.run(["repomix", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("repomix no está instalado. Instala con: npm install -g repomix")
            return ""
        
        try:
            with tempfile.NamedTemporaryFile(prefix="repomix-", suffix=".txt", delete=False) as tmp_file:
                output_path = tmp_file.name
            # Mejoramos el ignore para evitar archivos pesados o irrelevantes que puedan causar timeouts o exceder el token limit.
            subprocess.run(
                [
                    "repomix",
                    repo_path,
                    "--output",
                    output_path,
                    "--ignore",
                    "repomix-output*,json,repomix-last-output.txt"  
                ],
            )
            with open(output_path, "r", encoding="utf-8") as f:
                context = f.read()
            debug_output_path = os.path.join(os.getcwd(), "repomix-last-output.txt")
            with open(debug_output_path, "w", encoding="utf-8") as debug_file:
                debug_file.write(context)
            return context
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar repomix: {e.stderr}")
            return ""
        finally:
            if "output_path" in locals() and os.path.exists(output_path):
                os.remove(output_path)
    
    @staticmethod
    def get_repo_summary(repo_path: str = ".") -> str:
        context = RepoContextProvider.get_repo_context(repo_path)
        if context:
            return context
        
        # Fallback: resumen básico jijiji
        return RepoContextProvider._get_basic_summary(repo_path)
    
    @staticmethod
    def _get_basic_summary(repo_path: str) -> str:
        try:
            result = subprocess.run(
                ["find", repo_path, "-type", "f", "-name", "*.java", "-o", "-name", "*.py", "-o", "-name", "*.md"],
                capture_output=True,
                text=True,
                check=True
            )
            files = result.stdout.strip().split("\n")[:20]  # Primeros 20 archivos
            return f"Archivos del repositorio:\n" + "\n".join(files)
        except:
            return "No se pudo obtener información del repositorio"


class AIModelComparator:
    
    def __init__(self, api_key: str, default_repo_path: str = "."):
        if not api_key:
            raise ValueError("Google AI API key es requerida")
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model_name_one = os.getenv("MODEL_NAME_ONE") or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        self.model_name_two = os.getenv("MODEL_NAME_TWO") or os.getenv("GEMMA_MODEL_NAME", "gemma-3-27b-it")
        self.default_repo_path = default_repo_path
        # De todas maneras, usa 10000. 
        self.max_input_tokens = int(os.getenv("MAX_INPUT_TOKENS", "0"))

    def compare_with_model_one(self, request: CompareModelRequest) -> CompareModelResponse:
        return self._run_prompt(request, self.model_name_one, allow_system_message=True)

    def compare_with_model_two(self, request: CompareModelRequest) -> CompareModelResponse:
        return self._run_prompt(request, self.model_name_two, allow_system_message=False)

    # Alias para compatibilidad hacia atrás
    def compare_with_gemini(self, request: CompareModelRequest) -> CompareModelResponse:
        return self.compare_with_model_one(request)

    def compare_with_gemma(self, request: CompareModelRequest) -> CompareModelResponse:
        return self.compare_with_model_two(request)

    @staticmethod
    def _raise_if_model_not_found(model_name: str, error: Exception) -> None:
        error_text = str(error)
        if "is not found" in error_text or "404" in error_text:
            raise ValueError(
                f"Modelo '{model_name}' no existe o no soporta generateContent. "
                "Ajusta MODEL_NAME_ONE/MODEL_NAME_TWO en .env con un modelo valido."
            ) from error

    @staticmethod
    def _extract_total_tokens(token_result) -> Optional[int]:
        if isinstance(token_result, int):
            return token_result
        if hasattr(token_result, "total_tokens"):
            try:
                return int(token_result.total_tokens)
            except (TypeError, ValueError):
                return None
        if isinstance(token_result, dict):
            for key in ["total_tokens", "token_count", "tokens"]:
                if key in token_result:
                    try:
                        return int(token_result[key])
                    except (TypeError, ValueError):
                        return None
        return None

    def _count_tokens_safe(self, model: genai.GenerativeModel, text: str) -> Optional[int]:
        try:
            token_result = model.count_tokens(text)
            return self._extract_total_tokens(token_result)
        except Exception:
            return None

    def _build_prompt_with_context(self, user_prompt: str, repo_context: str) -> str:
        return f"""### Contexto del Repositorio:
{repo_context}

### Prompt del usuario:
{user_prompt}"""

    def _build_model_input(self, prompt: str, system_prompt: str, allow_system_message: bool) -> str:
        if allow_system_message:
            # Para estimar cuota consideramos system_prompt + prompt.
            return f"{system_prompt}\n\n{prompt}"
        return f"""Instrucciones:
{system_prompt}

Prompt del usuario:
{prompt}"""

    def _fit_prompt_to_token_limit(
        self,
        model: genai.GenerativeModel,
        user_prompt: str,
        system_prompt: str,
        allow_system_message: bool,
        repo_context: str
    ) -> str:
        if self.max_input_tokens <= 0:
            # Sin limite: incluir siempre el contexto completo del repositorio.
            return self._build_prompt_with_context(user_prompt, repo_context)

        prompt_with_context = self._build_prompt_with_context(user_prompt, repo_context)
        full_model_input = self._build_model_input(prompt_with_context, system_prompt, allow_system_message)
        token_count = self._count_tokens_safe(model, full_model_input)

        if token_count is not None and token_count <= self.max_input_tokens:
            return prompt_with_context

        # Fallback si no hay count_tokens: aproximación 1 token ~ 4 chars.
        if token_count is None:
            char_budget = self.max_input_tokens * 4
            if len(full_model_input) <= char_budget:
                return prompt_with_context

            overflow = len(full_model_input) - char_budget
            keep_chars = max(0, len(repo_context) - overflow)
            truncated_context = repo_context[:keep_chars]
            if keep_chars < len(repo_context):
                truncated_context += "\n...[repo context truncated]"
            return self._build_prompt_with_context(user_prompt, truncated_context) if keep_chars > 0 else user_prompt

        left = 0
        right = len(repo_context)
        best_prompt = user_prompt

        while left <= right:
            mid = (left + right) // 2
            candidate_context = repo_context[:mid]
            if mid < len(repo_context):
                candidate_context += "\n...[repo context truncated]"

            candidate_prompt = self._build_prompt_with_context(user_prompt, candidate_context) if mid > 0 else user_prompt
            candidate_input = self._build_model_input(candidate_prompt, system_prompt, allow_system_message)
            candidate_tokens = self._count_tokens_safe(model, candidate_input)

            if candidate_tokens is None:
                return candidate_prompt

            if candidate_tokens <= self.max_input_tokens:
                best_prompt = candidate_prompt
                left = mid + 1
            else:
                right = mid - 1

        return best_prompt
    
    def _run_prompt(
        self,
        request: CompareModelRequest,
        model_name: str,
        allow_system_message: bool
    ) -> CompareModelResponse:
        system_prompt = request.system_prompt or "Eres un asistente útil."

        model = genai.GenerativeModel(model_name=model_name)

        prompt = request.prompt
        if request.include_repo_context:
            repo_path = request.repo_path or self.default_repo_path
            repo_context = RepoContextProvider.get_repo_summary(repo_path)
            if repo_context:
                prompt = self._fit_prompt_to_token_limit(
                    model=model,
                    user_prompt=request.prompt,
                    system_prompt=system_prompt,
                    allow_system_message=allow_system_message,
                    repo_context=repo_context
                )

        model_input = prompt
        if allow_system_message:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                model_input = prompt
                start_time = time.time()
                response = model.generate_content(model_input)
                end_time = time.time()
            except Exception as e:
                self._raise_if_model_not_found(model_name, e)
                model_input = f"{system_prompt}\n\n{prompt}"
                start_time = time.time()
                response = model.generate_content(model_input)
                end_time = time.time()
        else:
            # Gemma no soporta system messages directamente
            model_input = f"""Instrucciones:
{system_prompt}

Prompt del usuario:
{prompt}"""
            try:
                start_time = time.time()
                response = model.generate_content(model_input)
                end_time = time.time()
            except Exception as e:
                self._raise_if_model_not_found(model_name, e)
                raise
        
        output = response.text if response else ""
        latency_ms = (end_time - start_time) * 1000
        
        return CompareModelResponse(
            provider="google-ai-studio",
            model=model_name,
            latency_ms=latency_ms,
            output=output,
            input_length=len(model_input),
            output_length=len(output) if output else 0
        )
