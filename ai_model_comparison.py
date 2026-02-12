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
            result = subprocess.run(
                ["repomix", repo_path, "--output", "json"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar repomix: {e.stderr}")
            return ""
    
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
    
    def _run_prompt(
        self,
        request: CompareModelRequest,
        model_name: str,
        allow_system_message: bool
    ) -> CompareModelResponse:
        # Construye el prompt incluindo contexto del repositorio si está habilitado (desde el request)
        prompt = request.prompt
        
        if request.include_repo_context:
            repo_path = request.repo_path or self.default_repo_path
            repo_context = RepoContextProvider.get_repo_summary(repo_path)
            if repo_context:
                prompt = f"""### Contexto del Repositorio:
{repo_context}

### Prompt del usuario:
{request.prompt}"""
        
        system_prompt = request.system_prompt or "Eres un asistente útil."
        
        # Crea el modelo
        model = genai.GenerativeModel(model_name=model_name)
        
        # Prepara el mensaje
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
