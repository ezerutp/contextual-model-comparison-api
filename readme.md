# Ejemplos de Requests JSON

Base URL sugerida: `http://localhost:8000`

## 1) `POST /api/ai-model-comparison/model-one`

```json
{
  "system_prompt": "Eres un arquitecto de software senior especializado en análisis de repositorios y evaluación de calidad de arquitectura.",
  "prompt": "Analiza la arquitectura del proyecto usando el contexto del repositorio proporcionado.\n\nSigue estrictamente estos pasos:\n\n1. Identifica el tipo de arquitectura (monolito, capas, hexagonal, MVC, microservicios, script, etc.).\n2. Describe los componentes principales y su responsabilidad.\n3. Analiza la separación de responsabilidades y el acoplamiento.\n4. Evalúa la organización del repositorio (carpetas, módulos, convenciones).\n5. Identifica posibles problemas arquitectónicos (code smells, dependencias incorrectas, violaciones de capas).\n6. Indica fortalezas de la arquitectura.\n7. Indica debilidades o riesgos técnicos.\n\nDevuelve el resultado EXCLUSIVAMENTE en formato JSON con la siguiente estructura:\n{\n  \"architecture_style\": \"\",\n  \"main_components\": [\n    { \"name\": \"\", \"responsibility\": \"\" }\n  ],\n  \"layering_quality\": \"\",\n  \"coupling_assessment\": \"\",\n  \"repository_structure_quality\": \"\",\n  \"strengths\": [\"\"],\n  \"weaknesses\": [\"\"],\n  \"overall_architecture_score\": \"number from 1 to 10\"\n}\n\nNo incluyas texto fuera del JSON.",
  "include_repo_context": true,
  "repo_path": "/home/ezer/VidarteTicliahuanca-2509af-main"
}
```

## 2) `POST /api/ai-model-comparison/model-two`

```json
{
  "system_prompt": "Eres un arquitecto de software senior especializado en análisis de repositorios y evaluación de calidad de arquitectura.",
  "prompt": "Analiza la arquitectura del proyecto usando el contexto del repositorio proporcionado.\n\nSigue estrictamente estos pasos:\n\n1. Identifica el tipo de arquitectura (monolito, capas, hexagonal, MVC, microservicios, script, etc.).\n2. Describe los componentes principales y su responsabilidad.\n3. Analiza la separación de responsabilidades y el acoplamiento.\n4. Evalúa la organización del repositorio (carpetas, módulos, convenciones).\n5. Identifica posibles problemas arquitectónicos (code smells, dependencias incorrectas, violaciones de capas).\n6. Indica fortalezas de la arquitectura.\n7. Indica debilidades o riesgos técnicos.\n\nDevuelve el resultado EXCLUSIVAMENTE en formato JSON con la siguiente estructura:\n{\n  \"architecture_style\": \"\",\n  \"main_components\": [\n    { \"name\": \"\", \"responsibility\": \"\" }\n  ],\n  \"layering_quality\": \"\",\n  \"coupling_assessment\": \"\",\n  \"repository_structure_quality\": \"\",\n  \"strengths\": [\"\"],\n  \"weaknesses\": [\"\"],\n  \"overall_architecture_score\": \"number from 1 to 10\"\n}\n\nNo incluyas texto fuera del JSON.",
  "include_repo_context": true,
  "repo_path": "/home/ezer/VidarteTicliahuanca-2509af-main"
}
```

## 3) `POST /api/ai-model-comparison/compare-both`

```json
{
  "system_prompt": "Eres un arquitecto de software senior especializado en análisis de repositorios y evaluación de calidad de arquitectura.",
  "prompt": "Analiza la arquitectura del proyecto usando el contexto del repositorio proporcionado y responde solo JSON.",
  "include_repo_context": true,
  "repo_path": "/home/ezer/VidarteTicliahuanca-2509af-main"
}
```

## 4) `POST /api/ai-model-comparison/compare-single-chunked`

```json
{
  "system_prompt": "Eres un arquitecto de software senior especializado en análisis de repositorios y evaluación de calidad de arquitectura.",
  "prompt": "Analiza la arquitectura del proyecto usando el contexto del repositorio proporcionado.\n\nSigue estrictamente estos pasos:\n\n1. Identifica el tipo de arquitectura (monolito, capas, hexagonal, MVC, microservicios, script, etc.).\n2. Describe los componentes principales y su responsabilidad.\n3. Analiza la separación de responsabilidades y el acoplamiento.\n4. Evalúa la organización del repositorio (carpetas, módulos, convenciones).\n5. Identifica posibles problemas arquitectónicos (code smells, dependencias incorrectas, violaciones de capas).\n6. Indica fortalezas de la arquitectura.\n7. Indica debilidades o riesgos técnicos.\n\nDevuelve el resultado EXCLUSIVAMENTE en formato JSON con la siguiente estructura:\n{\n  \"architecture_style\": \"\",\n  \"main_components\": [\n    { \"name\": \"\", \"responsibility\": \"\" }\n  ],\n  \"layering_quality\": \"\",\n  \"coupling_assessment\": \"\",\n  \"repository_structure_quality\": \"\",\n  \"strengths\": [\"\"],\n  \"weaknesses\": [\"\"],\n  \"overall_architecture_score\": \"number from 1 to 10\"\n}\n\nNo incluyas texto fuera del JSON.",
  "include_repo_context": true,
  "repo_path": "/home/ezer/VidarteTicliahuanca-2509af-main",
  "target_model": "model-two",
  "chunk_size": 13000,
  "chunk_overlap": 200,
  "max_chunks": 8
}
```

## 5) `POST /api/ai-model-comparison/compare-both-chunked`

```json
{
  "system_prompt": "Eres un arquitecto de software senior especializado en análisis de repositorios y evaluación de calidad de arquitectura.",
  "prompt": "Analiza la arquitectura del proyecto usando el contexto del repositorio proporcionado y responde solo JSON.",
  "include_repo_context": true,
  "repo_path": "/home/ezer/VidarteTicliahuanca-2509af-main",
  "chunk_size": 13000,
  "chunk_overlap": 200,
  "max_chunks": 8,
  "generate_final_analysis": false
}
```

## Notas rápidas

- `target_model` solo aplica en `compare-single-chunked` y acepta: `model-one` o `model-two`.
- Límites API:
  - `chunk_size`: `100` a `20000`
  - `chunk_overlap`: `0` a `5000`
  - `max_chunks`: `1` a `100`
  - `chunk_overlap` debe ser menor que `chunk_size`.
- `repo_path` debe existir y ser un directorio válido.
