# AI Model Comparison API - Evaluación de Repositorios de Candidatos

## Descripción General

Este proyecto extiende la **API REST de Comparación de Modelos de IA** con un sistema completo de evaluación de repositorios de candidatos usando **Gemini 2.5 Flash** y la **rúbrica oficial de HireLens**.

### Características Principales

- ✅ **Evaluación Individual**: Analiza un repositorio de candidato contra la rúbrica
- ✅ **Evaluación en Batch**: Evalúa múltiples repos etiquetados OK/BAD y genera métricas
- ✅ **Rúbrica Oficial**: Pesos exactos y criterios de HireLens
- ✅ **Detección de Alucinaciones**: Identifica criterios donde el LLM afirma sin evidencia
- ✅ **Métricas de Discriminación**: Mann-Whitney U, score gap, overlap analysis
- ✅ **Integración Repomix**: Extrae contexto automático del código fuente
- ✅ **API REST FastAPI**: Endpoints documentados con Swagger

---

## Estructura de Archivos

```
.
├── api_server.py                 # Main FastAPI app (sin cambios esenciales)
├── ai_model_comparison.py        # Comparador de modelos (sin cambios)
├── repo_evaluator.py             # [NUEVO] Evaluador individual
├── batch_evaluator.py            # [NUEVO] Evaluador en batch + métricas
├── routers/
│   ├── __init__.py
│   └── evaluation.py             # [NUEVO] Endpoints /api/repo-evaluation/*
├── prompts/
│   ├── __init__.py
│   └── evaluation_prompt.py       # [NUEVO] System prompt + rúbrica centralizada
├── run.sh                         # [NUEVO] Script de setup
├── README.md                      # Este archivo
├── .env.example                   # [GENERADO POR run.sh] Plantilla de config
└── venv/                          # [GENERADO POR run.sh] Virtual environment
```

---

## Instalación y Setup

### 1. Ejecutar Script de Setup

```bash
chmod +x run.sh
./run.sh
```

Este script:
- ✓ Verifica Python 3, Node.js, npm
- ✓ Instala repomix globalmente
- ✓ Crea virtualenv de Python
- ✓ Instala dependencias (fastapi, scipy, scikit-learn, etc.)
- ✓ Genera `.env.example`

### 2. Configurar Variables de Entorno

```bash
# Copiar template
cp .env.example .env

# Editar .env y agregar tu API key
# GOOGLE_AI_API_KEY=sk-xxxxx
```

**Variables disponibles**:
```env
# REQUERIDO
GOOGLE_AI_API_KEY=

# Modelos (defaults)
MODEL_NAME_ONE=gemini-2.5-flash
MODEL_NAME_TWO=gemma-3-27b-it
EVALUATOR_MODEL_NAME=gemini-2.5-flash

# Opcional
REPO_PATH=.
MAX_INPUT_TOKENS=0
```

### 3. Iniciar la API

```bash
source venv/bin/activate
uvicorn api_server:app --reload --port 8000
```

Acceder a:
- **API Docs**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Endpoints de Evaluación

### POST `/api/repo-evaluation/evaluate`

Evalúa un repositorio individual.

**Request**:
```json
{
  "repo_path": "/ruta/al/repo",
  "candidate_id": "candidato_01"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "candidate_id": "candidato_01",
    "model_used": "gemini-2.5-flash",
    "evaluation_timestamp": "2026-02-13T10:45:30Z",
    
    "backend": {
      "tecnologia_backend": {
        "value": "spring_boot",
        "peso": 0,
        "evidence": "..."
      },
      "segunda_etapa_folders": {
        "value": true,
        "peso": 1.5,
        "evidence": "FolderEntity.java línea 45 con @ManyToOne a TodoItem"
      },
      ...
    },
    
    "frontend": { ... },
    "other": { ... },
    
    "scores": {
      "backend_score": 3.45,
      "frontend_score": 2.80,
      "other_score": 1.00,
      "total_score": 7.25,
      "max_possible": 9.95
    },
    
    "decision": "PASS",
    
    "llm_metrics": {
      "hallucination_flags": [],
      "hallucination_score": 0.0,
      "confidence_score": 1.0,
      "low_evidence_criteria": []
    },
    
    "input_tokens_used": 2450,
    "latency_ms": 3250.45
  }
}
```

### POST `/api/repo-evaluation/batch`

Evalúa un dataset completo.

**Estructura de Dataset**:
```
dataset/
  candidato_01_OK/
    (código del candidato 1)
  candidato_02_BAD/
    (código del candidato 2)
  candidato_03_OK/
    (código del candidato 3)
```

**Request**:
```json
{
  "dataset_path": "/ruta/a/dataset"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "dataset_path": "/ruta/a/dataset",
    "total_repos": 15,
    "ok_repos": 10,
    "bad_repos": 5,
    
    "evaluations": [
      { ... }, // EvaluationResult por cada repo
      { ... }
    ],
    
    "metrics": {
      "discrimination": {
        "mean_score_ok": 7.45,
        "mean_score_bad": 3.20,
        "score_gap": 4.25,
        "mann_whitney_u": 32.5,
        "p_value": 0.0012,
        "discriminates_well": true
      },
      
      "classification": {
        "accuracy": 0.867,
        "precision": 0.875,
        "recall": 0.850,
        "f1_score": 0.862,
        "confusion_matrix": {
          "tp": 7,
          "fp": 1,
          "tn": 4,
          "fn": 3
        }
      },
      
      "hallucinations": {
        "mean_hallucination_score": 0.08,
        "max_hallucination_score": 0.25,
        "repos_with_hallucinations": 2,
        "most_hallucinated_criteria": [
          { "criterio": "uso_dtos", "frequency": 2 },
          { "criterio": "libreria_migraciones", "frequency": 1 }
        ]
      },
      
      "distribution": {
        "scores_ok": [6.5, 7.0, 7.25, 7.45, ...],
        "scores_bad": [2.5, 3.0, 3.5, 4.0, ...],
        "std_ok": 0.45,
        "std_bad": 0.65,
        "overlap_coefficient": 0.15
      }
    },
    
    "total_tokens_used": 45230,
    "total_latency_ms": 125000.0,
    "average_latency_ms": 8333.33
  }
}
```

### GET `/api/repo-evaluation/health`

Verifica estado del servicio.

**Response**:
```json
{
  "status": "ok",
  "model": "gemini-2.5-flash",
  "rpd_warning": "max 20 requests/day on free tier"
}
```

---

## La Rúbrica de HireLens

### BACKEND (máx. 4.20 puntos)

| Criterio | Peso | Descripción |
|----------|------|-------------|
| B1. tecnologia_backend | 0 | Framework usado (informativo) |
| B2. segunda_etapa_folders | 1.5 | Relación TodoItem ↔ Folders |
| B3. arquitectura_capas | 1.0 | Controller/Service/Repository separados |
| B4. uso_dtos | 0.5 | DTOs para request/response |
| B5. implemento_auth | 0.25 | Login/JWT/autenticación |
| B6. mappings_orm | 0.2 | Mappings BD + Repositories |
| B7. libreria_migraciones | 0.75 | Liquibase, Flyway, Alembic, etc. |

### FRONTEND (máx. 4.00 puntos)

| Criterio | Peso | Descripción |
|----------|------|-------------|
| F1. tecnologia_frontend | 0 | Framework usado (informativo) |
| F2. requests_clase_separada | 0.6 | Requests en servicio dedicado |
| F3. minimo_tres_componentes | 1.2 | ≥3 componentes |
| F4. manejo_asincronia | 0.6 | Async/await bien manejado |
| F5. sin_dom_manual | 0.4 | NO usa getElementById/innerHTML |
| F6. uso_router | 0.6 | Router para navegación |
| F7. libreria_estilos | 0.6 | Bootstrap/MUI/Tailwind/etc |

### OTHER (máx. 1.75 puntos)

| Criterio | Peso | Descripción |
|----------|------|-------------|
| O1. scripts_inicializacion | 0 | Setup/Dockerfile (informativo) |
| O2. deploy_plataforma | 0.75 | Deploy en Heroku/Railway/etc |
| O3. nombres_ingles | 0.5 | Código en inglés + expresivo |
| O4. uso_correcto_git | 0.5 | ≥3 commits con mensajes claros |

### Umbrales de Decisión

- **PASS**: score_total ≥ 7.0
- **REVIEW**: 4.5 ≤ score_total < 7.0
- **FAIL**: score_total < 4.5

---

## Detalles Técnicos

### RepoEvaluator

Clase principal que:
- Usa exclusivamente **Gemini 2.5 Flash**
- Integra **Repomix** para extraer contexto del código
- Fuerza respuesta en **JSON** con `response_mime_type: "application/json"`
- **Calcula scores en Python**, nunca confía en cálculos del LLM
- Detecta y registra **hallucinations** en post-procesamiento

**Método principal**:
```python
evaluator = RepoEvaluator(api_key="...", model_name="gemini-2.5-flash")
result = evaluator.evaluate(repo_path="/ruta", candidate_id="cand_01")
print(f"Score: {result.scores.total_score}, Decision: {result.decision}")
print(f"Hallucinations: {result.llm_metrics.hallucination_score}")
```

### BatchEvaluator

Extiende RepoEvaluator para evaluar datasets completos.

**Características**:
- Detecta automáticamente etiquetas OK/BAD en nombres de carpetas
- Respeta rate limits (sleep 3s entre requests)
- Calcula Mann-Whitney U, overlap coefficient, clasificación binaria
- Identifica criterios más frecuentemente alucinados
- Genera métricas de discriminación del modelo

**Método principal**:
```python
batch = BatchEvaluator()
result = batch.evaluate_dataset("/dataset/path")
print(f"Discriminates well: {result.metrics.discrimination.discriminates_well}")
print(f"Hallucination score mean: {result.metrics.hallucinations.mean_hallucination_score}")
```

### Detección de Alucinaciones

Un criterio se marca como **alucinación** si:
1. `value = True` (el LLM afirmó que sí)
2. `peso > 0` (es criterio contable)
3. **Y** alguno de:
   - `evidence` está vacío
   - `evidence` tiene < 15 caracteres
   - `evidence` es genérico ("N/A", "no encontrado", "sí", etc.)

**Cálculo**:
```python
hallucination_score = hallucinated_count / total_true_criteria
confidence_score = 1.0 - hallucination_score
```

---

## Ejemplos de Uso

### Evaluar Un Repositorio Individual

```python
from repo_evaluator import RepoEvaluator

evaluator = RepoEvaluator()
result = evaluator.evaluate(
    repo_path="/home/user/candidate-repo",
    candidate_id="candidato_01"
)

print(f"Tecnología Backend: {result.backend['tecnologia_backend']['value']}")
print(f"Score Total: {result.scores.total_score:.2f}")
print(f"Decision: {result.decision}")
print(f"Confianza: {result.llm_metrics.confidence_score:.2%}")
```

### Evaluar Dataset Completo

```python
from batch_evaluator import BatchEvaluator

batch = BatchEvaluator()
result = batch.evaluate_dataset("/dataset/candidates")

# Metrics
metrics = result.metrics.discrimination
print(f"Score OK promedio: {metrics.mean_score_ok:.2f}")
print(f"Score BAD promedio: {metrics.mean_score_bad:.2f}")
print(f"Discrimina bien: {metrics.discriminates_well}")

# Clasificación
clf = result.metrics.classification
print(f"Accuracy: {clf.accuracy:.1%}")
print(f"F1 Score: {clf.f1_score:.3f}")

# Alucinaciones
hall = result.metrics.hallucinations
print(f"Mean Hallucination: {hall.mean_hallucination_score:.2%}")
print(f"Top alucinado: {hall.most_hallucinated_criteria[0]['criterio']}")
```

### Evaluar vía API

```bash
# Evaluación individual
curl -X POST http://localhost:8000/api/repo-evaluation/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/ruta/al/repo",
    "candidate_id": "candidato_01"
  }' | jq .

# Batch
curl -X POST http://localhost:8000/api/repo-evaluation/batch \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/ruta/a/dataset"}' | jq '.data.metrics'
```

---

## Configuración Personalizada

### Cambiar Modelo de Evaluación

Editar `.env`:
```env
EVALUATOR_MODEL_NAME=gemini-2.0-pro
```

O pasar en código:
```python
evaluator = RepoEvaluator(model_name="gemini-2.0-pro")
```

### Límite de Tokens de Input

```env
# 0 = sin límite (recomendado)
MAX_INPUT_TOKENS=0

# O limitar a 10k tokens
MAX_INPUT_TOKENS=10000
```

### Ruta Default de Repositorio

```env
REPO_PATH=/home/user/repos
```

---

## Optimización de Repomix

El código utiliza estrategia de fallback:

1. **Intenta primero** `--include` con extensiones de código fuente:
   ```
   *.java, *.py, *.js, *.ts, *.jsx, *.tsx, *.go, *.rb, *.kt,
   *.md, *.sql, *.xml, *.gradle, *.yaml, *.yml
   ```

2. **Si falla**, fallback a `--ignore` con patrones simples:
   ```
   node_modules, dist, build, target, .git, *.lock, *.min.js,
   __pycache__, .venv, .idea
   ```

3. **Si timeout** (>30s), usa fallback: lista básica de archivos

### Logging de Repomix

```
Repomix output: 125000 chars, ~31250 estimated tokens
```

---

## Rate Limits y Consideraciones

### Gemini 2.5 Flash (Free Tier)

- **RPD (Requests Per Day)**: 20
- **TPM (Tokens Per Minute)**: 250,000
- **RPM (Requests Per Minute)**: 60

### Estrategia en BatchEvaluator

- Sleep 3 segundos entre requests (respeta RPM)
- NO reintenta automáticamente en fallos
- Registra errores y continúa con siguiente repo

### Estimación de Costo

Si cada evaluación usa ~2,500 tokens:
- 20 requests/día × 2,500 tokens = 50K tokens/día (✓ dentro de límite)
- Dataset de 10 repos = 25K tokens (~5% del límite)

---

## Troubleshooting

### Error: "GOOGLE_AI_API_KEY es requerida"

```bash
# Verificar .env
grep GOOGLE_AI_API_KEY .env

# Debe tener valor, no estar vacío
GOOGLE_AI_API_KEY=sk-xxxxx
```

### Error: "Modelo no retornó JSON válido"

- El LLM enviaba respuesta no-JSON
- Solución: Reintenta, el modelo es determinístico con `temperature: 0.2`
- Verifica que el prompt sea correcto

### Error: "repomix no está instalado"

```bash
npm install -g repomix
```

### Error: "Timeout al extraer contexto"

- El repo es muy grande
- Fallback a lista básica de archivos (sin contenido)
- Considera `--ignore` adicional para archivos pesados

### API no responde en /docs

Verifica:
```bash
# ¿Está corriendo en puerto 8000?
lsof -i :8000

# Logs
uvicorn api_server:app --reload --port 8000  # ver consola
```

---

## Próximas Mejoras

- [ ] Caché de evaluaciones por repo_path + hash
- [ ] Exportación de resultados a CSV/Excel
- [ ] Dashboard con gráficos de métricas batch
- [ ] Webhook para notificaciones de evaluación completa
- [ ] Soporte para múltiples lenguajes en prompts
- [ ] Evaluación incremental (delta) de cambios en repo

---

## Licencia

MIT - Ver LICENSE file

---

## Soporte

Para preguntas o issues:
1. Revisar este README
2. Consultar docstrings en código
3. Revisar logs de stderr/stdout
4. Verificar formato de dataset/repo_path

### Cositas para aclarar en base a las métricas obtenidas 

En discriminación, sale:
- mean_score (la media p)
- score_gap (la diferencia promedio entre los repos buenos y malos)
- Mann-Whitney U (no hacer caso, como el modelo realiza clasificación binaria, este número se encuentra contaminado con el criterio adicional (review))
- p_value va a la par con Mann-Whitney U, es la probabilidad de cuánto esa diferencia de medias sea accidental.
- overlap_coefficient: 0.0 es separacion perfecta, 1.0 es que ambos grupos son indistinguibles.

En clasificacion:
- TP (True positives)
- TN (True Negatives)
- FP (False Positives)
- FN (False negatives)

# Critico:
- El sistema muestra bajo accuracy y precision. Esto se debe a los puntajes clasificados como review (quiza falta calibrar mejor la rubrica, toda revisarlo)
- En general el modelo clasifica muy bien, pero toda aclarar el tema de la rubrica. 
