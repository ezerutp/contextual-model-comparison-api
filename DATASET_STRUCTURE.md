"""
EJEMPLO: Estructura de Dataset para Batch Evaluation

Para evaluar múltiples repositorios de candidatos, organiza tu dataset así:

dataset/
├── candidato_01_OK/
│   ├── backend/
│   │   ├── src/
│   │   │   ├── controller/
│   │   │   ├── service/
│   │   │   └── repository/
│   │   ├── pom.xml
│   │   └── build.gradle
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── services/
│   │   │   └── pages/
│   │   └── package.json
│   └── README.md
│
├── candidato_02_BAD/
│   ├── backend/
│   └── ...
│
├── candidato_03_OK/
│   └── ...
│
└── candidato_04_BAD/
    └── ...

REGLAS IMPORTANTES:

1. El nombre de cada carpeta DEBE contener "OK" o "BAD"
   - "candidato_01_OK" → label = 1 (bueno)
   - "candidato_02_BAD" → label = 0 (malo)
   - "candidato_03_EXCELLENT_OK" → label = 1 (contiene "OK")

2. No importa dónde esté "OK"/"BAD" en el nombre
   - "ok_candidate_01" ✓
   - "candidate_bad_01" ✓
   - "candidato_01" ✗ sin etiqueta (se asume BAD)

3. Cada carpeta debe ser un repositorio válido
   - Puede tener subcarpetas (backend, frontend, etc.)
   - Puede tener .git (se extrae el historio de commits)
   - Tiene que tener archivos de código fuente (.java, .py, .js, etc.)

EJEMPLO MÍNIMO:

dataset/
├── good_repo_OK/
│   ├── src/
│   │   ├── main.py
│   │   └── app.py
│   └── package.json
│
└── bad_repo_BAD/
    └── script.py


USO:

from batch_evaluator import BatchEvaluator

batch = BatchEvaluator()
result = batch.evaluate_dataset("/ruta/a/dataset")

print(f"Evaluados: {result.total_repos} repos")
print(f"OK: {result.ok_repos}, BAD: {result.bad_repos}")
print(f"Discriminates well: {result.metrics.discrimination.discriminates_well}")


CUÁNTOS REPOS NECESITAS:

- Mínimo: 5-10 repos (mezcla de OK/BAD)
- Recomendado: 15-20 repos para métricas confiables
- Nota: Cada repo consume ~20-30 segundos + wait de 3s entre requests


RATE LIMITS:

- Gemini 2.5 Flash free tier: max 20 requests/day
- 20 repos = 20 requests ✓ dentro del límite
- Alternativamente, usa gemini-2.0-pro si tienes mayor cuota


ESTRUCTURA DENTRO DE CADA REPO:

No hay restricciones. Puede ser:

1. Monorepo (backend + frontend):
   candidato_01_OK/
   ├── backend/
   └── frontend/

2. Solo backend:
   candidato_02_OK/
   └── src/

3. Solo frontend:
   candidato_03_OK/
   └── components/

4. Estructura original intacta:
   candidato_04_OK/
   ├── (archivo git clonado directamente)
   └── (sin cambios)


CÓMO GENERAR DATASET DE PRUEBA:

# Crear estructura
mkdir -p dataset
cd dataset

# Crear repos OK (fake - simplemente directorios con archivos)
mkdir -p good_candidate_01_OK/{src,tests}
echo "public class UserController {}" > good_candidate_01_OK/src/UserController.java
echo "class UserService: pass" > good_candidate_01_OK/src/user_service.py

mkdir -p good_candidate_02_OK/{backend,frontend}
echo "import React" > good_candidate_02_OK/frontend/App.jsx

# Crear repos BAD
mkdir -p bad_candidate_01_BAD
echo "var x = 1;" > bad_candidate_01_BAD/script.js

mkdir -p bad_candidate_02_BAD/{src}
echo "class Main {}" > bad_candidate_02_BAD/src/Main.java


VALIDACIÓN:

from batch_evaluator import BatchEvaluator

# Esto descubrirá automáticamente los repos
batch = BatchEvaluator()
repos = batch._discover_repos("/ruta/a/dataset")

for repo_name, repo_path, label in repos:
    label_str = "OK" if label == 1 else "BAD"
    print(f"{repo_name}: {label_str}")


SALIDA ESPERADA:

Evaluados: 4 repos
OK: 2, BAD: 2

Metrics (si los repos tienen contenido real):
- mean_score_ok: 7.25
- mean_score_bad: 3.45
- discriminates_well: true (p < 0.05, gap > 2.0)
"""
