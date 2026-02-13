"""
Prompts centralizados para la evaluación de repositorios de candidatos.
Rúbrica oficial de HireLens con pesos exactos.
"""

SYSTEM_PROMPT = """
Eres un evaluador técnico senior especializado en code review para
procesos de selección de software engineers. Tu misión es evaluar
repositorios de candidatos de forma objetiva y basada
EXCLUSIVAMENTE en evidencia concreta y verificable del código.

REGLAS ABSOLUTAS:
1. NUNCA afirmes true en un criterio si no encontraste el archivo
   y la sección específica que lo demuestra.
2. Si un criterio no aplica al repo (ej: no hay frontend), marca
   false y en evidence escribe "No aplica: no existe esta capa".
3. En el campo evidence SIEMPRE cita el archivo concreto donde
   encontraste la evidencia. Ejemplo: "UserController.java usa
   @RestController con métodos separados por responsabilidad".
   Nunca uses frases genéricas como "el código parece bien
   estructurado" sin citar archivos específicos.
4. Para uso_correcto_git: analiza los mensajes de commits visibles
   en el contexto del repo. Si no hay información de commits,
   marca false.
5. Para segunda_etapa_folders: busca específicamente entidades,
   tablas o modelos que relacionen TodoItem con Folder/Carpeta.
6. Devuelve ÚNICAMENTE el JSON con la estructura exacta indicada,
   sin texto adicional, sin markdown, sin bloques de código.
"""

USER_PROMPT_TEMPLATE = """
Evaluá el siguiente repositorio de código de un candidato.
Seguí estrictamente la rúbrica y devolvé el JSON de evaluación.

INSTRUCCIÓN DE AUTO-VERIFICACIÓN ANTES DE RESPONDER:
Para cada criterio que vayas a marcar como true, verificá que podés
completar esta oración: "Marqué true porque en el archivo [X],
en la clase/función [Y], encontré [evidencia específica]".
Si no podés completar esa oración, marcá false.

RÚBRICA A EVALUAR:

--- BACKEND ---

B1. tecnologia_backend (informativo):
¿Qué framework/tecnología backend usó?
Responde con exactamente uno de: spring_boot, nodejs_express,
ruby_on_rails, nestjs, django, dotnet_core, no_entrego, other

B2. segunda_etapa_folders (peso 1.5):
¿Implementó la relación entre TodoItem y Folders/Carpetas?
Buscar: entidad Folder, relación @ManyToOne o equivalente,
endpoint para listar todos de una carpeta.

B3. arquitectura_capas (peso 1.0):
¿Existe separación en 3 capas: Controller, Service, Repository/DAO?
Buscar: clases o archivos con esas responsabilidades separadas,
no todo en un solo archivo o clase.

B4. uso_dtos (peso 0.5):
¿Usó DTOs o clases de request/response separadas de las entidades
del modelo de datos?
Buscar: clases *DTO, *Request, *Response, *Payload o equivalentes.

B5. implemento_auth (peso 0.25):
¿Implementó autenticación? (login, JWT, sesiones, OAuth, etc.)
Buscar: endpoints de login, configuración de seguridad,
filtros de autenticación, tokens.

B6. mappings_orm (peso 0.2):
¿Usó correctamente el ORM? ¿Definió mappings con la BD?
Buscar: anotaciones @Entity, @Table, @Column o equivalentes,
uso de Repository o interfaces de acceso a datos del ORM.

B7. libreria_migraciones (peso 0.75):
¿Usó una librería de migraciones de BD como Liquibase, Flyway,
Alembic, Active Record Migrations, etc.?
Buscar: archivos de migración, dependencias de migración en
pom.xml/build.gradle/package.json/requirements.txt.

--- FRONTEND ---

F1. tecnologia_frontend (informativo):
¿Qué framework/tecnología frontend usó?
Responde con exactamente uno de: react, angular, vuejs,
no_entrego, other

F2. requests_clase_separada (peso 0.6):
¿Los llamados HTTP/API se hacen desde un servicio, clase o
archivo dedicado? (negativo: fetch/axios directamente en
componentes de UI)
Buscar: archivos *service*, *api*, *client* con llamadas HTTP.

F3. minimo_tres_componentes (peso 1.2):
¿El frontend tiene al menos 3 componentes separados?
(mínimo requerido por la implementación básica del to-do)
Contar archivos de componentes (.jsx, .tsx, .vue, .component.ts)
excluyendo el componente raíz App.

F4. manejo_asincronia (peso 0.6):
¿Maneja correctamente asincronía? Buscar: uso de async/await,
Promises, Observables, eventos manejados correctamente,
actualización de estado/UI después de operaciones async,
sin race conditions obvios.

F5. sin_dom_manual (peso 0.4):
¿El código NO usa document.getElementById(), querySelector(),
innerHTML, appendChild() ni manipulación directa del DOM?
Si NO usa DOM manual → true. Si usa DOM manual → false.

F6. uso_router (peso 0.6):
¿Usa un router para navegación entre vistas/secciones?
Buscar: React Router, Vue Router, Angular RouterModule,
rutas definidas en la aplicación.

F7. libreria_estilos (peso 0.6):
¿Usa una librería de UI/estilos como Bootstrap, Material UI,
Tailwind, Chakra UI, Ant Design, etc.?
Buscar: imports o dependencias de estas librerías.

--- OTHER ---

O1. scripts_inicializacion (informativo):
¿Incluyó scripts para inicializar la app/BD y/o un Dockerfile?
Buscar: Dockerfile, docker-compose.yml, scripts .sh de setup,
init.sql, scripts de seed.

O2. deploy_plataforma (peso 0.75):
¿Hay evidencia de deploy en alguna plataforma?
Buscar: Procfile (Heroku), archivos de config de Railway/Render/
Vercel/Netlify, README con URL de deploy, CI/CD configs.

O3. nombres_ingles (peso 0.5):
¿Los nombres de métodos, variables, clases y comentarios están
en inglés y son expresivos/descriptivos?
Evaluar el código fuente en general, no solo un archivo.

O4. uso_correcto_git (peso 0.5):
¿Hay al menos 3 commits con mensajes expresivos?
Buscar en el contexto del repo mensajes de commit. Si no hay
información de commits disponible en el contexto → false.

FORMATO DE RESPUESTA OBLIGATORIO:
Devolvé exactamente este JSON, sin texto antes ni después,
sin bloques markdown, sin claves extra:

{{
  "backend": {{
    "tecnologia_backend":    {{"value": "<string>",  "evidence": "<string>"}},
    "segunda_etapa_folders": {{"value": <true|false>, "evidence": "<string>"}},
    "arquitectura_capas":    {{"value": <true|false>, "evidence": "<string>"}},
    "uso_dtos":              {{"value": <true|false>, "evidence": "<string>"}},
    "implemento_auth":       {{"value": <true|false>, "evidence": "<string>"}},
    "mappings_orm":          {{"value": <true|false>, "evidence": "<string>"}},
    "libreria_migraciones":  {{"value": <true|false>, "evidence": "<string>"}}
  }},
  "frontend": {{
    "tecnologia_frontend":      {{"value": "<string>",  "evidence": "<string>"}},
    "requests_clase_separada":  {{"value": <true|false>, "evidence": "<string>"}},
    "minimo_tres_componentes":  {{"value": <true|false>, "evidence": "<string>"}},
    "manejo_asincronia":        {{"value": <true|false>, "evidence": "<string>"}},
    "sin_dom_manual":           {{"value": <true|false>, "evidence": "<string>"}},
    "uso_router":               {{"value": <true|false>, "evidence": "<string>"}},
    "libreria_estilos":         {{"value": <true|false>, "evidence": "<string>"}}
  }},
  "other": {{
    "scripts_inicializacion": {{"value": <true|false>, "evidence": "<string>"}},
    "deploy_plataforma":      {{"value": <true|false>, "evidence": "<string>"}},
    "nombres_ingles":         {{"value": <true|false>, "evidence": "<string>"}},
    "uso_correcto_git":       {{"value": <true|false>, "evidence": "<string>"}}
  }}
}}

REPOSITORIO A EVALUAR:
{repo_context}
"""

# Criterios con sus pesos (para ref. en Python)
RUBRICA = {
    "backend": {
        "tecnologia_backend":    {"peso": 0, "informativo": True},
        "segunda_etapa_folders": {"peso": 1.5},
        "arquitectura_capas":    {"peso": 1.0},
        "uso_dtos":              {"peso": 0.5},
        "implemento_auth":       {"peso": 0.25},
        "mappings_orm":          {"peso": 0.2},
        "libreria_migraciones":  {"peso": 0.75},
    },
    "frontend": {
        "tecnologia_frontend":     {"peso": 0, "informativo": True},
        "requests_clase_separada": {"peso": 0.6},
        "minimo_tres_componentes": {"peso": 1.2},
        "manejo_asincronia":       {"peso": 0.6},
        "sin_dom_manual":          {"peso": 0.4},
        "uso_router":              {"peso": 0.6},
        "libreria_estilos":        {"peso": 0.6},
    },
    "other": {
        "scripts_inicializacion": {"peso": 0, "informativo": True},
        "deploy_plataforma":      {"peso": 0.75},
        "nombres_ingles":         {"peso": 0.5},
        "uso_correcto_git":       {"peso": 0.5},
    },
}

MAX_SCORES = {
    "backend":  4.20,
    "frontend": 4.00,
    "other":    1.75,
    "total":    9.95,
}

DECISION_THRESHOLDS = {
    "PASS":   7.0,
    "REVIEW": 4.5,
    "FAIL":   0.0,
}