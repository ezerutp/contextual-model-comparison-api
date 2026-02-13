#!/bin/bash

set -e

echo "=========================================="
echo "AI Model Comparison API - Setup Script"
echo "=========================================="
echo ""

# ============================================
# FUNCIONES AUXILIARES
# ============================================

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 no está instalado"
        return 1
    fi
    return 0
}

# ============================================
# VERIFICAR REQUISITOS
# ============================================
echo "Checking system requirements..."
echo ""

# Verificar Python 3
if ! check_command python3; then
    echo "Por favor instala Python 3.8 o superior"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python 3 found: v$PYTHON_VERSION"

# Verificar Node.js
if ! check_command node; then
    echo "Por favor instala Node.js"
    exit 1
fi
NODE_VERSION=$(node --version | sed 's/v//')
echo "✓ Node.js found: v$NODE_VERSION"

# Verificar npm
if ! check_command npm; then
    echo "Por favor instala npm"
    exit 1
fi
NPM_VERSION=$(npm --version)
echo "✓ npm found: v$NPM_VERSION"

echo ""

# ============================================
# VERIFICAR E INSTALAR REPOMIX
# ============================================
echo "Checking repomix..."

if ! command -v repomix &> /dev/null; then
    echo "repomix no encontrado. Instalando globalmente..."
    npm install -g repomix
    echo "✓ repomix instalado"
else
    REPOMIX_VERSION=$(repomix --version 2>/dev/null || echo "unknown")
    echo "✓ repomix found: $REPOMIX_VERSION"
fi

echo ""

# ============================================
# CONFIGURAR PYTHON VIRTUALENV
# ============================================
echo "Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    echo "Creando virtualenv..."
    python3 -m venv venv
    echo "✓ Virtual environment creado"
else
    echo "✓ Virtual environment ya existe"
fi

# Activar virtualenv
source venv/bin/activate
echo "✓ Virtual environment activado"

echo ""

# ============================================
# INSTALAR DEPENDENCIAS PYTHON
# ============================================
echo "Installing Python dependencies..."

pip install --upgrade pip setuptools wheel > /dev/null 2>&1

pip install \
    fastapi \
    uvicorn \
    google-generativeai \
    python-dotenv \
    scipy \
    scikit-learn \
    numpy \
    pydantic

if [ $? -ne 0 ]; then
    echo "ERROR: Falló la instalación de dependencias Python"
    exit 1
fi

echo "✓ Python dependencies instaladas"
echo ""

# ============================================
# VERIFICAR / CREAR .env
# ============================================
echo "Setting up environment configuration..."

if [ ! -f ".env" ]; then
    echo "ADVERTENCIA: archivo .env no encontrado"
    echo ""
    echo "Creando .env.example como referencia..."
    cat > .env.example << 'EOF'
# API Key de Google AI Studio (REQUERIDO)
GOOGLE_AI_API_KEY=

# Modelos a usar
MODEL_NAME_ONE=gemini-2.5-flash
MODEL_NAME_TWO=gemma-3-27b-it

# Modelo para evaluación de repositorios
EVALUATOR_MODEL_NAME=gemini-2.5-flash

# Ruta default de repositorio
REPO_PATH=.

# Tokens máximo (0 = sin límite)
MAX_INPUT_TOKENS=0
EOF
    echo "✓ .env.example creado"
    echo ""
    echo "   DEBES crear un archivo .env con tu GOOGLE_AI_API_KEY"
    echo "    Copia .env.example a .env y completa los valores"
    echo ""
    exit 1
else
    echo "✓ .env file encontrado"
fi

# Verificar que tenga API KEY
if ! grep -q "GOOGLE_AI_API_KEY=" .env || grep "^GOOGLE_AI_API_KEY=$" .env > /dev/null; then
    echo "ERROR: GOOGLE_AI_API_KEY vacío en .env"
    exit 1
fi

echo ""

# ============================================
# CREAR ARCHIVOS __init__.py
# ============================================
echo "Setting up package structure..."

touch routers/__init__.py
touch prompts/__init__.py
echo "✓ Package structure initialized"

echo ""

# ============================================
# FINALIZACIÓN
# ============================================
echo "=========================================="
echo "✓ Setup completado exitosamente!"
echo "=========================================="
echo ""
echo "Para iniciar la API:"
echo ""
echo "  1. Asegúrate de tener .env configurado con GOOGLE_AI_API_KEY"
echo "  2. Ejecuta:"
echo ""
echo "     source venv/bin/activate"
echo "     uvicorn api_server:app --reload --port 8000"
echo ""
echo "La API estará disponible en: http://localhost:8000"
echo ""
echo "Endpoints disponibles:"
echo "  - POST /api/ai-model-comparison/model-one"
echo "  - POST /api/ai-model-comparison/model-two"
echo "  - POST /api/ai-model-comparison/compare-both"
echo "  - POST /api/ai-model-comparison/compare-both-chunked"
echo "  - POST /api/ai-model-comparison/compare-single-chunked"
echo "  - POST /api/repo-evaluation/evaluate"
echo "  - POST /api/repo-evaluation/batch"
echo "  - GET  /api/repo-evaluation/health"
echo ""
echo "Documentación interactiva:"
echo "  http://localhost:8000/docs"
echo ""
echo "=========================================="
