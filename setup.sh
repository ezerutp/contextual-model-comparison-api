#!/bin/bash

# Setup para API de Comparacion de Modelos
# Usage: bash setup.sh

set -e

echo "Configurando API de Comparacion de Modelos..."
echo ""

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no esta instalado"
    exit 1
fi
echo "OK: Python 3 encontrado: $(python3 --version)"

# Verifica Node.js (necesario para repomix)
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js no esta instalado"
    exit 1
fi
echo "OK: Node.js encontrado: $(node --version)"

# Crea venv
echo ""
echo "Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "OK: Entorno virtual creado"
else
    echo "OK: Entorno virtual ya existe"
fi

# Activa venv
source venv/bin/activate
echo "OK: Entorno virtual activado"

# Instala dependencias de Python
echo ""
echo "Instalando dependencias de Python..."
pip install --upgrade pip
pip install -r requirements.txt
echo "OK: Dependencias de Python instaladas"

# Instala repomix (para incluir contexto del repo en prompts)
echo ""
echo "Instalando repomix (para contexto de repositorio)..."
if command -v repomix &> /dev/null; then
    echo "OK: repomix ya esta instalado: $(repomix --version)"
else
    npm install -g repomix
    echo "OK: repomix instalado"
fi

# Crea archivo .env
echo ""
if [ ! -f ".env" ]; then
    echo "creating .env file..."
    echo "GOOGLE_AI_API_KEY=" > .env
    echo "OK: Archivo .env creado"
    echo "IMPORTANTE: Edita .env y agrega tu clave de API de Google"
else
    echo "OK: Archivo .env ya existe"
fi

echo ""
echo "OK: Configuracion completada!"
echo ""
echo "Proximos pasos:"
echo "1. Edita .env con tu GOOGLE_AI_API_KEY"
echo "2. Ejecuta: bash start_api.sh"
echo "3. La API estara en: http://localhost:8000"
echo ""
