#!/bin/bash

# Script para iniciar la API REST
# Usage: bash start_api.sh [--repo-path /ruta/al/repo]

if [ ! -d "venv" ]; then
    echo "ERROR: Entorno virtual no existe. Ejecuta primero: bash setup.sh"
    exit 1
fi

CLI_REPO_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-path)
            CLI_REPO_PATH="$2"
            shift 2
            ;;
        *)
            echo "ERROR: argumento no reconocido: $1"
            echo "Uso: bash start_api.sh [--repo-path /ruta/al/repo]"
            exit 1
            ;;
    esac
done

source venv/bin/activate

if [ ! -f ".env" ]; then
    echo "ERROR: Archivo .env no existe. Ejecuta primero: bash setup.sh"
    exit 1
fi

# Carga variables de entorno
export $(cat .env | grep -v '^#' | xargs)

if [ -z "$GOOGLE_AI_API_KEY" ]; then
    echo "ERROR: GOOGLE_AI_API_KEY no esta configurado en .env"
    exit 1
fi

if [ -n "$CLI_REPO_PATH" ]; then
    if [ ! -d "$CLI_REPO_PATH" ]; then
        echo "ERROR: --repo-path no es un directorio valido: $CLI_REPO_PATH"
        exit 1
    fi
    export REPO_PATH="$CLI_REPO_PATH"
fi

echo "Iniciando API de Comparacion de Modelos..."
echo ""
echo "Servidor disponible en: http://localhost:8000"
echo "Ruta repo por defecto: ${REPO_PATH:-.}"
echo ""
echo "Modelos disponibles:"
echo "  POST /api/ai-model-comparison/model-one"
echo "  POST /api/ai-model-comparison/model-two"
echo "  POST /api/ai-model-comparison/compare-both"
echo ""
echo "Presiona Ctrl+C para detener"
echo ""

python api_server.py
