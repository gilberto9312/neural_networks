#!/bin/bash

# Script de descarga de datasets para proyectos LLM

echo "📦 Descargando datasets..."

# Africa Galore
echo "Descargando Africa Galore..."
wget https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json \
    -O africa_galore.json

# Verificar descarga
if [ -f "africa_galore.json" ]; then
    echo "✅ Africa Galore descargado exitosamente"
else
    echo "❌ Error descargando Africa Galore"
    exit 1
fi

# Opcional: TinyStories (comentado por ahora)
# echo "Descargando TinyStories..."
# wget https://... -O tiny_stories.json

echo ""
echo "✅ Todos los datasets descargados exitosamente"
