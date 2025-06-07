#!/bin/bash
# Script para limpiar repositorio existente y mantener solo archivos esenciales

echo "🧹 LIMPIANDO REPOSITORIO EXISTENTE"
echo "=================================="

# Verificar que estamos en el directorio correcto
if [ ! -f "yolov8_recognition.py" ]; then
    echo "❌ No se encontró yolov8_recognition.py"
    echo "💡 Ejecuta este script desde: ~/Puzzlebot_Challenge/ros2_ws/src/yolobot_recognition/yolobot_recognition"
    exit 1
fi

echo "📁 Directorio actual: $(pwd)"

# 1. ELIMINAR CARPETAS PESADAS DE DATASETS
echo ""
echo "🗑️ ELIMINANDO DATASETS PESADOS..."

# Datasets de entrenamiento (varios GB)
rm -rf puzzlebot_roboflow_training/dataset/ 2>/dev/null && echo "   ✅ Eliminado: puzzlebot_roboflow_training/dataset/"
rm -rf puzzlebot_roboflow_training/roboflow_dataset/ 2>/dev/null && echo "   ✅ Eliminado: puzzlebot_roboflow_training/roboflow_dataset/"
rm -rf puzzlebot_yolo_training/ 2>/dev/null && echo "   ✅ Eliminado: puzzlebot_yolo_training/"
rm -rf extracted_dataset/ 2>/dev/null && echo "   ✅ Eliminado: extracted_dataset/"
rm -rf dataset/ 2>/dev/null && echo "   ✅ Eliminado: dataset/"
rm -rf input_dataset/ 2>/dev/null && echo "   ✅ Eliminado: input_dataset/"
rm -rf roboflow_dataset/ 2>/dev/null && echo "   ✅ Eliminado: roboflow_dataset/"

# Dataset específicos mencionados en tu estructura
rm -rf single_class_training/dataset/ 2>/dev/null && echo "   ✅ Eliminado: single_class_training/dataset/"
rm -rf single_class_training/input_dataset/ 2>/dev/null && echo "   ✅ Eliminado: single_class_training/input_dataset/"

# 2. ELIMINAR RESULTADOS DE ENTRENAMIENTO PESADOS
echo ""
echo "🗑️ ELIMINANDO RESULTADOS DE ENTRENAMIENTO PESADOS..."

# Mantener solo results.csv, eliminar resto
find . -name "results" -type d | while read dir; do
    if [ -d "$dir" ]; then
        # Conservar archivos CSV y PNG, eliminar el resto
        find "$dir" -type f ! -name "*.csv" ! -name "*.png" ! -name "*.yaml" -delete 2>/dev/null
        # Eliminar carpetas de experimentos completas excepto archivos específicos
        find "$dir" -name "train*" -type d -exec rm -rf {} \; 2>/dev/null
        find "$dir" -name "val*" -type d -exec rm -rf {} \; 2>/dev/null
        find "$dir" -name "weights" -type d -exec rm -rf {} \; 2>/dev/null
        echo "   ✅ Limpiado: $dir"
    fi
done

# Eliminar carpetas de resultados completas si están muy pesadas
rm -rf puzzlebot_roboflow_training/results/*/weights/ 2>/dev/null
rm -rf puzzlebot_roboflow_training/results/*/runs/ 2>/dev/null

# 3. ELIMINAR ARCHIVOS TEMPORALES Y CACHE
echo ""
echo "🗑️ ELIMINANDO ARCHIVOS TEMPORALES..."

# Python cache
find . -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null && echo "   ✅ Eliminado: __pycache__"
find . -name "*.pyc" -delete 2>/dev/null
find . -name "*.pyo" -delete 2>/dev/null

# Logs y temporales
find . -name "*.log" -delete 2>/dev/null && echo "   ✅ Eliminados: logs"
rm -rf wandb/ 2>/dev/null && echo "   ✅ Eliminado: wandb/"
rm -rf runs/ 2>/dev/null && echo "   ✅ Eliminado: runs/"

# Archivos de imagen temporales (mantener solo muestras)
find . -name "*.jpg" -size +1M -delete 2>/dev/null
find . -name "*.png" -size +1M -delete 2>/dev/null

# 4. LIMPIAR MODELOS DUPLICADOS
echo ""
echo "🎯 ORGANIZANDO MODELOS..."

# Crear carpeta models limpia si no existe
mkdir -p models/

# Mantener solo el modelo final principal
MAIN_MODEL="puzzlebot_traffic_signs_final.pt"
DATASET_YAML="dataset.yaml"

# Buscar y conservar modelo principal
if [ -f "$MAIN_MODEL" ]; then
    echo "   ✅ Modelo principal encontrado: $MAIN_MODEL"
elif [ -f "models/$MAIN_MODEL" ]; then
    echo "   ✅ Modelo principal en models/: $MAIN_MODEL"
else
    # Buscar en subdirectorios
    MODEL_FOUND=$(find . -name "$MAIN_MODEL" -type f | head -1)
    if [ ! -z "$MODEL_FOUND" ]; then
        cp "$MODEL_FOUND" models/
        echo "   ✅ Copiado modelo principal a models/: $MODEL_FOUND"
    else
        echo "   ⚠️ Modelo principal no encontrado: $MAIN_MODEL"
    fi
fi

# Conservar dataset.yaml
YAML_FOUND=$(find . -name "$DATASET_YAML" -type f | head -1)
if [ ! -z "$YAML_FOUND" ]; then
    cp "$YAML_FOUND" models/
    echo "   ✅ Copiado dataset.yaml a models/"
fi

# Eliminar modelos duplicados o innecesarios (mantener solo el principal)
find . -name "*.pt" -not -path "./models/*" -size +10M -delete 2>/dev/null && echo "   ✅ Eliminados: modelos duplicados"

# 5. CREAR .gitignore SI NO EXISTE
echo ""
echo "📝 CREANDO/ACTUALIZANDO .gitignore..."

cat > .gitignore << 'EOF'
# Dataset pesado (no subir a GitHub)
**/dataset/
**/datasets/
**/roboflow_dataset/
**/extracted_dataset/
**/input_dataset/
**/*_dataset/
*.zip
*.tar.gz

# Resultados de entrenamiento pesados
**/runs/
**/results/train*/
**/results/val*/
**/weights/
**/checkpoints/
**/logs/

# Mantener solo modelos finales específicos
models/**/*.pt
!models/puzzlebot_traffic_signs_final.pt
!models/oil_pan_final.pt

# Archivos temporales de Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Entornos virtuales
venv/
env/
ENV/
.venv/

# ROS2 build artifacts
build/
install/
log/
.colcon_workspace

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
*.tmp

# Logs y cache
*.log
wandb/
.cache/
.ultralytics/

# Imagenes temporales grandes
*.jpg
*.jpeg
*.png
*.bmp
*.mp4
*.avi
!docs/**/*.png
!examples/**/*.png
EOF

echo "   ✅ .gitignore creado/actualizado"

# 6. CREAR REQUIREMENTS.TXT
echo ""
echo "📦 CREANDO requirements.txt..."

cat > requirements.txt << 'EOF'
# Dependencias esenciales para YoloBot Recognition
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.21.0
pillow>=9.0.0
pyyaml>=6.0
pandas>=1.3.0
matplotlib>=3.5.0
albumentations>=1.3.0

# ROS2 dependencies (install via apt)
# ros-humble-cv-bridge
# ros-humble-sensor-msgs
# ros-humble-std-msgs
EOF

echo "   ✅ requirements.txt creado"

# 7. ACTUALIZAR README SI NO EXISTE
if [ ! -f "README.md" ]; then
    echo ""
    echo "📝 CREANDO README.md..."
    
    cat > README.md << 'EOF'
# YoloBot Recognition

🤖 **Sistema de detección de señales de tráfico para PuzzleBot usando YOLOv8 y ROS2**

## 🎯 Características

- ✅ Detección en tiempo real de señales de tráfico
- ✅ 6 clases: `fwd`, `left`, `right`, `stop`, `triDwn`, `triUp`
- ✅ 79.8% mAP@50, 99.5% precisión en direcciones
- ✅ 133 FPS (7.5ms/imagen)
- ✅ ROS2 Humble compatible

## 🚀 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt

# Construir ROS2 package
colcon build --packages-select yolobot_recognition
source install/setup.bash

# Ejecutar
ros2 run yolobot_recognition yolov8_recognition
```

## 📡 Topics ROS2

- **Input**: `image_raw/compressed`
- **Output**: `/Yolov8_Inference`, `/inference_result`

## 🎯 Modelo Entrenado

El modelo `puzzlebot_traffic_signs_final.pt` fue entrenado con:
- 67 epochs (early stopping)
- RTX A4000 GPU
- Dataset de Roboflow con ~28,000 imágenes augmentadas
- Optimizado para señales de tráfico (sin flip horizontal)

## 📊 Rendimiento

| Clase | mAP@50 | Descripción |
|-------|---------|-------------|
| left  | 99.5%  | Girar izquierda |
| right | 99.5%  | Girar derecha |
| fwd   | 78.7%  | Seguir adelante |
| stop  | 49.7%  | Alto/Pare |
| triDwn| 83.2%  | Ceder paso |
| triUp | 68.1%  | Construcción |
EOF

    echo "   ✅ README.md creado"
fi

# 8. MOSTRAR RESUMEN FINAL
echo ""
echo "📊 RESUMEN DE LIMPIEZA COMPLETADA:"
echo "================================="

# Calcular tamaño actual
TOTAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
echo "📁 Tamaño total actual: $TOTAL_SIZE"

echo ""
echo "✅ ARCHIVOS CONSERVADOS:"
ls -la *.py 2>/dev/null | awk '{print "   📄 " $9 " (" $5 " bytes)"}'
ls -la models/*.pt 2>/dev/null | awk '{print "   🎯 " $9 " (" $5 " bytes)"}'
ls -la models/*.yaml 2>/dev/null | awk '{print "   ⚙️ " $9 " (" $5 " bytes)"}'
ls -la config/*.yaml 2>/dev/null | awk '{print "   📋 " $9 " (" $5 " bytes)"}'

echo ""
echo "🗑️ ELIMINADO:"
echo "   ❌ Datasets pesados (varios GB)"
echo "   ❌ Resultados de entrenamiento"
echo "   ❌ Archivos temporales"
echo "   ❌ Cache de Python"
echo "   ❌ Modelos duplicados"

echo ""
echo "🚀 LISTO PARA GITHUB:"
echo "   git add ."
echo "   git commit -m 'Clean repository: remove heavy datasets and keep essentials'"
echo "   git push origin main"

echo ""
echo "📊 Contenido final estimado: <100MB"
echo "🎉 Repositorio limpio y listo para subir!"