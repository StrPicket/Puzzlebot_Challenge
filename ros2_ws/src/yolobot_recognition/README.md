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
