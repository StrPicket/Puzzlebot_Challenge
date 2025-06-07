from ultralytics import YOLO
import torch

print("🚀 Continuando entrenamiento con GPU - Configuración optimizada")
print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Cargar modelo desde epoch 20
model = YOLO('puzzlebot_yolo_training/results/puzzlebot_traffic_signs6/weights/last.pt')

# Entrenamiento optimizado para GTX 1050
try:
    results = model.train(
        data='puzzlebot_yolo_training/dataset.yaml',
        epochs=30,          # Total epochs deseados
        device='cuda',      # GPU
        batch=8,            # Reducido para GTX 1050
        imgsz=640,
        patience=20,
        save_period=5,      # Guardar cada 5 epochs
        project='puzzlebot_yolo_training/results',
        name='gpu_optimized',
        cache='disk',       # Usar disco en lugar de RAM
        workers=4,          # Reducir workers
        verbose=True
    )
    
    print(f"🎯 ¡ENTRENAMIENTO COMPLETADO!")
    print(f"📁 Modelo en: {results.save_dir}/weights/best.pt")
    
    # Copiar a models
    import shutil
    from pathlib import Path
    
    models_dir = Path('puzzlebot_yolo_training/models')
    models_dir.mkdir(exist_ok=True)
    final_model = models_dir / 'puzzlebot_traffic_signs_final.pt'
    
    shutil.copy(results.save_dir / 'weights' / 'best.pt', final_model)
    print(f"🎯 Modelo final copiado: {final_model}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("❌ Error de memoria GPU. Reduciendo batch size...")
        # Reintentar con batch más pequeño
        results = model.train(
            data='puzzlebot_yolo_training/dataset.yaml',
            epochs=30,
            device='cuda',
            batch=4,        # Batch aún más pequeño
            imgsz=640,
            project='puzzlebot_yolo_training/results',
            name='gpu_small_batch'
        )
    else:
        raise e

