#!/usr/bin/env python3
"""
YOLOv8 Training - LISTO PARA EJECUTAR
Con las 6 imágenes proporcionadas
"""

import os
import cv2
import numpy as np
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import albumentations as A

class PuzzleBotYOLOTrainer:
    def __init__(self):
        self.project_root = Path("puzzlebot_yolo_training")
        
        # 🎯 LAS 6 CLASES EXACTAS DE TUS IMÁGENES
        self.classes = [
            'stop',         # Imagen 1: Octágono rojo
            'road_work',    # Imagen 2: Triángulo con trabajador  
            'give_way',     # Imagen 3: Triángulo invertido
            'turn_left',    # Imagen 4: Flecha izquierda azul
            'go_straight',  # Imagen 5: Flecha arriba azul
            'turn_right'    # Imagen 6: Flecha derecha azul
        ]
        
        print("🎯 PuzzleBot YOLO Trainer Inicializado")
        print(f"📊 Clases a entrenar: {len(self.classes)}")
        for i, clase in enumerate(self.classes):
            print(f"   {i}: {clase}")
        
        self.setup_directories()
        self.setup_augmentation()

    def setup_directories(self):
        """Crear estructura completa"""
        dirs = [
            'dataset/images/train', 'dataset/images/val', 'dataset/images/test',
            'dataset/labels/train', 'dataset/labels/val', 'dataset/labels/test',
            'source_images', 'models', 'results'
        ]
        
        for dir_path in dirs:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Directorios creados en: {self.project_root}")

    def setup_augmentation(self):
        """Pipeline de augmentación optimizado para señales de tráfico"""
        self.augmentation = A.Compose([
            # Transformaciones geométricas
            A.HorizontalFlip(p=0.2),  # Menor probabilidad para señales direccionales
            A.Rotate(limit=25, p=0.8),
            A.RandomScale(scale_limit=0.4, p=0.9),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-20, 20), p=0.7),
            
            # Transformaciones de perspectiva (simular ángulos de cámara)
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            
            # Variaciones de iluminación y color
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.4),
            
            # Efectos de degradación (simular condiciones reales)
            A.GaussNoise(var_limit=20.0, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ImageCompression(quality_lower=60, p=0.4),
            
            # Redimensionar final
            A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.CenterCrop(height=640, width=640, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def create_base_images(self):
        """
        🎨 CREAR IMÁGENES BASE DESDE TUS UPLOADS
        
        Como no puedo acceder directamente a tus archivos subidos,
        creo versiones similares basadas en las descripciones
        
        ⚠️ IMPORTANTE: Reemplaza estas con tus imágenes reales
        """
        base_dir = self.project_root / 'source_images'
        
        print("🎨 Creando imágenes base...")
        print("⚠️ IMPORTANTE: Estas son plantillas temporales")
        print("   Reemplázalas con tus imágenes reales en:")
        print(f"   {base_dir}/")
        
        # STOP - Octágono rojo
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        # Crear octágono
        center = (200, 200)
        radius = 150
        angles = np.linspace(0, 2*np.pi, 9)[:-1] + np.pi/8
        points = []
        for angle in angles:
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        points = np.array(points, np.int32)
        cv2.fillPoly(img, [points], (0, 0, 200))  # Rojo
        cv2.putText(img, 'STOP', (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)
        cv2.imwrite(str(base_dir / 'stop.png'), img)
        
        # ROAD_WORK - Triángulo con trabajador
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        pts = np.array([[200, 50], [350, 350], [50, 350]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))  # Fondo blanco
        cv2.polylines(img, [pts], True, (0, 0, 200), 15)  # Borde rojo
        # Figura de trabajador simplificada
        cv2.circle(img, (200, 150), 20, (0, 0, 0), -1)  # Cabeza
        cv2.rectangle(img, (180, 170), (220, 250), (0, 0, 0), -1)  # Cuerpo
        cv2.line(img, (200, 250), (180, 300), (0, 0, 0), 8)  # Pierna izq
        cv2.line(img, (200, 250), (220, 300), (0, 0, 0), 8)  # Pierna der
        cv2.line(img, (180, 190), (160, 220), (0, 0, 0), 6)  # Brazo con pala
        cv2.line(img, (160, 220), (140, 240), (0, 0, 0), 4)  # Mango pala
        cv2.imwrite(str(base_dir / 'road_work.png'), img)
        
        # GIVE_WAY - Triángulo invertido
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        pts = np.array([[200, 350], [50, 50], [350, 50]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))  # Fondo blanco
        cv2.polylines(img, [pts], True, (0, 0, 200), 15)  # Borde rojo
        cv2.putText(img, 'GIVE', (140, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(img, 'WAY', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.imwrite(str(base_dir / 'give_way.png'), img)
        
        # Señales azules circulares con flechas
        for direction, angle_deg in [('turn_left', 180), ('go_straight', 90), ('turn_right', 0)]:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 240
            cv2.circle(img, (200, 200), 150, (200, 100, 0), -1)  # Azul
            cv2.circle(img, (200, 200), 150, (255, 255, 255), 8)  # Borde blanco
            
            # Dibujar flecha
            angle_rad = np.radians(angle_deg)
            if direction == 'go_straight':
                # Flecha hacia arriba
                cv2.arrowedLine(img, (200, 280), (200, 120), (255, 255, 255), 12, tipLength=0.3)
            elif direction == 'turn_left':
                # Flecha hacia izquierda
                cv2.arrowedLine(img, (280, 200), (120, 200), (255, 255, 255), 12, tipLength=0.3)
            else:  # turn_right
                # Flecha hacia derecha
                cv2.arrowedLine(img, (120, 200), (280, 200), (255, 255, 255), 12, tipLength=0.3)
            
            cv2.imwrite(str(base_dir / f'{direction}.png'), img)
        
        print("✅ Imágenes base creadas")
        print("\n📝 ACCIÓN REQUERIDA:")
        print("1. Ve a la carpeta: puzzlebot_yolo_training/source_images/")
        print("2. Reemplaza las imágenes con tus archivos reales:")
        print("   - stop.png/jpg")
        print("   - road_work.png/jpg") 
        print("   - give_way.png/jpg")
        print("   - turn_left.png/jpg")
        print("   - go_straight.png/jpg")
        print("   - turn_right.png/jpg")
        print("3. El script acepta PNG, JPG o JPEG")

    def load_source_images(self):
        """Cargar imágenes fuente (PNG o JPEG)"""
        source_dir = self.project_root / 'source_images'
        images_dict = {}
        
        print("📁 Cargando imágenes fuente...")
        
        for class_name in self.classes:
            # Buscar múltiples extensiones
            extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            img_found = False
            
            for ext in extensions:
                img_path = source_dir / f'{class_name}{ext}'
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        images_dict[class_name] = img
                        print(f"   ✅ {class_name}: {img_path.name} {img.shape}")
                        img_found = True
                        break
                    else:
                        print(f"   ❌ Error cargando: {img_path.name}")
            
            if not img_found:
                print(f"   ⚠️ No encontrado: {class_name}.[png|jpg|jpeg]")
        
        return images_dict

    def create_varied_backgrounds(self, num_backgrounds=200):
        """Crear fondos variados que simulen condiciones reales"""
        backgrounds = []
        
        print(f"🖼️ Creando {num_backgrounds} fondos variados...")
        
        for i in range(num_backgrounds):
            bg_type = random.choice(['road', 'urban', 'sky', 'concrete', 'nature'])
            
            if bg_type == 'road':
                # Simular asfalto/carretera
                base_color = [random.randint(40, 80), random.randint(40, 80), random.randint(40, 80)]
                bg = np.full((640, 640, 3), base_color, dtype=np.uint8)
                
                # Añadir líneas de carretera ocasionales
                if random.random() < 0.4:
                    for y in range(0, 640, random.randint(60, 120)):
                        cv2.line(bg, (0, y), (640, y), 
                               [c + random.randint(20, 40) for c in base_color], 
                               random.randint(2, 6))
                
            elif bg_type == 'urban':
                # Simular edificios/estructuras
                bg = np.full((640, 640, 3), 
                           [random.randint(100, 180), random.randint(100, 180), random.randint(80, 150)], 
                           dtype=np.uint8)
                
                # Añadir rectángulos como "ventanas" o "edificios"
                for _ in range(random.randint(3, 8)):
                    x1, y1 = random.randint(0, 500), random.randint(0, 500)
                    x2, y2 = x1 + random.randint(30, 140), y1 + random.randint(30, 140)
                    color = [random.randint(60, 200) for _ in range(3)]
                    cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)
                
            elif bg_type == 'sky':
                # Simular cielo
                bg = np.zeros((640, 640, 3), dtype=np.uint8)
                for y in range(640):
                    factor = y / 640.0
                    # Gradiente de azul
                    blue_intensity = int(180 + (255 - 180) * (1 - factor))
                    bg[y, :] = [blue_intensity * 0.7, blue_intensity * 0.8, blue_intensity]
                
            elif bg_type == 'concrete':
                # Simular concreto/pavimento
                base_gray = random.randint(120, 200)
                bg = np.full((640, 640, 3), [base_gray, base_gray, base_gray], dtype=np.uint8)
                
            else:  # nature
                # Simular vegetación
                green_base = random.randint(60, 120)
                bg = np.full((640, 640, 3), 
                           [random.randint(40, 80), green_base, random.randint(30, 70)], 
                           dtype=np.uint8)
            
            # Añadir ruido a todos los fondos
            noise = np.random.randint(-30, 30, (640, 640, 3), dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            backgrounds.append(bg)
        
        return backgrounds

    def place_sign_on_background(self, sign, background):
        """Colocar señal en fondo con variaciones realistas"""
        bg_h, bg_w = background.shape[:2]
        sign_h, sign_w = sign.shape[:2]
        
        # Tamaño aleatorio (más variado para simular distancias)
        min_size = 80
        max_size = min(300, bg_h//2, bg_w//2)
        new_size = random.randint(min_size, max_size)
        
        # Mantener proporción si la imagen no es cuadrada
        aspect_ratio = sign_w / sign_h
        if aspect_ratio > 1:
            new_w = new_size
            new_h = int(new_size / aspect_ratio)
        else:
            new_h = new_size
            new_w = int(new_size * aspect_ratio)
        
        sign_resized = cv2.resize(sign, (new_w, new_h))
        
        # Posición con margen para evitar bordes
        margin = 50
        max_x = max(0, bg_w - new_w - margin)
        max_y = max(0, bg_h - new_h - margin)
        x = random.randint(margin, max_x) if max_x > margin else 0
        y = random.randint(margin, max_y) if max_y > margin else 0
        
        # Colocación con blend ocasional para realismo
        if random.random() < 0.9:  # 90% reemplazo directo
            background[y:y+new_h, x:x+new_w] = sign_resized
        else:  # 10% blend suave
            alpha = random.uniform(0.8, 0.95)
            roi = background[y:y+new_h, x:x+new_w]
            blended = cv2.addWeighted(sign_resized, alpha, roi, 1-alpha, 0)
            background[y:y+new_h, x:x+new_w] = blended
        
        # Coordenadas YOLO (normalizadas)
        center_x = (x + new_w/2) / bg_w
        center_y = (y + new_h/2) / bg_h
        width = new_w / bg_w
        height = new_h / bg_h
        
        return background, (center_x, center_y, width, height)

    def generate_training_dataset(self, source_images, images_per_class=400):
        """Generar dataset completo para entrenamiento"""
        print(f"🏭 Generando dataset: {images_per_class} imágenes por clase")
        print(f"📊 Total objetivo: {images_per_class * len(source_images)} imágenes")
        
        backgrounds = self.create_varied_backgrounds(250)
        total_generated = 0
        
        for class_idx, (class_name, sign_image) in enumerate(source_images.items()):
            print(f"\n🎯 Clase {class_idx}: {class_name}")
            generated_count = 0
            
            for i in range(images_per_class):
                try:
                    # Elegir fondo aleatorio
                    background = random.choice(backgrounds).copy()
                    
                    # Colocar señal
                    img_with_sign, bbox = self.place_sign_on_background(sign_image, background)
                    
                    # Aplicar augmentaciones
                    augmented = self.augmentation(
                        image=img_with_sign,
                        bboxes=[bbox],
                        class_labels=[class_idx]
                    )
                    
                    if len(augmented['bboxes']) == 0:
                        continue  # La augmentación eliminó la bbox
                    
                    final_img = augmented['image']
                    final_bbox = augmented['bboxes'][0]
                    
                    # Determinar split (80/15/5)
                    rand_val = random.random()
                    if rand_val < 0.8:
                        split = 'train'
                    elif rand_val < 0.95:
                        split = 'val'
                    else:
                        split = 'test'
                    
                    # Guardar imagen y etiqueta
                    img_name = f"{class_name}_{i:04d}.jpg"
                    img_path = self.project_root / 'dataset' / 'images' / split / img_name
                    label_path = self.project_root / 'dataset' / 'labels' / split / f"{class_name}_{i:04d}.txt"
                    
                    cv2.imwrite(str(img_path), final_img)
                    
                    # Escribir etiqueta YOLO
                    with open(label_path, 'w') as f:
                        f.write(f"{class_idx} {final_bbox[0]:.6f} {final_bbox[1]:.6f} {final_bbox[2]:.6f} {final_bbox[3]:.6f}\n")
                    
                    generated_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"   📈 {i + 1}/{images_per_class}")
                        
                except Exception as e:
                    continue  # Saltar errores y seguir
            
            print(f"   ✅ Generadas: {generated_count} imágenes")
            total_generated += generated_count
        
        print(f"\n🎉 Dataset completo: {total_generated} imágenes")
        self.print_dataset_stats()

    def create_yaml_config(self):
        """Crear configuración YAML para YOLO"""
        config = {
            'path': str(self.project_root / 'dataset'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.project_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuración YAML: {yaml_path}")
        return yaml_path

    def train_yolo_model(self, epochs=30, batch_size=16):
        """Entrenar modelo YOLOv8"""
        print("🚀 INICIANDO ENTRENAMIENTO YOLOv8")
        print(f"⚙️ Epochs: {epochs}")
        print(f"⚙️ Batch size: {batch_size}")
        
        yaml_path = self.create_yaml_config()
        
        # Usar YOLOv8s para mejor balance velocidad/precisión
        model = YOLO('yolov8s.pt')
        print("📥 Modelo base: YOLOv8s (small)")
        
        # Configuración de entrenamiento optimizada
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.project_root / 'results'),
            name='puzzlebot_traffic_signs',
            save=True,
            plots=True,
            device='cuda',  # Cambiar a 'cuda' si tienes GPU
            patience=20,
            save_period=10,
            val=True,
            # Optimizaciones para señales de tráfico
            lr0=0.01,        # Learning rate inicial
            lrf=0.01,        # Learning rate final
            momentum=0.937,   # Momentum
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3,  # Epochs de warmup
            box=7.5,         # Box loss gain
            cls=0.5,         # Class loss gain
            dfl=1.5,         # DFL loss gain
            pose=12.0,       # Pose loss gain
            kobj=1.0,        # Keypoint obj loss gain
            label_smoothing=0.0,  # Label smoothing
            nbs=64,          # Nominal batch size
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            verbose=True
        )
        
        # Guardar modelo final
        model_save_path = self.project_root / 'models' / 'puzzlebot_traffic_signs.pt'
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            shutil.copy(best_model_path, model_save_path)
            print(f"🎯 Modelo guardado: {model_save_path}")
        
        return model_save_path, results

    def validate_model(self, model_path):
        """Validar modelo entrenado"""
        if not Path(model_path).exists():
            print(f"❌ Modelo no encontrado: {model_path}")
            return
        
        print("🧪 Validando modelo...")
        model = YOLO(str(model_path))
        
        # Validación completa
        metrics = model.val()
        
        print("\n📊 MÉTRICAS DE VALIDACIÓN:")
        print(f"   🎯 mAP50: {metrics.box.map50:.3f}")
        print(f"   🎯 mAP50-95: {metrics.box.map:.3f}")
        print(f"   🎯 Precisión: {metrics.box.mp:.3f}")
        print(f"   🎯 Recall: {metrics.box.mr:.3f}")
        
        return metrics

    def print_dataset_stats(self):
        """Mostrar estadísticas del dataset"""
        print("\n📊 ESTADÍSTICAS DEL DATASET:")
        
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val', 'test']:
            img_dir = self.project_root / 'dataset' / 'images' / split
            label_dir = self.project_root / 'dataset' / 'labels' / split
            
            img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
            
            print(f"   {split.upper()}: {img_count} imágenes, {label_count} etiquetas")
            total_images += img_count
            total_labels += label_count
        
        print(f"   TOTAL: {total_images} imágenes, {total_labels} etiquetas")

def main():
    """Pipeline completo de entrenamiento"""
    print("=" * 70)
    print("🎯 PUZZLEBOT YOLO TRAINING - MANCHESTER CHALLENGE")
    print("=" * 70)
    
    # Inicializar entrenador
    trainer = PuzzleBotYOLOTrainer()
    
    # Crear imágenes base (temporales)
    trainer.create_base_images()
    
    # Pausa para que el usuario reemplace las imágenes
    print("\n" + "="*50)
    print("⏸️ PAUSA REQUERIDA")
    print("="*50)
    print("1. Ve a: puzzlebot_yolo_training/source_images/")
    print("2. Reemplaza las 6 imágenes con tus archivos reales")
    print("3. Mantén los nombres exactos")
    
    input("\n⏸️ Presiona ENTER cuando hayas colocado tus imágenes...")
    
    # Cargar imágenes reales
    source_images = trainer.load_source_images()
    
    if len(source_images) < 6:
        print("❌ Faltan imágenes. Se necesitan las 6 señales.")
        return
    
    # Generar dataset
    print("\n🏭 Generando dataset de entrenamiento...")
    trainer.generate_training_dataset(source_images, images_per_class=400)
    
    # Entrenar modelo
    print("\n🚀 Iniciando entrenamiento...")
    model_path, results = trainer.train_yolo_model(epochs=30, batch_size=16)
    
    # Validar modelo
    print("\n🧪 Validando modelo final...")
    metrics = trainer.validate_model(model_path)
    
    # Resumen final
    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"🎯 Modelo entrenado: {model_path}")
    print(f"📊 Resultados en: {trainer.project_root}/results/")
    print("\n📝 PRÓXIMOS PASOS:")
    print("1. Copiar modelo a ROS2:")
    print(f"   cp {model_path} ~/ros2_ws/src/yolobot_recognition/models/")
    print("2. Actualizar launch file con el nuevo modelo")
    print("3. ¡Probar con el PuzzleBot!")

if __name__ == "__main__":
    main()