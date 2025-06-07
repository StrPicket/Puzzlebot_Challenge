#!/usr/bin/env python3
"""
YOLOv8 Training MODIFICADO - Para usar DATASET REAL
Procesa imágenes reales que el usuario proporciona
"""

import os
import cv2
import numpy as np
import yaml
import shutil
import random
import json
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from sklearn.model_selection import train_test_split

class RealDatasetYOLOTrainer:
    def __init__(self):
        self.project_root = Path("puzzlebot_real_training")
        
        # 🎯 LAS 6 CLASES EXACTAS
        self.classes = [
            'stop',         # 0
            'road_work',    # 1
            'give_way',     # 2
            'turn_left',    # 3
            'go_straight',  # 4
            'turn_right'    # 5
        ]
        
        print("🎯 PuzzleBot YOLO Trainer - DATASET REAL")
        print(f"📊 Clases a entrenar: {len(self.classes)}")
        for i, clase in enumerate(self.classes):
            print(f"   {i}: {clase}")
        
        self.setup_directories()
        self.setup_augmentation()

    def setup_directories(self):
        """Crear estructura para dataset real"""
        dirs = [
            'dataset/images/train', 'dataset/images/val', 'dataset/images/test',
            'dataset/labels/train', 'dataset/labels/val', 'dataset/labels/test',
            'input_dataset',  # Aquí van tus imágenes reales
            'models', 'results', 'validation_samples'
        ]
        
        for dir_path in dirs:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Directorios creados en: {self.project_root}")
        print(f"📁 Pon tus imágenes reales en: {self.project_root}/input_dataset/")

    def setup_augmentation(self):
        """Pipeline de augmentación para imágenes reales"""
        self.augmentation = A.Compose([
            # Transformaciones geométricas moderadas (no queremos distorsionar mucho las reales)
            A.HorizontalFlip(p=0.1),  # Muy bajo para señales direccionales
            A.Rotate(limit=15, p=0.7),  # Rotación moderada
            A.RandomScale(scale_limit=0.2, p=0.8),
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(-0.05, 0.05), 
                rotate=(-10, 10), 
                p=0.6
            ),
            
            # Variaciones de iluminación (simular diferentes condiciones)
            A.RandomBrightnessContrast(
                brightness_limit=0.3, 
                contrast_limit=0.3, 
                p=0.8
            ),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.3, 
                hue=0.1, 
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.4),
            
            # Efectos de degradación leves
            A.GaussNoise(var_limit=15.0, p=0.4),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ImageCompression(quality_lower=70, p=0.3),
            
            # Redimensionar a 640x640
            A.LongestMaxSize(max_size=640, p=1.0),
            A.PadIfNeeded(
                min_height=640, 
                min_width=640, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=[114, 114, 114],
                p=1.0
            ),
            
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_area=100,
            min_visibility=0.5
        ))

    def load_real_dataset(self):
        """
        Cargar dataset real de imágenes con estructura:
        
        input_dataset/
        ├── images/           # Todas las imágenes
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        └── labels/           # Etiquetas YOLO correspondientes
            ├── img001.txt
            ├── img002.txt
            └── ...
        
        O estructura alternativa por clases:
        input_dataset/
        ├── stop/
        │   ├── stop_001.jpg
        │   ├── stop_001.txt
        │   └── ...
        ├── road_work/
        │   ├── work_001.jpg
        │   ├── work_001.txt
        │   └── ...
        └── ...
        """
        input_dir = self.project_root / 'input_dataset'
        
        print("📁 Cargando dataset real...")
        print(f"📂 Buscando en: {input_dir}")
        
        # Verificar estructura del dataset
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'
        
        dataset_images = []
        dataset_labels = []
        
        if images_dir.exists() and labels_dir.exists():
            # Estructura: input_dataset/images/ y input_dataset/labels/
            print("📋 Detectada estructura: images/ y labels/")
            dataset_images, dataset_labels = self._load_from_images_labels_structure(images_dir, labels_dir)
            
        else:
            # Estructura por clases
            print("📋 Detectada estructura por clases")
            dataset_images, dataset_labels = self._load_from_class_structure(input_dir)
        
        if len(dataset_images) == 0:
            print("❌ No se encontraron imágenes válidas")
            print("📝 Estructura esperada:")
            print("   Opción 1: input_dataset/images/ + input_dataset/labels/")
            print("   Opción 2: input_dataset/clase1/, input_dataset/clase2/, etc.")
            return [], []
        
        print(f"✅ Cargadas {len(dataset_images)} imágenes con etiquetas")
        return dataset_images, dataset_labels

    def _load_from_images_labels_structure(self, images_dir, labels_dir):
        """Cargar desde estructura images/ y labels/"""
        dataset_images = []
        dataset_labels = []
        
        # Buscar todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_path in images_dir.iterdir():
            if img_path.suffix in image_extensions:
                # Buscar archivo de etiqueta correspondiente
                label_name = img_path.stem + '.txt'
                label_path = labels_dir / label_name
                
                if label_path.exists():
                    # Cargar imagen
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Cargar etiquetas
                        labels = self._load_yolo_labels(label_path)
                        if len(labels) > 0:
                            dataset_images.append({
                                'image': img,
                                'path': str(img_path),
                                'name': img_path.name
                            })
                            dataset_labels.append(labels)
                            print(f"   ✅ {img_path.name}: {len(labels)} objetos")
                        else:
                            print(f"   ⚠️ {img_path.name}: Sin etiquetas válidas")
                    else:
                        print(f"   ❌ Error cargando: {img_path.name}")
                else:
                    print(f"   ⚠️ {img_path.name}: Sin archivo de etiqueta")
        
        return dataset_images, dataset_labels

    def _load_from_class_structure(self, input_dir):
        """Cargar desde estructura por clases"""
        dataset_images = []
        dataset_labels = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = input_dir / class_name
            if not class_dir.exists():
                print(f"   ⚠️ Directorio no encontrado: {class_name}/")
                continue
            
            print(f"📂 Procesando clase: {class_name}")
            
            for img_path in class_dir.iterdir():
                if img_path.suffix in image_extensions:
                    # Buscar archivo de etiqueta correspondiente
                    label_name = img_path.stem + '.txt'
                    label_path = class_dir / label_name
                    
                    if label_path.exists():
                        # Cargar imagen
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            # Cargar etiquetas
                            labels = self._load_yolo_labels(label_path)
                            if len(labels) > 0:
                                dataset_images.append({
                                    'image': img,
                                    'path': str(img_path),
                                    'name': img_path.name,
                                    'class': class_name
                                })
                                dataset_labels.append(labels)
                                print(f"   ✅ {img_path.name}: {len(labels)} objetos")
                            else:
                                print(f"   ⚠️ {img_path.name}: Sin etiquetas válidas")
                        else:
                            print(f"   ❌ Error cargando: {img_path.name}")
                    else:
                        # Si no hay etiqueta, asumir que toda la imagen es de esta clase
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            # Crear bbox que cubra toda la imagen (con margen)
                            full_bbox = [class_idx, 0.5, 0.5, 0.8, 0.8]  # centro en 0.5,0.5 con 80% de cobertura
                            dataset_images.append({
                                'image': img,
                                'path': str(img_path),
                                'name': img_path.name,
                                'class': class_name
                            })
                            dataset_labels.append([full_bbox])
                            print(f"   ✅ {img_path.name}: Etiqueta automática (imagen completa)")
        
        return dataset_images, dataset_labels

    def _load_yolo_labels(self, label_path):
        """Cargar etiquetas en formato YOLO"""
        labels = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Validar que la etiqueta esté en rango válido
                            if (0 <= class_id < len(self.classes) and
                                0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                0 < width <= 1 and 0 < height <= 1):
                                labels.append([class_id, x_center, y_center, width, height])
                            else:
                                print(f"      ⚠️ Etiqueta inválida: {line}")
        except Exception as e:
            print(f"      ❌ Error leyendo etiquetas: {e}")
        
        return labels

    def generate_augmented_dataset(self, dataset_images, dataset_labels, augmentations_per_image=5):
        """Generar dataset augmentado desde imágenes reales"""
        print(f"🔄 Generando dataset augmentado...")
        print(f"📊 {len(dataset_images)} imágenes base × {augmentations_per_image} augmentaciones")
        print(f"📊 Total esperado: {len(dataset_images) * (augmentations_per_image + 1)} imágenes")
        
        total_generated = 0
        
        for idx, (img_data, labels) in enumerate(zip(dataset_images, dataset_labels)):
            original_img = img_data['image']
            img_name = img_data['name']
            
            print(f"\n🖼️ Procesando {idx+1}/{len(dataset_images)}: {img_name}")
            
            # 1. Guardar imagen original (sin augmentación)
            try:
                split = self._determine_split()
                
                # Redimensionar imagen original a 640x640 manteniendo aspecto
                resized_img = self._resize_image_with_padding(original_img)
                
                # Ajustar bounding boxes para la nueva resolución
                adjusted_labels = self._adjust_labels_for_resize(labels, original_img.shape, (640, 640))
                
                if len(adjusted_labels) > 0:
                    # Guardar imagen original
                    orig_img_name = f"orig_{idx:04d}_{img_name}"
                    self._save_image_and_labels(resized_img, adjusted_labels, orig_img_name, split)
                    total_generated += 1
                    
            except Exception as e:
                print(f"   ❌ Error con imagen original: {e}")
                continue
            
            # 2. Generar augmentaciones
            for aug_idx in range(augmentations_per_image):
                try:
                    # Aplicar augmentaciones
                    augmented = self.augmentation(
                        image=original_img,
                        bboxes=[[label[1], label[2], label[3], label[4]] for label in labels],
                        class_labels=[label[0] for label in labels]
                    )
                    
                    if len(augmented['bboxes']) == 0:
                        continue  # La augmentación eliminó todas las bboxes
                    
                    final_img = augmented['image']
                    final_bboxes = augmented['bboxes']
                    final_classes = augmented['class_labels']
                    
                    # Reconstruir etiquetas
                    final_labels = []
                    for bbox, class_id in zip(final_bboxes, final_classes):
                        final_labels.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                    
                    # Determinar split
                    split = self._determine_split()
                    
                    # Guardar imagen augmentada
                    aug_img_name = f"aug_{idx:04d}_{aug_idx:02d}_{img_name}"
                    self._save_image_and_labels(final_img, final_labels, aug_img_name, split)
                    total_generated += 1
                    
                except Exception as e:
                    continue  # Saltar errores en augmentación
            
            if (idx + 1) % 10 == 0:
                print(f"   📈 Progreso: {idx + 1}/{len(dataset_images)} ({total_generated} generadas)")
        
        print(f"\n🎉 Dataset generado: {total_generated} imágenes totales")
        self.print_dataset_stats()

    def _resize_image_with_padding(self, img):
        """Redimensionar imagen manteniendo aspecto y añadiendo padding"""
        h, w = img.shape[:2]
        
        # Calcular escala para que quepa en 640x640
        scale = min(640/w, 640/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(img, (new_w, new_h))
        
        # Crear imagen con padding
        padded = np.full((640, 640, 3), [114, 114, 114], dtype=np.uint8)
        
        # Centrar imagen redimensionada
        start_x = (640 - new_w) // 2
        start_y = (640 - new_h) // 2
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return padded

    def _adjust_labels_for_resize(self, labels, original_shape, target_shape):
        """Ajustar etiquetas para imagen redimensionada con padding"""
        orig_h, orig_w = original_shape[:2]
        target_h, target_w = target_shape
        
        # Calcular escala y offsets
        scale = min(target_w/orig_w, target_h/orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        adjusted_labels = []
        
        for label in labels:
            class_id, x_center, y_center, width, height = label
            
            # Convertir a coordenadas absolutas originales
            abs_x = x_center * orig_w
            abs_y = y_center * orig_h
            abs_w = width * orig_w
            abs_h = height * orig_h
            
            # Aplicar escala y offset
            new_abs_x = abs_x * scale + offset_x
            new_abs_y = abs_y * scale + offset_y
            new_abs_w = abs_w * scale
            new_abs_h = abs_h * scale
            
            # Convertir de vuelta a coordenadas normalizadas
            new_x_center = new_abs_x / target_w
            new_y_center = new_abs_y / target_h
            new_width = new_abs_w / target_w
            new_height = new_abs_h / target_h
            
            # Validar que esté dentro de los límites
            if (0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and
                new_width > 0 and new_height > 0):
                adjusted_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])
        
        return adjusted_labels

    def _determine_split(self):
        """Determinar split aleatoriamente"""
        rand_val = random.random()
        if rand_val < 0.7:
            return 'train'
        elif rand_val < 0.9:
            return 'val'
        else:
            return 'test'

    def _save_image_and_labels(self, img, labels, img_name, split):
        """Guardar imagen y sus etiquetas"""
        # Guardar imagen
        img_path = self.project_root / 'dataset' / 'images' / split / img_name
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Guardar etiquetas
        label_name = Path(img_name).stem + '.txt'
        label_path = self.project_root / 'dataset' / 'labels' / split / label_name
        
        with open(label_path, 'w') as f:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

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

    def train_yolo_model(self, epochs=30, batch_size=16):
        """Entrenar modelo YOLOv8"""
        print("🚀 INICIANDO ENTRENAMIENTO YOLOv8")
        print(f"⚙️ Epochs: {epochs}")
        print(f"⚙️ Batch size: {batch_size}")
        
        yaml_path = self.create_yaml_config()
        
        # Usar YOLOv8s para buen balance
        model = YOLO('yolov8s.pt')
        print("📥 Modelo base: YOLOv8s")
        
        # Entrenamiento
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.project_root / 'results'),
            name='puzzlebot_real_dataset',
            save=True,
            plots=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            patience=20,
            save_period=10,
            val=True,
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            verbose=True
        )
        
        # Guardar modelo
        model_save_path = self.project_root / 'models' / 'puzzlebot_real_trained.pt'
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            shutil.copy(best_model_path, model_save_path)
            print(f"🎯 Modelo guardado: {model_save_path}")
        
        return model_save_path, results

def main():
    """Pipeline para dataset real"""
    print("=" * 70)
    print("🎯 PUZZLEBOT YOLO TRAINING - DATASET REAL")
    print("=" * 70)
    
    # Verificar torch
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🔥 CUDA disponible: {torch.cuda.is_available()}")
    except ImportError:
        print("⚠️ PyTorch no encontrado")
        return
    
    # Inicializar entrenador
    trainer = RealDatasetYOLOTrainer()
    
    # Instrucciones para el usuario
    print("\n📋 INSTRUCCIONES:")
    print("Coloca tus imágenes en una de estas estructuras:")
    print("\n🔹 Opción 1: Estructura images/labels")
    print("   puzzlebot_real_training/input_dataset/")
    print("   ├── images/")
    print("   │   ├── img001.jpg")
    print("   │   ├── img002.jpg")
    print("   │   └── ...")
    print("   └── labels/")
    print("       ├── img001.txt  # Formato YOLO")
    print("       ├── img002.txt")
    print("       └── ...")
    
    print("\n🔹 Opción 2: Estructura por clases")
    print("   puzzlebot_real_training/input_dataset/")
    print("   ├── stop/")
    print("   │   ├── stop_01.jpg")
    print("   │   ├── stop_01.txt")
    print("   │   └── ...")
    print("   ├── road_work/")
    print("   │   ├── work_01.jpg")
    print("   │   ├── work_01.txt")
    print("   │   └── ...")
    print("   └── ...")
    
    input("\n⏸️ Presiona ENTER cuando hayas colocado tus imágenes...")
    
    # Cargar dataset real
    dataset_images, dataset_labels = trainer.load_real_dataset()
    
    if len(dataset_images) == 0:
        print("❌ No se cargaron imágenes. Verifica la estructura.")
        return
    
    # Configurar augmentaciones
    augmentations = int(input(f"\n🔄 Augmentaciones por imagen (recomendado 5): ") or "5")
    
    # Generar dataset augmentado
    trainer.generate_augmented_dataset(dataset_images, dataset_labels, augmentations)
    
    # Configurar entrenamiento
    epochs = int(input("\n⚙️ Número de epochs (recomendado 30): ") or "30")
    batch_size = int(input("⚙️ Batch size (recomendado 16): ") or "16")
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    model_path, results = trainer.train_yolo_model(epochs, batch_size)
    
    # Resumen
    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"🎯 Modelo: {model_path}")
    print(f"📊 Resultados: {trainer.project_root}/results/")
    print("\n📝 Para usar en ROS2:")
    print(f"   cp {model_path} ~/ros2_ws/src/yolobot_recognition/models/")

if __name__ == "__main__":
    main()