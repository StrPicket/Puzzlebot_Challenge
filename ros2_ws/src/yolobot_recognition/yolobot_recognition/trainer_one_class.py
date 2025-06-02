#!/usr/bin/env python3
"""
Single Class YOLO Trainer - Para entrenar solo turn_right
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

class SingleClassYOLOTrainer:
    def __init__(self, class_name='turn_right'):
        self.project_root = Path("single_class_training")
        self.class_name = class_name
        self.classes = [class_name]  # Solo una clase
        
        print(f"🎯 Entrenador para clase única: {class_name}")
        
        self.setup_directories()
        self.setup_augmentation()

    def setup_directories(self):
        """Crear estructura para entrenamiento de una sola clase"""
        dirs = [
            'dataset/images/train', 'dataset/images/val', 'dataset/images/test',
            'dataset/labels/train', 'dataset/labels/val', 'dataset/labels/test',
            'input_dataset',
            'models', 'results'
        ]
        
        for dir_path in dirs:
            (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Directorios creados en: {self.project_root}")

    def setup_augmentation(self):
        """Pipeline de augmentación para una sola clase"""
        self.augmentation = A.Compose([
            # Transformaciones geométricas moderadas
            A.HorizontalFlip(p=0.05),  # Muy bajo para señales direccionales
            A.Rotate(limit=20, p=0.8),
            A.RandomScale(scale_limit=0.3, p=0.9),
            A.Affine(
                scale=(0.8, 1.2), 
                translate_percent=(-0.1, 0.1), 
                rotate=(-15, 15), 
                p=0.7
            ),
            
            # Variaciones de iluminación
            A.RandomBrightnessContrast(
                brightness_limit=0.4, 
                contrast_limit=0.4, 
                p=0.9
            ),
            A.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.4, 
                hue=0.15, 
                p=0.8
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.6),
            A.CLAHE(clip_limit=2.0, p=0.5),
            
            # Efectos de degradación
            A.GaussNoise(var_limit=20.0, p=0.5),
            A.MotionBlur(blur_limit=5, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ImageCompression(quality_lower=60, p=0.4),
            
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
            min_visibility=0.4
        ))

    def copy_extracted_data(self, extracted_path):
        """Copiar datos extraídos de turn_right al proyecto de entrenamiento"""
        extracted_path = Path(extracted_path)
        
        print(f"\n📂 Copiando datos de {self.class_name} desde: {extracted_path}")
        
        # Crear directorios de destino
        input_images = self.project_root / 'input_dataset' / 'images'
        input_labels = self.project_root / 'input_dataset' / 'labels'
        input_images.mkdir(parents=True, exist_ok=True)
        input_labels.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        
        # Opción 1: Desde extracted_dataset/dataset/ (estructura organizada)
        source_images = extracted_path / 'dataset' / 'images'
        source_labels = extracted_path / 'dataset' / 'labels'
        
        if source_images.exists() and source_labels.exists():
            print("📋 Copiando desde dataset organizado...")
            
            # Buscar archivos que contengan 'turn_right'
            for img_file in source_images.glob(f'*{self.class_name}*.jpg'):
                label_name = img_file.stem + '.txt'
                label_file = source_labels / label_name
                
                if label_file.exists():
                    # Copiar imagen
                    dst_img = input_images / img_file.name
                    shutil.copy2(img_file, dst_img)
                    
                    # Copiar y convertir etiqueta (cambiar clase a 0)
                    dst_label = input_labels / label_name
                    self._copy_and_convert_label(label_file, dst_label)
                    
                    copied_count += 1
                    print(f"   ✅ {img_file.name}")
        
        # Opción 2: Desde extracted_frames/turn_right/ (por clase)
        elif (extracted_path / 'extracted_frames' / self.class_name).exists():
            source_frames = extracted_path / 'extracted_frames' / self.class_name
            print(f"📋 Copiando desde: {source_frames}")
            
            for img_file in source_frames.glob('*.jpg'):
                label_file = img_file.with_suffix('.txt')
                
                if label_file.exists():
                    # Copiar imagen
                    dst_img = input_images / img_file.name
                    shutil.copy2(img_file, dst_img)
                    
                    # Copiar y convertir etiqueta
                    dst_label = input_labels / label_file.name
                    self._copy_and_convert_label(label_file, dst_label)
                    
                    copied_count += 1
                    
                    if copied_count % 10 == 0:
                        print(f"   📈 Copiados: {copied_count}")
        
        else:
            print(f"❌ No se encontraron datos para {self.class_name}")
            print(f"   Buscado en: {source_images}")
            print(f"   Buscado en: {extracted_path}/extracted_frames/{self.class_name}")
            return False
        
        print(f"✅ Total copiado: {copied_count} pares imagen-etiqueta para {self.class_name}")
        return copied_count > 0

    def _copy_and_convert_label(self, src_label, dst_label):
        """Copiar etiqueta y convertir clase turn_right (5) a 0 (primera y única clase)"""
        try:
            with open(src_label, 'r') as f:
                lines = f.readlines()
            
            with open(dst_label, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        original_class = int(parts[0])
                        
                        # Solo procesar si es turn_right (clase 5) o si todas las clases son turn_right
                        if original_class == 5 or True:  # Acepta cualquier clase y la convierte a 0
                            # Convertir cualquier clase a 0 (primera y única clase)
                            f.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                        
        except Exception as e:
            print(f"   ⚠️ Error procesando etiqueta {src_label}: {e}")

    def generate_augmented_dataset(self, augmentations_per_image=8):
        """Generar dataset augmentado para turn_right"""
        input_images = self.project_root / 'input_dataset' / 'images'
        input_labels = self.project_root / 'input_dataset' / 'labels'
        
        # Cargar imágenes
        images_data = []
        
        print(f"\n📊 Cargando imágenes de {self.class_name}...")
        
        for img_file in input_images.glob('*.jpg'):
            label_file = input_labels / (img_file.stem + '.txt')
            
            if label_file.exists():
                img = cv2.imread(str(img_file))
                if img is not None:
                    # Cargar etiquetas
                    labels = self._load_labels(label_file)
                    if labels:
                        images_data.append({
                            'image': img,
                            'labels': labels,
                            'name': img_file.name
                        })
                        print(f"   ✅ {img_file.name}: {len(labels)} objetos")
        
        if not images_data:
            print("❌ No se encontraron imágenes válidas")
            return False
        
        print(f"\n🔄 Generando dataset para {self.class_name}:")
        print(f"📊 {len(images_data)} imágenes base × {augmentations_per_image + 1} (original + augmentaciones)")
        print(f"📊 Total esperado: {len(images_data) * (augmentations_per_image + 1)} imágenes")
        
        total_generated = 0
        
        for idx, img_data in enumerate(images_data):
            print(f"\n🖼️ Procesando {idx+1}/{len(images_data)}: {img_data['name']}")
            
            # 1. Guardar imagen original
            try:
                split = self._determine_split()
                
                # Redimensionar imagen original a 640x640 manteniendo aspecto
                resized_img = self._resize_image_with_padding(img_data['image'])
                
                # Ajustar bounding boxes para la nueva resolución
                adjusted_labels = self._adjust_labels_for_resize(
                    img_data['labels'], 
                    img_data['image'].shape, 
                    (640, 640)
                )
                
                if len(adjusted_labels) > 0:
                    orig_name = f"orig_{idx:04d}_{img_data['name']}"
                    self._save_image_and_labels(resized_img, adjusted_labels, orig_name, split)
                    total_generated += 1
                    print(f"   ✅ Original guardada: {orig_name}")
                    
            except Exception as e:
                print(f"   ❌ Error con imagen original: {e}")
                continue
            
            # 2. Generar augmentaciones
            for aug_idx in range(augmentations_per_image):
                try:
                    # Aplicar augmentaciones
                    bboxes = [[label[1], label[2], label[3], label[4]] for label in img_data['labels']]
                    class_labels = [label[0] for label in img_data['labels']]
                    
                    augmented = self.augmentation(
                        image=img_data['image'],
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    if len(augmented['bboxes']) > 0:
                        # Reconstruir etiquetas
                        final_labels = []
                        for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                            final_labels.append([class_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                        
                        # Guardar
                        split = self._determine_split()
                        aug_name = f"aug_{idx:04d}_{aug_idx:02d}_{img_data['name']}"
                        self._save_image_and_labels(
                            augmented['image'], 
                            final_labels, 
                            aug_name, 
                            split
                        )
                        total_generated += 1
                        
                except Exception as e:
                    continue
            
            progress = ((idx + 1) / len(images_data)) * 100
            print(f"   📈 Progreso: {progress:.1f}% - Total generadas: {total_generated}")
        
        print(f"\n🎉 Dataset completo para {self.class_name}: {total_generated} imágenes")
        self.print_dataset_stats()
        return True

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

    def _load_labels(self, label_file):
        """Cargar etiquetas YOLO"""
        labels = []
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([
                            int(parts[0]),  # class_id (siempre 0 para una clase)
                            float(parts[1]), float(parts[2]),  # center_x, center_y
                            float(parts[3]), float(parts[4])   # width, height
                        ])
        except Exception as e:
            print(f"   ❌ Error leyendo {label_file}: {e}")
        
        return labels

    def _determine_split(self):
        """Determinar split con distribución 70/20/10"""
        rand_val = random.random()
        if rand_val < 0.7:
            return 'train'
        elif rand_val < 0.9:
            return 'val'
        else:
            return 'test'

    def _save_image_and_labels(self, img, labels, img_name, split):
        """Guardar imagen y etiquetas en split correspondiente"""
        # Guardar imagen
        img_path = self.project_root / 'dataset' / 'images' / split / img_name
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Guardar etiquetas
        label_name = Path(img_name).stem + '.txt'
        label_path = self.project_root / 'dataset' / 'labels' / split / label_name
        
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    def create_yaml_config(self):
        """Crear configuración YAML para una sola clase"""
        config = {
            'path': str(self.project_root / 'dataset'),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Solo una clase
            'names': [self.class_name]
        }
        
        yaml_path = self.project_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuración YAML: {yaml_path}")
        return yaml_path

    def train_model(self, epochs=25, batch_size=16):
        """Entrenar modelo para una sola clase"""
        print(f"🚀 ENTRENANDO MODELO PARA: {self.class_name.upper()}")
        
        yaml_path = self.create_yaml_config()
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🔥 Usando dispositivo: {device}")
        except ImportError:
            device = 'cpu'
            print("⚠️ PyTorch no detectado, usando CPU")
        
        # Usar YOLOv8s
        model = YOLO('yolov8s.pt')
        print("📥 Modelo base: YOLOv8s")
        
        # Entrenamiento optimizado para una clase
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.project_root / 'results'),
            name=f'{self.class_name}_model',
            save=True,
            plots=True,
            device=device,
            patience=15,
            save_period=5,
            val=True,
            
            # Parámetros optimizados
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            verbose=True
        )
        
        # Guardar modelo
        model_save_path = self.project_root / 'models' / f'{self.class_name}_trained.pt'
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            shutil.copy(best_model_path, model_save_path)
            print(f"🎯 Modelo guardado: {model_save_path}")
        
        return model_save_path, results

    def print_dataset_stats(self):
        """Mostrar estadísticas del dataset"""
        print(f"\n📊 ESTADÍSTICAS - {self.class_name.upper()}:")
        
        total_images = 0
        for split in ['train', 'val', 'test']:
            img_dir = self.project_root / 'dataset' / 'images' / split
            img_count = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            print(f"   {split.upper()}: {img_count} imágenes")
            total_images += img_count
        
        print(f"   TOTAL: {total_images} imágenes")

def main():
    """Entrenamiento automático para turn_right"""
    print("=" * 60)
    print("🎯 SINGLE CLASS YOLO TRAINER - TURN_RIGHT")
    print("=" * 60)
    
    # 🎯 RUTA AUTOMÁTICA CONFIGURADA
    extracted_path = "/home/strpicket/ManchesterChallenge/ros2_ws/src/yolobot_recognition/yolobot_recognition/extracted_dataset"
    
    print(f"📁 Usando dataset automáticamente desde:")
    print(f"   {extracted_path}")
    
    # Verificar que la ruta existe
    if not Path(extracted_path).exists():
        print(f"❌ No se encontró el dataset en: {extracted_path}")
        print("🔧 Verifica la ruta o modifica el script")
        return
    
    # Verificar que existe turn_right
    turn_right_path = Path(extracted_path) / 'extracted_frames' / 'turn_right'
    if not turn_right_path.exists():
        print(f"❌ No se encontró carpeta turn_right en: {turn_right_path}")
        print("📂 Estructura esperada:")
        print(f"   {extracted_path}/extracted_frames/turn_right/")
        return
    
    # Contar archivos disponibles
    jpg_files = list(turn_right_path.glob('*.jpg'))
    txt_files = list(turn_right_path.glob('*.txt'))
    
    print(f"📊 Datos encontrados para turn_right:")
    print(f"   📷 Imágenes: {len(jpg_files)}")
    print(f"   📄 Etiquetas: {len(txt_files)}")
    
    if len(jpg_files) == 0:
        print("❌ No se encontraron imágenes .jpg")
        return
    
    if len(txt_files) == 0:
        print("⚠️ No se encontraron etiquetas .txt")
        print("   El script intentará crear etiquetas automáticas")
    
    # Configurar clase a entrenar
    class_name = 'turn_right'
    trainer = SingleClassYOLOTrainer(class_name)
    
    print(f"\n✅ Inicializando entrenador para: {class_name}")
    
    # Copiar datos automáticamente
    print(f"\n🔄 Copiando datos automáticamente...")
    if not trainer.copy_extracted_data(extracted_path):
        print("❌ No se pudieron copiar los datos")
        return
    
    # Configuración automática con valores optimizados
    augmentations = 10  # Más augmentaciones para mejor dataset
    epochs = 30         # Más epochs para mejor convergencia
    batch_size = 16     # Tamaño estándar
    
    print(f"\n⚙️ Configuración automática:")
    print(f"   🔄 Augmentaciones por imagen: {augmentations}")
    print(f"   🏋️ Epochs: {epochs}")
    print(f"   📦 Batch size: {batch_size}")
    
    # Confirmar antes de proceder
    confirm = input(f"\n🚀 ¿Proceder con el entrenamiento automático? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Entrenamiento cancelado")
        return
    
    # Generar dataset augmentado
    print(f"\n🔄 Generando dataset augmentado...")
    if not trainer.generate_augmented_dataset(augmentations):
        print("❌ No se pudo generar el dataset")
        return
    
    # Entrenar modelo automáticamente
    print(f"\n🚀 Iniciando entrenamiento automático...")
    try:
        model_path, results = trainer.train_model(epochs, batch_size)
        
        # Resumen exitoso
        print("\n" + "="*60)
        print(f"🎉 ENTRENAMIENTO COMPLETADO - {class_name.upper()}")
        print("="*60)
        print(f"🎯 Modelo entrenado: {model_path}")
        print(f"📊 Resultados en: {trainer.project_root}/results/")
        
        # Instrucciones para ROS2
        print(f"\n📝 PARA USAR EN ROS2:")
        print(f"1. Copiar modelo:")
        print(f"   cp {model_path} ~/ros2_ws/src/yolobot_recognition/models/")
        print(f"2. Configurar en launch file:")
        print(f"   model_name: {class_name}_trained.pt")
        print(f"   confidence_threshold: 0.3")
        print(f"3. ¡Probar con PuzzleBot!")
        
        # Mostrar comando directo
        ros_model_path = f"~/ros2_ws/src/yolobot_recognition/models/{class_name}_trained.pt"
        print(f"\n🔧 Comando directo para copiar:")
        print(f"cp {model_path} {ros_model_path}")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        print("🔧 Verifica que tengas PyTorch y ultralytics instalados:")
        print("   pip install torch ultralytics albumentations")
        
if __name__ == "__main__":
    main()