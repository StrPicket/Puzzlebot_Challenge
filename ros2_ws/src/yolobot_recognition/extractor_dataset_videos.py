#!/usr/bin/env python3
"""
Video to Dataset Extractor - PuzzleBot
Extrae frames de videos del PuzzleBot para crear dataset de entrenamiento
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk

class VideoDatasetExtractor:
    def __init__(self):
        self.output_dir = Path("extracted_dataset")
        self.classes = [
            'stop',         # 0
            'road_work',    # 1  
            'give_way',     # 2
            'turn_left',    # 3
            'go_straight',  # 4
            'turn_right'    # 5
        ]
        
        # Configuración optimizada para videos cortos con cámara de amplio campo de visión
        self.frame_skip = 5   # Extraer cada 5 frames (más denso para videos cortos)
        self.min_frame_diff = 20  # Diferencia mínima entre frames (más sensible)
        self.blur_threshold = 100   # Umbral estándar (imagen ya viene corregida)
        self.brightness_range = (30, 220)  # Rango estándar para buena iluminación
        self.fisheye_correction = False  # DESACTIVADO - imagen ya viene corregida
        
        self.setup_directories()
        print("🎥 Video Dataset Extractor inicializado")

    def setup_directories(self):
        """Crear estructura de directorios"""
        dirs = [
            'videos',           # Videos originales
            'extracted_frames', # Frames extraídos temporalmente
            'dataset/images',   # Imágenes finales
            'dataset/labels',   # Etiquetas YOLO
            'previews',         # Previews para verificación
        ]
        
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Directorios creados en: {self.output_dir}")

    def extract_frames_from_video(self, video_path, class_name, output_subdir=None):
        """Extraer frames de un video específico"""
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Video no encontrado: {video_path}")
            return []
        
        print(f"\n🎥 Procesando video: {video_path.name}")
        print(f"🎯 Clase: {class_name}")
        
        # Crear subdirectorio si se especifica
        if output_subdir:
            frames_dir = self.output_dir / 'extracted_frames' / output_subdir
        else:
            frames_dir = self.output_dir / 'extracted_frames' / class_name
        
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error abriendo video: {video_path}")
            return []
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        last_saved_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Saltar frames según configuración
            if frame_count % self.frame_skip != 0:
                continue
            
            # Evaluar calidad del frame
            quality_score = self._evaluate_frame_quality(frame)
            
            if quality_score > 0.5:  # Umbral más bajo para videos cortos
                # Verificar diferencia con frame anterior
                if self._is_frame_different(frame, last_saved_frame):
                    
                    # No aplicar corrección fish-eye (imagen ya viene corregida)
                    final_frame = frame
                    
                    # Guardar frame
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    frame_name = f"{class_name}_{saved_count:04d}_t{timestamp:.2f}.jpg"
                    frame_path = frames_dir / frame_name
                    
                    cv2.imwrite(str(frame_path), final_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    extracted_frames.append({
                        'path': str(frame_path),
                        'frame_number': frame_count,
                        'timestamp': timestamp,
                        'quality_score': quality_score,
                        'class': class_name,
                        'corrected': False  # No se aplicó corrección
                    })
                    
                    last_saved_frame = frame.copy()
                    saved_count += 1
                    
                    if saved_count % 10 == 0:  # Reportar cada 10 frames
                        print(f"   📸 Extraídos: {saved_count} frames")
                        
            # Para videos cortos, mostrar progreso más frecuentemente
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   🎬 Progreso: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        print(f"✅ Completado: {saved_count} frames extraídos de {total_frames}")
        
        return extracted_frames

    def apply_fisheye_correction(self, frame):
        """Aplicar corrección básica para lente fish-eye"""
        if not self.fisheye_correction:
            return frame
        
        try:
            h, w = frame.shape[:2]
            
            # Parámetros básicos de corrección fish-eye
            # Estos valores pueden necesitar ajuste según tu lente específica
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K=np.array([[w*0.8, 0, w/2], [0, h*0.8, h/2], [0, 0, 1]], dtype=np.float32),
                D=np.array([0.1, 0.05, 0.01, 0.0], dtype=np.float32),  # Coeficientes de distorsión
                R=np.eye(3),
                P=np.array([[w*0.7, 0, w/2], [0, h*0.7, h/2], [0, 0, 1]], dtype=np.float32),
                size=(w, h),
                m1type=cv2.CV_32FC1
            )
            
            corrected = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            return corrected
            
        except Exception as e:
            print(f"   ⚠️ Error en corrección fish-eye: {e}")
            return frame

    def detect_traffic_signs_region(self, frame):
        """Detectar regiones donde probablemente estén las señales de tráfico"""
        h, w = frame.shape[:2]
        
        # Para cámara con amplio campo de visión, las señales pueden estar en varias posiciones
        # Región más amplia que cubre la parte central-superior donde típicamente aparecen
        roi_top = int(h * 0.05)      # 5% desde arriba
        roi_bottom = int(h * 0.75)   # hasta 75% de la altura
        roi_left = int(w * 0.15)     # 15% desde la izquierda  
        roi_right = int(w * 0.85)    # hasta 85% del ancho
        
        # Crear máscara para la ROI
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[roi_top:roi_bottom, roi_left:roi_right] = 255
        
        return mask, (roi_left, roi_top, roi_right, roi_bottom)
    def _evaluate_frame_quality(self, frame):
        """Evaluar calidad del frame optimizado para fish-eye y videos cortos"""
        if frame is None:
            return 0.0
        
        # Aplicar corrección fish-eye si está habilitada
        corrected_frame = self.apply_fisheye_correction(frame)
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(corrected_frame, cv2.COLOR_BGR2GRAY)
        
        # Obtener región de interés para señales
        roi_mask, roi_coords = self.detect_traffic_signs_region(corrected_frame)
        roi_left, roi_top, roi_right, roi_bottom = roi_coords
        
        # Evaluar solo en la región de interés
        roi_gray = gray[roi_top:roi_bottom, roi_left:roi_right]
        
        # 1. Detectar blur usando varianza del Laplaciano (ajustado para fish-eye)
        laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / self.blur_threshold, 1.0)
        
        # 2. Evaluar brillo en ROI
        mean_brightness = np.mean(roi_gray)
        if self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]:
            brightness_score = 1.0
        else:
            if mean_brightness < self.brightness_range[0]:
                brightness_score = mean_brightness / self.brightness_range[0]
            else:
                brightness_score = (255 - mean_brightness) / (255 - self.brightness_range[1])
            brightness_score = max(0, brightness_score)
        
        # 3. Evaluar contraste en ROI
        contrast_score = roi_gray.std() / 128.0
        contrast_score = min(contrast_score, 1.0)
        
        # 4. Detectar formas circulares/rectangulares (típicas de señales)
        edges = cv2.Canny(roi_gray, 30, 100)  # Umbrales más bajos para fish-eye
        
        # Detectar círculos (señales circulares)
        circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                 param1=50, param2=30, minRadius=10, maxRadius=100)
        
        # Detectar contornos (señales rectangulares/triangulares)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Evaluar presencia de formas geométricas
        shape_score = 0.0
        if circles is not None:
            shape_score += min(len(circles[0]) * 0.3, 1.0)
        
        # Filtrar contornos por área y forma
        geometric_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Área razonable para señales
                # Aproximar contorno
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Si tiene 3-8 vértices, podría ser una señal
                if 3 <= len(approx) <= 8:
                    geometric_shapes += 1
        
        shape_score += min(geometric_shapes * 0.2, 1.0)
        shape_score = min(shape_score, 1.0)
        
        # 5. Evaluar densidad de bordes en ROI
        edge_density = np.sum(edges > 0) / edges.size
        content_score = min(edge_density * 15, 1.0)  # Factor ajustado para fish-eye
        
        # Combinar scores con pesos optimizados para detección de señales
        final_score = (
            blur_score * 0.35 +        # Menos peso al blur (fish-eye puede ser menos nítida)
            brightness_score * 0.25 +   # Brillo sigue siendo importante
            contrast_score * 0.15 +     # Contraste moderado
            content_score * 0.1 +       # Contenido general
            shape_score * 0.15          # NUEVO: Presencia de formas geométricas
        )
        
        return final_score

    def _is_frame_different(self, frame1, frame2):
        """Verificar si dos frames son suficientemente diferentes"""
        if frame2 is None:
            return True
        
        # Redimensionar para comparación rápida
        small1 = cv2.resize(frame1, (64, 64))
        small2 = cv2.resize(frame2, (64, 64))
        
        # Calcular diferencia
        diff = cv2.absdiff(small1, small2)
        mean_diff = np.mean(diff)
        
        return mean_diff > self.min_frame_diff

    def manual_frame_selection(self, video_path, class_name):
        """Selección manual de frames usando interfaz gráfica"""
        print(f"\n🖱️ Selección manual para: {class_name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error abriendo video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Crear ventana de selección
        root = tk.Tk()
        root.title(f"Selección Manual - {class_name}")
        root.geometry("800x700")
        
        selected_frames = []
        current_frame_idx = [0]
        
        # Variables de la interfaz
        frame_label = tk.Label(root)
        frame_label.pack(pady=10)
        
        info_label = tk.Label(root, text="", font=("Arial", 10))
        info_label.pack()
        
        def update_frame_display():
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx[0])
            ret, frame = cap.read()
            
            if ret:
                # Redimensionar para mostrar
                display_frame = cv2.resize(frame, (640, 480))
                display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a formato Tkinter
                pil_image = Image.fromarray(display_frame_rgb)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                frame_label.configure(image=tk_image)
                frame_label.image = tk_image
                
                # Actualizar info
                timestamp = current_frame_idx[0] / fps if fps > 0 else current_frame_idx[0]
                quality = self._evaluate_frame_quality(frame)
                info_text = f"Frame: {current_frame_idx[0]}/{total_frames} | Tiempo: {timestamp:.2f}s | Calidad: {quality:.2f}"
                info_label.configure(text=info_text)
                
                return frame
            return None
        
        def save_current_frame():
            frame = update_frame_display()
            if frame is not None:
                timestamp = current_frame_idx[0] / fps if fps > 0 else current_frame_idx[0]
                frame_name = f"{class_name}_manual_{len(selected_frames):04d}_t{timestamp:.2f}.jpg"
                
                frames_dir = self.output_dir / 'extracted_frames' / class_name
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frames_dir / frame_name
                
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                selected_frames.append({
                    'path': str(frame_path),
                    'frame_number': current_frame_idx[0],
                    'timestamp': timestamp,
                    'quality_score': self._evaluate_frame_quality(frame),
                    'class': class_name
                })
                
                print(f"💾 Guardado: {frame_name}")
                messagebox.showinfo("Guardado", f"Frame guardado: {frame_name}")
        
        def next_frame():
            if current_frame_idx[0] < total_frames - 1:
                current_frame_idx[0] += 1
                update_frame_display()
        
        def prev_frame():
            if current_frame_idx[0] > 0:
                current_frame_idx[0] -= 1
                update_frame_display()
        
        def jump_frame():
            new_frame = simpledialog.askinteger("Saltar", f"Frame (0-{total_frames-1}):")
            if new_frame is not None and 0 <= new_frame < total_frames:
                current_frame_idx[0] = new_frame
                update_frame_display()
        
        def skip_frames(n):
            current_frame_idx[0] = min(current_frame_idx[0] + n, total_frames - 1)
            update_frame_display()
        
        # Botones de control
        controls_frame = tk.Frame(root)
        controls_frame.pack(pady=10)
        
        tk.Button(controls_frame, text="◀◀ -50", command=lambda: skip_frames(-50)).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="◀ -10", command=lambda: skip_frames(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="◀ Anterior", command=prev_frame).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="💾 GUARDAR", command=save_current_frame, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(controls_frame, text="Siguiente ▶", command=next_frame).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="+10 ▶", command=lambda: skip_frames(10)).pack(side=tk.LEFT, padx=2)
        tk.Button(controls_frame, text="+50 ▶▶", command=lambda: skip_frames(50)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(root, text="🔍 Saltar a Frame", command=jump_frame).pack(pady=5)
        tk.Button(root, text="✅ Terminar", command=root.quit, bg="lightcoral").pack(pady=10)
        
        # Mostrar primer frame
        update_frame_display()
        
        # Ejecutar interfaz
        root.mainloop()
        root.destroy()
        
        cap.release()
        print(f"✅ Selección manual completada: {len(selected_frames)} frames")
        
        return selected_frames

    def create_smart_bbox_annotations(self, frames_data):
        """Crear anotaciones inteligentes basadas en detección de formas"""
        print(f"\n🤖 Creando anotaciones inteligentes...")
        
        annotated_data = []
        
        for frame_info in frames_data:
            frame_path = Path(frame_info['path'])
            class_name = frame_info['class']
            class_id = self.classes.index(class_name)
            
            # Cargar imagen
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Obtener región de interés
            roi_mask, roi_coords = self.detect_traffic_signs_region(img)
            roi_left, roi_top, roi_right, roi_bottom = roi_coords
            roi_gray = gray[roi_top:roi_bottom, roi_left:roi_right]
            
            # Intentar detectar formas automáticamente
            best_bbox = None
            
            # 1. Para señal STOP (octagonal) - detectar como círculo aproximado
            if class_name == 'stop':
                circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                                         param1=50, param2=30, minRadius=25, maxRadius=150)
                
                if circles is not None and len(circles[0]) > 0:
                    # Tomar el círculo más grande y bien definido
                    circle = max(circles[0], key=lambda c: c[2])
                    cx, cy, radius = circle
                    
                    # Convertir coordenadas relativas a imagen completa
                    abs_cx = (cx + roi_left) / w
                    abs_cy = (cy + roi_top) / h
                    bbox_w = (radius * 2.3) / w  # Un poco más grande para cubrir el octágono
                    bbox_h = (radius * 2.3) / h
                    
                    # Asegurar que esté dentro de límites
                    bbox_w = min(bbox_w, 0.7)
                    bbox_h = min(bbox_h, 0.7)
                    
                    best_bbox = [abs_cx, abs_cy, bbox_w, bbox_h]
            
            # 2. Para señales circulares con flechas
            elif class_name in ['turn_left', 'go_straight', 'turn_right']:
                circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                         param1=50, param2=30, minRadius=20, maxRadius=130)
                
                if circles is not None and len(circles[0]) > 0:
                    circle = max(circles[0], key=lambda c: c[2])
                    cx, cy, radius = circle
                    
                    abs_cx = (cx + roi_left) / w
                    abs_cy = (cy + roi_top) / h
                    bbox_w = (radius * 2.2) / w
                    bbox_h = (radius * 2.2) / h
                    
                    bbox_w = min(bbox_w, 0.6)
                    bbox_h = min(bbox_h, 0.6)
                    
                    best_bbox = [abs_cx, abs_cy, bbox_w, bbox_h]
            
            # 3. Para señales triangulares
            elif class_name in ['road_work', 'give_way']:
                edges = cv2.Canny(roi_gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                triangular_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 800 < area < 12000:  # Área razonable para triángulos
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        if len(approx) >= 3 and len(approx) <= 4:  # Triángulo o aproximación
                            triangular_contours.append((contour, area))
                
                if triangular_contours:
                    best_contour = max(triangular_contours, key=lambda x: x[1])[0]
                    x, y, w_cont, h_cont = cv2.boundingRect(best_contour)
                    
                    abs_cx = ((x + w_cont/2) + roi_left) / w
                    abs_cy = ((y + h_cont/2) + roi_top) / h
                    bbox_w = (w_cont * 1.4) / w
                    bbox_h = (h_cont * 1.4) / h
                    
                    bbox_w = min(bbox_w, 0.7)
                    bbox_h = min(bbox_h, 0.7)
                    
                    best_bbox = [abs_cx, abs_cy, bbox_w, bbox_h]
            
            # 4. Si no se detectó nada, usar bbox por defecto optimizada por clase
            if best_bbox is None:
                if class_name == 'stop':
                    # STOP suele estar prominente y centrado
                    best_bbox = [0.5, 0.4, 0.5, 0.5]
                elif class_name in ['turn_left', 'go_straight', 'turn_right']:
                    # Señales de dirección suelen estar en posición media
                    best_bbox = [0.5, 0.35, 0.4, 0.4]
                elif class_name in ['road_work', 'give_way']:
                    # Señales triangulares pueden estar un poco más arriba
                    best_bbox = [0.5, 0.3, 0.4, 0.45]
                else:
                    # Default genérico
                    best_bbox = [0.5, 0.4, 0.45, 0.45]
            
            # Validar y ajustar bbox si es necesario
            center_x, center_y, bbox_width, bbox_height = best_bbox
            
            # Asegurar que esté completamente dentro de la imagen
            half_w = bbox_width / 2
            half_h = bbox_height / 2
            
            center_x = max(half_w, min(1 - half_w, center_x))
            center_y = max(half_h, min(1 - half_h, center_y))
            
            if (0 < center_x < 1 and 0 < center_y < 1 and 
                0 < bbox_width < 1 and 0 < bbox_height < 1):
                
                # Crear archivo de etiqueta YOLO
                label_name = frame_path.stem + '.txt'
                label_path = frame_path.parent / label_name
                
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                
                annotated_data.append({
                    'image_path': str(frame_path),
                    'label_path': str(label_path),
                    'class': class_name,
                    'bbox': [center_x, center_y, bbox_width, bbox_height],
                    'detection_method': 'smart_auto'
                })
                
                print(f"   🎯 {class_name}: bbox creada ({center_x:.2f}, {center_y:.2f}) {bbox_width:.2f}x{bbox_height:.2f}")
            else:
                print(f"   ⚠️ {class_name}: bbox inválida")
        
        print(f"✅ Creadas {len(annotated_data)} anotaciones inteligentes")
        return annotated_data
        """Crear anotaciones de bounding boxes para frames extraídos"""
        print(f"\n📦 Creando anotaciones de bounding boxes...")
        
        annotated_data = []
        
        for frame_info in frames_data:
            frame_path = Path(frame_info['path'])
            class_name = frame_info['class']
            class_id = self.classes.index(class_name)
            
            # Cargar imagen para obtener dimensiones
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Por defecto, crear bbox que cubra el centro de la imagen
            # Esto asume que la señal está principalmente en el centro
            center_x = 0.5
            center_y = 0.5
            bbox_width = 0.6   # 60% del ancho
            bbox_height = 0.6  # 60% del alto
            
            # Crear archivo de etiqueta YOLO
            label_name = frame_path.stem + '.txt'
            label_path = frame_path.parent / label_name
            
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            annotated_data.append({
                'image_path': str(frame_path),
                'label_path': str(label_path),
                'class': class_name,
                'bbox': [center_x, center_y, bbox_width, bbox_height]
            })
        
        print(f"✅ Creadas {len(annotated_data)} anotaciones automáticas")
        return annotated_data

    def create_manual_bbox_annotations(self, frames_data):
        """Crear anotaciones de bounding boxes manualmente usando interfaz gráfica"""
        print(f"\n🖱️ Anotación manual de bounding boxes...")
        
        annotated_data = []
        
        for i, frame_info in enumerate(frames_data):
            frame_path = Path(frame_info['path'])
            class_name = frame_info['class']
            class_id = self.classes.index(class_name)
            
            print(f"\n📦 Anotando {i+1}/{len(frames_data)}: {frame_path.name}")
            
            # Cargar imagen
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            original_h, original_w = img.shape[:2]
            
            # Redimensionar para mostrar (manteniendo aspecto)
            display_scale = min(800 / original_w, 600 / original_h)
            display_w = int(original_w * display_scale)
            display_h = int(original_h * display_scale)
            display_img = cv2.resize(img, (display_w, display_h))
            
            # Variables para el bbox
            bbox_coords = {'start': None, 'end': None, 'drawing': False}
            current_img = display_img.copy()
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal current_img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    bbox_coords['start'] = (x, y)
                    bbox_coords['drawing'] = True
                    
                elif event == cv2.EVENT_MOUSEMOVE and bbox_coords['drawing']:
                    current_img = display_img.copy()
                    cv2.rectangle(current_img, bbox_coords['start'], (x, y), (0, 255, 0), 2)
                    cv2.imshow('Anotar BBox', current_img)
                    
                elif event == cv2.EVENT_LBUTTONUP:
                    bbox_coords['end'] = (x, y)
                    bbox_coords['drawing'] = False
                    cv2.rectangle(current_img, bbox_coords['start'], bbox_coords['end'], (0, 255, 0), 2)
                    cv2.imshow('Anotar BBox', current_img)
            
            # Crear ventana y configurar callback
            cv2.namedWindow('Anotar BBox', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Anotar BBox', display_w, display_h)
            cv2.setMouseCallback('Anotar BBox', mouse_callback)
            
            # Mostrar instrucciones
            instructions = current_img.copy()
            cv2.putText(instructions, f"Clase: {class_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(instructions, "Arrastra para crear bbox", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(instructions, "ENTER: Guardar | ESC: Saltar | Q: Terminar", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Anotar BBox', instructions)
            
            # Esperar interacción del usuario
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == 13:  # ENTER - Guardar
                    if bbox_coords['start'] and bbox_coords['end']:
                        # Convertir coordenadas a formato YOLO
                        x1 = min(bbox_coords['start'][0], bbox_coords['end'][0]) / display_scale
                        y1 = min(bbox_coords['start'][1], bbox_coords['end'][1]) / display_scale
                        x2 = max(bbox_coords['start'][0], bbox_coords['end'][0]) / display_scale
                        y2 = max(bbox_coords['start'][1], bbox_coords['end'][1]) / display_scale
                        
                        # Convertir a formato YOLO (normalizado)
                        center_x = ((x1 + x2) / 2) / original_w
                        center_y = ((y1 + y2) / 2) / original_h
                        bbox_width = (x2 - x1) / original_w
                        bbox_height = (y2 - y1) / original_h
                        
                        # Validar bbox
                        if (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                            0 < bbox_width <= 1 and 0 < bbox_height <= 1):
                            
                            # Crear archivo de etiqueta
                            label_name = frame_path.stem + '.txt'
                            label_path = frame_path.parent / label_name
                            
                            with open(label_path, 'w') as f:
                                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
                            
                            annotated_data.append({
                                'image_path': str(frame_path),
                                'label_path': str(label_path),
                                'class': class_name,
                                'bbox': [center_x, center_y, bbox_width, bbox_height]
                            })
                            
                            print(f"   ✅ Guardado: {frame_path.name}")
                            break
                        else:
                            print(f"   ⚠️ BBox inválida, intenta de nuevo")
                            bbox_coords = {'start': None, 'end': None, 'drawing': False}
                            current_img = display_img.copy()
                            cv2.imshow('Anotar BBox', current_img)
                    else:
                        print(f"   ⚠️ Dibuja una bounding box primero")
                
                elif key == 27:  # ESC - Saltar
                    print(f"   ⏭️ Saltado: {frame_path.name}")
                    break
                
                elif key == ord('q'):  # Q - Terminar
                    cv2.destroyAllWindows()
                    return annotated_data
            
            cv2.destroyAllWindows()
        
        print(f"✅ Anotación manual completada: {len(annotated_data)} imágenes")
        return annotated_data

    def organize_final_dataset(self, annotated_data):
        """Organizar dataset final en estructura para el script de entrenamiento"""
        print(f"\n📋 Organizando dataset final...")
        
        # Crear estructura final
        final_images_dir = self.output_dir / 'dataset' / 'images'
        final_labels_dir = self.output_dir / 'dataset' / 'labels'
        
        final_images_dir.mkdir(parents=True, exist_ok=True)
        final_labels_dir.mkdir(parents=True, exist_ok=True)
        
        organized_count = 0
        
        for data in annotated_data:
            src_img_path = Path(data['image_path'])
            src_label_path = Path(data['label_path'])
            
            if src_img_path.exists() and src_label_path.exists():
                # Crear nombres únicos
                final_name = f"{data['class']}_{organized_count:04d}.jpg"
                
                # Copiar imagen
                dst_img_path = final_images_dir / final_name
                shutil.copy2(src_img_path, dst_img_path)
                
                # Copiar etiqueta
                dst_label_path = final_labels_dir / (final_name.replace('.jpg', '.txt'))
                shutil.copy2(src_label_path, dst_label_path)
                
                organized_count += 1
        
        print(f"✅ Dataset organizado: {organized_count} pares imagen-etiqueta")
        
        # Crear resumen
        summary = {
            'total_images': organized_count,
            'classes': self.classes,
            'extraction_date': datetime.now().isoformat(),
            'dataset_structure': 'images/ + labels/',
        }
        
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Resumen guardado: {summary_path}")
        
        return organized_count

    def process_videos_batch(self, videos_info, extraction_mode='auto'):
        """Procesar múltiples videos en lote"""
        print(f"\n🎬 Procesando {len(videos_info)} videos en modo: {extraction_mode}")
        
        all_frames = []
        
        for video_info in videos_info:
            video_path = video_info['path']
            class_name = video_info['class']
            
            if extraction_mode == 'auto':
                frames = self.extract_frames_from_video(video_path, class_name)
            elif extraction_mode == 'manual':
                frames = self.manual_frame_selection(video_path, class_name)
            else:
                print(f"❌ Modo de extracción desconocido: {extraction_mode}")
                continue
            
            all_frames.extend(frames)
        
        print(f"\n📊 Total frames extraídos: {len(all_frames)}")
        
        return all_frames

def main():
    """Pipeline principal para extracción de dataset desde videos"""
    print("=" * 70)
    print("🎥 VIDEO TO DATASET EXTRACTOR - PUZZLEBOT")
    print("=" * 70)
    
    extractor = VideoDatasetExtractor()
    
    # 🎯 RUTAS PREDEFINIDAS DE LOS VIDEOS
    base_path = "/home/strpicket/Videos/Screencasts/Pista_prueba/IOS"
    predefined_videos = {
        'stop': f"{base_path}/Stop_ios.mp4",
        'road_work': f"{base_path}/Road_work_ios.mp4", 
        'give_way': f"{base_path}/Give_way_ios.mp4",
        'turn_left': f"{base_path}/Turn_left_ios.mp4",
        'go_straight': f"{base_path}/Go_straight_ios.mp4",
        'turn_right': f"{base_path}/Turn_right_ios.mp4"
    }
    
    print(f"\n📁 Videos encontrados en: {base_path}")
    
    videos_info = []
    
    # Verificar cada video y configurar automáticamente
    for class_name in extractor.classes:
        video_path = predefined_videos.get(class_name)
        
        if video_path and Path(video_path).exists():
            videos_info.append({
                'path': video_path,
                'class': class_name
            })
            print(f"   ✅ {class_name}: {Path(video_path).name}")
        else:
            print(f"   ❌ {class_name}: No encontrado - {video_path}")
    
    if not videos_info:
        print("❌ No se encontraron videos válidos")
        return
    
    print(f"\n📊 Videos configurados automáticamente: {len(videos_info)}")
    for video in videos_info:
        print(f"   🎬 {video['class']}: {Path(video['path']).name}")
    
    # Confirmar antes de proceder
    confirm = input(f"\n🚀 ¿Proceder con estos {len(videos_info)} videos? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelado por el usuario")
        return
    
    # Seleccionar modo de extracción
    print(f"\n🔧 Modos de extracción disponibles:")
    print("1. auto   - Extracción automática (recomendado)")
    print("2. manual - Selección manual de frames")
    
    mode = input("Selecciona modo (auto/manual) [auto]: ").strip().lower() or 'auto'
    
    if mode not in ['auto', 'manual']:
        print("❌ Modo inválido, usando 'auto'")
        mode = 'auto'
    
    # Procesar videos
    all_frames = extractor.process_videos_batch(videos_info, mode)
    
    if not all_frames:
        print("❌ No se extrajeron frames")
        return
    
    # Seleccionar modo de anotación
    print(f"\n🔧 Modos de anotación disponibles:")
    print("1. smart  - Detección automática de formas (RECOMENDADO para fish-eye)")
    print("2. auto   - Bounding boxes automáticas simples")
    print("3. manual - Dibujar bounding boxes manualmente")
    
    annotation_mode = input("Selecciona modo de anotación (smart/auto/manual) [smart]: ").strip().lower() or 'smart'
    
    # Crear anotaciones
    if annotation_mode == 'manual':
        annotated_data = extractor.create_manual_bbox_annotations(all_frames)
    elif annotation_mode == 'smart':
        annotated_data = extractor.create_smart_bbox_annotations(all_frames)
    else:
        annotated_data = extractor.create_bounding_box_annotations(all_frames)
    
    if not annotated_data:
        print("❌ No se crearon anotaciones")
        return
    
    # Organizar dataset final
    final_count = extractor.organize_final_dataset(annotated_data)
    
    # Resumen final
    print("\n" + "="*70)
    print("🎉 EXTRACCIÓN COMPLETADA")
    print("="*70)
    print(f"📊 Total imágenes: {final_count}")
    print(f"📁 Dataset en: {extractor.output_dir}/dataset/")
    print(f"   📷 Imágenes: {extractor.output_dir}/dataset/images/")
    print(f"   📄 Etiquetas: {extractor.output_dir}/dataset/labels/")
    
    print(f"\n📝 PRÓXIMOS PASOS:")
    print(f"1. Copiar dataset al script de entrenamiento:")
    print(f"   cp -r {extractor.output_dir}/dataset/* puzzlebot_real_training/input_dataset/")
    print(f"2. Ejecutar script de entrenamiento con dataset real")
    print(f"3. ¡Entrenar el modelo!")

if __name__ == "__main__":
    main()