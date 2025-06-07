#!/usr/bin/env python3

import sys
import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CompressedImage
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

class EnhancedYoloV8Detection(Node):
    def __init__(self):
        super().__init__('enhanced_yolov8_detection')
        
        # Declare parameters - OPTIMIZADOS PARA TURN_RIGHT
        self.declare_parameter('model_name', 'turn_right_trained.pt')
        self.declare_parameter('confidence_threshold', 0.25)    # Más bajo para detectar a distancia
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('update_rate', 15.0)             # Más frecuente
        self.declare_parameter('image_size', 640)
        self.declare_parameter('enhance_image', True)           # Mejoras automáticas
        self.declare_parameter('temporal_smoothing', True)      # Filtrado temporal
        self.declare_parameter('min_detection_count', 2)        # Confirmaciones mínimas
        
        # Retrieve parameters
        self.model_name = self.get_parameter('model_name').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.update_rate = self.get_parameter('update_rate').value
        self.image_size = self.get_parameter('image_size').value
        self.enhance_image = self.get_parameter('enhance_image').value
        self.temporal_smoothing = self.get_parameter('temporal_smoothing').value
        self.min_detection_count = self.get_parameter('min_detection_count').value
        
        # Initialize variables
        self.bridge = CvBridge()
        self.image = None
        self.model = None
        self.timer = None
        
        # Temporal smoothing variables
        self.detection_history = defaultdict(lambda: deque(maxlen=5))
        self.detection_counts = defaultdict(int)
        
        # Image enhancement setup
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Create timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Publishers
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 10)
        self.image_pub = self.create_publisher(Image, "/inference_result", 10)
        self.enhanced_image_pub = self.create_publisher(Image, "/enhanced_image", 10)
        
        # Subscriber
        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed', 
            self.image_callback, 
            qos.qos_profile_sensor_data
        )
        
        self.get_logger().info('Enhanced YoloV8Detection for turn_right started.')

    def _load_yolo_model(self):
        """Load YOLO model from package share directory."""
        try:
            package_share_dir = get_package_share_directory('yolobot_recognition')
            model_path = os.path.join(package_share_dir, 'models', self.model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}.")
                
            self.model = YOLO(model_path)
            self.get_logger().info(f'Enhanced YOLO model loaded: {self.model_name}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

    def enhance_image_quality(self, image):
        """Aplicar mejoras automáticas de imagen para mejor detección"""
        if not self.enhance_image:
            return image
            
        try:
            # 1. Convertir a LAB para mejor manejo de iluminación
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 2. Aplicar CLAHE para mejorar contraste local
            l_enhanced = self.clahe.apply(l)
            
            # 3. Merge channels back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # 4. Mejora adicional de brillo si está muy oscuro
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 80:  # Imagen muy oscura
                # Aumentar brillo
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=30)
            elif mean_brightness > 180:  # Imagen muy brillante
                # Reducir brillo
                enhanced = cv2.convertScaleAbs(enhanced, alpha=0.9, beta=-20)
            
            # 5. Aplicar filtro bilateral para reducir ruido pero mantener bordes
            enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            # 6. Leve sharpening para mejorar definición de bordes
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.5
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # Asegurar valores válidos
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            self.get_logger().warn(f"Image enhancement failed: {e}")
            return image

    def image_callback(self, msg):
        """Callback to convert ROS image to OpenCV format and store it."""
        try:
            # Decode compressed image
            raw_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Check if image is grayscale and convert to BGR if needed
            if len(raw_image.shape) == 2:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            elif raw_image.shape[2] == 1:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            
            # Apply image enhancement
            self.image = self.enhance_image_quality(raw_image)
            
            # Publish enhanced image for debugging
            if self.enhance_image:
                try:
                    enhanced_msg = self.bridge.cv2_to_imgmsg(self.image, encoding='bgr8')
                    enhanced_msg.header.stamp = self.get_clock().now().to_msg()
                    enhanced_msg.header.frame_id = "camera_frame"
                    self.enhanced_image_pub.publish(enhanced_msg)
                except Exception as e:
                    pass  # No es crítico
                
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")

    def is_detection_stable(self, class_name, bbox, confidence):
        """Check if detection is stable over time using temporal smoothing."""
        if not self.temporal_smoothing:
            return True
            
        # Add current detection to history
        detection_info = {'bbox': bbox, 'confidence': confidence}
        self.detection_history[class_name].append(detection_info)
        self.detection_counts[class_name] += 1
        
        # Check if we have enough consistent detections
        if len(self.detection_history[class_name]) >= self.min_detection_count:
            # Calculate average confidence of recent detections
            recent_detections = list(self.detection_history[class_name])[-self.min_detection_count:]
            avg_confidence = sum(d['confidence'] for d in recent_detections) / len(recent_detections)
            
            # Check consistency of bounding boxes
            recent_boxes = [d['bbox'] for d in recent_detections]
            avg_iou = 0
            comparisons = 0
            
            for i in range(len(recent_boxes)):
                for j in range(i + 1, len(recent_boxes)):
                    iou = self.calculate_iou(recent_boxes[i], recent_boxes[j])
                    avg_iou += iou
                    comparisons += 1
            
            if comparisons > 0:
                avg_iou /= comparisons
                # Detection is stable if good IoU and confidence
                return avg_iou > 0.3 and avg_confidence > (self.confidence_threshold * 0.8)
                
        return self.detection_counts[class_name] >= self.min_detection_count

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
            
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def timer_callback(self):
        """Main timer function to process images and detect objects."""
        if self.image is None or self.model is None:
            return

        try:
            # Run YOLO inference with enhanced parameters
            results = self.model(
                self.image,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                half=False,      # Use FP32 for better accuracy
                augment=True,    # Enable test-time augmentation for better detection
                agnostic_nms=False
            )
            
            # Create the inference message
            yolov8_inference = Yolov8Inference()
            yolov8_inference.header.frame_id = "camera_frame"
            yolov8_inference.header.stamp = self.get_clock().now().to_msg()
            
            # Process detection results
            detection_count = 0
            confirmed_detections = 0
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence score
                        conf = float(box.conf[0].to('cpu').detach().numpy())
                        
                        if conf >= self.confidence_threshold:
                            # Get box coordinates and class
                            b = box.xyxy[0].to('cpu').detach().numpy().copy()
                            c = int(box.cls[0].to('cpu').detach().numpy())
                            class_name = self.model.names[c]
                            
                            bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                            
                            # Check temporal stability
                            is_stable = self.is_detection_stable(class_name, bbox, conf)
                            
                            if is_stable:
                                inference_result = InferenceResult()
                                inference_result.class_name = class_name
                                inference_result.left = bbox[0]
                                inference_result.top = bbox[1]
                                inference_result.right = bbox[2]
                                inference_result.bottom = bbox[3]
                                
                                yolov8_inference.yolov8_inference.append(inference_result)
                                confirmed_detections += 1
                                
                                # Log detección confirmada
                                self.get_logger().info(f"CONFIRMED: {class_name} (conf: {conf:.2f})")
                            
                            detection_count += 1
            
            # Create annotated image and publish
            annotated_frame = results[0].plot(
                conf=True,
                line_width=3,
                font_size=1.2,
                pil=False
            )
            
            # Convert from RGB to BGR for ROS
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Add info text
            if self.temporal_smoothing:
                info_text = f"Raw: {detection_count} | Confirmed: {confirmed_detections}"
                cv2.putText(annotated_frame_bgr, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Add confidence threshold info
            conf_text = f"Conf threshold: {self.confidence_threshold:.2f}"
            cv2.putText(annotated_frame_bgr, conf_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            image_msg = self.bridge.cv2_to_imgmsg(annotated_frame_bgr, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_frame"
            
            self.image_pub.publish(image_msg)
            self.yolov8_pub.publish(yolov8_inference)
            
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Handle parameter updates"""
        for param in params:
            if param.name == 'confidence_threshold':
                if 0.0 <= param.value <= 1.0:
                    self.confidence_threshold = float(param.value)
                    self.get_logger().info(f"Confidence threshold updated: {self.confidence_threshold}")
                else:
                    return SetParametersResult(successful=False, reason="Invalid confidence threshold")
            
            # Add other parameter handlers as needed
            
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)

    try:
        node = EnhancedYoloV8Detection()
    except Exception as e:
        print(f"[FATAL] Enhanced YoloV8Detection failed to initialize: {e}", file=sys.stderr)
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()