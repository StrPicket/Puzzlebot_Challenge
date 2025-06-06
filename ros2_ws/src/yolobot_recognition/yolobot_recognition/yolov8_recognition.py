#!/usr/bin/env python3

import sys
import os
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CompressedImage
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

class YoloV8Detection(Node):
    def __init__(self):
        super().__init__('yolov8_detection')
        
        # Declare parameters
        self.declare_parameter('model_name', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('update_rate', 10.0)
        
        # Retrieve parameters
        self.model_name = self.get_parameter('model_name').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.update_rate = self.get_parameter('update_rate').value
        
        # Initialize variables
        self.bridge = CvBridge()
        self.image = None
        self.model = None
        self.timer = None  
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Create timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Publishers
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 10)
        self.image_pub = self.create_publisher(Image, "/inference_result", 10)
        
        # Subscriber
        self.subscription = self.create_subscription(
            CompressedImage,
            'image_raw/compressed', 
            self.image_callback, 
            qos.qos_profile_sensor_data
        )
        
        self.get_logger().info('YoloV8Detection Start.')

    def _load_yolo_model(self):
        """Load YOLO model from package share directory."""
        try:
            package_share_dir = get_package_share_directory('yolobot_recognition')
            model_path = os.path.join(package_share_dir, 'models', self.model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}.")
                
            self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO model loaded: {self.model_name}.')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

    def image_callback(self, msg):
        """Callback to convert ROS image to OpenCV format and store it."""
        try:
            # First decode the compressed image
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Check if image is grayscale and convert to BGR if needed
            if len(self.image.shape) == 2:  # Grayscale image
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            elif self.image.shape[2] == 1:  # Single channel
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")

    def timer_callback(self):
        """Main timer function to process images and detect objects."""
        if self.image is None or self.model is None:
            return

        try:
            # Run YOLO inference
            results = self.model(self.image)
            
            # Create the inference message
            yolov8_inference = Yolov8Inference()
            yolov8_inference.header.frame_id = "camera_frame"
            yolov8_inference.header.stamp = self.get_clock().now().to_msg()
            
            # Process detection results
            detection_count = 0
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get confidence score
                        conf = float(box.conf[0].to('cpu').detach().numpy())
                        
                        # Only include detections above confidence threshold
                        if conf >= self.confidence_threshold:
                            inference_result = InferenceResult()
                            
                            # Get box coordinates and class
                            b = box.xyxy[0].to('cpu').detach().numpy().copy()
                            c = int(box.cls[0].to('cpu').detach().numpy())
                            
                            # Fill the message
                            inference_result.class_name = self.model.names[c]
                            inference_result.left = int(b[0])
                            inference_result.top = int(b[1])
                            inference_result.right = int(b[2])
                            inference_result.bottom = int(b[3])
                            
                            yolov8_inference.yolov8_inference.append(inference_result)
                            detection_count += 1
            
            # Create annotated image and publish
            annotated_frame = results[0].plot()
            # Convertir de RGB a BGR para ROS
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            image_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "camera_frame"
            
            self.image_pub.publish(image_msg)
            self.yolov8_pub.publish(yolov8_inference)
            
            if detection_count > 0:
                self.get_logger().debug(f"Detected {detection_count} objects.")
            
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Validates and applies updated node parameters."""
        model_changed = False
        
        for param in params:
            if param.name == 'model_name':
                if not isinstance(param.value, str) or len(param.value.strip()) == 0:
                    return SetParametersResult(
                        successful=False,
                        reason="model_name must be a non-empty string."
                    )
                if param.value != self.model_name:
                    self.model_name = param.value
                    model_changed = True
                    self.get_logger().info(f"model_name updated: {self.model_name}.")

            elif param.name == 'confidence_threshold':
                if not isinstance(param.value, (int, float)) or param.value < 0.0 or param.value > 1.0:
                    return SetParametersResult(
                        successful=False,
                        reason="confidence_threshold must be between 0.0 and 1.0."
                    )
                self.confidence_threshold = float(param.value)
                self.get_logger().info(f"confidence_threshold updated: {self.confidence_threshold}.")

            elif param.name == 'update_rate':
                if not isinstance(param.value, (int, float)) or param.value <= 0.0:
                    return SetParametersResult(
                        successful=False,
                        reason="update_rate must be > 0."
                    )
                self.update_rate = float(param.value)
                # Only cancel timer if it exists
                if hasattr(self, 'timer') and self.timer is not None:
                    self.timer.cancel()
                    self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"update_rate updated: {self.update_rate} Hz.")

        # Reload model if model_name changed
        if model_changed:
            try:
                self._load_yolo_model()
            except Exception as e:
                return SetParametersResult(
                    successful=False,
                    reason=f"Failed to load new model: {e}"
                )

        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)

    try:
        node = YoloV8Detection()
    except Exception as e:
        print(f"[FATAL] YoloV8Detection failed to initialize: {e}", file=sys.stderr)
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
