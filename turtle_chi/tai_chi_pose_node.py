import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
import cv2
import cv_bridge
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import time

#from mlp_loader import load_mlp_model, extract_features

class MLPBinaryClassifier:
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        h1, h2 = hidden_dims
        
        # Initialize weights
        self.W1 = np.zeros((input_dim, h1))
        self.b1 = np.zeros((1, h1))
        
        self.W2 = np.zeros((h1, h2))
        self.b2 = np.zeros((1, h2))
        
        self.W3 = np.zeros((h2, 1))
        self.b3 = np.zeros((1, 1))
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        
        z3 = a2 @ self.W3 + self.b3
        y_hat = self.sigmoid(z3)
        
        return y_hat
    
    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat >= 0.5).astype(int).ravel()


def load_mlp_model(model_path, scaler_path, input_dim=43, hidden_dims=(64, 32)):
    # Load model weights
    model_data = np.load(model_path)
    
    # Load scaler
    scaler_data = np.load(scaler_path)
    mean = scaler_data["mean"]
    std = scaler_data["std"]
    
    # Create model
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=hidden_dims)
    
    # Load weights
    model.W1 = model_data["W1"]
    model.b1 = model_data["b1"]
    model.W2 = model_data["W2"]
    model.b2 = model_data["b2"]
    model.W3 = model_data["W3"]
    model.b3 = model_data["b3"]
    
    return model, mean, std


def normalize_keypoints(kpts):
    kpts = np.array(kpts)[:, :2]  # Only use (x, y) and also drop confidence
    
    #  key points
    ls = kpts[5]  # left shoulder
    rs = kpts[6]  # right shoulder
    lh = kpts[11]  # left hip
    rh = kpts[12]  # right hip
    
    # Center on torso
    torso_center = (ls + rs + lh + rh) / 4.0
    centered = kpts - torso_center
    
    # scale by torso size
    shoulder_width = np.linalg.norm(ls - rs)
    torso_len = np.linalg.norm((ls + rs)/2 - (lh + rh)/2)
    scale = max((shoulder_width + torso_len)/2.0, 1e-6)
    
    return centered / scale


def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features(kpts):
    # Normalize keypoints
    k = normalize_keypoints(kpts)
    
    # Get key joints
    ls = k[5]   # left shoulder
    rs = k[6]   # right shoulder
    le = k[7]   # left elbow
    re = k[8]   # right elbow
    lw = k[9]   # left wrist
    rw = k[10]  # right wrist
    lh = k[11]  # left hip
    rh = k[12]  # right hip
    la = k[15]  # left ankle
    ra = k[16]  # right ankle
    
    # Calculate angles
    left_elbow_angle = angle(ls, le, lw)
    right_elbow_angle = angle(rs, re, rw)
    left_sh_angle = angle(lh, ls, le)
    right_sh_angle = angle(rh, rs, re)
    
    # Torso orientation
    torso_vec = ((ls + rs)/2) - ((lh + rh)/2)
    torso_tilt_x = torso_vec[0]
    torso_tilt_y = torso_vec[1]
    
    # Arm heights relative to shoulders
    left_arm_height = lw[1] - ls[1]
    right_arm_height = rw[1] - rs[1]
    
    # Feet distance
    feet_distance = np.linalg.norm(la - ra)
    
    # Combine all features
    features = np.concatenate([
        k.flatten(),  # 17*2 = 34 features (normalized x, y coords) ?!!?!?
        [
            left_elbow_angle,
            right_elbow_angle,
            left_sh_angle,
            right_sh_angle,
            torso_tilt_x,
            torso_tilt_y,
            left_arm_height,
            right_arm_height,
            feet_distance
        ]
    ])  # so in roral 43 features
    
    return features


class TaiChiPoseNode(Node):
    def __init__(self):
        super().__init__('tai_chi_pose_node')
        
        self.declare_parameter('camera_topic', '/tb03/oakd/rgb/preview/image_raw') # i am #lazy
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('model_path', 'models/ver3/movement_1_mlp_scratch_ver3.npz')
        self.declare_parameter('scaler_path', 'models/ver3/movement_1_scaler_scratch_ver3.npz')
        self.declare_parameter('min_keypoint_confidence', 0.3)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.scaler_path = self.get_parameter('scaler_path').value
        self.min_keypoint_confidence = self.get_parameter('min_keypoint_confidence').value
        
        self.get_logger().info("="*60)
        self.get_logger().info("Tai Chi Pose Estimation Node Starting...")
        self.get_logger().info("="*60)
        
        
        self.bridge = cv_bridge.CvBridge()

        # Initialize the debugging window
        cv2.namedWindow("window", 1)
        
        self.get_logger().info("Loading MoveNet...")
        self.load_movenet()
        self.get_logger().info(f"MoveNet loaded (input size: {self.input_size})")
        
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.get_logger().info(f"Loading binary classifier...")
            self.model, self.feature_mean, self.feature_std = load_mlp_model(
                self.model_path,
                self.scaler_path,
                input_dim=43,
                hidden_dims=(64, 32)
            )
            self.get_logger().info(f"Binary Classifier loaded")
            self.get_logger().info(f"  Architecture: 43 to 64 to 32 to 1")
            self.get_logger().info(f"  Evaluates ANY pose as correct/incorrect")
        else:
            self.get_logger().error(f"Model files not found!")
            self.get_logger().error(f"  Model: {self.model_path}")
            self.get_logger().error(f"  Scaler: {self.scaler_path}")
            raise FileNotFoundError("Model or scaler file not found")
        
        self.latest_image = None
        self.capture_ready = False
        
        # Result publisher
        self.result_pub = self.create_publisher(String, self.result_topic, 10)
        
        # Camera subscriber
        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Trigger subscriber
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info("Pose Estimation Node Ready!")
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Trigger: {self.trigger_topic}")
        self.get_logger().info(f"  Result: {self.result_topic}")
        self.get_logger().info("  Waiting for trigger...")
        self.get_logger().info("="*60)
    
    def load_movenet(self):
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        
        def movenet(input_image):
            model = module.signatures['serving_default']
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            return outputs['output_0'].numpy()
        
        self.movenet = movenet
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            self.capture_ready = True
            cv2.imshow("window", cv_image)
            cv2.waitKey(3)
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")
    
    def trigger_callback(self, msg):
        if not msg.data:
            return
        
        self.get_logger().info("TRIGGER RECEIVED - Capturing pose...")
        
        if not self.capture_ready or self.latest_image is None:
            self.get_logger().warn("No image available yet!")
            result_msg = String()
            result_msg.data = "no_person"
            self.result_pub.publish(result_msg)
            return
        
        try:
            cv_image = self.latest_image.copy()
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            input_image = tf.expand_dims(rgb_image, axis=0)
            input_image = tf.image.resize_with_pad(
                input_image,
                self.input_size,
                self.input_size
            )
            
            keypoints_with_scores = self.movenet(input_image)
            
            avg_conf = np.mean(keypoints_with_scores[0, 0, :, 2])
            if avg_conf < self.min_keypoint_confidence:
                self.get_logger().warn(f"No person detected (conf: {avg_conf:.2f})")
                result_msg = String()
                result_msg.data = "no_person"
                self.result_pub.publish(result_msg)
                return
            
            # Extract keypoints [17, 3] - (y, x, confidence)
            kpts = keypoints_with_scores[0, 0]
            self.get_logger().info(f"keypoints: {kpts}")

            # Convert to [17, 3] with (x, y, confidence) for feature extraction
            kpts_reordered = np.zeros_like(kpts)
            kpts_reordered[:, 0] = kpts[:, 1]  # x
            kpts_reordered[:, 1] = kpts[:, 0]  # y
            kpts_reordered[:, 2] = kpts[:, 2]  # confidence

            # Extract features (43 features)
            features = extract_features(kpts_reordered)

            # Normalize features
            features_norm = (features - self.feature_mean.flatten()) / self.feature_std.flatten()
            
            # Predict (binary: 1=correct, 0=incorrect)
            prediction = self.model.predict(features_norm.reshape(1, -1))[0]
            
            # Get probability
            prob = self.model.forward(features_norm.reshape(1, -1))[0, 0]
            
            # Publish result
            result_msg = String()
            if prediction == 1:
                result_msg.data = "correct"
                self.get_logger().info(f" CORRECT (confidence: {prob:.3f})")
            else:
                result_msg.data = "incorrect"
                self.get_logger().info(f"  INCORRECT (confidence: {1-prob:.3f})")

            self.result_pub.publish(result_msg)
            self.get_logger().info(f"  Keypoint avg confidence: {avg_conf:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error classifying pose: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
            result_msg = String()
            result_msg.data = "error"
            self.result_pub.publish(result_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TaiChiPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
