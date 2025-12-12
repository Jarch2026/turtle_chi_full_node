"""
Tai Chi Pose Estimation and Classification Node

This ROS2 node performs real-time pose estimation and evaluation for Tai Chi movements.
It uses Google's MoveNet for skeletal keypoint detection and custom MLP classifiers to
determine if a user's pose matches the target movement.

The system:
1. Receives camera images from TurtleBot4's OAK-D camera
2. Detects human pose using MoveNet (17 keypoints)
3. Extracts normalized features from keypoints (positions, angles, heights)
4. Classifies pose using movement-specific MLP models
5. Publishes evaluation results (correct/incorrect/no_person)

- Multi-model architecture: Different classifiers for each of 3 Tai Chi movements
- Secondary evaluation model for fine-grained feedback (low vs incorrect) and (narrow leg stance vs incorrect)
- Real-time pose visualization
- Dynamic model switching based on which movement is being taught

"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool, Int32
import cv2
import cv_bridge
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import time


class MLPBinaryClassifier:
    """
    Custom Multi-Layer Perceptron (MLP) for binary pose classification.
    
    This is a simple 3-layer neural network implemented from scratch using NumPy.
    It classifies whether a pose is "correct" (1) or "incorrect" (0) for a given
    Tai Chi movement.
    
    Architecture:
        Input Layer: 43 features (normalized keypoints + derived features)
        Hidden Layer 1: 64 neurons with ReLU activation
        Hidden Layer 2: 32 neurons with ReLU activation
        Output Layer: 1 neuron with sigmoid activation (binary classification)
    
    The model weights are loaded from .npz files trained offline using scikit-learn.
    """
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        h1, h2 = hidden_dims
        
        # Initialize weights. Layer 1
        self.W1 = np.zeros((input_dim, h1))
        self.b1 = np.zeros((1, h1))
        # Layer 2
        self.W2 = np.zeros((h1, h2))
        self.b2 = np.zeros((1, h2))
        # Layer 3
        self.W3 = np.zeros((h2, 1))
        self.b3 = np.zeros((1, 1))
    
    @staticmethod
    def relu(x):
        """
        ReLU (Rectified Linear Unit) activation function.
        Returns maximum of 0 and x, introducing non-linearity.
        """
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function for binary classification output.
        Maps any input value to range (0, 1), interpreted as probability
        of the pose being "correct".
        
        Note:
            Clips input to [-50, 50] to prevent overflow in exp() calculation
        """
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X):
        """
        Forward pass through the network to compute output probabilities.
        
        Computation flow:
            X -> [Linear -> ReLU] -> [Linear -> ReLU] -> [Linear -> Sigmoid] -> y_hat
        """
        # Layer 1
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        # Layer 3
        z3 = a2 @ self.W3 + self.b3
        y_hat = self.sigmoid(z3)
        
        return y_hat
    
    def predict(self, X):
        """
        Make binary predictions (0 or 1) from input features.
        
        Uses 0.5 as decision threshold:
            - Probability >= 0.5 -> Predict 1 (correct)
            - Probability < 0.5 -> Predict 0 (incorrect)
        """
        y_hat = self.forward(X)
        return (y_hat >= 0.5).astype(int).ravel()


def load_mlp_model(model_path, scaler_path, input_dim=43, hidden_dims=(64, 32)):
    """
    Load a trained MLP model and its associated feature scaler from disk.
    
    The model and scaler were trained offline using scikit-learn and saved as .npz files.
    The scaler ensures features are normalized to zero mean and unit variance, matching
    the distribution seen during training.
    """
    # Load saved weights
    model_data = np.load(model_path)
    
    # Load normalization parameters
    scaler_data = np.load(scaler_path)
    mean = scaler_data["mean"]
    std = scaler_data["std"]
    
    # Create and populate model
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=hidden_dims)
    
    # Load weights from training
    model.W1 = model_data["W1"]
    model.b1 = model_data["b1"]
    model.W2 = model_data["W2"]
    model.b2 = model_data["b2"]
    model.W3 = model_data["W3"]
    model.b3 = model_data["b3"]
    
    return model, mean, std


def normalize_keypoints(kpts):
    """
    Normalize skeletal keypoints to be invariant to position, scale, and rotation.
    
    This normalization ensures the model focuses on pose SHAPE instead of other environment differences like
    where the person is in the frame or how far they are from the camera etc.
    
    Normalization steps:
    1. Center on torso (remove translation)
    2. Scale by body size (remove scale variation)
    
    Keypoint indices (MoveNet format):
        0: nose, 1-2: eyes, 3-4: ears
        5: left shoulder, 6: right shoulder
        7: left elbow, 8: right elbow
        9: left wrist, 10: right wrist
        11: left hip, 12: right hip
        13: left knee, 14: right knee
        15: left ankle, 16: right ankle
    """
    kpts = np.array(kpts)[:, :2]  # Only use (y, x) and drop confidence - matches what we used for training!
    
    # Key points
    ls = kpts[5]  # left shoulder
    rs = kpts[6]  # right shoulder
    lh = kpts[11]  # left hip
    rh = kpts[12]  # right hip
    
    # Center on torso
    torso_center = (ls + rs + lh + rh) / 4.0
    centered = kpts - torso_center
    
    # Scale by torso size
    # Calculate characteristic body dimensions
    shoulder_width = np.linalg.norm(ls - rs)
    torso_len = np.linalg.norm((ls + rs)/2 - (lh + rh)/2)

    # Use average of width and height as scale factor
    scale = max((shoulder_width + torso_len)/2.0, 1e-6)
    # Divide by scale to normalize
    return centered / scale


def angle(a, b, c):
    """
    Calculate angle at point b formed by points a-b-c.
    
    This is used to compute joint angles (like elbow angle, shoulder angle)
    which are important features for pose classification.
    """
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features(kpts):
    """
    Extract a comprehensive 43-dimensional feature vector from pose keypoints.
    
    It then transforms raw 17 keypoints into a rich feature representation
    that captures both spatial relationships and pose geometry.

    Design rationale:
        - Normalized coordinates: Capture overall pose shape
        - Joint angles: Critical for arm position evaluation
        - Torso orientation: Indicates body alignment
        - Arm heights: Important for distinguishing high vs low arm positions
        - Feet distance: Shows stance width/balance
    """
    # Normalize keypoints (keeps [y, x] format)
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
        k.flatten(),  # 17*2 = 34 features (normalized y, x coords - matches training!)
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
    ])  # Total: 43 features
    
    return features


class MultiModelTaiChiPoseNode(Node):
    """
    ROS2 node for multi-model Tai Chi pose estimation and evaluation.
    This node acts as the "vision system" for the Turtle Chi robot, providing
    real-time feedback on whether a user's pose matches the target movement.
    
    Key points:
    1. Receive and process camera images
    2. Detect human pose using MoveNet
    3. Extract features from detected pose
    4. Classify pose using movement-specific MLP models
    5. Publish evaluation results
    
    The node maintains multiple models (one per movement + a secondary evaluator for Movement 1 and 2)
    and switches between them based on commands from the teaching node.
    """
    def __init__(self):
        super().__init__('tai_chi_pose_node')
        # --- paramter declarations ---
        self.declare_parameter('camera_topic', '/tb03/oakd/rgb/preview/image_raw')
        self.declare_parameter('trigger_topic', '/trigger_capture')
        self.declare_parameter('result_topic', '/pose_result')
        self.declare_parameter('model_select_topic', '/select_movement')
        self.declare_parameter('models_dir', 'models/ver3/')
        self.declare_parameter('min_keypoint_confidence', 0.3)

        # ----- retrieve parameter values ----
        self.camera_topic = self.get_parameter('camera_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.result_topic = self.get_parameter('result_topic').value
        self.model_select_topic = self.get_parameter('model_select_topic').value
        self.models_dir = self.get_parameter('models_dir').value
        self.min_keypoint_confidence = self.get_parameter('min_keypoint_confidence').value

        self.get_logger().info("="*60)
        self.get_logger().info("Multi-Model Tai Chi Pose Estimation Node Starting...")
        self.get_logger().info("="*60)

        # ----- opencv setup -----
        self.bridge = cv_bridge.CvBridge() #ros image to opencv converter
        cv2.namedWindow("window", 1) # create window for visualization

        # ----- load movement classification models and movenet pose detector ---
        self.get_logger().info("Loading MoveNet...")
        self.load_movenet()
        self.get_logger().info(f"MoveNet loaded (input size: {self.input_size})")
        
        # Load all models at startup  
        self.models = {}
        self.get_logger().info("\nLoading movement models...")
         
        for i in range(1, 5):
            model_path = os.path.join(self.models_dir, f'movement_{i}_mlp_scratch_ver3.npz')
            scaler_path = os.path.join(self.models_dir, f'movement_{i}_scaler_scratch_ver3.npz')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model, mean, std = load_mlp_model(model_path, scaler_path, input_dim=43, hidden_dims=(64, 32))
                self.models[i] = {
                    'model': model,
                    'mean': mean,
                    'std': std,
                    'path': model_path
                }
                self.get_logger().info(f"  Movement {i} model loaded: {os.path.basename(model_path)}")
            else:
                self.get_logger().warn(f"  Movement {i} model not found!")
        
        # ---- Load secondary evaluation model ----- 
        model_5_path = os.path.join(self.models_dir, 'movement_1_low_mlp.npz')
        scaler_5_path = os.path.join(self.models_dir, 'movement_1_low_scaler.npz')
        
        if os.path.exists(model_5_path) and os.path.exists(scaler_5_path):
            model, mean, std = load_mlp_model(model_5_path, scaler_5_path, input_dim=43, hidden_dims=(64, 32))
            self.models[5] = {
                'model': model,
                'mean': mean,
                'std': std,
                'path': model_5_path
            }
            self.get_logger().info(f"  Model 1 (low/incorrect) loaded: {os.path.basename(model_5_path)}")
        else:
            self.get_logger().warn(f"  Model 1 (low/incorrect) not found - secondary evaluation disabled")
        # --- state variables -----
        self.current_movement = 1
        self.get_logger().info(f"\n Currently evaluating: Movement {self.current_movement}")
        self.get_logger().info(f"   Total models loaded: {len(self.models)}")
        
        self.latest_image = None
        self.capture_ready = False
        
        # Publishers
        self.result_pub = self.create_publisher(String, self.result_topic, 10)
        
        # Subscribers
        # Subscribe to camera feed
        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Subscribe to evaluation triggers
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10
        )

        # Subscribe to model selection commands
        self.model_select_sub = self.create_subscription(
            Int32,
            self.model_select_topic,
            self.model_select_callback,
            10
        )
        
        self.get_logger().info("="*60)
        self.get_logger().info("Multi-Model Pose Estimation Node Ready!")
        self.get_logger().info(f"  Camera: {self.camera_topic}")
        self.get_logger().info(f"  Trigger: {self.trigger_topic}")
        self.get_logger().info(f"  Model Select: {self.model_select_topic}")
        self.get_logger().info(f"  Result: {self.result_topic}")
        self.get_logger().info("  Waiting for commands...")
        self.get_logger().info("="*60)
    
    def load_movenet(self):
        """
        Load Google's MoveNet pose detection model from TensorFlow Hub.
        
        MoveNet is a fast, accurate pose estimation model that detects 17 skeletal
        keypoints from RGB images. We use the "lightning" variant which is optimized
        for speed over accuracy (suitable for real-time applications).
            
        The loaded model is stored as self.movenet function that can be called
        directly on input images.
        
        Note: Downloads model on first run (~13MB) and subsequent runs load from cache. 
              Input images are automatically resized and padded to 192×192
        """
        # Load pre-trained model from TensorFlow Hub
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.input_size = 192
        
        def movenet(input_image):
            """
            Wrapper function for MoveNet inference.
            """
            model = module.signatures['serving_default']
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            return outputs['output_0'].numpy()
        
        self.movenet = movenet
    
    def model_select_callback(self, msg):
        """
        Callback to switch between different movement classification models.
        
        Called when the teaching node wants to evaluate a different movement.
        For example, when transitioning from teaching Movement 1 to Movement 2.
        
        Args:
            msg (Int32): Movement number (1-5)
                        1-3: Main movement models
                        4: Secondary evaluator for Movement 2 (reserved)
                        5: Secondary evaluator for Movement 1
        """
        movement_num = msg.data
        if movement_num in self.models:
            self.current_movement = movement_num
            self.get_logger().info(f" Switched to Movement {movement_num} model")
        else:
            self.get_logger().warn(f"  Movement {movement_num} model not loaded!")
    
    def image_callback(self, msg):
        """
        Callback to receive and store camera images.
        
        Continuously receives images from the camera and stores the latest one.
        Also displays the image in an OpenCV window for debugging/visualization.
        """
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Store for evaluation
            self.latest_image = cv_image
            self.capture_ready = True

            #display in window
            cv2.imshow("window", cv_image)
            cv2.waitKey(3)
        except Exception as e:
            self.get_logger().error(f"Error receiving image: {e}")
    
    def trigger_callback(self, msg):
        """
        Callback to trigger pose capture and evaluation.
        
        This is the main evaluation pipeline, called when the teaching node
        wants to evaluate the user's current pose.

        Basically, it does:
        
        1. Verify image is available
        2. Verify model is loaded
        3. Run MoveNet to detect pose
        4. Check if person is detected (confidence threshold)
        5. Extract features from keypoints
        6. Normalize features using model's scaler
        7. Classify using current movement's model
        8. Publish result
        """
        # Ignore false triggers
        if not msg.data:
            return
        
        self.get_logger().info(f" TRIGGER - Evaluating Movement {self.current_movement}...")

        
        # --- validate image availability ----
        if not self.capture_ready or self.latest_image is None:
            self.get_logger().warn("No image available yet!")
            result_msg = String()
            result_msg.data = "no_person"
            self.result_pub.publish(result_msg)
            return

        # --- validate model availability ---- 
        if self.current_movement not in self.models:
            self.get_logger().error(f"Movement {self.current_movement} model not loaded!")
            result_msg = String()
            result_msg.data = "error"
            self.result_pub.publish(result_msg)
            return
        
        try:
            # ---- prepare image for movenet ----
            cv_image = self.latest_image.copy()
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # convert to TensorFlow tensor and add batch dimension
            input_image = tf.expand_dims(rgb_image, axis=0)

            # resize to MoveNet input size (192×192) with padding to preserve aspect ratio
            input_image = tf.image.resize_with_pad(
                input_image,
                self.input_size,
                self.input_size
            )

            # --- run pose detection ----
            # return shape: (1, 1, 17, 3) where last dim is [y, x, confidence]
            keypoints_with_scores = self.movenet(input_image)

            # --- check person detection
            # calculate average confidence across all keypoints
            avg_conf = np.mean(keypoints_with_scores[0, 0, :, 2])
            if avg_conf < self.min_keypoint_confidence:
                # low confidence
                self.get_logger().warn(f"No person detected (conf: {avg_conf:.2f})")
                result_msg = String()
                result_msg.data = "no_person"
                self.result_pub.publish(result_msg)
                return

            # extract keypoints
            kpts = keypoints_with_scores[0, 0]

            # feature extraction
            features = extract_features(kpts)

            # Get current model
            model_data = self.models[self.current_movement]
            model = model_data['model']
            mean = model_data['mean']
            std = model_data['std']

            # Normalize features
            features_norm = (features - mean.flatten()) / std.flatten()
            
            # Predict (binary: 1=correct, 0=incorrect)
            prediction = model.predict(features_norm.reshape(1, -1))[0]
            
            # Get probability
            prob = model.forward(features_norm.reshape(1, -1))[0, 0]
            
            # Publish result
            result_msg = String()
            if prediction == 1:
                result_msg.data = "correct"
                self.get_logger().info(f" Movement {self.current_movement} CORRECT (confidence: {prob:.3f})")
            else:
                result_msg.data = "incorrect"
                self.get_logger().info(f" Movement {self.current_movement} INCORRECT (confidence: {1-prob:.3f})")

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
        node = MultiModelTaiChiPoseNode()
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
