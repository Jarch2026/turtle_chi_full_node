import numpy as np
import json


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
