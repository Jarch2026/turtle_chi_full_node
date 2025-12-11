import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import math

def load_keypoints_list(filepath):
    with open(filepath) as f:
        data = json.load(f)
        
    kpts_list = list(data.values())

    return kpts_list


# feature extraction
def normalize_keypoints(kpts):
    kpts = np.array(kpts)[:, :2]

    ls = kpts[5]; rs = kpts[6]
    lh = kpts[11]; rh = kpts[12]

    torso_center = (ls + rs + lh + rh) / 4
    k = kpts - torso_center

    shoulder_width = np.linalg.norm(ls - rs)
    torso_len = np.linalg.norm((ls + rs)/2 - (lh + rh)/2)
    scale = max((shoulder_width + torso_len)/2, 1e-6)

    return k / scale


def angle(a,b,c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def extract_features(kpts):
    k = normalize_keypoints(kpts)

    ls = k[5]; rs = k[6]
    le = k[7]; re = k[8]
    lw = k[9]; rw = k[10]
    lh = k[11]; rh = k[12]
    la = k[15]; ra = k[16]

    left_elbow_angle = angle(ls, le, lw)
    right_elbow_angle = angle(rs, re, rw)
    left_sh_angle = angle(lh, ls, le)
    right_sh_angle = angle(rh, rs, re)

    torso_vec = ((ls + rs)/2) - ((lh + rh)/2)
    torso_tilt_x = torso_vec[0]
    torso_tilt_y = torso_vec[1]

    left_arm_height = lw[1] - ls[1]
    right_arm_height = rw[1] - rs[1]

    feet_dist = np.linalg.norm(la - ra)

    features = np.concatenate([
        k.flatten(),
        [left_elbow_angle, right_elbow_angle,
         left_sh_angle, right_sh_angle,
         torso_tilt_x, torso_tilt_y,
         left_arm_height, right_arm_height,
         feet_dist]
    ])

    return features


# train the classifier
def train_correctness_classifier(correct_json, incorrect_json, movement_name):
    correct_kpts = load_keypoints_list(correct_json)
    incorrect_kpts = load_keypoints_list(incorrect_json)

    X = []
    Y = []
    
    # correct = 1
    for kpts in correct_kpts:
        X.append(extract_features(kpts))
        Y.append(1)

    # incorrect = 0
    for kpts in incorrect_kpts:
        X.append(extract_features(kpts))
        Y.append(0)

    X = np.array(X)
    Y = np.array(Y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=600,
        learning_rate_init=1e-3
    )

    clf.fit(X_scaled, Y)
    y_pred = clf.predict(X_scaled)

    print(classification_report(Y, y_pred, target_names=["incorrect", "correct"]))

    joblib.dump(clf, f"{movement_name}_correctness_mlp.pkl")
    joblib.dump(scaler, f"{movement_name}_scaler.pkl")

    print(f"Saved: {movement_name}_correctness_mlp.pkl")
    print(f"Saved: {movement_name}_scaler.pkl")


# run
if __name__ == "__main__":
    train_correctness_classifier(
        correct_json="dataset/movement_1/correct.json",
        incorrect_json="dataset/movement_1/incorrect.json",
        movement_name="movement_1"
    )