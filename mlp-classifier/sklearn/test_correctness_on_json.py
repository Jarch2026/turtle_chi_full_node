import json
import joblib
import numpy as np
from sklearn.train_correctness_classifier import extract_features

# Load model + scaler
clf = joblib.load("movement_1_correctness_mlp.pkl")
scaler = joblib.load("movement_1_scaler.pkl")

def load_keypoints_dict(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return list(data.values())

correct_kpts = load_keypoints_dict("dataset/movement_1/correct.json")
incorrect_kpts = load_keypoints_dict("dataset/movement_1/incorrect.json")

# Test on correct examples
print("Correct samples:")
for i, kpts in enumerate(correct_kpts[:30]):
    feat = extract_features(kpts)
    pred = clf.predict(scaler.transform([feat]))[0]
    print(f"Correct sample {i}: prediction={pred}")

# Test on incorrect examples
print("Incorrect samples:")
for i, kpts in enumerate(incorrect_kpts[:30]):
    feat = extract_features(kpts)
    pred = clf.predict(scaler.transform([feat]))[0]
    print(f"Incorrect sample {i}: prediction={pred}")
