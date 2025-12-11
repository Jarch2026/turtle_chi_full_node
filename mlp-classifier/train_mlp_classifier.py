import json
import numpy as np

# parsing json to return a list of keypoint arrays
def load_keypoints_dict(filepath):
    # bc the json file has the structure of:
    # { "image1": [...], "image2": [...], ... }

    with open(filepath) as f:
        data = json.load(f)
    return list(data.values())


# feature extraction
def normalize_keypoints(kpts):
    kpts = np.array(kpts)[:, :2]  # we only use (17,2), the json is in (17, 3) but the last column is for confidence

    ls = kpts[5]; rs = kpts[6]
    lh = kpts[11]; rh = kpts[12]

    torso_center = (ls + rs + lh + rh) / 4.0
    centered = kpts - torso_center

    shoulder_width = np.linalg.norm(ls - rs)
    torso_len = np.linalg.norm((ls + rs)/2 - (lh + rh)/2)
    scale = max((shoulder_width + torso_len)/2.0, 1e-6)

    return centered / scale


def angle(a, b, c):
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

    feet_distance = np.linalg.norm(la - ra)

    features = np.concatenate([
        k.flatten(),
        [
            left_elbow_angle, right_elbow_angle,
            left_sh_angle, right_sh_angle,
            torso_tilt_x, torso_tilt_y,
            left_arm_height, right_arm_height,
            feet_distance
        ]
    ])
    return features


# MLP implementation
class MLPBinaryClassifier:
    def __init__(self, input_dim, hidden_dims=(64, 32), lr=5e-3, seed=0):
        np.random.seed(seed)
        h1, h2 = hidden_dims

        # Xavier/He initialization
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, h1))

        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))

        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, 1))

        self.lr = lr

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def bce_loss(y_true, y_pred):
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        y_hat = self.sigmoid(z3)

        return y_hat, (X, z1, a1, z2, a2, z3, y_hat)

    def backward(self, cache, y_true, y_pred):
        X, z1, a1, z2, a2, z3, y_hat = cache
        N = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        dL_dy = (y_pred - y_true) / N
        dz3 = dL_dy * y_hat * (1 - y_hat)

        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_grad(z2)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # gradient step
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=2000):
        y = y.reshape(-1, 1)

        for epoch in range(1, epochs+1):
            # shuffle every epoch <-- added this bc the results were not 100% accuracy
            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]

            y_hat, cache = self.forward(X)
            loss = self.bce_loss(y, y_hat)

            self.backward(cache, y, y_hat)

            if epoch % 100 == 0:
                pred = (y_hat >= 0.5).astype(int)
                acc = np.mean(pred == y)
                print(f"Epoch {epoch:4d}  Loss={loss:.4f}  Acc={acc:.3f}")

    def predict(self, X):
        y_hat, _ = self.forward(X)
        return (y_hat >= 0.5).astype(int).ravel()


# Training wrapper
def train_correctness_mlp(correct_json, incorrect_json, save_model="models/movement_4/movement_4_mlp.npz"):
    # load dataset
    correct_kpts = load_keypoints_dict(correct_json)
    incorrect_kpts = load_keypoints_dict(incorrect_json)

    X = []
    Y = []

    for k in correct_kpts:
        X.append(extract_features(k))
        Y.append(1)

    for k in incorrect_kpts:
        X.append(extract_features(k))
        Y.append(0)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)

    print("Dataset size:", X.shape, "Labels:", np.bincount(Y))

    # feature scaling
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    # added this, inspired from that sklearn MLP classifier has this feature
    np.savez("models/movement_4/movement_4_scaler.npz", mean=mean, std=std)
    print("Saved scaler to movement_4_scaler.npz")

    # train mlp
    input_dim = X.shape[1]
    # increased hidden layer, lowered learning rate
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=(64,32), lr=5e-3)
    model.fit(X, Y, epochs=600) # from 500 --> 1000 --> 2000

    # final training accuracy
    preds = model.predict(X)
    acc = np.mean(preds == Y)
    print("Final training accuracy:", acc)

    # save parameters
    np.savez(save_model,
             W1=model.W1, b1=model.b1,
             W2=model.W2, b2=model.b2,
             W3=model.W3, b3=model.b3)
    print(f"Saved model → {save_model}")


# run
if __name__ == "__main__":
    train_correctness_mlp("dataset/movement_4/correct.json", "dataset/incorrect.json")
