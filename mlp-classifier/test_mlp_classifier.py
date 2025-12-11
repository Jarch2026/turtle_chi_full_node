# import numpy as np
# import json
# from train_mlp_classifier import (
#     load_keypoints_dict,
#     extract_features,
#     MLPBinaryClassifier
# )

# # load scaler and model weights
# def load_mlp_from_npz(model_path, scaler_path, input_dim, hidden_dims=(64,32)):
#     # load parameters
#     data = np.load(model_path)
#     scaler = np.load(scaler_path)

#     mean = scaler["mean"]
#     std = scaler["std"]

#     # create model object
#     model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=hidden_dims)

#     # load weights
#     model.W1 = data["W1"]
#     model.b1 = data["b1"]
#     model.W2 = data["W2"]
#     model.b2 = data["b2"]
#     model.W3 = data["W3"]
#     model.b3 = data["b3"]

#     return model, mean, std


# # main test
# if __name__ == "__main__":
#     correct_kpts = load_keypoints_dict("dataset/movement_1/correct.json")
#     incorrect_kpts = load_keypoints_dict("dataset/movement_1/incorrect.json")

#     X_correct = [
#         [
#             0.194268137216568,
#             0.36178767681121826,
#             0.8199368715286255
#         ],
#         [
#             0.18171802163124084,
#             0.3815325200557709,
#             0.6974732875823975
#         ],
#         [
#             0.17885719239711761,
#             0.3461880385875702,
#             0.8221679329872131
#         ],
#         [
#             0.19237589836120605,
#             0.40522301197052,
#             0.847095251083374
#         ],
#         [
#             0.19348251819610596,
#             0.3263118863105774,
#             0.7905223369598389
#         ],
#         [
#             0.2751682996749878,
#             0.43929561972618103,
#             0.8710607290267944
#         ],
#         [
#             0.26490235328674316,
#             0.2893701195716858,
#             0.7169162034988403
#         ],
#         [
#             0.3151388168334961,
#             0.45100438594818115,
#             0.4796256124973297
#         ],
#         [
#             0.3107684850692749,
#             0.2759146988391876,
#             0.5081459879875183
#         ],
#         [
#             0.3351489007472992,
#             0.4351035952568054,
#             0.531028687953949
#         ],
#         [
#             0.32368457317352295,
#             0.27885305881500244,
#             0.3877490162849426
#         ],
#         [
#             0.5406951904296875,
#             0.40077608823776245,
#             0.8500264286994934
#         ],
#         [
#             0.5386444926261902,
#             0.3030703067779541,
#             0.8918479084968567
#         ],
#         [
#             0.7368545532226562,
#             0.4293789863586426,
#             0.8270569443702698
#         ],
#         [
#             0.7411866188049316,
#             0.2709406912326813,
#             0.825921893119812
#         ],
#         [
#             0.9160959124565125,
#             0.46751484274864197,
#             0.8430129885673523
#         ],
#         [
#             0.922856330871582,
#             0.26678192615509033,
#             0.6624086499214172
#         ]
#     ]
    
#     # X_incorrect = np.array([extract_features(k) for k in incorrect_kpts])

#     input_dim = X_correct[1]

#     # load model + scaler
#     model, mean, std = load_mlp_from_npz(
#         "models/ver3/movement_1_mlp_scratch_ver3.npz",
#         "models/ver3/movement_1_scaler_scratch_ver3.npz",
#         input_dim
#     )

#     # normalize features with same scaler as training
#     X_correct_norm = (X_correct - mean) / std
#     # X_incorrect_norm = (X_incorrect - mean) / std

#     print("Correct samples:")
#     # for i in range(30):
#         # pred = model.predict(X_correct_norm[i:i+1])[0]
#         # print(f"Correct sample {i}: Pred={pred}")
        
#     pred = model.predict(X_correct_norm)[0]
#     print(f"Correct sample : Pred={pred}")

#     # print("Incorrect samples:")
#     # for i in range(30):
#     #     pred = model.predict(X_incorrect_norm[i:i+1])[0]
#     #     print(f"Incorrect sample {i}: Pred={pred}")

import numpy as np
from train_mlp_classifier import (
    load_keypoints_dict,
    extract_features,
    MLPBinaryClassifier
)

def load_mlp_from_npz(model_path, scaler_path, input_dim, hidden_dims=(64,32)):
    data = np.load(model_path)
    scaler = np.load(scaler_path)
    
    mean = scaler["mean"]
    std = scaler["std"]

    # create model object
    model = MLPBinaryClassifier(input_dim=input_dim, hidden_dims=hidden_dims)

    # load weights
    model.W1 = data["W1"]
    model.b1 = data["b1"]
    model.W2 = data["W2"]
    model.b2 = data["b2"]
    model.W3 = data["W3"]
    model.b3 = data["b3"]

    return model, mean, std


# ---------- main ----------
if __name__ == "__main__":

    # Example ONE pose (17 keypoints × [x,y,score])
    # Put your 17 keypoints here:
    keypoints = np.array([
        [
            0.19834792613983154,
            0.5183044672012329,
            0.576542854309082
        ],
        [
            0.18479840457439423,
            0.5374801158905029,
            0.5270482301712036
        ],
        [
            0.1863769292831421,
            0.5015411376953125,
            0.4749681353569031
        ],
        [
            0.2079753279685974,
            0.5591772198677063,
            0.6914658546447754
        ],
        [
            0.20907919108867645,
            0.48289400339126587,
            0.6629880666732788
        ],
        [
            0.29398804903030396,
            0.6074073314666748,
            0.8092374801635742
        ],
        [
            0.28489407896995544,
            0.44266995787620544,
            0.8630126118659973
        ],
        [
            0.4067838788032532,
            0.6468639969825745,
            0.8451031446456909
        ],
        [
            0.19620832800865173,
            0.3802909553050995,
            0.8826054334640503
        ],
        [
            0.4583977460861206,
            0.5426298975944519,
            0.797065019607544
        ],
        [
            0.1432250440120697,
            0.5004735589027405,
            0.6539157629013062
        ],
        [
            0.5301929712295532,
            0.5756518840789795,
            0.8375738859176636
        ],
        [
            0.531144380569458,
            0.4780629277229309,
            0.8366470336914062
        ],
        [
            0.743642270565033,
            0.6070961356163025,
            0.8102280497550964
        ],
        [
            0.7384619116783142,
            0.4514741897583008,
            0.7667038440704346
        ],
        [
            0.9141036868095398,
            0.6243351101875305,
            0.8678454160690308
        ],
        [
            0.912327229976654,
            0.449473112821579,
            0.8879901170730591
        ]
    ])

    # ---------- extract 43 features ----------
    features = extract_features(keypoints)

    input_dim = features.shape[0]    # should be 43

    # ---------- load trained model ----------
    model, mean, std = load_mlp_from_npz(
        "models/movement_4/movement_4_mlp.npz",
        "models/movement_4/movement_4_scaler.npz",
        input_dim
    )

    # ---------- normalize ----------
    features_norm = (features - mean) / std

    # ---------- predict ----------
    pred = model.predict(features_norm.reshape(1, -1))[0]
    print(keypoints)
    print("\n=== Pose Classification Result ===")
    print("Prediction:", "CORRECT" if pred == 1 else "INCORRECT")