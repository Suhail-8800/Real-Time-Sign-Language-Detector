import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= PATH =================
KEYPOINTS_ROOT = r"E:\SignLanguageProject\data\WLASL_20_keypoints"

# ================= LOAD DATA =================
X = []
y = []

labels = sorted(os.listdir(KEYPOINTS_ROOT))

print("Labels found:", labels)

for label in labels:
    label_path = os.path.join(KEYPOINTS_ROOT, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            file_path = os.path.join(label_path, file)
            data = np.load(file_path)

            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ================= ENCODE LABELS =================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Encoded labels:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# ================= TRAIN / TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("\nTrain set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# ================= SAVE DATA =================
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("label_classes.npy", label_encoder.classes_)

print("\n✅ Dataset preparation completed.")
