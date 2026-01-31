import numpy as np
from tensorflow.keras.models import load_model

# ================= PATHS =================
MODEL_PATH = r"E:\SignLanguageProject\sign_language_lstm.h5"
LABELS_PATH = r"E:\SignLanguageProject\label_classes.npy"

# ================= LOAD MODEL & LABELS =================
model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

print("Model loaded successfully.")
print("Labels:", labels)

# ================= TEST WITH ONE SAMPLE =================
# Pick ANY .npy file from WLASL_20_keypoints manually
# Example path below (change only the file path if needed)

sample_path = r"E:\SignLanguageProject\data\WLASL_20_keypoints\go\24940.npy"

sample = np.load(sample_path)

# Model expects shape: (1, 30, features)
sample = np.expand_dims(sample, axis=0)

# ================= PREDICT =================
predictions = model.predict(sample)
predicted_index = np.argmax(predictions)
predicted_word = labels[predicted_index]

print("\nPredicted index:", predicted_index)
print("Predicted word:", predicted_word)
