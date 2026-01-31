import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

# ================= PATHS =================
MODEL_PATH = r"E:\SignLanguageProject\sign_language_lstm.h5"
LABEL_PATH = r"E:\SignLanguageProject\label_classes.npy"

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
labels = np.load(LABEL_PATH)

print("Model & labels loaded.")
print("Labels:", labels)

# ================= FUNCTIONS =================

def predict_word(sequence):
    """
    sequence: np.array of shape (30, features)
    """
    sequence = np.expand_dims(sequence, axis=0)
    preds = model.predict(sequence, verbose=0)
    return labels[np.argmax(preds)]

def smooth_predictions(pred_list, window=5):
    """
    Majority voting over a sliding window
    """
    smoothed = []
    for i in range(len(pred_list)):
        start = max(0, i - window + 1)
        window_preds = pred_list[start:i+1]
        most_common = Counter(window_preds).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

def build_sentence(predictions):
    """
    Remove consecutive duplicates
    """
    sentence = []
    prev = None
    for word in predictions:
        if word != prev:
            sentence.append(word)
            prev = word
    return " ".join(sentence)


# def refine_sentence(sentence):
#     """
#     Convert sign-language gloss into readable English
#     """
#     s = sentence.lower()

#     rules = {
#         "no like": "I don't like",
#         "no go": "I cannot go",
#         "me want": "I want",
#         "no understand": "I don't understand",
#         "help": "I need help",
#         "go home": "I want to go home",
#         "me go": "I am going",
#     }

#     for k, v in rules.items():
#         s = s.replace(k, v)

#     # Capitalize and punctuate
#     s = s.capitalize()
#     if not s.endswith("."):
#         s += "."

#     return s


def refine_sentence(words):
    """
    Convert sign-language gloss into natural English
    """
    words = [w.lower() for w in words]
    s = " ".join(words)

    # -------- INTENT RULES -------- #

    # dislike + inability + help
    if "no" in words and "like" in words and "help" in words:
        return "I don’t like this, and I need help."

    # cannot go
    if "no" in words and "go" in words:
        return "I can’t go."

    # want something
    if "me" in words and "want" in words:
        obj = [w for w in words if w not in ["me", "want"]]
        if obj:
            return f"I want {' '.join(obj)}."
        return "I want something."

    # go home
    if "go" in words and "home" in words:
        return "I want to go home."

    # confusion
    if "no" in words and "understand" in words:
        return "Sorry, I don’t understand."

    # help only
    if words == ["help"]:
        return "I need help."

    # -------- FALLBACK -------- #
    sentence = " ".join(words).capitalize()
    if not sentence.endswith("."):
        sentence += "."

    return sentence


# ================= TEST WITH MULTIPLE FILES =================
# Pick multiple .npy files manually (can be same or different words)

sample_files = [
    r"E:\SignLanguageProject\data\WLASL_20_keypoints\go\24940.npy",
    r"E:\SignLanguageProject\data\WLASL_20_keypoints\go\24941.npy",
    r"E:\SignLanguageProject\data\WLASL_20_keypoints\home\27765.npy",
]

raw_predictions = []

for path in sample_files:
    seq = np.load(path)
    word = predict_word(seq)
    raw_predictions.append(word)

print("\nRaw predictions:")
print(raw_predictions)

smoothed = smooth_predictions(raw_predictions)

print("\nSmoothed predictions:")
print(smoothed)

# sentence = build_sentence(smoothed)

# print("\nFinal sentence:")
# print(sentence)

sentence = build_sentence(smoothed)
refined = refine_sentence(sentence)

print("\nFinal sentence:")
print(refined)

