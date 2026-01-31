import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model

# ================= PATHS =================
MODEL_PATH = r"E:\SignLanguageProject\sign_language_lstm.h5"
LABEL_PATH = r"E:\SignLanguageProject\label_classes.npy"

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
labels = np.load(LABEL_PATH)

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ================= PARAMETERS =================
NUM_FRAMES = 30
PREDICTION_WINDOW = 7   # smoothing window

# ================= STORAGE =================
sequence = deque(maxlen=NUM_FRAMES)
predictions = deque(maxlen=PREDICTION_WINDOW)
sentence = []

# ================= LANDMARK SETTINGS =================
POSE_LANDMARKS = 12 * 4
HAND_LANDMARKS = 21 * 3
FEATURES_PER_FRAME = POSE_LANDMARKS + 2 * HAND_LANDMARKS

def extract_landmarks(results):
    pose = np.zeros(POSE_LANDMARKS)
    lh = np.zeros(HAND_LANDMARKS)
    rh = np.zeros(HAND_LANDMARKS)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark[:12]
        pose = np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks]).flatten()

    if results.left_hand_landmarks:
        lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()

    if results.right_hand_landmarks:
        rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, lh, rh])

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


WINDOW_NAME = "Sign Language Live Demo"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)



cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(static_image_mode=False) as holistic:

    print("🎥 Live demo started. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        keypoints = extract_landmarks(results)
        sequence.append(keypoints)

        # Only predict when we have full sequence
        if len(sequence) == NUM_FRAMES:
            input_data = np.expand_dims(sequence, axis=0)
            preds = model.predict(input_data, verbose=0)
            pred_index = np.argmax(preds)
            pred_word = labels[pred_index]

            predictions.append(pred_word)

            # Smooth prediction
            if len(predictions) == PREDICTION_WINDOW:
                most_common = Counter(predictions).most_common(1)[0][0]

                if len(sentence) == 0 or most_common != sentence[-1]:
                    sentence.append(most_common)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display text
        h, w, _ = frame.shape

        # Top banner
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)

        refined_sentence = refine_sentence(sentence[-5:])

        cv2.putText(
            frame,
            "Sentence: " + refined_sentence,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

       
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
