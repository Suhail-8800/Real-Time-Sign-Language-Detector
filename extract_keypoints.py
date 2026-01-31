import os
import cv2
import numpy as np
import mediapipe as mp


# ================= PATHS =================
FRAMES_ROOT = r"E:\SignLanguageProject\data\WLASL_20_frames"
KEYPOINTS_ROOT = r"E:\SignLanguageProject\data\WLASL_20_keypoints"

NUM_FRAMES = 30


# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic.Holistic


# ================= LANDMARK SIZES =================
POSE_LANDMARKS = 12 * 4    # upper body only
HAND_LANDMARKS = 21 * 3    # one hand

FEATURES_PER_FRAME = POSE_LANDMARKS + 2 * HAND_LANDMARKS

# ================= FUNCTIONS =================
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

# ================= MAIN =================
with mp_holistic(static_image_mode=True) as holistic_model:

    for label in os.listdir(FRAMES_ROOT):
        label_path = os.path.join(FRAMES_ROOT, label)
        out_label_path = os.path.join(KEYPOINTS_ROOT, label)
        os.makedirs(out_label_path, exist_ok=True)

        for video in os.listdir(label_path):
            video_path = os.path.join(label_path, video)
            frames = sorted(os.listdir(video_path))[:NUM_FRAMES]

            sequence = []

            for frame_name in frames:
                frame_path = os.path.join(video_path, frame_name)
                image = cv2.imread(frame_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = holistic_model.process(image)
                keypoints = extract_landmarks(results)
                sequence.append(keypoints)

            while len(sequence) < NUM_FRAMES:
                sequence.append(np.zeros(FEATURES_PER_FRAME))

            sequence = np.array(sequence)
            np.save(os.path.join(out_label_path, video + ".npy"), sequence)

            print(f"[OK] {label}/{video}")

print("\n✅ Keypoint extraction completed.")
