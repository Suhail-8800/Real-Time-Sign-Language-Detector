import cv2
import os
import numpy as np

VIDEOS_ROOT = r"E:\SignLanguageProject\data\WLASL_20"
FRAMES_ROOT = r"E:\SignLanguageProject\data\WLASL_20_frames"

FRAMES_PER_VIDEO = 30  # fixed-length sequence

os.makedirs(FRAMES_ROOT, exist_ok=True)

for label in os.listdir(VIDEOS_ROOT):
    label_path = os.path.join(VIDEOS_ROOT, label)
    if not os.path.isdir(label_path):
        continue

    output_label_dir = os.path.join(FRAMES_ROOT, label)
    os.makedirs(output_label_dir, exist_ok=True)

    for video_file in os.listdir(label_path):
        if not video_file.endswith(".mp4"):
            continue

        video_id = video_file.replace(".mp4", "")
        video_path = os.path.join(label_path, video_file)

        output_video_dir = os.path.join(output_label_dir, video_id)
        os.makedirs(output_video_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            continue

        frame_indices = np.linspace(
            0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int
        )

        current_frame = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame in frame_indices:
                frame_resized = cv2.resize(frame, (224, 224))
                frame_name = f"frame_{saved_count:02d}.jpg"
                frame_path = os.path.join(output_video_dir, frame_name)
                cv2.imwrite(frame_path, frame_resized)
                saved_count += 1

            current_frame += 1

            if saved_count >= FRAMES_PER_VIDEO:
                break

        cap.release()

        print(f"[DONE] {label}/{video_id} → {saved_count} frames")

print("\nFrame extraction completed.")
