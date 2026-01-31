import json
import os
import shutil


WLASL_JSON = r"E:\SignLanguageProject\data\WLASL_original\WLASL_v0.3.json"
VIDEOS_SRC = r"E:\SignLanguageProject\data\WLASL_original\videos"
DEST_ROOT = r"E:\SignLanguageProject\data\WLASL_20"

GLOSS_MAP = {
    "i": "i",
    "you": "you",
    "we": "we",
    "go": "go",
    "come": "come",
    "want": "want",
    "need": "need",
    "help": "help",
    "like": "like",
    "yes": "yes",
    "no": "no",
    "what": "what",
    "who": "who",
    "where": "where",
    "why": "why",
    "home": "home",
    "food": "food",
    "water": "water",
    "stop": "stop",
    "thank you": "thank_you"
}

with open(WLASL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== COPYING WLASL_20 DATASET ===\n")

count = 0

for entry in data:
    gloss = entry["gloss"]
    if gloss not in GLOSS_MAP:
        continue

    dest_folder = os.path.join(DEST_ROOT, GLOSS_MAP[gloss])

    for inst in entry["instances"]:
        video_id = inst["video_id"]
        src_file = os.path.join(VIDEOS_SRC, f"{video_id}.mp4")
        dst_file = os.path.join(dest_folder, f"{video_id}.mp4")

        if os.path.exists(src_file):
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"[COPIED] {src_file} -> {dst_file}")
                count += 1


print(f"\nTotal videos copied: {count}")

