# 🖐️ Real-Time Sign Language Recognition & Sentence Formation

A real-time **Sign Language Recognition system** that converts hand gestures into meaningful words and sentences using deep learning.

The system uses **MediaPipe for landmark extraction** and an **LSTM neural network** to learn temporal gesture patterns from live webcam input. The application runs directly from the terminal and performs live gesture detection, word prediction, and sentence formation in real time.

---
<p align="center">
  <img width="40%" alt="Screenshot 2026-02-01 152628" src="https://github.com/user-attachments/assets/9581e290-31cb-4086-8384-875ffeb0d581"/>
  <img width="40%" alt="Screenshot 2026-02-01 152729" src="https://github.com/user-attachments/assets/6aa7cc2f-fdf9-4b2c-989d-510ab886b043"  />
</p>



🚀 Features

✅ Real-time sign recognition using webcam 

✅ Hand & body landmark extraction using MediaPipe 

✅ LSTM-based temporal gesture learning 

✅ Sentence formation from continuous predictions

✅ Prediction smoothing for stable output 

✅ Lightweight CPU-based inference (no GPU required)

✅ Live webcam demo for real-time translation

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Computer Vision | OpenCV |
| Landmark Detection | MediaPipe Holistic |
| Data Handling | NumPy, scikit-learn |
| IDE / Editor | VS Code |
| Environment | Anaconda (Python 3.10) |
| Dataset | WLASL (subset) |


## 📁 Project Structure

```
SignLanguageProject/
│
├── data/
│   ├── WLASL_original/
│   ├── WLASL_20/
│   ├── WLASL_20_frames/
│   └── WLASL_20_keypoints/
│
├── copy_wlasl_20.py          # Dataset filtering
├── extract_frames.py         # Frame extraction
├── extract_keypoints.py      # MediaPipe keypoints extraction
├── prepare_dataset.py        # Dataset preparation
├── train_lstm.py             # LSTM model training
├── predict_word.py           # Single word prediction
├── predict_sequence.py       # Sentence formation testing
├── live_demo.py              # Real-time webcam demo
│
├── sign_language_lstm.h5     # Trained model
├── label_classes.npy         # Label mapping
└── README.md
```



🧩 How to Run the Project
Prerequisites
Python (Anaconda recommended)
Webcam
Installed libraries:
TensorFlow
MediaPipe
OpenCV
NumPy
scikit-learn

Step 1 — Activate Environment

Open Anaconda Prompt: conda activate signlang

Step 2 — Run Live Demo
python live_demo.py

A webcam window will open, and gestures will be translated into words and sentences.

Press Q to exit.

🖥️ Example Output
Sentence: HELP NO HELP

Displayed live while performing gestures.

🧠 Learning Highlights

Built complete ML pipeline from raw videos to live inference

Applied temporal deep learning using LSTM networks

Implemented real-time gesture recognition

Developed sentence formation from predictions

Worked with MediaPipe for landmark detection

Designed CPU-efficient inference pipeline

🎓 Academic Contribution

This project demonstrates how deep learning and computer vision can be combined to build real-time assistive communication systems for the hearing-impaired community.

It serves as a practical example of human-computer interaction using AI.

🧑‍💻 Author

Suhail Rajput
📧 suhailrajput325@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/suhail-rajput-64158722b/

💻 GitHub: https://github.com/Suhail-8800

🔮 Future Enhancements

Increase vocabulary size using larger datasets

Add grammar correction using NLP

Deploy as web or mobile application

Add multilingual text output

Improve accuracy using BiLSTM or attention models

Add speech output for translated sentences

💡 Inspiration

This project was developed as a capstone project to explore how AI can help bridge communication gaps using real-time gesture recognition and deep learning.

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub.
Your support motivates further open-source development!
