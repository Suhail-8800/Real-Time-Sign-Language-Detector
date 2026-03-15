# 🖐️ Real-Time Sign Language Recognition & Sentence Formation

A real-time **Sign Language Recognition system** that converts hand gestures into meaningful words and sentences using deep learning.

The system uses **MediaPipe for landmark extraction** and an **LSTM neural network** to learn temporal gesture patterns from live webcam input. The application performs **live gesture detection, word prediction, and sentence formation in real time**, helping bridge the communication gap between hearing-impaired individuals and people who do not understand sign language.

---

## 📌 Project Overview

Sign language is widely used by deaf and mute individuals, but many people are not familiar with it.
This project uses **computer vision and deep learning techniques** to automatically recognize sign language gestures.

The system processes sign language videos, extracts **hand landmarks using MediaPipe**, and trains a **Long Short-Term Memory (LSTM) neural network** to recognize gestures from sequences of frames.

The trained model can then be used for **real-time gesture detection using a webcam**.

---

## 🚀 Features

* Real-time **sign recognition using webcam**
* **MediaPipe hand landmark detection**
* **LSTM-based temporal gesture learning**
* **Sentence formation from continuous predictions**
* Prediction smoothing for stable output
* Lightweight **CPU-based inference**
* Live webcam demo for real-time translation

---

## 📊 Dataset

This project uses a subset of the **WLASL (Word-Level American Sign Language) dataset**.

Selected gesture classes:

```
come
food
go
help
home
i
like
need
no
stop
thank_you
want
water
we
what
where
who
why
yes
you
```

### Dataset Processing Pipeline

```
WLASL Videos
     ↓
Frame Extraction
     ↓
MediaPipe Hand Landmark Detection
     ↓
Keypoint Dataset Creation
     ↓
Sequence Dataset
     ↓
LSTM Model Training
```

---

## 🛠️ Tech Stack

| Component          | Technology                |
| ------------------ | ------------------------- |
| Language           | Python                    |
| Deep Learning      | TensorFlow / Keras (LSTM) |
| Computer Vision    | OpenCV                    |
| Landmark Detection | MediaPipe                 |
| Data Handling      | NumPy, scikit-learn       |
| IDE                | VS Code                   |
| Environment        | Anaconda                  |
| Dataset            | WLASL (subset)            |

---

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
├── copy_wlasl_20.py
├── extract_frames.py
├── extract_keypoints.py
├── prepare_dataset.py
├── train_lstm.py
├── predict_word.py
├── predict_sequence.py
├── live_demo.py
├── test_webcam.py
│
├── sign_language_lstm.h5
├── label_classes.npy
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Suhail-8800/Real-Time-Sign-Language-Detector.git
cd Real-Time-Sign-Language-Detector
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Activate Environment

```
conda activate signlang
```

### Step 2 — Run the Live Demo

```
python live_demo.py
```

A webcam window will open, and gestures will be translated into **words and sentences in real time**.

Press **Q** to exit.

---

## 🧠 Model Architecture

The system uses a **Long Short-Term Memory (LSTM) neural network**.

Sign language gestures are **sequential movements over time**, and LSTM networks are designed to learn patterns from sequential data.

Model workflow:

```
Video Frames
     ↓
MediaPipe Hand Keypoints
     ↓
Sequence Creation
     ↓
LSTM Neural Network
     ↓
Gesture Prediction
```

---

## 🖥️ Example Output

Example prediction during live inference:

```
Sentence: HELP NO HELP
```

The sentence is displayed live while performing gestures.

---

## 🎓 Learning Highlights

* Built a complete ML pipeline from **raw videos to real-time inference**
* Applied **temporal deep learning using LSTM networks**
* Implemented **real-time gesture recognition**
* Developed **sentence formation from continuous predictions**
* Integrated **MediaPipe landmark detection**
* Designed a **CPU-efficient inference pipeline**

---

## 🔮 Future Improvements

* Increase vocabulary using larger datasets
* Implement **full sentence translation**
* Recognize **numeric gestures (0–9)**
* Add **speech output for predicted gestures**
* Deploy as a **web or mobile application**
* Improve accuracy using **BiLSTM or attention models**
* Add multilingual output

---

## 👨‍💻 Author

**Suhail Rajput**
Computer Science and Engineering
VIT Bhopal University

📧 [suhailrajput325@gmail.com](mailto:suhailrajput325@gmail.com)

🔗 LinkedIn
https://www.linkedin.com/in/suhail-rajput-64158722b/

💻 GitHub
https://github.com/Suhail-8800

---

## ⭐ Support

If you like this project, consider giving it a **⭐ on GitHub**.
Your support motivates further open-source development.

---

## 📜 License

This project is for **educational and research purposes**.
