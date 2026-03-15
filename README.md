# Sign Language Detection System Using Deep Learning

This project implements a **real-time Sign Language Detection System** using **MediaPipe, Deep Learning, and an LSTM neural network**.
The system detects hand gestures from video input and translates them into meaningful words.

The goal of the project is to help **bridge the communication gap between hearing-impaired individuals and people who do not understand sign language**.

---

# Project Overview

Sign language is widely used by deaf and mute individuals, but many people are not familiar with it.
This project uses **computer vision and deep learning techniques** to recognize sign language gestures automatically.

The system processes sign language videos, extracts **hand landmarks using MediaPipe**, and trains a **Long Short-Term Memory (LSTM) model** to recognize gestures from sequences of frames.

The trained model can then be used for **real-time gesture detection using a webcam**.

---

# Key Features

* Real-time **webcam-based sign detection**
* **MediaPipe hand landmark detection**
* **LSTM deep learning model** for gesture recognition
* **Sequence-based gesture prediction**
* Support for **20 sign language words**
* Modular pipeline for **dataset preparation, training, and prediction**
* Real-time gesture prediction interface

---

# Dataset

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

Dataset processing pipeline:

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

# Project Structure

```
SignLanguageProject/

data/
 ├── WLASL_20
 ├── WLASL_20_frames
 ├── WLASL_20_keypoints
 └── WLASL_original

copy_wlasl_20.py
extract_frames.py
extract_keypoints.py
prepare_dataset.py
train_lstm.py

predict_word.py
predict_sequence.py
live_demo.py
test_webcam.py

requirements.txt
README.md
```

---

# Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MediaPipe**
* **OpenCV**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**

---

# Model Architecture

The gesture recognition system uses a **Long Short-Term Memory (LSTM) neural network**.

Why LSTM?

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

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

### 1. Activate environment (if using Conda)

```
conda activate signlang
```

### 2. Run the real-time detection system

```
python live_demo.py
```

This will open the webcam and start detecting sign language gestures.

---

# Training the Model

If you want to train the model from scratch:

### Step 1 – Extract frames from videos

```
python extract_frames.py
```

### Step 2 – Extract hand landmarks

```
python extract_keypoints.py
```

### Step 3 – Prepare sequence dataset

```
python prepare_dataset.py
```

### Step 4 – Train the LSTM model

```
python train_lstm.py
```

---

# Example Prediction

Example output from the system:

```
Input Gesture: HELP
Predicted Word: help
Confidence Score: 0.95
```

---

# Future Improvements

Possible future improvements include:

* Increasing the number of sign language gestures
* Implementing **full sentence translation**
* Recognizing **numeric gestures (0–9)**
* Adding **speech output for predicted gestures**
* Developing a **complete user-friendly application**
* Deploying the system as a **mobile or web application**

---

# Author

**Suhail Rajput**
Computer Science and Engineering
VIT Bhopal University

---

# License

This project is for educational and research purposes.
