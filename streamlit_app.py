import streamlit as st
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Sign Language Detector")

st.title("🤟 Sign Language Detection Demo")
st.write("Upload a `.npy` keypoints file to test the model")

MODEL_URL = "https://drive.google.com/uc?id=1hOAGUc1DO1gr5ENBuD5qdisgI5Ip9IMN"
MODEL_PATH = "sign_language_lstm.h5"

@st.cache_resource
def load():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)
    labels = np.load("label_classes.npy")
    return model, labels

model, labels = load()

uploaded_file = st.file_uploader("Upload .npy file", type=["npy"])

if uploaded_file is not None:
    data = np.load(uploaded_file)
    data = np.expand_dims(data, axis=0)
    preds = model.predict(data)
    st.success(f"Prediction: {labels[np.argmax(preds)]}")
