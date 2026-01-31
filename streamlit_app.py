import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("🤟 Sign Language Detection Demo")

@st.cache_resource
def load():
    model = load_model("sign_language_lstm.h5")
    labels = np.load("label_classes.npy")
    return model, labels

model, labels = load()

file = st.file_uploader("Upload .npy file", type=["npy"])

if file:
    data = np.load(file)
    data = np.expand_dims(data, axis=0)
    pred = model.predict(data)
    st.success(f"Prediction: {labels[np.argmax(pred)]}")
