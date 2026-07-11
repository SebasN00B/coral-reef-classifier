import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# Page config
st.set_page_config(
    page_title="Coral Reef Health Classifier",
    page_icon="🪸",
    layout="centered"
)

# Title
st.title("🪸 Coral Reef Health Classifier")
st.markdown("Upload an underwater coral image to classify its health state.")
st.markdown("**Model:** ConvNeXtTiny (94% accuracy) | **Dataset:** BHD Corals")

# Model download
MODEL_URL = "https://github.com/SebasN00B/coral-reef-classifier/releases/download/v1.0/ConvNeXtTiny_final.keras"
MODEL_PATH = "ConvNeXtTiny_final.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... (this may take a minute)"):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Class labels
CLASSES = ["Bleached", "Dead", "Healthy"]
CLASS_COLORS = {"Healthy": "🟢", "Bleached": "🟡", "Dead": "🔴"}
CLASS_DESC = {
    "Healthy": "The coral is in good condition with normal coloration and tissue integrity.",
    "Bleached": "The coral has expelled its symbiotic algae and appears white or pale.",
    "Dead": "The coral tissue has died, leaving exposed skeleton colonized by algae."
}

# Upload image
uploaded_file = st.file_uploader("Choose a coral image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

    # Results
    st.markdown("---")
    st.subheader("Classification Result")
    st.markdown(f"### {CLASS_COLORS[predicted_class]} **{predicted_class} Coral**")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.info(CLASS_DESC[predicted_class])

    # All probabilities
    st.markdown("**Class Probabilities:**")
    for i, cls in enumerate(CLASSES):
        prob = float(predictions[0][i]) * 100
        st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

    st.markdown("---")
    st.caption("Sebastian Felipe Caviedes Ortega | Machine Learning Final Project | University of Europe for Applied Sciences")