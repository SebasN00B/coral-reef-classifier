import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Coral Reef Health Classifier",
    page_icon="🪸",
    layout="centered"
)

# Title
st.title("🪸 Coral Reef Health Classifier")
st.markdown("Upload an underwater coral image to classify its health state.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ConvNeXtTiny_final.keras")
    return model

model = load_model()

# Class labels
CLASSES = ["Bleached", "Dead", "Healthy"]
CLASS_COLORS = {"Healthy": "🟢", "Bleached": "🟡", "Dead": "🔴"}

# Upload image
uploaded_file = st.file_uploader("Choose a coral image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
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

    # All probabilities
    st.markdown("**Class Probabilities:**")
    for i, cls in enumerate(CLASSES):
        prob = float(predictions[0][i]) * 100
        st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")