import cv2
import numpy as np
import streamlit as st
from mlops.infer import predict

# Set the page configuration
st.set_page_config(page_title="Emotion Recognition", layout="centered", initial_sidebar_state="collapsed")

# Title and description
st.title("Emotion Recognition (Upload Image)")

# Choose model
model_key = st.selectbox("Choose Model", ["cnn", "resnet50"])

# File uploader to upload image
uploaded = st.file_uploader("Upload a face image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Read the image
    img_bytes = uploaded.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        st.error("Failed to read image.")
    else:
        # If grayscale, convert to RGB
        if len(frame_bgr.shape) == 2:  # grayscale
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)  # Convert to RGB

        # Display the uploaded image
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Input Image (RGB)", use_column_width=True)

        # Predict and show results
        with st.spinner('Processing...'):
            label, conf, probs, meta = predict(frame_bgr, model_key=model_key)

        # Display prediction results
        st.markdown("### Prediction Result")
        st.write(f"**Predicted Emotion:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2f}`")

        # Display model info
        st.markdown("### Model Info")
        st.caption(f"Model: {model_key.upper()} | Version: {meta.get('version')} | Input Size: {meta.get('img_size')} | Color Mode: {meta.get('color_mode')}")

        # Show class probabilities (optional)
        with st.expander("Probability Details"):
            LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            for i, p in enumerate(probs):
                st.write(f"{LABELS[i]}: {p:.3f}")
