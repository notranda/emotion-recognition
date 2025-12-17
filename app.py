import cv2
import numpy as np
import streamlit as st
from mlops.infer import predict

# --- Set up the page config ---
st.set_page_config(page_title="Emotion Recognition", layout="centered", initial_sidebar_state="collapsed")

# --- Custom CSS for modern look ---
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .container {
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            padding: 15px 32px;
            border-radius: 12px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .sidebar .sidebar-content {
            background-color: #f1f1f1;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title and description ---
st.markdown('<div class="title">Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown("""
    **Upload an image of a face**, and the model will predict the emotion displayed on the face.
    You can choose between two models: **CNN** or **ResNet50**.
""")

# --- Model selection ---
model_key = st.selectbox("Choose a Model", ["cnn", "resnet50"], index=0, help="Select the model to use for emotion recognition.")

# --- File upload section ---
uploaded = st.file_uploader("Upload a face image (jpg / png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # --- Read and process the image ---
    img_bytes = uploaded.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        st.error("Failed to read image.")
    else:
        # --- Convert to RGB if grayscale ---
        if len(frame_bgr.shape) == 2:  # grayscale
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)  # Convert to RGB

        # --- Display the uploaded image ---
        st.image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            caption="Input Image (RGB)",
            use_column_width=True
        )

        # --- Predict and show results ---
        with st.spinner('Processing...'):
            label, conf, probs, meta = predict(frame_bgr, model_key=model_key)

        # --- Display prediction results ---
        st.markdown("### Prediction Result")
        st.write(f"**Predicted Emotion:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2f}`")

        # --- Display model info ---
        st.markdown("### Model Info")
        st.caption(
            f"Model: {model_key.upper()} | "
            f"Version: {meta.get('version')} | "
            f"Input Size: {meta.get('img_size')} | "
            f"Color Mode: {meta.get('color_mode')}"
        )

        # --- Display all class probabilities ---
        with st.expander("Probability Details"):
            LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            for i, p in enumerate(probs):
                st.write(f"{LABELS[i]}: {p:.3f}")

# --- Additional Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: #fafafa;
        }
    </style>
""", unsafe_allow_html=True)
