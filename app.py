import cv2
import numpy as np
import streamlit as st

from mlops.infer import predict


st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("Emotion Recognition (Upload Image)")

# pilih model
model_key = st.selectbox("Pilih Model", ["cnn", "resnet50"])

uploaded = st.file_uploader(
    "Upload gambar wajah (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # read image
    img_bytes = uploaded.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        st.error("Gagal membaca gambar.")
    else:
        # tampilkan gambar (RGB)
        st.image(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            caption="Gambar input (RGB)",
            use_column_width=True
        )


        # prediksi
        label, conf, probs, meta = predict(frame_bgr, model_key=model_key)

        st.markdown("### Hasil Prediksi")
        st.write(f"**Emosi:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2f}`")

        st.markdown("### Info Model")
        st.caption(
            f"Model: {model_key.upper()} | "
            f"Version: {meta.get('version')} | "
            f"Input: {meta.get('img_size')} | "
            f"Color mode: {meta.get('color_mode')}"
        )

        # optional: tampilkan semua probabilitas
        with st.expander("Detail probabilitas"):
            LABELS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
            for i, p in enumerate(probs):
                st.write(f"{LABELS[i]}: {p:.3f}")

