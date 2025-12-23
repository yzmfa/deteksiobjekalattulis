import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Deteksi Alat Tulis - YOLO",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Deteksi Objek Alat Tulis")
st.write("Aplikasi deteksi objek menggunakan **YOLO** dengan model `best.pt`")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
source_option = st.sidebar.radio(
    "Pilih Sumber Input",
    ("Image", "Video", "Kamera")
)

# =========================
# FUNGSI DETEKSI
# =========================
def detect_image(image):
    results = model.predict(image, conf=confidence)
    annotated = results[0].plot()
    return annotated

# =========================
# IMAGE DETECTION
# =========================
if source_option == "Image":
    st.subheader("🖼️ Deteksi dari Gambar")

    uploaded_image = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)

        with col2:
            result_img = detect_image(img_array)
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

# =========================
# VIDEO DETECTION
# =========================
elif source_option == "Video":
    st.subheader("🎥 Deteksi dari Video")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
        os.remove(temp_video.name)

# =========================
# WEBCAM DETECTION
# =========================
elif source_option == "Kamera":
    st.subheader("📷 Deteksi dari Kamera (Webcam)")

    run = st.checkbox("▶️ Jalankan Kamera")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak dapat mengakses kamera")
            break

        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()

        stframe.image(
            annotated_frame,
            channels="BGR",
            use_container_width=True
        )

    cap.release()
