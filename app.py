import streamlit as st
import cv2
import numpy as np
from deteksi_warna import YarnColorDetector
from PIL import Image
from datetime import datetime
import os

st.set_page_config(page_title="Deteksi Warna Manual", layout="centered")
st.title("📷 Deteksi Warna Benang (Snapshot Manual)")

detector = YarnColorDetector("yarn_colors_database.csv")
capture = st.button("📸 Ambil Gambar dari Kamera")

frame = None

if capture:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.flip(frame, 1)
        rgb, nama, kode, confidence = detector.get_dominant_color(frame)
        lab = detector.rgb_to_lab(rgb)

        st.image(frame, caption="📷 Snapshot", channels="BGR", use_column_width=True)

        st.markdown("### 🎯 Hasil Deteksi Warna")
        st.write(f"**Nama Warna:** {nama}")
        st.write(f"**Kode Warna:** {kode}")
        st.write(f"**RGB:** {rgb}")
        st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        st.markdown(
            f"<div style='width:100px;height:100px;background-color:rgb{rgb};border:1px solid #000'></div>",
            unsafe_allow_html=True
        )

        # Tombol Simpan
        if st.button("💾 Simpan Hasil"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("results", exist_ok=True)
            filename = f"results/detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            detector.save_detection_result(frame, nama, kode, rgb, confidence)

            st.success(f"Hasil disimpan ke: `{filename}`")
    else:
        st.error("Gagal mengambil gambar dari kamera.")
else:
    st.info("Klik tombol di atas untuk mulai ambil gambar dari webcam.")
