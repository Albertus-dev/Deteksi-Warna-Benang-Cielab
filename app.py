import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
from datetime import datetime
from deteksi_warna import YarnColorDetector
import os
from PIL import Image

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Warna Benang", layout="centered")
st.title("🎥 Deteksi Warna Benang via Webcam")

# Load detektor
detector = YarnColorDetector("yarn_colors_database.csv")

# Konfigurasi STUN untuk Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# State lokal untuk simpan hasil
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None
if "frame_terakhir" not in st.session_state:
    st.session_state.frame_terakhir = None

area = detector.sample_area

# Video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame = img.copy()

        rgb, nama, kode, confidence = detector.get_dominant_color(img)

        # Simpan hasil deteksi ke session
        st.session_state.hasil_deteksi = {
            "rgb": rgb,
            "nama": nama,
            "kode": kode,
            "confidence": confidence
        }
        st.session_state.frame_terakhir = self.frame.copy()

        # Tampilkan ke video
        cv2.rectangle(
            img,
            (area["x"], area["y"]),
            (area["x"] + area["width"], area["y"] + area["height"]),
            (0, 255, 0), 2
        )
        teks = f"{nama} ({kode}) - {confidence:.1f}%" if rgb else "Tidak terdeteksi"
        cv2.putText(img, teks, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Jalankan Streamlit WebRTC
ctx = webrtc_streamer(
    key="warna-webcam",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# UI hasil deteksi
hasil = st.session_state.get("hasil_deteksi", None)

if hasil and hasil["rgb"] is not None:
    st.markdown("### 🎯 Hasil Deteksi")
    st.write(f"**Nama Warna:** {hasil['nama']}")
    st.write(f"**Kode Warna:** {hasil['kode']}")
    st.write(f"**RGB:** {hasil['rgb']}")
    lab = detector.rgb_to_lab(hasil["rgb"])
    st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
    st.write(f"**Confidence:** {hasil['confidence']:.1f}%")
    st.markdown(
        f"<div style='width:100px;height:100px;background-color:rgb{hasil['rgb']};border:1px solid #000'></div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Ambil Screenshot"):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("results", exist_ok=True)
            path = f"results/screenshot_{now}.jpg"
            cv2.imwrite(path, st.session_state.frame_terakhir)
            st.image(path, caption="Screenshot Berhasil", use_column_width=True)
            st.success(f"Screenshot disimpan di `{path}`")

    with col2:
        if st.button("💾 Simpan Hasil Deteksi"):
            path = detector.save_detection_result(
                st.session_state.frame_terakhir,
                hasil["nama"],
                hasil["kode"],
                hasil["rgb"],
                hasil["confidence"]
            )
            st.success(f"Hasil deteksi disimpan di `{path}`")
else:
    st.info("⏳ Tunggu kamera mendeteksi warna...")
