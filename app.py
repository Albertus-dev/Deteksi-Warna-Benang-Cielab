import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
from deteksi_warna import YarnColorDetector
from datetime import datetime
import os
from PIL import Image
import time

# Konfigurasi
st.set_page_config(page_title="🎨 Deteksi Warna Benang", layout="centered")
st.title("🎥 Deteksi Warna Benang Realtime")

# Load detektor warna
detector = YarnColorDetector("yarn_colors_database.csv")
area = detector.sample_area

# Konfigurasi STUN WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Buffer state hasil deteksi
if "result_state" not in st.session_state:
    st.session_state.result_state = None

# Kelas proses frame video
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.last_frame = img.copy()

        rgb_color, color_name, color_code, confidence = detector.get_dominant_color(img)

        # Simpan ke session_state
        st.session_state.result_state = {
            "rgb": rgb_color,
            "color_name": color_name,
            "color_code": color_code,
            "confidence": confidence,
            "frame": img
        }

        # Tambahkan UI pada frame
        cv2.rectangle(img, (area['x'], area['y']), (area['x'] + area['width'], area['y'] + area['height']), (0, 255, 0), 2)
        if rgb_color:
            text = f"{color_name} ({color_code}) — {confidence:.1f}%"
        else:
            text = "Tidak terdeteksi"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Tampilkan stream webcam
ctx = webrtc_streamer(
    key="deteksi-warna",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Delay singkat agar webcam bisa inisialisasi
time.sleep(1)

# Cek hasil dari session_state
result = st.session_state.get("result_state")

if result and result["rgb"] is not None:
    st.markdown("### 🎯 Hasil Deteksi Warna")
    st.write(f"**Nama Warna:** {result['color_name']}")
    st.write(f"**Kode Warna:** {result['color_code']}")
    st.write(f"**RGB:** {result['rgb']}")
    lab = detector.rgb_to_lab(result["rgb"])
    st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
    st.write(f"**Confidence:** {result['confidence']:.1f}%")
    st.markdown(
        f"<div style='width:100px;height:100px;background-color:rgb{result['rgb']};border:1px solid #000'></div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📸 Ambil Screenshot"):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"results/screenshot_{now}.jpg"
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(screenshot_path, result["frame"])
            st.success(f"📸 Screenshot disimpan: `{screenshot_path}`")
            st.image(screenshot_path, caption="Hasil Screenshot", use_column_width=True)

    with col2:
        if st.button("💾 Simpan Hasil Deteksi"):
            filepath = detector.save_detection_result(
                result["frame"],
                result["color_name"],
                result["color_code"],
                result["rgb"],
                result["confidence"]
            )
            st.success(f"✅ Hasil deteksi disimpan: `{filepath}`")
else:
    st.info("⏳ Tunggu kamera mendeteksi warna...")
