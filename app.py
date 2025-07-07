import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
from deteksi_warna import YarnColorDetector
from datetime import datetime
import os

# Inisialisasi detektor
detector = YarnColorDetector("yarn_colors_database.csv")

st.set_page_config(page_title="Yarn Color Detector", layout="centered")
st.title("🎥 Yarn Color Detection - Real-time via Webcam")

# Konfigurasi STUN agar bisa digunakan di Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Area sampling default
area = detector.sample_area

# Kelas untuk proses video dari webcam
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_result = None
        self.last_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.last_frame = img.copy()

        rgb_color, color_name, color_code, confidence = detector.get_dominant_color(img)

        self.last_result = {
            "rgb": rgb_color,
            "color_name": color_name,
            "color_code": color_code,
            "confidence": confidence
        }

        # Gambar area sampling
        cv2.rectangle(
            img,
            (area['x'], area['y']),
            (area['x'] + area['width'], area['y'] + area['height']),
            (0, 255, 0), 2
        )

        # Gambar hasil
        text = f"{color_name} ({color_code}) - {confidence:.1f}%" if rgb_color else "Tidak terdeteksi"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Jalankan kamera
ctx = webrtc_streamer(
    key="yarn-color-detector",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Menampilkan hasil deteksi di UI Streamlit
if ctx.video_processor:
    result = ctx.video_processor.last_result

    if result and result["rgb"]:
        st.markdown("### 🎯 Hasil Deteksi Warna")
        st.write(f"**Nama Warna:** {result['color_name']}")
        st.write(f"**Kode Warna:** {result['color_code']}")
        st.write(f"**RGB:** {result['rgb']}")
        lab = detector.rgb_to_lab(result['rgb'])
        st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
        st.write(f"**Confidence:** {result['confidence']:.1f}%")
        st.markdown(
            f"<div style='width:100px;height:100px;background-color:rgb{result['rgb']};border:1px solid #000'></div>",
            unsafe_allow_html=True
        )

        # Kolom tombol: Screenshot & Simpan Deteksi
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("📸 Ambil Screenshot"):
                if ctx.video_processor and ctx.video_processor.last_frame is not None:
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"results/screenshot_{now}.jpg"
                    os.makedirs("results", exist_ok=True)
                    cv2.imwrite(screenshot_path, ctx.video_processor.last_frame)
                    st.success(f"Screenshot disimpan: `{screenshot_path}`")

        with col2:
            if st.button("💾 Simpan Hasil Deteksi"):
                frame = ctx.video_processor.last_frame
                filepath = detector.save_detection_result(
                    frame,
                    result["color_name"],
                    result["color_code"],
                    result["rgb"],
                    result["confidence"]
                )
                st.success(f"Hasil deteksi disimpan: `{filepath}`")
    else:
        st.info("Tunggu kamera mendeteksi warna...")
else:
    st.warning("Kamera belum aktif atau belum dimuat...")
