import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from datetime import datetime
from deteksi_warna import YarnColorDetector
import av
import os
import threading

st.set_page_config(page_title="Deteksi Warna Benang", layout="centered")
st.title("🎨 Deteksi Warna Benang Sederhana")

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Load detektor
@st.cache_resource
def load_detector():
    try:
        detector = YarnColorDetector("yarn_colors_database.csv")
        st.success(f"Database warna berhasil dimuat: {len(detector.color_db)} warna")
        return detector
    except Exception as e:
        st.error(f"Gagal memuat detector: {e}")
        return None

detector = load_detector()

if detector is None:
    st.stop()

# Thread lock untuk sinkronisasi
lock = threading.Lock()

# Session state untuk menyimpan hasil
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# Video Processor yang lebih robust
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector
        self.frame_count = 0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            # Konversi frame ke numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Flip horizontal untuk mirror effect
            img = cv2.flip(img, 1)
            
            # Proses setiap 5 frame untuk mengurangi beban
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                # Deteksi warna
                rgb_color, color_name, color_code, confidence = self.detector.get_dominant_color(img)
                
                # Update session state dengan thread lock
                with lock:
                    if rgb_color is not None:
                        st.session_state.hasil_deteksi = {
                            "rgb": rgb_color,
                            "nama": color_name,
                            "kode": color_code,
                            "confidence": confidence,
                            "waktu": datetime.now().strftime("%H:%M:%S")
                        }
                        st.session_state.last_frame = img.copy()
            
            # Gambar area sampling
            area = self.detector.sample_area
            h, w = img.shape[:2]
            
            # Pastikan area sampling dalam batas frame
            x = max(0, min(area['x'], w - area['width']))
            y = max(0, min(area['y'], h - area['height']))
            width = min(area['width'], w - x)
            height = min(area['height'], h - y)
            
            # Gambar kotak hijau untuk area sampling
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Tambahkan text info
            cv2.putText(img, "Area Deteksi", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan hasil deteksi di video jika ada
            if st.session_state.hasil_deteksi:
                hasil = st.session_state.hasil_deteksi
                text = f"{hasil['nama']} ({hasil['confidence']:.1f}%)"
                cv2.putText(img, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error dalam video processing: {e}")
            return frame

# Tampilkan instruksi
st.markdown("""
### Instruksi Penggunaan:
1. **Klik tombol "START"** untuk memulai kamera
2. **Arahkan benang** ke dalam kotak hijau di layar
3. **Tunggu beberapa detik** untuk hasil deteksi muncul
4. **Hasil akan tampil** di bagian bawah secara otomatis

---
""")

# Kolom untuk layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Kamera")
    
    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="yarn-color-detector",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("⚙️ Pengaturan")
    
    # Pengaturan area sampling
    if st.checkbox("Sesuaikan Area Sampling"):
        detector.sample_area['x'] = st.slider("X Position", 0, 500, detector.sample_area['x'])
        detector.sample_area['y'] = st.slider("Y Position", 0, 400, detector.sample_area['y'])
        detector.sample_area['width'] = st.slider("Width", 50, 200, detector.sample_area['width'])
        detector.sample_area['height'] = st.slider("Height", 50, 200, detector.sample_area['height'])
    
    # Pengaturan sensitivitas
    if st.checkbox("Pengaturan Lanjutan"):
        detector.brightness_min = st.slider("Brightness Min", 0, 100, detector.brightness_min)
        detector.brightness_max = st.slider("Brightness Max", 100, 255, detector.brightness_max)
        detector.saturation_min = st.slider("Saturation Min", 0, 100, detector.saturation_min)

# Hasil deteksi
st.header("📊 Hasil Deteksi")

# Placeholder untuk hasil
hasil_placeholder = st.empty()

# Update hasil secara real-time
if st.session_state.hasil_deteksi:
    hasil = st.session_state.hasil_deteksi
    
    with hasil_placeholder.container():
        # Tampilkan dalam format yang lebih menarik
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Nama Warna", hasil['nama'])
            st.metric("Confidence", f"{hasil['confidence']:.1f}%")
        
        with col2:
            st.metric("Kode Warna", hasil['kode'])
            st.metric("Waktu Deteksi", hasil['waktu'])
        
        with col3:
            # Tampilkan warna
            rgb = hasil['rgb']
            st.markdown(f"""
            <div style='
                width: 100px; 
                height: 100px; 
                background-color: rgb{rgb}; 
                border: 2px solid #333; 
                border-radius: 10px;
                margin: 10px auto;
            '></div>
            """, unsafe_allow_html=True)
            
            st.write(f"**RGB:** {rgb}")
        
        # Tombol aksi
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💾 Simpan Hasil", key="save_result"):
                if st.session_state.last_frame is not None:
                    try:
                        os.makedirs("results", exist_ok=True)
                        filename = f"results/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, st.session_state.last_frame)
                        st.success(f"✅ Disimpan: {filename}")
                    except Exception as e:
                        st.error(f"❌ Gagal menyimpan: {e}")
        
        with col2:
            if st.button("🔄 Reset", key="reset_result"):
                st.session_state.hasil_deteksi = None
                st.session_state.last_frame = None
                st.experimental_rerun()

else:
    with hasil_placeholder.container():
        st.info("🎯 Arahkan benang ke dalam kotak hijau untuk memulai deteksi...")
        
        # Status debugging
        if st.checkbox("Show Debug Info"):
            st.write("**Debug Information:**")
            st.write(f"- Detector loaded: {detector is not None}")
            st.write(f"- Color database size: {len(detector.color_db) if detector else 0}")
            st.write(f"- Sample area: {detector.sample_area if detector else 'N/A'}")
            st.write(f"- WebRTC context: {ctx.state if 'ctx' in locals() else 'N/A'}")

# Footer
st.markdown("---")
st.markdown("🎨 **Yarn Color Detector** - Deteksi warna benang secara real-time")