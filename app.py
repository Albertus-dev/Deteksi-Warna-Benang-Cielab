import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
import cv2
from datetime import datetime
from deteksi_warna import YarnColorDetector
import os
from PIL import Image
import csv

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Warna Benang", layout="wide")

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .color-display {
        border: 2px solid #000;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    .control-panel {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        flex: 1;
        margin: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🎥 Deteksi Warna Benang dengan CIELAB</h1><p>Sistem deteksi warna benang menggunakan teknologi computer vision dan database warna lengkap</p></div>', unsafe_allow_html=True)

# Load detector
@st.cache_resource
def load_detector():
    return YarnColorDetector("yarn_colors_database.csv")

detector = load_detector()

# Konfigurasi STUN untuk Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Initialize session state
if "hasil_deteksi" not in st.session_state:
    st.session_state.hasil_deteksi = None
if "frame_terakhir" not in st.session_state:
    st.session_state.frame_terakhir = None
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "sampling_area" not in st.session_state:
    st.session_state.sampling_area = detector.sample_area.copy()
if "filter_settings" not in st.session_state:
    st.session_state.filter_settings = {
        "brightness_min": detector.brightness_min,
        "brightness_max": detector.brightness_max,
        "saturation_min": detector.saturation_min
    }

# Sidebar untuk kontrol
with st.sidebar:
    st.header("⚙️ Kontrol & Pengaturan")
    
    # Area sampling controls
    st.subheader("📏 Area Sampling")
    col1, col2 = st.columns(2)
    with col1:
        new_width = st.slider("Lebar", 50, 200, st.session_state.sampling_area['width'], 10)
        new_x = st.slider("Posisi X", 0, 640-new_width, st.session_state.sampling_area['x'], 10)
    with col2:
        new_height = st.slider("Tinggi", 50, 200, st.session_state.sampling_area['height'], 10)
        new_y = st.slider("Posisi Y", 0, 480-new_height, st.session_state.sampling_area['y'], 10)
    
    # Update sampling area
    st.session_state.sampling_area = {
        'x': new_x, 'y': new_y, 'width': new_width, 'height': new_height
    }
    detector.sample_area = st.session_state.sampling_area
    
    if st.button("🎯 Reset ke Tengah"):
        st.session_state.sampling_area = {
            'x': (640 - 100) // 2,
            'y': (480 - 100) // 2,
            'width': 100,
            'height': 100
        }
        detector.sample_area = st.session_state.sampling_area
        st.rerun()
    
    # Filter settings
    st.subheader("🎛️ Filter Warna")
    brightness_min = st.slider("Brightness Min", 0, 100, st.session_state.filter_settings['brightness_min'])
    brightness_max = st.slider("Brightness Max", 100, 255, st.session_state.filter_settings['brightness_max'])
    saturation_min = st.slider("Saturation Min", 0, 50, st.session_state.filter_settings['saturation_min'])
    
    # Update filter settings
    st.session_state.filter_settings = {
        "brightness_min": brightness_min,
        "brightness_max": brightness_max,
        "saturation_min": saturation_min
    }
    detector.brightness_min = brightness_min
    detector.brightness_max = brightness_max
    detector.saturation_min = saturation_min
    
    # Database info
    st.subheader("📊 Database Info")
    st.info(f"Total warna: {len(detector.color_db)}")
    
    # Categories
    categories = {}
    for color in detector.color_db:
        cat = color['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    st.write("**Kategori Warna:**")
    for cat, count in categories.items():
        st.write(f"• {cat}: {count} warna")

# Video processor class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None
        self.frame = None
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame = img.copy()

        # Update detector settings
        detector.sample_area = st.session_state.sampling_area
        detector.brightness_min = st.session_state.filter_settings['brightness_min']
        detector.brightness_max = st.session_state.filter_settings['brightness_max']
        detector.saturation_min = st.session_state.filter_settings['saturation_min']

        # Detect color
        rgb, nama, kode, confidence = detector.get_dominant_color(img)

        # Save detection result
        if rgb is not None:
            st.session_state.hasil_deteksi = {
                "rgb": rgb,
                "nama": nama,
                "kode": kode,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            st.session_state.frame_terakhir = self.frame.copy()
            
            # Add to history (limit to last 10)
            if len(st.session_state.detection_history) >= 10:
                st.session_state.detection_history.pop(0)
            st.session_state.detection_history.append(st.session_state.hasil_deteksi.copy())

        # Draw interface on video
        self.draw_enhanced_interface(img, rgb, nama, kode, confidence)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def draw_enhanced_interface(self, frame, rgb_color, color_name, color_code, confidence):
        """Enhanced interface drawing with more information"""
        # Background untuk informasi
        info_height = 160
        cv2.rectangle(frame, (10, 10), (500, 10 + info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, 10 + info_height), (255, 255, 255), 2)
        
        if rgb_color:
            # Hitung LAB values
            lab = detector.rgb_to_lab(rgb_color)
            
            # Informasi teks
            info_lines = [
                f"Warna: {color_name}",
                f"Kode: {color_code if color_code else 'N/A'}",
                f"RGB: ({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})",
                f"LAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}",
                f"Confidence: {confidence:.1f}%",
                f"Area: {detector.sample_area['width']}x{detector.sample_area['height']}",
                f"Filter: B({detector.brightness_min}-{detector.brightness_max}) S({detector.saturation_min})"
            ]
            
            # Gambar teks
            for i, text in enumerate(info_lines):
                y_pos = 30 + i * 20
                cv2.putText(frame, text, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Color box
            color_box_bgr = (rgb_color[2], rgb_color[1], rgb_color[0])  # RGB to BGR
            cv2.rectangle(frame, (510, 10), (610, 110), color_box_bgr, -1)
            cv2.rectangle(frame, (510, 10), (610, 110), (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((confidence / 100) * 100)
            cv2.rectangle(frame, (510, 120), (510 + bar_width, 140), (0, 255, 0), -1)
            cv2.rectangle(frame, (510, 120), (610, 140), (255, 255, 255), 2)
            cv2.putText(frame, f"{confidence:.0f}%", (620, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Tidak ada warna terdeteksi", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sampling area with enhanced visualization
        area = detector.sample_area
        # Main rectangle
        cv2.rectangle(frame, (area['x'], area['y']), 
                     (area['x'] + area['width'], area['y'] + area['height']), 
                     (0, 255, 0), 2)
        
        # Corner markers
        corner_size = 10
        corners = [
            (area['x'], area['y']),
            (area['x'] + area['width'], area['y']),
            (area['x'], area['y'] + area['height']),
            (area['x'] + area['width'], area['y'] + area['height'])
        ]
        
        for corner in corners:
            cv2.circle(frame, corner, corner_size, (0, 255, 0), -1)
        
        # Center cross
        center_x = area['x'] + area['width'] // 2
        center_y = area['y'] + area['height'] // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)
        
        # Instructions
        instructions = [
            "Letakkan benang di area hijau untuk deteksi",
            "Gunakan sidebar untuk mengatur area dan filter"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 30 + i * 15
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Detection")
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="yarn-color-detector",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🎯 Hasil Deteksi")
    
    # Current detection results
    hasil = st.session_state.get("hasil_deteksi", None)
    
    if hasil and hasil["rgb"] is not None:
        # Color display
        rgb_str = f"rgb({hasil['rgb'][0]}, {hasil['rgb'][1]}, {hasil['rgb'][2]})"
        st.markdown(f"""
        <div class="color-display" style="background-color: {rgb_str}; height: 100px; margin: 10px 0;">
            <h3 style="margin: 0; padding-top: 35px;">{hasil['nama']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed information
        with st.container():
            st.write("**📋 Detail Informasi:**")
            st.write(f"🏷️ **Nama:** {hasil['nama']}")
            st.write(f"🎨 **Kode:** {hasil['kode']}")
            st.write(f"🔴 **RGB:** {hasil['rgb']}")
            
            lab = detector.rgb_to_lab(hasil["rgb"])
            st.write(f"🔬 **CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
            st.write(f"📊 **Confidence:** {hasil['confidence']:.1f}%")
            
            # Confidence bar
            confidence_pct = hasil['confidence'] / 100
            st.progress(confidence_pct)
            
            # Timestamp
            if 'timestamp' in hasil:
                st.write(f"🕐 **Waktu:** {hasil['timestamp'].strftime('%H:%M:%S')}")
        
        # Action buttons
        st.subheader("🔧 Aksi")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📸 Screenshot", use_container_width=True):
                if st.session_state.frame_terakhir is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("results", exist_ok=True)
                    filename = f"results/screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, st.session_state.frame_terakhir)
                    
                    # Display screenshot
                    img_rgb = cv2.cvtColor(st.session_state.frame_terakhir, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Screenshot berhasil diambil", use_column_width=True)
                    st.success(f"Screenshot disimpan: {filename}")
        
        with col_btn2:
            if st.button("💾 Simpan Hasil", use_container_width=True):
                if st.session_state.frame_terakhir is not None:
                    filepath = detector.save_detection_result(
                        st.session_state.frame_terakhir,
                        hasil["nama"],
                        hasil["kode"],
                        hasil["rgb"],
                        hasil["confidence"]
                    )
                    st.success(f"Hasil disimpan: {filepath}")
    
    else:
        st.info("⏳ Menunggu deteksi warna...")
        st.markdown("""
        <div class="info-box">
            <h4>📋 Petunjuk Penggunaan:</h4>
            <ul>
                <li>Pastikan webcam aktif</li>
                <li>Letakkan benang di area hijau</li>
                <li>Tunggu hingga warna terdeteksi</li>
                <li>Gunakan sidebar untuk mengatur area dan filter</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Detection history
if st.session_state.detection_history:
    st.subheader("📈 Riwayat Deteksi")
    
    # Show last 5 detections
    with st.expander("Lihat Riwayat", expanded=False):
        for i, detection in enumerate(reversed(st.session_state.detection_history[-5:])):
            with st.container():
                col_hist1, col_hist2, col_hist3 = st.columns([1, 2, 1])
                
                with col_hist1:
                    rgb_str = f"rgb({detection['rgb'][0]}, {detection['rgb'][1]}, {detection['rgb'][2]})"
                    st.markdown(f"""
                    <div style="background-color: {rgb_str}; height: 40px; border-radius: 5px; border: 1px solid #000;"></div>
                    """, unsafe_allow_html=True)
                
                with col_hist2:
                    st.write(f"**{detection['nama']}**")
                    st.write(f"Confidence: {detection['confidence']:.1f}%")
                
                with col_hist3:
                    if 'timestamp' in detection:
                        st.write(detection['timestamp'].strftime('%H:%M:%S'))
                
                st.divider()

# Statistics
st.subheader("📊 Statistik")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

with col_stat1:
    st.metric("Total Warna DB", len(detector.color_db))

with col_stat2:
    st.metric("Deteksi Hari Ini", len(st.session_state.detection_history))

with col_stat3:
    if hasil and hasil["rgb"] is not None:
        st.metric("Confidence Saat Ini", f"{hasil['confidence']:.1f}%")
    else:
        st.metric("Confidence Saat Ini", "0%")

with col_stat4:
    area = st.session_state.sampling_area
    st.metric("Area Sampling", f"{area['width']}×{area['height']}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎨 Yarn Color Detector with CIELAB Technology</p>
    <p>Menggunakan computer vision dan database warna lengkap untuk deteksi warna benang yang akurat</p>
</div>
""", unsafe_allow_html=True)