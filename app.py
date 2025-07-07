import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
from deteksi_warna import YarnColorDetector
import threading
import queue

# Konfigurasi halaman
st.set_page_config(
    page_title="Yarn Color Detector - Realtime",
    page_icon="🎨",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .color-box {
        width: 150px;
        height: 150px;
        border: 3px solid #333;
        border-radius: 10px;
        margin: 10px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        width: 100%;
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Yarn Color Detector - Realtime</h1>
    <p>Deteksi warna benang secara real-time menggunakan CIELAB color space</p>
</div>
""", unsafe_allow_html=True)

# Inisialisasi session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

class StreamlitYarnDetector:
    def __init__(self, csv_file='yarn_colors_database.csv'):
        self.detector = YarnColorDetector(csv_file)
        self.cap = None
        self.is_running = False
        
    def start_camera(self):
        """Memulai kamera"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.is_running = True
            return True
        except Exception as e:
            st.error(f"Gagal mengakses kamera: {e}")
            return False
    
    def stop_camera(self):
        """Menghentikan kamera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame_with_detection(self):
        """Mendapatkan frame dengan deteksi warna"""
        if not self.cap or not self.is_running:
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        # Flip horizontal untuk mirror effect
        frame = cv2.flip(frame, 1)
        
        # Deteksi warna
        rgb_color, color_name, color_code, confidence = self.detector.get_dominant_color(frame)
        
        # Gambar sampling area
        x, y = self.detector.sample_area['x'], self.detector.sample_area['y']
        w, h = self.detector.sample_area['width'], self.detector.sample_area['height']
        
        # Gambar rectangle sampling area
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Sampling Area", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tambahkan info pada frame
        if rgb_color:
            info_text = f"{color_name} ({confidence:.1f}%)"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Konversi BGR ke RGB untuk Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detection_result = {
            'rgb_color': rgb_color,
            'color_name': color_name,
            'color_code': color_code,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        return frame_rgb, detection_result

def camera_thread():
    """Thread untuk menangani kamera"""
    while st.session_state.camera_active and st.session_state.detector:
        frame, result = st.session_state.detector.get_frame_with_detection()
        if frame is not None and result is not None:
            # Simpan ke queue dengan non-blocking
            try:
                st.session_state.frame_queue.put((frame, result), block=False)
            except queue.Full:
                # Jika queue penuh, ambil yang lama dan masukkan yang baru
                try:
                    st.session_state.frame_queue.get(block=False)
                    st.session_state.frame_queue.put((frame, result), block=False)
                except queue.Empty:
                    pass
        time.sleep(0.033)  # ~30 FPS

# Layout dengan kolom
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    
    # Kontrol kamera
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🎥 Start Camera", type="primary"):
            if not st.session_state.camera_active:
                st.session_state.detector = StreamlitYarnDetector()
                if st.session_state.detector.start_camera():
                    st.session_state.camera_active = True
                    # Mulai thread kamera
                    camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
                    camera_thread_obj.start()
                    st.success("Kamera dimulai!")
                    st.rerun()
                else:
                    st.error("Gagal memulai kamera")
    
    with col_btn2:
        if st.button("⏹️ Stop Camera"):
            if st.session_state.camera_active:
                st.session_state.camera_active = False
                if st.session_state.detector:
                    st.session_state.detector.stop_camera()
                st.success("Kamera dihentikan!")
                st.rerun()
    
    with col_btn3:
        if st.button("📸 Capture"):
            if st.session_state.camera_active:
                st.info("Screenshot disimpan! (fitur akan dikembangkan)")
    
    # Placeholder untuk video stream
    video_placeholder = st.empty()
    
    # Pengaturan area sampling
    st.subheader("⚙️ Sampling Area Settings")
    
    if st.session_state.detector:
        col_x, col_y = st.columns(2)
        with col_x:
            new_x = st.slider("X Position", 0, 500, 
                            st.session_state.detector.detector.sample_area['x'])
            new_width = st.slider("Width", 50, 200, 
                                st.session_state.detector.detector.sample_area['width'])
        
        with col_y:
            new_y = st.slider("Y Position", 0, 350, 
                            st.session_state.detector.detector.sample_area['y'])
            new_height = st.slider("Height", 50, 200, 
                                 st.session_state.detector.detector.sample_area['height'])
        
        # Update area sampling
        st.session_state.detector.detector.sample_area.update({
            'x': new_x,
            'y': new_y,
            'width': new_width,
            'height': new_height
        })

with col2:
    st.subheader("🎯 Detection Results")
    
    # Placeholder untuk hasil deteksi
    result_placeholder = st.empty()
    
    # History deteksi
    st.subheader("📊 Detection History")
    history_placeholder = st.empty()

# Loop utama untuk menampilkan frame
if st.session_state.camera_active:
    try:
        # Ambil frame dari queue
        frame, result = st.session_state.frame_queue.get(block=False)
        
        # Tampilkan frame
        video_placeholder.image(frame, channels="RGB", use_column_width=True)
        
        # Tampilkan hasil deteksi
        if result and result['rgb_color']:
            rgb_color = result['rgb_color']
            color_name = result['color_name']
            color_code = result['color_code']
            confidence = result['confidence']
            
            # Simpan ke history
            st.session_state.detection_results.append({
                'time': time.strftime('%H:%M:%S'),
                'color': color_name,
                'confidence': f"{confidence:.1f}%"
            })
            
            # Batasi history
            if len(st.session_state.detection_results) > 10:
                st.session_state.detection_results.pop(0)
            
            # Tampilkan hasil saat ini
            with result_placeholder.container():
                st.markdown(f"""
                <div class="info-card">
                    <h4>🎨 {color_name}</h4>
                    <p><strong>Kode Warna:</strong> {color_code if color_code else 'N/A'}</p>
                    <p><strong>RGB:</strong> {rgb_color}</p>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Color box
                color_style = f"background-color: rgb{rgb_color};"
                st.markdown(f"""
                <div class="color-box" style="{color_style}">
                    {color_name}
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%"></div>
                </div>
                <p style="text-align: center; margin-top: 5px;">Confidence: {confidence:.1f}%</p>
                """, unsafe_allow_html=True)
            
            # Tampilkan history
            if st.session_state.detection_results:
                history_df = pd.DataFrame(st.session_state.detection_results)
                history_placeholder.dataframe(history_df, use_container_width=True)
        
        # Auto-refresh
        time.sleep(0.1)
        st.rerun()
        
    except queue.Empty:
        # Jika tidak ada frame baru, tampilkan pesan loading
        video_placeholder.info("📷 Menunggu frame kamera...")
        time.sleep(0.1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {e}")

else:
    video_placeholder.info("📷 Klik 'Start Camera' untuk memulai deteksi warna")
    result_placeholder.info("🎯 Hasil deteksi akan muncul di sini")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎨 Yarn Color Detector menggunakan CIELAB color space untuk akurasi tinggi</p>
    <p>📱 Pastikan kamera web Anda terhubung dengan baik</p>
</div>
""", unsafe_allow_html=True)