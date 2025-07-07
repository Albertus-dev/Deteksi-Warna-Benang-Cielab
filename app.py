import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
from datetime import datetime
import csv
from collections import Counter
from skimage import color

# Konfigurasi halaman
st.set_page_config(
    page_title="Yarn Color Detector",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling sederhana
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .color-display {
        width: 100%;
        height: 100px;
        border: 2px solid #333;
        border-radius: 10px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class YarnColorDetector:
    def __init__(self, csv_file='yarn_colors_database.csv'):
        """Inisialisasi detector dengan database warna dari CSV"""
        
        # Area sampling yang bisa disesuaikan
        self.sample_area = {
            'x': 270,
            'y': 190,
            'width': 100,
            'height': 100
        }
        
        # Load database warna dari CSV
        self.load_color_database(csv_file)
        
        # Parameter untuk filtering
        self.brightness_min = 20
        self.brightness_max = 235
        self.saturation_min = 10
        
    def load_color_database(self, csv_file):
        """Memuat database warna dari file CSV"""
        try:
            if not os.path.exists(csv_file):
                st.error(f"File {csv_file} tidak ditemukan!")
                st.stop()
            
            self.color_db = []
            with open(csv_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    cleaned_lines.append(line)
                
                if not cleaned_lines:
                    st.error("File CSV kosong!")
                    st.stop()
                
                header = cleaned_lines[0].split(',')
                header = [h.strip() for h in header]
                
                for line in cleaned_lines[1:]:
                    if not line.strip():
                        continue
                        
                    values = line.split(',')
                    
                    if len(values) != len(header):
                        continue
                    
                    try:
                        row_dict = {}
                        for i, col_name in enumerate(header):
                            row_dict[col_name] = values[i].strip()
                        
                        color_data = {
                            'color_name': row_dict['color_name'],
                            'color_code': row_dict['color_code'],
                            'r': int(float(row_dict['r'])),
                            'g': int(float(row_dict['g'])),
                            'b': int(float(row_dict['b'])),
                            'l_lab': float(row_dict['l_lab']),
                            'a_lab': float(row_dict['a_lab']),
                            'b_lab': float(row_dict['b_lab']),
                            'category': row_dict['category']
                        }
                        self.color_db.append(color_data)
                        
                    except (ValueError, KeyError):
                        continue
            
            if len(self.color_db) == 0:
                st.error("Tidak ada data warna yang valid ditemukan di CSV!")
                st.stop()
                
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.stop()
    
    def rgb_to_lab(self, rgb):
        """Konversi RGB ke CIELAB"""
        try:
            rgb_norm = np.array([[rgb]]) / 255.0
            lab = color.rgb2lab(rgb_norm)[0][0]
            return lab
        except:
            return np.array([0, 0, 0])
    
    def calculate_delta_e(self, lab1, lab2):
        """Menghitung Delta E (CIE76)"""
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        return np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
    
    def find_closest_color(self, target_rgb):
        """Mencari warna terdekat berdasarkan Delta E"""
        if target_rgb is None:
            return None, "Tidak terdeteksi", None, 0
        
        target_lab = self.rgb_to_lab(target_rgb)
        min_delta_e = float('inf')
        best_match = None
        
        for color_data in self.color_db:
            db_lab = np.array([color_data['l_lab'], color_data['a_lab'], color_data['b_lab']])
            delta_e = self.calculate_delta_e(target_lab, db_lab)
            
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                best_match = color_data
        
        if best_match is not None:
            confidence = max(0, 100 - min_delta_e * 2)
            return best_match, best_match['color_name'], best_match['color_code'], confidence
        
        return None, "Tidak dikenal", None, 0
    
    def preprocess_sample(self, sample):
        """Preprocessing area sample"""
        blurred = cv2.GaussianBlur(sample, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        mask = np.ones(hsv.shape[:2], dtype=bool)
        mask &= (hsv[:,:,2] > self.brightness_min) & (hsv[:,:,2] < self.brightness_max)
        mask &= hsv[:,:,1] > self.saturation_min
        
        return blurred, mask
    
    def get_dominant_color(self, image):
        """Mendapatkan warna dominan"""
        sample = image[
            self.sample_area['y']:self.sample_area['y'] + self.sample_area['height'],
            self.sample_area['x']:self.sample_area['x'] + self.sample_area['width']
        ]
        
        processed_sample, mask = self.preprocess_sample(sample)
        valid_pixels = processed_sample[mask]
        
        if len(valid_pixels) == 0:
            return None, "Tidak terdeteksi", None, 0
        
        pixels_bgr = valid_pixels.reshape((-1, 3))
        dominant_color_bgr = np.median(pixels_bgr, axis=0)
        
        dominant_color_rgb = (int(dominant_color_bgr[2]), 
                            int(dominant_color_bgr[1]), 
                            int(dominant_color_bgr[0]))
        
        best_match, color_name, color_code, confidence = self.find_closest_color(dominant_color_rgb)
        
        return dominant_color_rgb, color_name, color_code, confidence
    
    def process_frame(self, frame):
        """Memproses frame dan mengembalikan hasil deteksi"""
        frame_flipped = cv2.flip(frame, 1)
        
        # Deteksi warna
        rgb_color, color_name, color_code, confidence = self.get_dominant_color(frame_flipped)
        
        # Gambar sampling area
        x, y = self.sample_area['x'], self.sample_area['y']
        w, h = self.sample_area['width'], self.sample_area['height']
        
        cv2.rectangle(frame_flipped, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame_flipped, "Sampling Area", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Tambahkan info detection
        if rgb_color and color_name != "Tidak terdeteksi":
            info_text = f"{color_name} ({confidence:.1f}%)"
            cv2.putText(frame_flipped, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_flipped, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return frame_flipped, rgb_color, color_name, color_code, confidence

# Inisialisasi session state
if 'detector' not in st.session_state:
    st.session_state.detector = YarnColorDetector()
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🎨 Yarn Color Detector</h1>
    <p>Deteksi warna benang menggunakan CIELAB color space</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk kontrol
st.sidebar.header("⚙️ Pengaturan")

# Kontrol kamera
st.sidebar.subheader("📹 Kamera")
camera_source = st.sidebar.selectbox("Pilih sumber kamera:", [0, 1, 2], index=0)

# Pengaturan sampling area
st.sidebar.subheader("🎯 Area Sampling")
sample_x = st.sidebar.slider("X Position", 0, 500, st.session_state.detector.sample_area['x'])
sample_y = st.sidebar.slider("Y Position", 0, 350, st.session_state.detector.sample_area['y'])
sample_width = st.sidebar.slider("Width", 50, 200, st.session_state.detector.sample_area['width'])
sample_height = st.sidebar.slider("Height", 50, 200, st.session_state.detector.sample_area['height'])

# Update area sampling
st.session_state.detector.sample_area.update({
    'x': sample_x,
    'y': sample_y,
    'width': sample_width,
    'height': sample_height
})

# Pengaturan filter
st.sidebar.subheader("🔧 Filter")
brightness_min = st.sidebar.slider("Brightness Min", 0, 100, st.session_state.detector.brightness_min)
brightness_max = st.sidebar.slider("Brightness Max", 150, 255, st.session_state.detector.brightness_max)
saturation_min = st.sidebar.slider("Saturation Min", 0, 50, st.session_state.detector.saturation_min)

# Update filter
st.session_state.detector.brightness_min = brightness_min
st.session_state.detector.brightness_max = brightness_max
st.session_state.detector.saturation_min = saturation_min

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera")
    
    # Kontrol kamera
    col_start, col_stop, col_capture = st.columns(3)
    
    with col_start:
        start_camera = st.button("🎥 Start Camera", type="primary")
    
    with col_stop:
        stop_camera = st.button("⏹️ Stop Camera")
    
    with col_capture:
        capture_frame = st.button("📸 Capture")
    
    # Placeholder untuk video
    frame_placeholder = st.empty()
    
    # Kamera logic
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        # Inisialisasi kamera
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Proses frame
                processed_frame, rgb_color, color_name, color_code, confidence = st.session_state.detector.process_frame(frame)
                
                # Konversi BGR ke RGB untuk Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Tampilkan frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Simpan hasil deteksi untuk ditampilkan di kolom kanan
                st.session_state.current_detection = {
                    'rgb_color': rgb_color,
                    'color_name': color_name,
                    'color_code': color_code,
                    'confidence': confidence,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                
                # Capture frame
                if capture_frame and rgb_color:
                    # Simpan ke history
                    st.session_state.detection_history.append(st.session_state.current_detection.copy())
                    
                    # Batasi history
                    if len(st.session_state.detection_history) > 20:
                        st.session_state.detection_history.pop(0)
                    
                    # Simpan gambar
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"yarn_detection_{color_name.replace(' ', '_')}_{timestamp}.jpg"
                    os.makedirs("results", exist_ok=True)
                    cv2.imwrite(os.path.join("results", filename), processed_frame)
                    
                    st.success(f"Gambar disimpan: {filename}")
                
                # Auto refresh
                time.sleep(0.1)
                st.rerun()
        
        cap.release()
    
    else:
        frame_placeholder.info("📷 Klik 'Start Camera' untuk memulai deteksi")

with col2:
    st.subheader("🎯 Hasil Deteksi")
    
    # Tampilkan hasil deteksi terkini
    if 'current_detection' in st.session_state:
        detection = st.session_state.current_detection
        
        if detection['rgb_color'] and detection['color_name'] != "Tidak terdeteksi":
            # Color display
            color_style = f"background-color: rgb{detection['rgb_color']};"
            st.markdown(f"""
            <div class="color-display" style="{color_style}">
                {detection['color_name']}
            </div>
            """, unsafe_allow_html=True)
            
            # Info cards
            st.markdown(f"""
            <div class="info-box">
                <h4>🏷️ {detection['color_name']}</h4>
                <p><strong>Kode:</strong> {detection['color_code'] if detection['color_code'] else 'N/A'}</p>
                <p><strong>RGB:</strong> {detection['rgb_color']}</p>
                <p><strong>Confidence:</strong> {detection['confidence']:.1f}%</p>
                <p><strong>Waktu:</strong> {detection['timestamp']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            confidence_width = detection['confidence']
            st.markdown(f"""
            <div style="background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="width: {confidence_width}%; height: 20px; background: linear-gradient(90deg, #ff6b6b, #4ecdc4);"></div>
            </div>
            <p style="text-align: center; margin-top: 5px;">Confidence: {detection['confidence']:.1f}%</p>
            """, unsafe_allow_html=True)
        
        else:
            st.info("Tidak ada warna yang terdeteksi")
    
    # Database info
    st.subheader("📊 Database Info")
    st.write(f"Total warna: {len(st.session_state.detector.color_db)}")
    
    # Tampilkan beberapa warna dari database
    if st.expander("Lihat warna database"):
        for i, color in enumerate(st.session_state.detector.color_db[:10]):
            st.write(f"• {color['color_name']} - {color['color_code']}")
        if len(st.session_state.detector.color_db) > 10:
            st.write(f"... dan {len(st.session_state.detector.color_db) - 10} warna lainnya")
    
    # History
    st.subheader("📈 History Deteksi")
    if st.session_state.detection_history:
        history_df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(history_df[['timestamp', 'color_name', 'confidence']], use_container_width=True)
        
        # Tombol clear history
        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.success("History dibersihkan!")
            st.rerun()
    else:
        st.info("Belum ada history deteksi")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>🎨 Yarn Color Detector menggunakan CIELAB color space untuk akurasi tinggi</p>
    <p>Pastikan kamera web Anda terhubung dengan baik dan berikan izin akses kamera</p>
</div>
""", unsafe_allow_html=True)