import streamlit as st
import cv2
import numpy as np
import csv
import os
from collections import Counter
from skimage import color
from datetime import datetime
import pandas as pd
from PIL import Image
import io
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Yarn Color Detector - CIELAB",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
        self.color_db = []
        self.load_color_database(csv_file)
        
        # Parameter untuk filtering
        self.brightness_min = 20
        self.brightness_max = 235
        self.saturation_min = 10
        
        # Untuk tracking hasil deteksi
        self.current_detection = {
            'rgb_color': None,
            'color_name': 'Tidak terdeteksi',
            'color_code': None,
            'confidence': 0,
            'lab_values': None
        }
        
    def load_color_database(self, csv_file):
        """Memuat database warna dari file CSV"""
        try:
            if not os.path.exists(csv_file):
                st.error(f"File {csv_file} tidak ditemukan!")
                return
            
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
                    return
                
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
                        
                    except (ValueError, KeyError) as e:
                        continue
            
            if len(self.color_db) > 0:
                st.success(f"Database warna berhasil dimuat: {len(self.color_db)} warna")
            else:
                st.error("Tidak ada data warna yang valid ditemukan!")
                
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    
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
        if image is None:
            return None, "Tidak terdeteksi", None, 0
            
        height, width = image.shape[:2]
        
        # Pastikan sample area dalam batas frame
        x = max(0, min(self.sample_area['x'], width - self.sample_area['width']))
        y = max(0, min(self.sample_area['y'], height - self.sample_area['height']))
        w = min(self.sample_area['width'], width - x)
        h = min(self.sample_area['height'], height - y)
        
        sample = image[y:y+h, x:x+w]
        
        if sample.size == 0:
            return None, "Tidak terdeteksi", None, 0
        
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
    
    def draw_interface(self, frame):
        """Menggambar interface dengan area sampling"""
        if frame is None:
            return frame
            
        height, width = frame.shape[:2]
        
        # Pastikan sample area dalam batas
        x = max(0, min(self.sample_area['x'], width - self.sample_area['width']))
        y = max(0, min(self.sample_area['y'], height - self.sample_area['height']))
        w = min(self.sample_area['width'], width - x)
        h = min(self.sample_area['height'], height - y)
        
        # Gambar area sampling
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label area sampling
        cv2.putText(frame, "Area Sampling", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = YarnColorDetector()
        self.lock = threading.Lock()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip horizontal untuk mirror effect
        img = cv2.flip(img, 1)
        
        # Deteksi warna
        rgb_color, color_name, color_code, confidence = self.detector.get_dominant_color(img)
        
        # Update current detection dengan thread safety
        with self.lock:
            self.detector.current_detection = {
                'rgb_color': rgb_color,
                'color_name': color_name,
                'color_code': color_code,
                'confidence': confidence,
                'lab_values': self.detector.rgb_to_lab(rgb_color) if rgb_color else None
            }
        
        # Gambar interface
        img_with_interface = self.detector.draw_interface(img)
        
        return img_with_interface

def save_detection_result(detection_data, frame_data=None):
    """Menyimpan hasil deteksi"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Buat folder results jika belum ada
    os.makedirs("results", exist_ok=True)
    
    # Simpan data ke CSV log
    log_file = os.path.join("results", "detection_log.csv")
    log_exists = os.path.exists(log_file)
    
    rgb_color = detection_data.get('rgb_color')
    
    with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'color_name', 'color_code', 'rgb_r', 'rgb_g', 'rgb_b', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not log_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': timestamp,
            'color_name': detection_data.get('color_name', ''),
            'color_code': detection_data.get('color_code', ''),
            'rgb_r': rgb_color[0] if rgb_color else 0,
            'rgb_g': rgb_color[1] if rgb_color else 0,
            'rgb_b': rgb_color[2] if rgb_color else 0,
            'confidence': detection_data.get('confidence', 0)
        })
    
    return f"Hasil disimpan pada {timestamp}"

def load_detection_history():
    """Memuat riwayat deteksi"""
    log_file = os.path.join("results", "detection_log.csv")
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def main():
    st.title("🧶 Yarn Color Detector - CIELAB")
    st.markdown("**Deteksi Warna Benang Menggunakan Algoritma CIELAB**")
    
    # Sidebar untuk kontrol
    st.sidebar.header("⚙️ Pengaturan")
    
    # Informasi database
    detector = YarnColorDetector()
    st.sidebar.info(f"Database: {len(detector.color_db)} warna dimuat")
    
    # Pengaturan area sampling
    st.sidebar.subheader("Area Sampling")
    sample_width = st.sidebar.slider("Lebar Area", 50, 200, detector.sample_area['width'])
    sample_height = st.sidebar.slider("Tinggi Area", 50, 200, detector.sample_area['height'])
    
    # Pengaturan filter
    st.sidebar.subheader("Filter Warna")
    brightness_min = st.sidebar.slider("Brightness Min", 0, 100, detector.brightness_min)
    brightness_max = st.sidebar.slider("Brightness Max", 200, 255, detector.brightness_max)
    saturation_min = st.sidebar.slider("Saturation Min", 0, 50, detector.saturation_min)
    
    # Layout utama
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Live Camera")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="yarn-detector",
            mode="sendrecv",
            rtc_configuration=RTC_CONFIGURATION,
            video_transformer_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Kontrol
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            if st.button("📸 Capture & Save"):
                if webrtc_ctx.video_transformer:
                    with webrtc_ctx.video_transformer.lock:
                        detection = webrtc_ctx.video_transformer.detector.current_detection
                    
                    if detection['rgb_color'] is not None:
                        result = save_detection_result(detection)
                        st.success(result)
                    else:
                        st.warning("Tidak ada warna yang terdeteksi")
        
        with col1b:
            if st.button("🔄 Reset Area"):
                if webrtc_ctx.video_transformer:
                    webrtc_ctx.video_transformer.detector.sample_area = {
                        'x': 270,
                        'y': 190,
                        'width': sample_width,
                        'height': sample_height
                    }
                st.success("Area sampling direset")
        
        with col1c:
            if st.button("⚙️ Apply Settings"):
                if webrtc_ctx.video_transformer:
                    webrtc_ctx.video_transformer.detector.sample_area.update({
                        'width': sample_width,
                        'height': sample_height
                    })
                    webrtc_ctx.video_transformer.detector.brightness_min = brightness_min
                    webrtc_ctx.video_transformer.detector.brightness_max = brightness_max
                    webrtc_ctx.video_transformer.detector.saturation_min = saturation_min
                st.success("Pengaturan diterapkan")
    
    with col2:
        st.subheader("📊 Hasil Deteksi")
        
        # Container untuk hasil real-time
        result_container = st.container()
        
        # Update hasil deteksi secara real-time
        if webrtc_ctx.video_transformer:
            with webrtc_ctx.video_transformer.lock:
                detection = webrtc_ctx.video_transformer.detector.current_detection
            
            with result_container:
                # Nama warna
                st.markdown(f"**Warna:** {detection['color_name']}")
                
                # Kode warna
                if detection['color_code']:
                    st.markdown(f"**Kode:** {detection['color_code']}")
                
                # RGB values
                if detection['rgb_color']:
                    r, g, b = detection['rgb_color']
                    st.markdown(f"**RGB:** ({r}, {g}, {b})")
                    
                    # Color preview
                    color_html = f"""
                    <div style="width: 100px; height: 60px; background-color: rgb({r},{g},{b}); 
                         border: 2px solid #ccc; border-radius: 5px; margin: 10px 0;"></div>
                    """
                    st.markdown(color_html, unsafe_allow_html=True)
                
                # LAB values
                if detection['lab_values'] is not None:
                    lab = detection['lab_values']
                    st.markdown(f"**LAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
                
                # Confidence
                confidence = detection['confidence']
                st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Confidence progress bar
                st.progress(confidence / 100)
                
                # Confidence indicator
                if confidence > 80:
                    st.success("🎯 Deteksi Sangat Akurat")
                elif confidence > 60:
                    st.warning("⚠️ Deteksi Cukup Akurat")
                else:
                    st.error("❌ Deteksi Kurang Akurat")
    
    # Bagian bawah - Database dan History
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Database Warna", "📈 Riwayat Deteksi", "ℹ️ Informasi"])
    
    with tab1:
        st.subheader("Database Warna")
        if detector.color_db:
            # Konversi ke DataFrame untuk ditampilkan
            df_colors = pd.DataFrame(detector.color_db)
            
            # Filter berdasarkan kategori
            categories = ['Semua'] + list(df_colors['category'].unique())
            selected_category = st.selectbox("Filter Kategori:", categories)
            
            if selected_category != 'Semua':
                df_colors = df_colors[df_colors['category'] == selected_category]
            
            # Tampilkan dengan color preview
            st.dataframe(
                df_colors[['color_name', 'color_code', 'category', 'r', 'g', 'b']],
                use_container_width=True
            )
            
            st.info(f"Menampilkan {len(df_colors)} dari {len(detector.color_db)} warna")
        else:
            st.error("Database warna tidak berhasil dimuat")
    
    with tab2:
        st.subheader("Riwayat Deteksi")
        df_history = load_detection_history()
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
            
            # Statistik
            st.subheader("Statistik")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Deteksi", len(df_history))
            
            with col2:
                avg_confidence = df_history['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            with col3:
                most_common = df_history['color_name'].mode()
                if not most_common.empty:
                    st.metric("Warna Terbanyak", most_common.iloc[0])
        else:
            st.info("Belum ada riwayat deteksi")
    
    with tab3:
        st.subheader("Informasi Sistem")
        
        st.markdown("""
        ### 🎯 Cara Penggunaan:
        1. **Pastikan kamera aktif** - Izinkan akses kamera pada browser
        2. **Letakkan yarn di area hijau** - Area sampling ditandai dengan kotak hijau
        3. **Tunggu deteksi** - Sistem akan menampilkan hasil secara real-time
        4. **Capture hasil** - Klik tombol "Capture & Save" untuk menyimpan
        
        ### 🔧 Fitur Utama:
        - **Real-time detection** menggunakan algoritma CIELAB
        - **Database lengkap** dengan {len(detector.color_db)} warna yarn
        - **Confidence scoring** untuk akurasi deteksi
        - **Riwayat deteksi** tersimpan otomatis
        - **Filter warna** dapat disesuaikan
        
        ### 📊 Akurasi Deteksi:
        - **80-100%**: Sangat akurat ✅
        - **60-79%**: Cukup akurat ⚠️
        - **<60%**: Kurang akurat ❌
        
        ### 💡 Tips untuk Hasil Terbaik:
        - Gunakan pencahayaan yang cukup dan merata
        - Letakkan yarn di area sampling dengan rapi
        - Hindari bayangan atau pantulan cahaya
        - Sesuaikan pengaturan filter sesuai kondisi
        """.format(len(detector.color_db)))

if __name__ == "__main__":
    main()