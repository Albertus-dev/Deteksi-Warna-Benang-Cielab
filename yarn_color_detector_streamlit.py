import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
from datetime import datetime
import os
from skimage import color
import tempfile

class YarnColorDetector:
    def __init__(self, csv_file='yarn_colors_database.csv'):
        """Inisialisasi detector dengan database warna dari CSV"""
        self.load_color_database(csv_file)
        
        # Parameter untuk filtering
        self.brightness_min = 20
        self.brightness_max = 235
        self.saturation_min = 10
        
    def load_color_database(self, csv_file):
        """Memuat database warna dari file CSV"""
        try:
            if not os.path.exists(csv_file):
                # Jika file tidak ada, buat dari data default
                self.create_default_database()
                return
            
            self.color_db = []
            with open(csv_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
                # Bersihkan baris dari tanda kutip berlebih
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    cleaned_lines.append(line)
                
                if not cleaned_lines:
                    raise ValueError("File CSV kosong!")
                
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
            
            if len(self.color_db) == 0:
                self.create_default_database()
                
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            self.create_default_database()
    
    def create_default_database(self):
        """Membuat database default jika CSV tidak tersedia"""
        self.color_db = [
            {'color_name': 'White', 'color_code': '#FFFFFF', 'r': 255, 'g': 255, 'b': 255, 'l_lab': 100.0, 'a_lab': 0.0, 'b_lab': 0.0, 'category': 'Basic'},
            {'color_name': 'Black', 'color_code': '#000000', 'r': 0, 'g': 0, 'b': 0, 'l_lab': 0.0, 'a_lab': 0.0, 'b_lab': 0.0, 'category': 'Basic'},
            {'color_name': 'Red', 'color_code': '#FF0000', 'r': 255, 'g': 0, 'b': 0, 'l_lab': 53.2, 'a_lab': 80.1, 'b_lab': 67.2, 'category': 'Primary'},
            {'color_name': 'Green', 'color_code': '#008000', 'r': 0, 'g': 128, 'b': 0, 'l_lab': 46.2, 'a_lab': -51.7, 'b_lab': 49.9, 'category': 'Primary'},
            {'color_name': 'Blue', 'color_code': '#0000FF', 'r': 0, 'g': 0, 'b': 255, 'l_lab': 32.3, 'a_lab': 79.2, 'b_lab': -107.9, 'category': 'Primary'},
            {'color_name': 'Yellow', 'color_code': '#FFFF00', 'r': 255, 'g': 255, 'b': 0, 'l_lab': 97.1, 'a_lab': -21.6, 'b_lab': 94.5, 'category': 'Primary'},
        ]
    
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
    
    def get_dominant_color_from_area(self, image, x, y, width, height):
        """Mendapatkan warna dominan dari area tertentu"""
        h, w = image.shape[:2]
        
        # Pastikan area dalam batas gambar
        x = max(0, min(x, w - width))
        y = max(0, min(y, h - height))
        width = min(width, w - x)
        height = min(height, h - y)
        
        sample = image[y:y + height, x:x + width]
        
        if sample.size == 0:
            return None, "Tidak terdeteksi", None, 0
        
        processed_sample, mask = self.preprocess_sample(sample)
        valid_pixels = processed_sample[mask]
        
        if len(valid_pixels) == 0:
            return None, "Tidak terdeteksi", None, 0
        
        pixels_bgr = valid_pixels.reshape((-1, 3))
        dominant_color_bgr = np.median(pixels_bgr, axis=0)
        
        # Konversi BGR ke RGB
        dominant_color_rgb = (int(dominant_color_bgr[2]), 
                            int(dominant_color_bgr[1]), 
                            int(dominant_color_bgr[0]))
        
        best_match, color_name, color_code, confidence = self.find_closest_color(dominant_color_rgb)
        
        return dominant_color_rgb, color_name, color_code, confidence

def create_download_link(data, filename, text):
    """Membuat link download untuk data"""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

def main():
    st.set_page_config(
        page_title="Yarn Color Detector",
        page_icon="🧶",
        layout="wide"
    )
    
    st.title("🧶 Yarn Color Detector")
    st.markdown("---")
    
    # Sidebar untuk kontrol
    st.sidebar.header("⚙️ Pengaturan")
    
    # Inisialisasi detector
    if 'detector' not in st.session_state:
        st.session_state.detector = YarnColorDetector()
    
    detector = st.session_state.detector
    
    # Tampilkan informasi database
    st.sidebar.success(f"Database: {len(detector.color_db)} warna dimuat")
    
    # Parameter sampling area
    st.sidebar.subheader("📏 Area Sampling")
    sample_width = st.sidebar.slider("Lebar Area", 50, 300, 100)
    sample_height = st.sidebar.slider("Tinggi Area", 50, 300, 100)
    
    # Parameter filtering
    st.sidebar.subheader("🎛️ Filter")
    detector.brightness_min = st.sidebar.slider("Brightness Min", 0, 100, 20)
    detector.brightness_max = st.sidebar.slider("Brightness Max", 100, 255, 235)
    detector.saturation_min = st.sidebar.slider("Saturation Min", 0, 50, 10)
    
    # Opsi input
    st.sidebar.subheader("📷 Input")
    input_option = st.sidebar.selectbox(
        "Pilih sumber input:",
        ["Upload Gambar", "Kamera (Live)", "Webcam Snapshot"]
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if input_option == "Upload Gambar":
            uploaded_file = st.file_uploader(
                "Pilih gambar yarn",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload gambar yarn yang ingin dideteksi warnanya"
            )
            
            if uploaded_file is not None:
                # Baca gambar
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Konversi RGB ke BGR untuk OpenCV
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Tampilkan gambar dengan area sampling
                display_image = img_array.copy()
                h, w = display_image.shape[:2]
                
                # Hitung posisi area sampling (tengah gambar)
                sample_x = max(0, (w - sample_width) // 2)
                sample_y = max(0, (h - sample_height) // 2)
                
                # Gambar kotak area sampling
                cv2.rectangle(display_image, 
                            (sample_x, sample_y),
                            (sample_x + sample_width, sample_y + sample_height),
                            (0, 255, 0), 3)
                
                # Konversi kembali ke RGB untuk display
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                st.image(display_image, 
                        caption="Gambar dengan area sampling (kotak hijau)",
                        use_column_width=True)
                
                # Deteksi warna
                rgb_color, color_name, color_code, confidence = detector.get_dominant_color_from_area(
                    img_array, sample_x, sample_y, sample_width, sample_height
                )
                
                # Tampilkan hasil deteksi
                if rgb_color:
                    with col2:
                        st.subheader("🎨 Hasil Deteksi")
                        
                        # Color box
                        color_html = f"""
                        <div style="
                            width: 100%;
                            height: 100px;
                            background-color: rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]});
                            border: 2px solid #333;
                            border-radius: 10px;
                            margin-bottom: 10px;
                        "></div>
                        """
                        st.markdown(color_html, unsafe_allow_html=True)
                        
                        # Informasi warna
                        st.metric("Nama Warna", color_name)
                        st.metric("Kode Warna", color_code if color_code else "N/A")
                        st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Informasi RGB dan LAB
                        st.text(f"RGB: ({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})")
                        lab = detector.rgb_to_lab(rgb_color)
                        st.text(f"LAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
                        
                        # Progress bar confidence
                        st.progress(confidence / 100)
                        
                        # Tombol save
                        if st.button("💾 Simpan Hasil"):
                            # Simpan gambar dengan hasil deteksi
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"yarn_detection_{color_name.replace(' ', '_')}_{timestamp}.jpg"
                            
                            # Konversi gambar ke bytes
                            result_image = Image.fromarray(display_image)
                            img_bytes = io.BytesIO()
                            result_image.save(img_bytes, format='JPEG')
                            img_bytes = img_bytes.getvalue()
                            
                            # Download link
                            st.markdown(
                                create_download_link(img_bytes, filename, "📥 Download Gambar"),
                                unsafe_allow_html=True
                            )
                            
                            # Simpan ke log CSV
                            log_data = {
                                'timestamp': timestamp,
                                'color_name': color_name,
                                'color_code': color_code,
                                'rgb_r': rgb_color[0],
                                'rgb_g': rgb_color[1],
                                'rgb_b': rgb_color[2],
                                'confidence': confidence,
                                'filename': filename
                            }
                            
                            # Buat CSV log
                            df_log = pd.DataFrame([log_data])
                            csv_bytes = df_log.to_csv(index=False).encode()
                            
                            st.markdown(
                                create_download_link(csv_bytes, f"detection_log_{timestamp}.csv", "📥 Download Log CSV"),
                                unsafe_allow_html=True
                            )
                            
                            st.success("Hasil deteksi siap didownload!")
        
        elif input_option == "Kamera (Live)":
            st.info("🔄 Fitur kamera live akan segera tersedia")
            st.markdown("""
            Untuk menggunakan fitur kamera live, Anda perlu:
            1. Menjalankan aplikasi di local server
            2. Memberikan permission akses kamera
            3. Menggunakan browser yang mendukung WebRTC
            """)
        
        elif input_option == "Webcam Snapshot":
            # Webcam snapshot menggunakan streamlit-webrtc (jika tersedia)
            st.info("📸 Fitur webcam snapshot")
            st.markdown("""
            Untuk menggunakan fitur webcam:
            1. Klik tombol untuk mengambil foto
            2. Foto akan otomatis diproses
            3. Hasil deteksi akan muncul di panel kanan
            """)
    
    # Bagian database viewer
    st.markdown("---")
    st.subheader("📊 Database Warna")
    
    # Filter berdasarkan kategori
    categories = list(set([color['category'] for color in detector.color_db]))
    selected_category = st.selectbox("Filter berdasarkan kategori:", ["Semua"] + categories)
    
    # Tampilkan database
    if selected_category == "Semua":
        display_colors = detector.color_db
    else:
        display_colors = [color for color in detector.color_db if color['category'] == selected_category]
    
    # Buat DataFrame untuk ditampilkan
    df_colors = pd.DataFrame(display_colors)
    
    # Tampilkan dalam bentuk tabel dengan color preview
    st.dataframe(
        df_colors[['color_name', 'color_code', 'category', 'r', 'g', 'b']],
        use_container_width=True
    )
    
    # Color palette preview
    st.subheader("🎨 Preview Warna")
    
    # Tampilkan warna dalam grid
    cols = st.columns(8)
    for i, color in enumerate(display_colors[:32]):  # Batasi 32 warna untuk preview
        with cols[i % 8]:
            color_html = f"""
            <div style="
                width: 60px;
                height: 60px;
                background-color: {color['color_code']};
                border: 1px solid #333;
                border-radius: 5px;
                margin: 2px;
            " title="{color['color_name']}"></div>
            <small>{color['color_name']}</small>
            """
            st.markdown(color_html, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Yarn Color Detector** - Aplikasi deteksi warna yarn menggunakan algoritma CIELAB Delta E
    
    ✨ **Fitur:**
    - Deteksi warna akurat dengan database lengkap
    - Preprocessing gambar dengan filter brightness & saturation
    - Perhitungan confidence score
    - Export hasil deteksi
    - Database viewer dengan filter kategori
    """)

if __name__ == "__main__":
    main()