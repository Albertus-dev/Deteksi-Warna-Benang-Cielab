import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from datetime import datetime
from deteksi_warna import YarnColorDetector
import av
import os
import threading
from PIL import Image
import io

st.set_page_config(page_title="Deteksi Warna Benang", layout="wide")
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
if "hasil_deteksi_upload" not in st.session_state:
    st.session_state.hasil_deteksi_upload = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Fungsi untuk memperbarui area deteksi ke tengah
def update_sample_area_to_center(detector, img_shape):
    """Update area deteksi ke tengah gambar"""
    h, w = img_shape[:2]
    area = detector.sample_area
    
    # Tempatkan area di tengah gambar
    width = min(area.get('width', 100), w)
    height = min(area.get('height', 100), h)
    x = (w - width) // 2
    y = (h - height) // 2
    
    # Update area detector
    detector.sample_area['x'] = x
    detector.sample_area['y'] = y
    detector.sample_area['width'] = width
    detector.sample_area['height'] = height
    
    return x, y, width, height

# Fungsi untuk memproses gambar upload
def process_uploaded_image(image_file):
    """Memproses gambar yang diupload untuk deteksi warna"""
    try:
        # Baca gambar
        image = Image.open(image_file)
        
        # Konversi ke RGB jika perlu
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Konversi ke numpy array
        img_array = np.array(image)
        
        # Konversi RGB ke BGR untuk OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Simpan gambar asli
        st.session_state.uploaded_image = img_bgr.copy()
        
        # Update area deteksi ke tengah gambar
        update_sample_area_to_center(detector, img_bgr.shape)
        
        # Deteksi warna
        rgb_color, color_name, color_code, confidence = detector.get_dominant_color(img_bgr)
        
        if rgb_color is not None:
            # Dapatkan analisis warna lengkap
            analysis = detector.get_color_analysis(rgb_color)
            
            st.session_state.hasil_deteksi_upload = {
                "rgb": rgb_color,
                "nama": color_name,
                "kode": color_code,
                "confidence": confidence,
                "waktu": datetime.now().strftime("%H:%M:%S"),
                "analysis": analysis
            }
            
            return True
        else:
            st.error("Tidak dapat mendeteksi warna dari gambar")
            return False
            
    except Exception as e:
        st.error(f"Error memproses gambar: {e}")
        return False

# Fungsi untuk menampilkan hasil deteksi
def display_detection_results(hasil, analysis, rgb, is_upload=False):
    """Menampilkan hasil deteksi dalam format yang konsisten"""
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.markdown("### 🎯 Identifikasi")
        st.metric("Nama Warna", hasil['nama'])
        st.metric("Confidence", f"{hasil['confidence']:.1f}%")
        st.metric("Waktu Deteksi", hasil['waktu'])
    
    with col2:
        st.markdown("### 🔢 Kode & RGB")
        st.metric("Kode Warna", hasil['kode'])
        st.write(f"**RGB:** {rgb[0]}, {rgb[1]}, {rgb[2]}")
        if analysis and 'hex' in analysis:
            st.write(f"**HEX:** {analysis['hex']}")
    
    with col3:
        st.markdown("### 🌈 CIELAB")
        if analysis and 'lab' in analysis:
            lab = analysis['lab']
            st.write(f"**L*:** {lab[0]:.1f}")
            st.write(f"**a*:** {lab[1]:.1f}")
            st.write(f"**b*:** {lab[2]:.1f}")
        else:
            st.write("Data LAB tidak tersedia")
        
        # Tambahan HSV jika ada
        if analysis and 'hsv' in analysis:
            hsv = analysis['hsv']
            st.write(f"**HSV:** {hsv[0]}, {hsv[1]}, {hsv[2]}")
    
    with col4:
        st.markdown("### 🎨 Preview")
        # Tampilkan warna
        st.markdown(f"""
        <div style='
            width: 120px; 
            height: 120px; 
            background-color: rgb{rgb}; 
            border: 3px solid #333; 
            border-radius: 10px;
            margin: 10px auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        '></div>
        """, unsafe_allow_html=True)
        
        # Kategori warna
        if analysis and 'category' in analysis:
            st.write(f"**Kategori:** {analysis['category']}")
    
    # Informasi tambahan dalam expander
    with st.expander("🔍 Informasi Detail"):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Spesifikasi Warna")
            st.write(f"**Nama:** {hasil['nama']}")
            st.write(f"**Kode:** {hasil['kode']}")
            st.write(f"**RGB:** ({rgb[0]}, {rgb[1]}, {rgb[2]})")
            if analysis and 'hex' in analysis:
                st.write(f"**HEX:** {analysis['hex']}")
            if analysis and 'lab' in analysis:
                lab = analysis['lab']
                st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
            if analysis and 'hsv' in analysis:
                hsv = analysis['hsv']
                st.write(f"**HSV:** H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")
        
        with col2:
            st.markdown("#### 🎭 Warna Serupa")
            if analysis and 'similar_colors' in analysis:
                similar = analysis['similar_colors'][:3]  # Ambil 3 teratas
                if similar:
                    for i, color in enumerate(similar):
                        st.write(f"**{i+1}.** {color['name']} ({color['code']}) - ΔE: {color['delta_e']:.1f}")
                else:
                    st.write("Tidak ada warna serupa ditemukan")
            else:
                st.write("Data warna serupa tidak tersedia")
    
    # Tombol aksi
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        save_key = "save_result_upload" if is_upload else "save_result"
        if st.button("💾 Simpan Hasil", key=save_key):
            image_to_save = st.session_state.uploaded_image if is_upload else st.session_state.last_frame
            if image_to_save is not None:
                try:
                    os.makedirs("results", exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    source_type = "upload" if is_upload else "camera"
                    
                    # Simpan gambar
                    img_filename = f"results/detection_{source_type}_{timestamp}.jpg"
                    cv2.imwrite(img_filename, image_to_save)
                    
                    # Simpan info detail
                    txt_filename = f"results/detection_{source_type}_{timestamp}.txt"
                    with open(txt_filename, 'w', encoding='utf-8') as f:
                        f.write("=== HASIL DETEKSI WARNA BENANG ===\n")
                        f.write(f"Sumber: {'Upload Gambar' if is_upload else 'Kamera Real-time'}\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Nama Warna: {hasil['nama']}\n")
                        f.write(f"Kode Warna: {hasil['kode']}\n")
                        f.write(f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})\n")
                        if analysis and 'hex' in analysis:
                            f.write(f"HEX: {analysis['hex']}\n")
                        if analysis and 'lab' in analysis:
                            lab = analysis['lab']
                            f.write(f"CIELAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}\n")
                        if analysis and 'hsv' in analysis:
                            hsv = analysis['hsv']
                            f.write(f"HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}\n")
                        f.write(f"Confidence: {hasil['confidence']:.1f}%\n")
                        f.write(f"Kategori: {analysis.get('category', 'N/A')}\n")
                        f.write(f"File Gambar: {img_filename}\n")
                        
                        # Warna serupa
                        if analysis and 'similar_colors' in analysis:
                            f.write("\n=== WARNA SERUPA ===\n")
                            for i, color in enumerate(analysis['similar_colors'][:5]):
                                f.write(f"{i+1}. {color['name']} ({color['code']}) - ΔE: {color['delta_e']:.1f}\n")
                    
                    st.success(f"✅ Hasil disimpan:")
                    st.write(f"- Gambar: {img_filename}")
                    st.write(f"- Detail: {txt_filename}")
                    
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan: {e}")
    
    with col2:
        copy_key = "copy_info_upload" if is_upload else "copy_info"
        if st.button("📋 Copy Info", key=copy_key):
            info_text = f"Warna: {hasil['nama']}\nKode: {hasil['kode']}\nRGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})\n"
            if analysis and 'hex' in analysis:
                info_text += f"HEX: {analysis['hex']}\n"
            if analysis and 'lab' in analysis:
                lab = analysis['lab']
                info_text += f"CIELAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}\n"
            info_text += f"Confidence: {hasil['confidence']:.1f}%"
            
            st.code(info_text, language="text")
            st.success("📋Info siap di-copy!")
    
    with col3:
        reset_key = "reset_result_upload" if is_upload else "reset_result"
        if st.button("🔄 Reset", key=reset_key):
            if is_upload:
                st.session_state.hasil_deteksi_upload = None
                st.session_state.uploaded_image = None
            else:
                st.session_state.hasil_deteksi = None
                st.session_state.last_frame = None
            st.rerun()

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
                # Update area deteksi ke tengah gambar
                update_sample_area_to_center(self.detector, img.shape)
                
                # Deteksi warna
                rgb_color, color_name, color_code, confidence = self.detector.get_dominant_color(img)
                
                # Update session state dengan thread lock
                with lock:
                    if rgb_color is not None:
                        # Dapatkan analisis warna lengkap
                        analysis = self.detector.get_color_analysis(rgb_color)
                        
                        st.session_state.hasil_deteksi = {
                            "rgb": rgb_color,
                            "nama": color_name,
                            "kode": color_code,
                            "confidence": confidence,
                            "waktu": datetime.now().strftime("%H:%M:%S"),
                            "analysis": analysis
                        }
                        st.session_state.last_frame = img.copy()
            
            # Gambar area sampling di tengah
            h, w = img.shape[:2]
            area = self.detector.sample_area
            
            # Tempatkan area di tengah gambar
            width = min(area['width'], w)
            height = min(area['height'], h)
            x = (w - width) // 2
            y = (h - height) // 2
            
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
                
                # Tampilkan kode warna
                cv2.putText(img, f"Code: {hasil['kode']}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, f"Code: {hasil['kode']}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error dalam video processing: {e}")
            return frame

# Tampilkan instruksi
st.markdown("""
### Instruksi Penggunaan:
1. **Kamera Real-time**: Klik tombol "START" untuk memulai kamera dan arahkan benang ke dalam kotak
2. **Upload Gambar**: Upload gambar benang untuk deteksi warna
3. **Tunggu beberapa detik** untuk hasil deteksi muncul

---
""")

# Tabs untuk memisahkan fitur
tab1, tab2 = st.tabs(["📹 Kamera Real-time", "📁 Upload Gambar"])

with tab1:
    # Kolom untuk layout kamera
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
        
        # Pengaturan area sampling (akan ditempatkan di tengah)
        if st.checkbox("Sesuaikan Area Sampling"):
            detector.sample_area['width'] = st.slider("Width", 50, 300, detector.sample_area.get('width', 100))
            detector.sample_area['height'] = st.slider("Height", 50, 300, detector.sample_area.get('height', 100))
            st.info("Area deteksi akan ditempatkan di tengah gambar secara otomatis")
        
        # Pengaturan sensitivitas
        if st.checkbox("Pengaturan Lanjutan"):
            detector.brightness_min = st.slider("Brightness Min", 0, 100, detector.brightness_min)
            detector.brightness_max = st.slider("Brightness Max", 100, 255, detector.brightness_max)
            detector.saturation_min = st.slider("Saturation Min", 0, 100, detector.saturation_min)
    
    # Hasil deteksi kamera
    st.header("📊 Hasil Deteksi Kamera")
    
    # Placeholder untuk hasil kamera
    hasil_placeholder = st.empty()
    
    # Update hasil secara real-time
    if st.session_state.hasil_deteksi:
        hasil = st.session_state.hasil_deteksi
        analysis = hasil.get('analysis', {})
        rgb = hasil['rgb']
        
        with hasil_placeholder.container():
            display_detection_results(hasil, analysis, rgb, is_upload=False)
    
    else:
        with hasil_placeholder.container():
            st.info("Arahkan benang ke dalam kotak hijau untuk memulai deteksi...")
            
            # Status debugging
            if st.checkbox("Show Debug Info"):
                st.write("**Debug Information:**")
                st.write(f"- Detector loaded: {detector is not None}")
                st.write(f"- Color database size: {len(detector.color_db) if detector else 0}")
                st.write(f"- Sample area: {detector.sample_area if detector else 'N/A'}")
                st.write(f"- WebRTC context: {ctx.state if 'ctx' in locals() else 'N/A'}")

with tab2:
    st.subheader("📁 Upload Gambar")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar benang untuk deteksi warna",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload gambar dengan format JPG, PNG, BMP, atau TIFF"
    )
    
    if uploaded_file is not None:
        # Tampilkan gambar yang diupload
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Gambar Asli")
            
            # Tampilkan gambar
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Info gambar
            st.write(f"**Nama file:** {uploaded_file.name}")
            st.write(f"**Ukuran:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Mode:** {image.mode}")
            
            # Tombol untuk memproses
            if st.button("🔍 Deteksi Warna", key="detect_upload"):
                with st.spinner("Memproses gambar..."):
                    success = process_uploaded_image(uploaded_file)
                    if success:
                        st.success("✅ Deteksi berhasil!")
                        st.rerun()
        
        with col2:
            st.subheader("🎨 Gambar dengan Area Deteksi")
            
            if st.session_state.uploaded_image is not None:
                # Tampilkan gambar dengan area sampling
                img_with_area = st.session_state.uploaded_image.copy()
                
                # Gambar area sampling di tengah
                h, w = img_with_area.shape[:2]
                area = detector.sample_area
                
                # Tempatkan area di tengah gambar
                width = min(area['width'], w)
                height = min(area['height'], h)
                x = (w - width) // 2
                y = (h - height) // 2
                
                # Gambar kotak hijau untuk area sampling
                cv2.rectangle(img_with_area, (x, y), (x + width, y + height), (0, 255, 0), 3)
                
                # Tambahkan text info
                cv2.putText(img_with_area, "Area Deteksi", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Konversi BGR ke RGB untuk tampilan
                img_rgb = cv2.cvtColor(img_with_area, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, use_container_width=True)
            else:
                st.info("Klik tombol 'Deteksi Warna' untuk memproses gambar")
    
    # Hasil deteksi upload
    st.header("📊 Hasil Deteksi Upload")
    
    if st.session_state.hasil_deteksi_upload:
        hasil = st.session_state.hasil_deteksi_upload
        analysis = hasil.get('analysis', {})
        rgb = hasil['rgb']
        
        display_detection_results(hasil, analysis, rgb, is_upload=True)
    
    else:
        st.info("Upload gambar dan klik 'Deteksi Warna' untuk melihat hasil deteksi...")

# Footer
st.markdown("---")
st.markdown("**Yarn Color Detector** - Deteksi warna benang secara real-time dan dari gambar upload")