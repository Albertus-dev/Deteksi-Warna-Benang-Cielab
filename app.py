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
st.set_page_config(page_title="Deteksi Warna Benang", layout="wide")
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
if "area_config" not in st.session_state:
    st.session_state.area_config = detector.sample_area.copy()
if "filter_config" not in st.session_state:
    st.session_state.filter_config = {
        "brightness_min": detector.brightness_min,
        "brightness_max": detector.brightness_max,
        "saturation_min": detector.saturation_min
    }

# Sidebar untuk kontrol
st.sidebar.header("🎛️ Kontrol Deteksi")

# Kontrol Area Sampling
st.sidebar.subheader("Area Sampling")
col1, col2 = st.sidebar.columns(2)

with col1:
    new_x = st.number_input("X Position", 
                           min_value=0, max_value=640-100, 
                           value=st.session_state.area_config["x"], 
                           step=10)
    new_width = st.number_input("Width", 
                               min_value=50, max_value=200, 
                               value=st.session_state.area_config["width"], 
                               step=10)

with col2:
    new_y = st.number_input("Y Position", 
                           min_value=0, max_value=480-100, 
                           value=st.session_state.area_config["y"], 
                           step=10)
    new_height = st.number_input("Height", 
                                min_value=50, max_value=200, 
                                value=st.session_state.area_config["height"], 
                                step=10)

# Update area config
st.session_state.area_config = {
    "x": new_x,
    "y": new_y,
    "width": new_width,
    "height": new_height
}

# Tombol reset area ke tengah
if st.sidebar.button("🎯 Reset Area ke Tengah"):
    st.session_state.area_config = {
        "x": (640 - 100) // 2,
        "y": (480 - 100) // 2,
        "width": 100,
        "height": 100
    }
    st.experimental_rerun()

# Kontrol Filter
st.sidebar.subheader("Filter Preprocessing")
new_brightness_min = st.sidebar.slider("Brightness Min", 
                                      min_value=0, max_value=100, 
                                      value=st.session_state.filter_config["brightness_min"])
new_brightness_max = st.sidebar.slider("Brightness Max", 
                                      min_value=150, max_value=255, 
                                      value=st.session_state.filter_config["brightness_max"])
new_saturation_min = st.sidebar.slider("Saturation Min", 
                                      min_value=0, max_value=50, 
                                      value=st.session_state.filter_config["saturation_min"])

# Update filter config
st.session_state.filter_config = {
    "brightness_min": new_brightness_min,
    "brightness_max": new_brightness_max,
    "saturation_min": new_saturation_min
}

# Monitoring dan statistik real-time
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_detections": 0,
        "avg_confidence": 0,
        "start_time": datetime.now()
    }

# Update statistik jika ada deteksi
if hasil and hasil["rgb"] is not None:
    st.session_state.stats["total_detections"] += 1
    # Hitung rata-rata confidence
    if st.session_state.stats["total_detections"] > 1:
        st.session_state.stats["avg_confidence"] = (
            (st.session_state.stats["avg_confidence"] * (st.session_state.stats["total_detections"] - 1) + 
             hasil["confidence"]) / st.session_state.stats["total_detections"]
        )
    else:
        st.session_state.stats["avg_confidence"] = hasil["confidence"]

# Sidebar statistik
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistik Sesi")
st.sidebar.metric("Total Deteksi", st.session_state.stats["total_detections"])
st.sidebar.metric("Rata-rata Confidence", f"{st.session_state.stats['avg_confidence']:.1f}%")

# Waktu sesi
session_time = datetime.now() - st.session_state.stats["start_time"]
st.sidebar.metric("Waktu Sesi", f"{session_time.seconds//60}m {session_time.seconds%60}s")

# Reset statistik
if st.sidebar.button("🔄 Reset Statistik"):
    st.session_state.stats = {
        "total_detections": 0,
        "avg_confidence": 0,
        "start_time": datetime.now()
    }
    st.experimental_rerun()

# Informasi sistem
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Info Sistem")
st.sidebar.info(f"""
**Konfigurasi Aktif:**
- Area: {st.session_state.area_config['width']}×{st.session_state.area_config['height']}
- Posisi: ({st.session_state.area_config['x']}, {st.session_state.area_config['y']})
- Brightness: {st.session_state.filter_config['brightness_min']}-{st.session_state.filter_config['brightness_max']}
- Saturation: {st.session_state.filter_config['saturation_min']}+

**Database:**
- Total warna: {len(detector.color_db)}
- Algoritma: CIELAB Delta-E
""")

# Video processor dengan kontrol yang bisa disesuaikan
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None
        self.frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame = img.copy()

        # Update detector dengan konfigurasi terbaru
        detector.sample_area = st.session_state.area_config.copy()
        detector.brightness_min = st.session_state.filter_config["brightness_min"]
        detector.brightness_max = st.session_state.filter_config["brightness_max"]
        detector.saturation_min = st.session_state.filter_config["saturation_min"]

        rgb, nama, kode, confidence = detector.get_dominant_color(img)

        # Simpan hasil deteksi ke session
        st.session_state.hasil_deteksi = {
            "rgb": rgb,
            "nama": nama,
            "kode": kode,
            "confidence": confidence
        }
        st.session_state.frame_terakhir = self.frame.copy()

        # Tampilkan area sampling dengan warna yang berbeda berdasarkan confidence
        area = detector.sample_area
        
        # Warna border berdasarkan confidence
        if confidence > 80:
            border_color = (0, 255, 0)  # Hijau - confidence tinggi
        elif confidence > 60:
            border_color = (0, 255, 255)  # Kuning - confidence sedang
        else:
            border_color = (0, 0, 255)  # Merah - confidence rendah
        
        # Gambar border area sampling
        cv2.rectangle(
            img,
            (area["x"], area["y"]),
            (area["x"] + area["width"], area["y"] + area["height"]),
            border_color, 3
        )
        
        # Gambar crosshair di tengah area
        center_x = area["x"] + area["width"] // 2
        center_y = area["y"] + area["height"] // 2
        cv2.line(img, (center_x - 10, center_y), (center_x + 10, center_y), border_color, 2)
        cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 10), border_color, 2)

        # Tampilkan informasi di video
        if rgb:
            lab = detector.rgb_to_lab(rgb)
            info_lines = [
                f"Warna: {nama}",
                f"Kode: {kode if kode else 'N/A'}",
                f"RGB: {rgb}",
                f"LAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}",
                f"Confidence: {confidence:.1f}%"
            ]
            
            # Background untuk teks
            cv2.rectangle(img, (10, 10), (500, 140), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (500, 140), (255, 255, 255), 2)
            
            # Tulis informasi
            for i, text in enumerate(info_lines):
                y_pos = 30 + i * 22
                cv2.putText(img, text, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Color box
            color_box_bgr = (rgb[2], rgb[1], rgb[0])  # RGB to BGR
            cv2.rectangle(img, (510, 10), (610, 110), color_box_bgr, -1)
            cv2.rectangle(img, (510, 10), (610, 110), (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((confidence / 100) * 100)
            if confidence > 80:
                bar_color = (0, 255, 0)  # Hijau
            elif confidence > 60:
                bar_color = (0, 255, 255)  # Kuning
            else:
                bar_color = (0, 0, 255)  # Merah
            
            cv2.rectangle(img, (510, 120), (510 + bar_width, 140), bar_color, -1)
            cv2.rectangle(img, (510, 120), (610, 140), (255, 255, 255), 2)
            cv2.putText(img, f"{confidence:.0f}%", (620, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            cv2.putText(img, "Tidak terdeteksi", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Instruksi
        instructions = [
            "Letakkan benang di area persegi",
            "Gunakan kontrol sidebar untuk adjust"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = img.shape[0] - 40 + i * 20
            cv2.putText(img, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout utama
col1, col2 = st.columns([2, 1])

with col1:
    # Jalankan Streamlit WebRTC
    ctx = webrtc_streamer(
        key="warna-webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    # UI hasil deteksi
    st.subheader("🎯 Hasil Deteksi")
    
    hasil = st.session_state.get("hasil_deteksi", None)
    
    if hasil and hasil["rgb"] is not None:
        # Tampilkan color box besar
        st.markdown(
            f"""
            <div style='width:150px;height:150px;background-color:rgb{hasil['rgb']};
                        border:3px solid #000;border-radius:10px;margin:10px 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);'></div>
            """,
            unsafe_allow_html=True
        )
        
        # Informasi detail
        st.write(f"**Nama Warna:** {hasil['nama']}")
        st.write(f"**Kode Warna:** {hasil['kode']}")
        st.write(f"**RGB:** {hasil['rgb']}")
        
        lab = detector.rgb_to_lab(hasil["rgb"])
        st.write(f"**CIELAB:** L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}")
        
        # Confidence dengan progress bar
        confidence = hasil['confidence']
        st.write(f"**Confidence:** {confidence:.1f}%")
        
        # Progress bar dengan warna berdasarkan confidence
        if confidence > 80:
            st.success(f"Confidence: {confidence:.1f}%")
        elif confidence > 60:
            st.warning(f"Confidence: {confidence:.1f}%")
        else:
            st.error(f"Confidence: {confidence:.1f}%")
        
        # Progress bar visual
        progress_bar = st.progress(confidence / 100)
        
        # Tombol aksi
        st.subheader("💾 Aksi")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("📸 Screenshot", use_container_width=True):
                if st.session_state.frame_terakhir is not None:
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs("results", exist_ok=True)
                    path = f"results/screenshot_{now}.jpg"
                    cv2.imwrite(path, st.session_state.frame_terakhir)
                    st.success(f"Screenshot disimpan!")
                    
                    # Tampilkan preview
                    st.image(path, caption="Screenshot", use_column_width=True)
                else:
                    st.error("Tidak ada frame untuk screenshot")
        
        with col_btn2:
            if st.button("💾 Simpan Hasil", use_container_width=True):
                if st.session_state.frame_terakhir is not None:
                    path = detector.save_detection_result(
                        st.session_state.frame_terakhir,
                        hasil["nama"],
                        hasil["kode"],
                        hasil["rgb"],
                        hasil["confidence"]
                    )
                    st.success(f"Hasil disimpan!")
                    st.info(f"File: {os.path.basename(path)}")
                else:
                    st.error("Tidak ada frame untuk disimpan")
        
        # Tampilkan statistik LAB
        st.subheader("📊 Statistik Warna")
        
        # LAB values dalam format yang mudah dibaca
        lab_col1, lab_col2, lab_col3 = st.columns(3)
        
        with lab_col1:
            st.metric("Lightness (L*)", f"{lab[0]:.1f}", help="0-100, hitam ke putih")
        
        with lab_col2:
            st.metric("Green-Red (a*)", f"{lab[1]:.1f}", help="Negatif=hijau, Positif=merah")
        
        with lab_col3:
            st.metric("Blue-Yellow (b*)", f"{lab[2]:.1f}", help="Negatif=biru, Positif=kuning")
        
        # Kategori warna
        # Cari kategori dari database
        for color_data in detector.color_db:
            if color_data['color_name'] == hasil['nama']:
                st.write(f"**Kategori:** {color_data['category']}")
                break
    
    else:
        st.info("⏳ Tunggu kamera mendeteksi warna...")
        st.markdown(
            """
            <div style='width:150px;height:150px;background-color:#f0f0f0;
                        border:3px dashed #ccc;border-radius:10px;margin:10px 0;
                        display:flex;align-items:center;justify-content:center;'>
                <span style='color:#999;'>Tidak ada warna</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# Informasi tambahan di bagian bawah
st.markdown("---")
st.subheader("📝 Informasi Sistem")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.write("**Database Warna:**")
    st.write(f"- Total warna: {len(detector.color_db)}")
    st.write(f"- Area sampling: {st.session_state.area_config['width']}x{st.session_state.area_config['height']}")

with info_col2:
    st.write("**Filter Aktif:**")
    st.write(f"- Brightness: {st.session_state.filter_config['brightness_min']}-{st.session_state.filter_config['brightness_max']}")
    st.write(f"- Saturation min: {st.session_state.filter_config['saturation_min']}")

with info_col3:
    st.write("**Hasil Folder:**")
    if os.path.exists("results"):
        files = [f for f in os.listdir("results") if f.endswith(('.jpg', '.csv'))]
        st.write(f"- File tersimpan: {len(files)}")
    else:
        st.write("- Folder: Belum ada")

# Riwayat deteksi
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# Tambahkan ke riwayat jika ada deteksi baru
if hasil and hasil["rgb"] is not None:
    # Cek apakah ini deteksi yang berbeda dari yang terakhir
    if (not st.session_state.detection_history or 
        st.session_state.detection_history[-1]["nama"] != hasil["nama"]):
        
        # Batasi riwayat maksimal 10 item
        if len(st.session_state.detection_history) >= 10:
            st.session_state.detection_history.pop(0)
        
        # Tambahkan deteksi baru
        st.session_state.detection_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "nama": hasil["nama"],
            "kode": hasil["kode"],
            "rgb": hasil["rgb"],
            "confidence": hasil["confidence"]
        })

# Tampilkan riwayat
if st.session_state.detection_history:
    st.subheader("📋 Riwayat Deteksi")
    
    for i, detection in enumerate(reversed(st.session_state.detection_history[-5:])):  # 5 terakhir
        with st.expander(f"🕐 {detection['timestamp']} - {detection['nama']} ({detection['confidence']:.1f}%)"):
            col_hist1, col_hist2 = st.columns([1, 2])
            
            with col_hist1:
                st.markdown(
                    f"""
                    <div style='width:50px;height:50px;background-color:rgb{detection['rgb']};
                                border:2px solid #000;border-radius:5px;'></div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_hist2:
                st.write(f"**Nama:** {detection['nama']}")
                st.write(f"**Kode:** {detection['kode']}")
                st.write(f"**RGB:** {detection['rgb']}")
                st.write(f"**Confidence:** {detection['confidence']:.1f}%")
    
    # Tombol clear history
    if st.button("🗑️ Clear History"):
        st.session_state.detection_history = []
        st.experimental_rerun()

# Database explorer
with st.expander("🎨 Database Warna"):
    st.write(f"Total warna dalam database: **{len(detector.color_db)}**")
    
    # Filter berdasarkan kategori
    categories = list(set([color['category'] for color in detector.color_db]))
    selected_category = st.selectbox("Filter berdasarkan kategori:", ["Semua"] + sorted(categories))
    
    # Tampilkan warna berdasarkan kategori
    filtered_colors = detector.color_db if selected_category == "Semua" else [
        color for color in detector.color_db if color['category'] == selected_category
    ]
    
    # Tampilkan dalam grid
    cols_per_row = 5
    for i in range(0, len(filtered_colors), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, color in enumerate(filtered_colors[i:i+cols_per_row]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div style='width:60px;height:60px;background-color:rgb({color['r']},{color['g']},{color['b']});
                                border:1px solid #000;border-radius:5px;margin:5px auto;'></div>
                    <div style='text-align:center;font-size:10px;'>
                        <strong>{color['color_name']}</strong><br>
                        {color['color_code']}<br>
                        Conf: {100 - (abs(color['l_lab']-50) + abs(color['a_lab']) + abs(color['b_lab']))/3:.0f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Tips penggunaan
with st.expander("💡 Tips Penggunaan"):
    st.markdown("""
    **Untuk hasil terbaik:**
    1. **Pencahayaan:** Gunakan pencahayaan yang cukup dan merata
    2. **Posisi:** Letakkan benang di tengah area persegi
    3. **Background:** Gunakan background yang kontras dengan warna benang
    4. **Jarak:** Jaga jarak yang konsisten antara kamera dan benang
    5. **Stabilitas:** Hindari gerakan yang berlebihan saat deteksi
    
    **Kontrol Area Sampling:**
    - Sesuaikan posisi dan ukuran area sampling sesuai kebutuhan
    - Warna border: Hijau (confidence tinggi), Kuning (sedang), Merah (rendah)
    - Crosshair menunjukkan titik tengah area sampling
    
    **Filter Preprocessing:**
    - Brightness: Filter berdasarkan kecerahan pixel (0-255)
    - Saturation: Filter untuk menghindari warna pucat (0-255)
    - Reset ke default jika hasil tidak optimal
    
    **Fitur Tambahan:**
    - Riwayat deteksi: Melihat 5 deteksi terakhir
    - Database explorer: Menjelajahi semua warna dalam database
    - Screenshot: Menyimpan gambar dengan informasi deteksi
    - Export hasil: Menyimpan data ke CSV log
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
        🎯 Yarn Color Detector v2.0 | Menggunakan CIELAB Color Space untuk akurasi maksimal
    </div>
    """,
    unsafe_allow_html=True
)