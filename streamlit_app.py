import streamlit as st
from PIL import Image
import numpy as np
import io
from deteksi_warna import YarnColorDetector

st.set_page_config(page_title="Yarn Color Detector", layout="centered")

st.title("🎨 Deteksi Warna Yarn (Streamlit Version)")

uploaded_file = st.file_uploader("Upload gambar yarn", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Convert PIL Image ke NumPy array (BGR)
    image_np = np.array(image)
    image_bgr = image_np[:, :, ::-1]  # RGB to BGR

    # Inisialisasi detektor
    detector = YarnColorDetector("yarn_colors_database.csv")

    # Gunakan seluruh gambar sebagai area sampling
    detector.sample_area = {
        'x': 0,
        'y': 0,
        'width': image_bgr.shape[1],
        'height': image_bgr.shape[0]
    }

    rgb_color, color_name, color_code, confidence = detector.get_dominant_color(image_bgr)

    st.subheader("🎯 Hasil Deteksi Warna")
    st.write(f"**Nama Warna**: {color_name}")
    st.write(f"**Kode Warna**: {color_code if color_code else 'Tidak tersedia'}")
    st.write(f"**RGB**: {rgb_color}")
    st.write(f"**Confidence**: {confidence:.2f}%")
    st.markdown(f"<div style='width:100px;height:100px;background-color:rgb{rgb_color};border:1px solid #000'></div>", unsafe_allow_html=True)
