import cv2
import numpy as np
import csv
import os
from collections import Counter
from skimage import color
from datetime import datetime
import streamlit as st

class YarnColorDetector:
    def __init__(self, csv_file='yarn_colors_database.csv'):
        """Inisialisasi detector dengan database warna dari CSV"""
        # Area sampling yang bisa disesuaikan
        self.sample_area = {
            'x': 250,
            'y': 150,
            'width': 120,
            'height': 120
        }
        
        # Parameter untuk filtering
        self.brightness_min = 20
        self.brightness_max = 235
        self.saturation_min = 10
        
        # Load database warna dari CSV
        self.color_db = []
        self.load_color_database(csv_file)
        
    def load_color_database(self, csv_file):
        """Memuat database warna dari file CSV"""
        try:
            if not os.path.exists(csv_file):
                st.error(f"File {csv_file} tidak ditemukan!")
                return
            
            with open(csv_file, 'r', encoding='utf-8') as file:
                # Baca semua baris
                lines = file.readlines()
                
                if not lines:
                    st.error("File CSV kosong!")
                    return
                
                # Bersihkan dan parse header
                header_line = lines[0].strip()
                if header_line.startswith('"') and header_line.endswith('"'):
                    header_line = header_line[1:-1]
                
                header = [h.strip().strip('"') for h in header_line.split(',')]
                
                # Parse data
                for line in lines[1:]:
                    if not line.strip():
                        continue
                    
                    # Bersihkan line
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    
                    # Split dan bersihkan values
                    values = [v.strip().strip('"') for v in line.split(',')]
                    
                    if len(values) != len(header):
                        continue
                    
                    try:
                        # Buat dictionary
                        row_dict = dict(zip(header, values))
                        
                        # Konversi ke format yang diperlukan
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
                        # Skip baris yang error
                        continue
            
            if len(self.color_db) == 0:
                st.error("Tidak ada data warna yang valid ditemukan!")
                return
            
            print(f"Loaded {len(self.color_db)} colors from database")
                
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return
    
    def rgb_to_lab(self, rgb):
        """Konversi RGB ke CIELAB"""
        try:
            # Normalisasi RGB ke range 0-1
            rgb_norm = np.array([[rgb]], dtype=np.float32) / 255.0
            # Konversi ke LAB
            lab = color.rgb2lab(rgb_norm)[0][0]
            return lab
        except Exception as e:
            print(f"Error converting RGB to LAB: {e}")
            return np.array([0, 0, 0])
    
    def calculate_delta_e(self, lab1, lab2):
        """Menghitung Delta E (CIE76) untuk perbandingan warna"""
        try:
            delta_l = lab1[0] - lab2[0]
            delta_a = lab1[1] - lab2[1]
            delta_b = lab1[2] - lab2[2]
            
            return np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
        except:
            return float('inf')
    
    def find_closest_color(self, target_rgb):
        """Mencari warna terdekat berdasarkan Delta E"""
        if target_rgb is None or len(self.color_db) == 0:
            return None, "Tidak terdeteksi", None, 0
        
        try:
            target_lab = self.rgb_to_lab(target_rgb)
            min_delta_e = float('inf')
            best_match = None
            
            for color_data in self.color_db:
                try:
                    db_lab = np.array([color_data['l_lab'], color_data['a_lab'], color_data['b_lab']])
                    delta_e = self.calculate_delta_e(target_lab, db_lab)
                    
                    if delta_e < min_delta_e:
                        min_delta_e = delta_e
                        best_match = color_data
                except:
                    continue
            
            if best_match is not None:
                # Konversi Delta E ke confidence score
                confidence = max(0, min(100, 100 - min_delta_e * 2))
                return best_match, best_match['color_name'], best_match['color_code'], confidence
            
        except Exception as e:
            print(f"Error finding closest color: {e}")
        
        return None, "Tidak dikenal", None, 0
    
    def preprocess_sample(self, sample):
        """Preprocessing area sample untuk meningkatkan akurasi"""
        if sample is None or sample.size == 0:
            return None, None
        
        try:
            # Gaussian blur untuk mengurangi noise
            blurred = cv2.GaussianBlur(sample, (5, 5), 0)
            
            # Konversi ke HSV untuk filtering
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # Buat mask untuk filtering
            mask = np.ones(hsv.shape[:2], dtype=bool)
            
            # Filter berdasarkan brightness (V channel)
            mask &= (hsv[:,:,2] > self.brightness_min) & (hsv[:,:,2] < self.brightness_max)
            
            # Filter berdasarkan saturation (S channel)
            mask &= hsv[:,:,1] > self.saturation_min
            
            return blurred, mask
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None, None
    
    def get_dominant_color(self, image):
        """Mendapatkan warna dominan dari area sampling"""
        if image is None:
            return None, "Tidak terdeteksi", None, 0
        
        try:
            # Dapatkan dimensi image
            h, w = image.shape[:2]
            
            # Pastikan area sampling dalam batas frame
            x = max(0, min(self.sample_area['x'], w - self.sample_area['width']))
            y = max(0, min(self.sample_area['y'], h - self.sample_area['height']))
            width = min(self.sample_area['width'], w - x)
            height = min(self.sample_area['height'], h - y)
            
            # Ekstrak area sample
            sample = image[y:y + height, x:x + width]
            
            if sample.size == 0:
                return None, "Tidak terdeteksi", None, 0
            
            # Preprocessing
            processed_sample, mask = self.preprocess_sample(sample)
            
            if processed_sample is None or mask is None:
                return None, "Tidak terdeteksi", None, 0
            
            # Ambil pixel yang valid
            if not np.any(mask):
                # Jika tidak ada pixel valid, gunakan seluruh sample
                valid_pixels = processed_sample.reshape((-1, 3))
            else:
                valid_pixels = processed_sample[mask]
            
            if len(valid_pixels) == 0:
                return None, "Tidak terdeteksi", None, 0
            
            # Clustering sederhana untuk mendapatkan warna dominan
            # Gunakan mean untuk mendapatkan warna rata-rata
            dominant_color_bgr = np.mean(valid_pixels, axis=0)
            
            # Konversi BGR ke RGB
            dominant_color_rgb = (
                int(dominant_color_bgr[2]), 
                int(dominant_color_bgr[1]), 
                int(dominant_color_bgr[0])
            )
            
            # Cari warna terdekat di database
            best_match, color_name, color_code, confidence = self.find_closest_color(dominant_color_rgb)
            
            return dominant_color_rgb, color_name, color_code, confidence
            
        except Exception as e:
            print(f"Error in get_dominant_color: {e}")
            return None, "Error", None, 0
    
    def get_color_analysis(self, rgb_color):
        """Mendapatkan analisis lengkap warna"""
        if rgb_color is None:
            return None
        
        try:
            analysis = {
                'rgb': rgb_color,
                'hex': f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}",
                'lab': self.rgb_to_lab(rgb_color),
                'hsv': None,
                'category': None,
                'similar_colors': []
            }
            
            # Konversi ke HSV
            rgb_array = np.uint8([[rgb_color]])
            hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]
            analysis['hsv'] = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
            
            # Cari warna terdekat untuk kategori
            best_match, _, _, _ = self.find_closest_color(rgb_color)
            if best_match:
                analysis['category'] = best_match['category']
                
            # Cari warna serupa
            target_lab = self.rgb_to_lab(rgb_color)
            similar_colors = []
            
            for color_data in self.color_db:
                try:
                    db_lab = np.array([color_data['l_lab'], color_data['a_lab'], color_data['b_lab']])
                    delta_e = self.calculate_delta_e(target_lab, db_lab)
                    
                    if delta_e < 20:  # Threshold untuk warna serupa
                        similar_colors.append({
                            'name': color_data['color_name'],
                            'code': color_data['color_code'],
                            'delta_e': delta_e
                        })
                except:
                    continue
            
            # Urutkan berdasarkan Delta E
            similar_colors.sort(key=lambda x: x['delta_e'])
            analysis['similar_colors'] = similar_colors[:5]
            
            return analysis
            
        except Exception as e:
            print(f"Error in color analysis: {e}")
            return None
    
    def save_detection_result(self, image, rgb_color, color_name, color_code, confidence):
        """Menyimpan hasil deteksi"""
        try:
            if image is None:
                return None, None
            
            # Buat folder results
            os.makedirs("results", exist_ok=True)
            
            # Timestamp untuk filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Simpan gambar
            img_filename = f"results/detection_{timestamp}.jpg"
            cv2.imwrite(img_filename, image)
            
            # Simpan info hasil
            info_filename = f"results/detection_{timestamp}.txt"
            with open(info_filename, 'w', encoding='utf-8') as f:
                f.write(f"Hasil Deteksi Warna Benang\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Nama Warna: {color_name}\n")
                f.write(f"Kode Warna: {color_code}\n")
                f.write(f"RGB: {rgb_color}\n")
                f.write(f"Confidence: {confidence:.1f}%\n")
                f.write(f"File Gambar: {img_filename}\n")
            
            return img_filename, info_filename
            
        except Exception as e:
            print(f"Error saving detection result: {e}")
            return None, None