import cv2
import numpy as np
import csv
import os
import sys
from collections import Counter
from skimage import color
from datetime import datetime
import streamlit as st

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
        
        # Parameter untuk filtering
        self.brightness_min = 20
        self.brightness_max = 235
        self.saturation_min = 10
        
        # Load database warna dari CSV
        self.load_color_database(csv_file)
        
    def load_color_database(self, csv_file):
        """Memuat database warna dari file CSV - Handle berbagai format CSV"""
        try:
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"File {csv_file} tidak ditemukan!")
            
            self.color_db = []
            with open(csv_file, 'r', encoding='utf-8') as file:
                # Baca semua baris
                lines = file.readlines()
                
                # Bersihkan baris dari tanda kutip berlebih
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    # Hapus tanda kutip di awal dan akhir baris jika ada
                    if line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    cleaned_lines.append(line)
                
                # Parse header
                if not cleaned_lines:
                    raise ValueError("File CSV kosong!")
                
                header = cleaned_lines[0].split(',')
                # Bersihkan header dari spasi
                header = [h.strip() for h in header]
                
                # Parse data
                for line in cleaned_lines[1:]:
                    if not line.strip():  # Skip baris kosong
                        continue
                        
                    values = line.split(',')
                    
                    # Pastikan jumlah kolom sesuai
                    if len(values) != len(header):
                        continue
                    
                    try:
                        # Buat dictionary berdasarkan header
                        row_dict = {}
                        for i, col_name in enumerate(header):
                            row_dict[col_name] = values[i].strip()
                        
                        # Validasi dan konversi data
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
                raise ValueError("Tidak ada data warna yang valid ditemukan di CSV!")
                
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            st.error("Pastikan file 'yarn_colors_database.csv' ada dan format sesuai.")
            st.stop()
    
    def rgb_to_lab(self, rgb):
        """Konversi RGB ke CIELAB dengan handling yang lebih baik"""
        try:
            # Normalisasi RGB ke range 0-1
            rgb_norm = np.array([[rgb]]) / 255.0
            # Konversi ke LAB
            lab = color.rgb2lab(rgb_norm)[0][0]
            return lab
        except:
            return np.array([0, 0, 0])
    
    def calculate_delta_e(self, lab1, lab2):
        """Menghitung Delta E (CIE76) untuk perbandingan warna yang lebih akurat"""
        delta_l = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        return np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)
    
    def find_closest_color(self, target_rgb):
        """Mencari warna terdekat berdasarkan Delta E di ruang CIELAB"""
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
            # Konversi Delta E ke confidence score (0-100)
            confidence = max(0, min(100, 100 - min_delta_e * 1.5))
            return best_match, best_match['color_name'], best_match['color_code'], confidence
        
        return None, "Tidak dikenal", None, 0
    
    def preprocess_sample(self, sample):
        """Preprocessing area sample untuk meningkatkan akurasi"""
        if sample is None or sample.size == 0:
            return None, None
            
        # Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(sample, (5, 5), 0)
        
        # Konversi ke HSV untuk filtering yang lebih baik
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Filter berdasarkan brightness dan saturation
        mask = np.ones(hsv.shape[:2], dtype=bool)
        
        # Filter brightness (V channel)
        mask &= (hsv[:,:,2] > self.brightness_min) & (hsv[:,:,2] < self.brightness_max)
        
        # Filter saturation untuk menghindari warna yang terlalu pucat
        mask &= hsv[:,:,1] > self.saturation_min
        
        return blurred, mask
    
    def get_dominant_color(self, image):
        """Mendapatkan warna dominan dengan preprocessing yang lebih baik"""
        if image is None:
            return None, "Tidak terdeteksi", None, 0
            
        # Pastikan area sampling dalam batas frame
        h, w = image.shape[:2]
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
        
        # Ambil pixel yang valid saja
        valid_pixels = processed_sample[mask]
        
        if len(valid_pixels) == 0:
            return None, "Tidak terdeteksi", None, 0
        
        # Clustering sederhana untuk mendapatkan warna dominan
        pixels_bgr = valid_pixels.reshape((-1, 3))
        
        # Gunakan median untuk mendapatkan warna representatif
        dominant_color_bgr = np.median(pixels_bgr, axis=0)
        
        # Konversi BGR ke RGB
        dominant_color_rgb = (int(dominant_color_bgr[2]), 
                            int(dominant_color_bgr[1]), 
                            int(dominant_color_bgr[0]))
        
        # Cari warna terdekat di database
        best_match, color_name, color_code, confidence = self.find_closest_color(dominant_color_rgb)
        
        return dominant_color_rgb, color_name, color_code, confidence
    
    def get_color_analysis(self, rgb_color):
        """Mendapatkan analisis lengkap warna"""
        if rgb_color is None:
            return None
            
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
        
        # Cari warna terdekat untuk mendapatkan kategori