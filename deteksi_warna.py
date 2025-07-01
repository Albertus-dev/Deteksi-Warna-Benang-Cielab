import cv2
import numpy as np
import csv
import os
import sys
from collections import Counter
from skimage import color
from datetime import datetime

class YarnColorDetector:
    def __init__(self, csv_file='yarn_colors_database.csv'):
        """Inisialisasi detector dengan database warna dari CSV"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
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
                        print(f"Warning: Baris diabaikan (kolom tidak sesuai): {line}")
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
                        print(f"Warning: Baris diabaikan (error parsing): {line} - {e}")
                        continue
            
            if len(self.color_db) == 0:
                raise ValueError("Tidak ada data warna yang valid ditemukan di CSV!")
                
            print(f"Database warna dimuat dari CSV: {len(self.color_db)} warna")
            print("Contoh warna yang dimuat:")
            for i, color in enumerate(self.color_db[:3]):  # Tampilkan 3 warna pertama
                print(f"  {i+1}. {color['color_name']} - {color['color_code']}")
            if len(self.color_db) > 3:
                print(f"  ... dan {len(self.color_db)-3} warna lainnya")
                
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Program tidak dapat berjalan tanpa database warna yang valid.")
            print("Pastikan format CSV sesuai dengan yang diharapkan.")
            sys.exit("Program dihentikan karena database warna tidak tersedia.")
    
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
            confidence = max(0, 100 - min_delta_e * 2)  # Konversi ke confidence score
            return best_match, best_match['color_name'], best_match['color_code'], confidence
        
        return None, "Tidak dikenal", None, 0
    
    def preprocess_sample(self, sample):
        """Preprocessing area sample untuk meningkatkan akurasi"""
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
        # Ekstrak area sample
        sample = image[
            self.sample_area['y']:self.sample_area['y'] + self.sample_area['height'],
            self.sample_area['x']:self.sample_area['x'] + self.sample_area['width']
        ]
        
        # Preprocessing
        processed_sample, mask = self.preprocess_sample(sample)
        
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
    
    def draw_interface(self, frame, rgb_color, color_name, color_code, confidence):
        """Menggambar interface dengan informasi yang lebih lengkap"""
        # Background untuk informasi
        info_height = 140
        cv2.rectangle(frame, (10, 10), (450, 10 + info_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 10 + info_height), (255, 255, 255), 2)
        
        if rgb_color:
            # Hitung LAB values
            lab = self.rgb_to_lab(rgb_color)
            
            # Informasi teks
            info_lines = [
                f"Warna: {color_name}",
                f"Kode: {color_code if color_code else 'N/A'}",
                f"RGB: ({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})",
                f"LAB: L*={lab[0]:.1f}, a*={lab[1]:.1f}, b*={lab[2]:.1f}",
                f"Confidence: {confidence:.1f}%"
            ]
            
            # Gambar teks
            for i, text in enumerate(info_lines):
                y_pos = 30 + i * 22
                cv2.putText(frame, text, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Color box
            color_box_bgr = (rgb_color[2], rgb_color[1], rgb_color[0])  # RGB to BGR
            cv2.rectangle(frame, (460, 10), (560, 110), color_box_bgr, -1)
            cv2.rectangle(frame, (460, 10), (560, 110), (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int((confidence / 100) * 100)
            cv2.rectangle(frame, (460, 120), (460 + bar_width, 140), (0, 255, 0), -1)
            cv2.rectangle(frame, (460, 120), (560, 140), (255, 255, 255), 2)
            cv2.putText(frame, f"{confidence:.0f}%", (570, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Sampling area
        cv2.rectangle(
            frame,
            (self.sample_area['x'], self.sample_area['y']),
            (self.sample_area['x'] + self.sample_area['width'], 
             self.sample_area['y'] + self.sample_area['height']),
            (0, 0, 0), 2
        )
        
        # Instructions
        instructions = [
            "Letakkan yarn di area hijau",
            "Q: Keluar | S: Screenshot | R: Reset area"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 40 + i * 20
            cv2.putText(frame, instruction, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_detection_result(self, frame, color_name, color_code, rgb_color, confidence):
        """Menyimpan hasil deteksi"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yarn_detection_{color_name.replace(' ', '_')}_{timestamp}.jpg"
        
        # Buat folder results jika belum ada
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        cv2.imwrite(filepath, frame)
        
        # Simpan juga data ke CSV log
        log_file = os.path.join("results", "detection_log.csv")
        log_exists = os.path.exists(log_file)
        
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'color_name', 'color_code', 'rgb_r', 'rgb_g', 'rgb_b', 'confidence', 'filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not log_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': timestamp,
                'color_name': color_name,
                'color_code': color_code,
                'rgb_r': rgb_color[0] if rgb_color else 0,
                'rgb_g': rgb_color[1] if rgb_color else 0,
                'rgb_b': rgb_color[2] if rgb_color else 0,
                'confidence': confidence,
                'filename': filename
            })
        
        print(f"Hasil disimpan: {filepath}")
        return filepath
    
    def run(self):
        """Menjalankan deteksi warna yarn"""
        print("=== Program Deteksi Warna Yarn dengan CIELAB ===")
        print(f"Database: {len(self.color_db)} warna dimuat")
        print("Kontrol:")
        print("  Q: Keluar")
        print("  S: Screenshot & simpan hasil")
        print("  R: Reset posisi area sampling")
        print("  +/-: Adjust ukuran area sampling")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Gagal mengakses kamera")
                break
            
            # Flip horizontal untuk mirror effect
            frame = cv2.flip(frame, 1)
            
            # Deteksi warna
            rgb_color, color_name, color_code, confidence = self.get_dominant_color(frame)
            
            # Gambar interface
            self.draw_interface(frame, rgb_color, color_name, color_code, confidence)
            
            # Tampilkan frame
            cv2.imshow('Yarn Color Detector - CIELAB Database', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                if rgb_color:
                    self.save_detection_result(frame, color_name, color_code, rgb_color, confidence)
                else:
                    print("Tidak ada warna yang terdeteksi untuk disimpan")
            elif key == ord('r') or key == ord('R'):
                # Reset posisi area sampling ke tengah
                self.sample_area['x'] = (frame.shape[1] - self.sample_area['width']) // 2
                self.sample_area['y'] = (frame.shape[0] - self.sample_area['height']) // 2
                print("Area sampling direset ke tengah")
            elif key == ord('+') or key == ord('='):
                # Perbesar area sampling
                if self.sample_area['width'] < 200:
                    self.sample_area['width'] += 10
                    self.sample_area['height'] += 10
                    print(f"Area sampling: {self.sample_area['width']}x{self.sample_area['height']}")
            elif key == ord('-'):
                # Perkecil area sampling
                if self.sample_area['width'] > 50:
                    self.sample_area['width'] -= 10
                    self.sample_area['height'] -= 10
                    print(f"Area sampling: {self.sample_area['width']}x{self.sample_area['height']}")
        
        self.cleanup()
    
    def cleanup(self):
        """Membersihkan resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program selesai")

def main():
    """Fungsi utama - dengan validasi CSV wajib"""
    try:
        csv_file = 'yarn_colors_database.csv'
        
        # Validasi file CSV WAJIB ada
        if not os.path.exists(csv_file):
            print(f"ERROR: File {csv_file} tidak ditemukan!")
            print("File CSV database warna WAJIB ada untuk menjalankan program.")
            print(f"Pastikan file '{csv_file}' ada di folder yang sama dengan program ini.")
            return
        
        # Buat detector dan jalankan
        detector = YarnColorDetector(csv_file)
        detector.run()
        
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user")
    except FileNotFoundError:
        print("File database warna tidak ditemukan!")
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan:")
        print("1. Kamera terhubung dengan baik")
        print("2. Semua library sudah terinstall")
        print("3. File CSV 'yarn_colors_database.csv' ada dan valid")

if __name__ == "__main__":
    main()