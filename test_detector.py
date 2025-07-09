import cv2
import numpy as np
from deteksi_warna import YarnColorDetector

def test_detector():
    """Test detector dengan warna solid"""
    print("Testing YarnColorDetector...")
    
    # Inisialisasi detector
    detector = YarnColorDetector("yarn_colors_database.csv")
    
    if len(detector.color_db) == 0:
        print("ERROR: Database warna kosong!")
        return
    
    print(f"Database loaded: {len(detector.color_db)} colors")
    
    # Test dengan beberapa warna solid
    test_colors = [
        (255, 0, 0),    # Merah
        (0, 255, 0),    # Hijau
        (0, 0, 255),    # Biru
        (255, 255, 0),  # Kuning
        (255, 255, 255), # Putih
        (0, 0, 0),      # Hitam
        (128, 128, 128) # Abu-abu
    ]
    
    for rgb in test_colors:
        # Buat image dummy dengan warna solid
        h, w = 480, 640
        test_image = np.full((h, w, 3), rgb[::-1], dtype=np.uint8)  # BGR format
        
        # Test deteksi
        result_rgb, color_name, color_code, confidence = detector.get_dominant_color(test_image)
        
        print(f"Input RGB: {rgb}")
        print(f"Detected: {color_name} ({color_code})")
        print(f"Output RGB: {result_rgb}")
        print(f"Confidence: {confidence:.1f}%")
        print("-" * 40)

def test_with_camera():
    """Test dengan kamera real"""
    print("Testing dengan kamera...")
    
    # Inisialisasi detector
    detector = YarnColorDetector("yarn_colors_database.csv")
    
    if len(detector.color_db) == 0:
        print("ERROR: Database warna kosong!")
        return
    
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Tidak bisa membuka kamera!")
        return
    
    print("Kamera terbuka. Tekan 'q' untuk keluar, 's' untuk screenshot...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Deteksi warna
        rgb_color, color_name, color_code, confidence = detector.get_dominant_color(frame)
        
        # Gambar area sampling
        area = detector.sample_area
        cv2.rectangle(frame, (area['x'], area['y']),
                     (area['x'] + area['width'], area['y'] + area['height']),
                     (0, 255, 0), 2)
        
        # Tampilkan hasil
        if rgb_color:
            text = f"{color_name} ({confidence:.1f}%)"
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"RGB: {rgb_color}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tampilkan frame
        cv2.imshow('Yarn Color Detector Test', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if rgb_color:
                print(f"Screenshot: {color_name} - {rgb_color} - {confidence:.1f}%")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== Yarn Color Detector Test ===")
    
    # Test 1: Warna solid
    print("\n1. Testing dengan warna solid:")
    test_detector()
    
    # Test 2: Kamera (uncomment untuk test dengan kamera)
    # print("\n2. Testing dengan kamera:")
    # test_with_camera()