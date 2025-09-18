"""
Script untuk melatih model YOLOv8 untuk deteksi mobil
Jalankan script ini untuk melatih model dengan dataset yang tersedia
"""

from ultralytics import YOLO
import os

def train_yolo_model():
    """
    Melatih model YOLOv8 dengan dataset mobil
    """
    print("Memulai pelatihan model YOLOv8...")
    
    # Load model YOLOv8n (nano version - lebih cepat)
    model = YOLO('yolov8n.pt')
    
    # Mulai pelatihan
    results = model.train(
        data='data.yaml',           # File konfigurasi dataset
        epochs=100,                 # Jumlah epoch
        imgsz=640,                  # Ukuran gambar
        batch=16,                   # Batch size
        name='car_detection',       # Nama eksperimen
        save=True,                  # Simpan model
        device='CPU',               # Gunakan CPU (ubah ke 'cuda' jika ada GPU)
        workers=4,                  # Jumlah worker untuk data loading
        patience=20,                # Early stopping patience
        save_period=10              # Simpan checkpoint setiap 10 epoch
    )
    
    print("Pelatihan selesai!")
    print(f"Model terbaik disimpan di: {results.save_dir}/weights/best.pt")
    
    # Copy model terbaik ke folder weights
    import shutil
    os.makedirs('weights', exist_ok=True)
    shutil.copy(f"{results.save_dir}/weights/best.pt", "weights/best.pt")
    print("Model berhasil disalin ke folder weights/")

if __name__ == "__main__":
    train_yolo_model()
