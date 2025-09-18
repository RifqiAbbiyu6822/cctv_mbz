"""
Modul untuk deteksi dan penghitungan mobil menggunakan YOLO - Fixed Version dengan debugging
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
import os

class CarCounter:
    """
    Kelas untuk deteksi dan penghitungan mobil dalam video stream - Fixed Version
    """
    
    def __init__(self, model_path, debug=True):
        """
        Inisialisasi CarCounter
        
        Args:
            model_path (str): Path ke model YOLO yang sudah dilatih
            debug (bool): Mode debugging untuk troubleshooting
        """
        self.debug = debug
        
        # Debug: Cek file model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {model_path}")
        
        print(f"Loading YOLO model dari: {model_path}")
        try:
            self.model = YOLO(model_path)
            if self.debug:
                print(f"Model berhasil dimuat: {self.model.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        # ID yang sudah dihitung untuk menghindari double count
        self.counted_ids = set()
        
        # Simpan posisi sebelumnya per track untuk deteksi arah
        self.previous_centers = {}
        
        # Dictionary jumlah per-arah dan total
        self.counts = {'mobil': 0, 'jakarta': 0, 'bandung': 0}
        
        # Posisi satu garis penghitungan (y) - akan diset otomatis
        self.line_y = None
        self.tolerance = 15  # Tingkatkan toleransi untuk deteksi yang lebih mudah
        
        # Fallback anti-duplikasi saat tidak ada ID tracking
        self.recent_cross_events = []  # list of dict: {t, x, y, direction}
        self.event_timeout = 3.0  # Tingkatkan timeout
        
        # Area Region of Interest (ROI) untuk filtering deteksi
        self.roi_defined = False
        self.roi_points = None
        
        # Frame counter untuk debugging
        self.frame_count = 0
        
        # Class IDs untuk kendaraan - diperluas dan disesuaikan COCO classes
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO dataset)
        # Note: Class 0 adalah 'person', bukan kendaraan
        
        # Confidence threshold yang lebih rendah untuk deteksi yang lebih sensitif
        self.conf_threshold = 0.3
        self.iou_threshold = 0.5
        
        if self.debug:
            print(f"Initialized with vehicle classes: {self.vehicle_classes}")
            print(f"Confidence threshold: {self.conf_threshold}")
    
    def test_model(self, test_frame=None):
        """Test apakah model bisa melakukan deteksi"""
        print("\n=== TESTING MODEL ===")
        
        if test_frame is None:
            # Buat test frame dummy
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Gambar kotak sederhana untuk simulasi
            cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        try:
            # Test deteksi tanpa filter kelas
            results = self.model(test_frame, verbose=True)
            print(f"Test detection results: {len(results)} results")
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                print(f"Boxes detected: {len(boxes)}")
                if hasattr(boxes, 'cls'):
                    classes = boxes.cls.cpu().numpy()
                    print(f"Classes detected: {classes}")
                    unique_classes = np.unique(classes)
                    print(f"Unique classes: {unique_classes}")
                    
                    # Cek apakah ada kendaraan
                    vehicle_detected = any(cls in self.vehicle_classes for cls in classes)
                    print(f"Vehicle detected: {vehicle_detected}")
                else:
                    print("No classes detected")
            else:
                print("No boxes detected")
                
        except Exception as e:
            print(f"Error in model test: {e}")
            return False
            
        print("=== END TEST ===\n")
        return True
    
    def set_counting_line(self, frame_height, line_ratio=0.5):
        """
        Set posisi satu garis penghitungan di tengah frame.
        """
        self.line_y = int(frame_height * float(line_ratio))
        if self.debug:
            print(f"Garis penghitungan diset pada Y={self.line_y} (ratio: {line_ratio})")
    
    def is_in_roi(self, x, y, frame_width, frame_height):
        """
        Cek apakah titik berada dalam ROI
        """
        if not self.roi_defined:
            # Default ROI: lebih permisif, hanya hindari tepi ekstrem
            margin_x = int(frame_width * 0.05)  # 5% margin
            margin_y = int(frame_height * 0.05)  # 5% margin
            return (margin_x <= x <= (frame_width - margin_x) and 
                   margin_y <= y <= (frame_height - margin_y))
        return True
    
    def process_frame(self, frame, tracking=True):
        """
        Proses frame untuk deteksi dan penghitungan mobil - dengan debugging
        """
        self.frame_count += 1
        
        if self.debug and self.frame_count % 30 == 1:  # Debug setiap 30 frame
            print(f"\n--- Processing frame {self.frame_count} ---")
        
        # Set garis penghitungan jika belum diset
        if self.line_y is None:
            self.set_counting_line(frame.shape[0])
        
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Deteksi kendaraan dengan YOLO - parameter yang lebih permisif
            if tracking:
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=self.vehicle_classes,
                    tracker="bytetrack.yaml",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    save=False,
                    show=False
                )
            else:
                results = self.model(
                    frame,
                    classes=self.vehicle_classes,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    save=False,
                    show=False
                )
                
            if self.debug and self.frame_count % 30 == 1:
                print(f"YOLO results: {len(results)} results")
                
        except Exception as e:
            print(f"Error dalam deteksi YOLO: {e}")
            if self.debug:
                print(f"Model path: {self.model}")
                print(f"Frame shape: {frame.shape}")
            return frame, self.counts
        
        # Gambar garis penghitungan dengan zona toleransi
        self._draw_counting_line(frame, frame_width)
        
        # Proses hasil deteksi
        detection_count = 0
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detection_count = len(boxes)
            
            if self.debug and self.frame_count % 30 == 1:
                print(f"Boxes detected: {detection_count}")
                if hasattr(boxes, 'cls'):
                    classes = boxes.cls.cpu().numpy()
                    print(f"Classes: {classes}")
            
            if tracking and hasattr(boxes, 'id') and boxes.id is not None:
                # Mode tracking dengan ID
                self._process_tracking_mode(frame, boxes, frame_width, frame_height)
            else:
                # Mode deteksi tanpa ID tracking
                self._process_detection_mode(frame, boxes, frame_width, frame_height)
        
        # Tambahkan informasi debug yang lebih lengkap
        self._draw_debug_info(frame, detection_count)
        
        return frame, self.counts
    
    def _draw_counting_line(self, frame, frame_width):
        """Gambar garis penghitungan dengan zona toleransi"""
        # Garis utama
        cv2.line(frame, (0, self.line_y), (frame_width, self.line_y), (0, 255, 0), 3)
        
        # Zona toleransi
        cv2.line(frame, (0, self.line_y - self.tolerance), 
                (frame_width, self.line_y - self.tolerance), (0, 200, 0), 1)
        cv2.line(frame, (0, self.line_y + self.tolerance), 
                (frame_width, self.line_y + self.tolerance), (0, 200, 0), 1)
        
        # Label
        cv2.putText(frame, f'Counting Line Y={self.line_y}', (10, self.line_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _process_tracking_mode(self, frame, boxes, frame_width, frame_height):
        """Proses deteksi dengan tracking mode - Fixed Version"""
        box_coords = boxes.xyxy.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        detected_vehicles = 0
        
        for i, (box, track_id, conf) in enumerate(zip(box_coords, track_ids, confidences)):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Filter berdasarkan ROI
            if not self.is_in_roi(cx, cy, frame_width, frame_height):
                continue
                
            detected_vehicles += 1
            
            # Debug info per vehicle
            if self.debug and self.frame_count % 30 == 1:
                print(f"Vehicle ID:{track_id} at ({cx},{cy}), conf:{conf:.2f}")
            
            # Ambil posisi sebelumnya untuk deteksi arah
            prev_cy = self.previous_centers.get(track_id)
            self.previous_centers[track_id] = cy
            
            # Deteksi crossing hanya jika ada posisi sebelumnya
            if prev_cy is not None:
                # Deteksi crossing ke arah Jakarta (bergerak turun melewati garis)
                if self._check_line_crossing(prev_cy, cy, self.line_y, 'down'):
                    if track_id not in self.counted_ids:
                        self.counts['jakarta'] += 1
                        self.counted_ids.add(track_id)
                        self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                        print(f"[TRACKING] ✓ Kendaraan ID:{track_id} ke arah Jakarta! Total JKT:{self.counts['jakarta']}")
                
                # Deteksi crossing ke arah Bandung (bergerak naik melewati garis)
                elif self._check_line_crossing(prev_cy, cy, self.line_y, 'up'):
                    if track_id not in self.counted_ids:
                        self.counts['bandung'] += 1
                        self.counted_ids.add(track_id)
                        self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                        print(f"[TRACKING] ✓ Kendaraan ID:{track_id} ke arah Bandung! Total BDG:{self.counts['bandung']}")
            
            # Gambar bounding box dan info
            color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
            self._draw_detection(frame, x1, y1, x2, y2, cx, cy, 
                               f'ID:{track_id} ({conf:.2f})', color)
            
            # Gambar garis trajectory jika ada posisi sebelumnya
            if prev_cy is not None:
                cv2.arrowedLine(frame, (cx, prev_cy), (cx, cy), (255, 255, 0), 2, tipLength=0.3)
        
        if self.debug and detected_vehicles > 0 and self.frame_count % 30 == 1:
            print(f"Processed {detected_vehicles} vehicles in tracking mode")
    
    def _process_detection_mode(self, frame, boxes, frame_width, frame_height):
        """Proses deteksi tanpa tracking mode - Fixed Version"""
        box_coords = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        # Bersihkan event lama
        now = time.time()
        self.recent_cross_events = [e for e in self.recent_cross_events 
                                  if now - e['t'] < self.event_timeout]
        
        detected_vehicles = 0
        
        for i, (box, conf) in enumerate(zip(box_coords, confidences)):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Filter berdasarkan ROI
            if not self.is_in_roi(cx, cy, frame_width, frame_height):
                continue
                
            detected_vehicles += 1
            
            # Gambar bounding box dan info
            self._draw_detection(frame, x1, y1, x2, y2, cx, cy, f'Vehicle ({conf:.2f})')
            
            # Fallback counting: jika pusat dekat garis dan belum ada event duplikat
            self._check_line_proximity_counting(cx, cy, now, frame_width)
        
        if self.debug and detected_vehicles > 0 and self.frame_count % 30 == 1:
            print(f"Processed {detected_vehicles} vehicles in detection mode")
    
    def _check_line_crossing(self, prev_y, curr_y, line_y, direction):
        """
        Cek apakah terjadi line crossing - dengan debugging
        """
        crossed = False
        if direction == 'down':
            # Bergerak turun: dari atas garis ke bawah garis
            crossed = prev_y < (line_y - self.tolerance) and curr_y > (line_y + self.tolerance)
        elif direction == 'up':
            # Bergerak naik: dari bawah garis ke atas garis
            crossed = prev_y > (line_y + self.tolerance) and curr_y < (line_y - self.tolerance)
        
        if self.debug and crossed:
            print(f"Line crossing detected: {direction}, prev_y:{prev_y}, curr_y:{curr_y}, line_y:{line_y}")
        
        return crossed
    
    def _check_line_proximity_counting(self, cx, cy, current_time, frame_width):
        """Cek proximity counting untuk mode tanpa tracking - Fixed"""
        
        def is_duplicate_event(direction):
            """Cek apakah ada event duplikat dalam waktu dekat"""
            for event in self.recent_cross_events:
                if (event['direction'] == direction and 
                    abs(event['x'] - cx) < 100 and  # Perluas area duplikasi
                    abs(event['y'] - self.line_y) < self.tolerance * 2):
                    return True
            return False
        
        # Cek proximity ke garis dengan toleransi yang lebih besar
        if abs(cy - self.line_y) <= self.tolerance:
            # Tentukan arah berdasarkan posisi X (lebih deterministik)
            if cx < frame_width // 2:
                direction = 'jakarta'  # Kiri menuju Jakarta
            else:
                direction = 'bandung'  # Kanan menuju Bandung
            
            if not is_duplicate_event(direction):
                if direction == 'jakarta':
                    self.counts['jakarta'] += 1
                else:
                    self.counts['bandung'] += 1
                
                self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                self.recent_cross_events.append({
                    't': current_time, 
                    'x': cx, 
                    'y': self.line_y, 
                    'direction': direction
                })
                print(f"[PROXIMITY] ✓ Kendaraan ke arah {direction.title()}! Total: {self.counts[direction]}")
    
    def _draw_detection(self, frame, x1, y1, x2, y2, cx, cy, label, color=(255, 0, 0)):
        """Gambar bounding box dan informasi deteksi"""
        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label dengan confidence
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gambar titik pusat
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        
        # Info posisi relatif terhadap garis
        distance_to_line = abs(cy - self.line_y)
        if distance_to_line <= self.tolerance:
            cv2.putText(frame, f'ON LINE ({distance_to_line}px)', (cx-50, cy-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def _draw_debug_info(self, frame, detection_count):
        """Gambar informasi debug yang lengkap"""
        # Info utama
        debug_text = f"Frame: {self.frame_count} | Detected: {detection_count} | Total: {self.counts['mobil']} | JKT: {self.counts['jakarta']} | BDG: {self.counts['bandung']}"
        cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Info model
        cv2.putText(frame, f"Model: YOLO | Conf: {self.conf_threshold} | Classes: {len(self.vehicle_classes)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Info tracking
        cv2.putText(frame, f"Tracked IDs: {len(self.previous_centers)} | Events: {len(self.recent_cross_events)}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Info garis dan toleransi
        frame_height = frame.shape[0]
        cv2.putText(frame, f"Line Y: {self.line_y} | Tolerance: ±{self.tolerance}px", 
                   (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def reset_counter(self):
        """Reset counter dan ID yang sudah dihitung"""
        self.counted_ids.clear()
        self.previous_centers.clear()
        self.recent_cross_events.clear()
        self.counts = {'mobil': 0, 'jakarta': 0, 'bandung': 0}
        self.frame_count = 0
        print("✓ Counter direset!")
    
    def get_count(self):
        """Mendapatkan jumlah mobil yang terdeteksi"""
        return self.counts
    
    def set_roi(self, points):
        """Set Region of Interest untuk filtering deteksi"""
        self.roi_points = points
        self.roi_defined = True
        print(f"✓ ROI diset dengan {len(points)} points")
    
    def get_statistics(self):
        """Mendapatkan statistik deteksi"""
        return {
            'total_vehicles': self.counts['mobil'],
            'jakarta_direction': self.counts['jakarta'],
            'bandung_direction': self.counts['bandung'],
            'frames_processed': self.frame_count,
            'tracked_objects': len(self.previous_centers),
            'recent_events': len(self.recent_cross_events),
            'detection_rate': detection_rate if self.frame_count > 0 else 0
        }

# Contoh penggunaan dengan debugging
def main():
    """Contoh penggunaan CarCounter dengan debugging"""
    
    # Ganti dengan path model YOLO Anda
    model_path = "yolov8n.pt"  # atau "yolov8s.pt", "yolov8m.pt", dll
    
    try:
        # Inisialisasi dengan debug mode
        counter = CarCounter(model_path, debug=True)
        
        # Test model
        counter.test_model()
        
        # Buka video atau webcam
        cap = cv2.VideoCapture(0)  # Ganti dengan path video file jika perlu
        
        if not cap.isOpened():
            print("Error: Tidak bisa membuka video source")
            return
        
        print("Tekan 'q' untuk quit, 'r' untuk reset counter")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Proses frame
            processed_frame, counts = counter.process_frame(frame, tracking=True)
            
            # Tampilkan hasil
            cv2.imshow('Car Counter - Debug Mode', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset_counter()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        stats = counter.get_statistics()
        print(f"\n=== FINAL STATISTICS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()