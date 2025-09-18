"""
Modul untuk deteksi dan penghitungan mobil menggunakan YOLO
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO

class CarCounter:
    """
    Kelas untuk deteksi dan penghitungan mobil dalam video stream
    """
    
    def __init__(self, model_path):
        """
        Inisialisasi CarCounter
        
        Args:
            model_path (str): Path ke model YOLO yang sudah dilatih
        """
        self.model = YOLO(model_path)
        # ID yang sudah dihitung per-garis untuk menghindari double count
        self.counted_ids_jakarta = set()
        self.counted_ids_bandung = set()
        # Simpan posisi sebelumnya per track untuk deteksi arah
        self.previous_centers = {}
        # Dictionary jumlah per-arah dan total
        self.counts = {'mobil': 0, 'jakarta': 0, 'bandung': 0}
        # Posisi dua garis penghitungan (y)
        self.line_y_jakarta = None
        self.line_y_bandung = None
        self.tolerance = 15  # Toleransi untuk deteksi garis penghitungan
        # Fallback anti-duplikasi saat tidak ada ID tracking
        self.recent_cross_events = []  # list of dict: {t, x, y, line}
        
    def set_counting_lines(self, frame_height, jakarta_ratio=0.4, bandung_ratio=0.6):
        """
        Set posisi dua garis penghitungan untuk dua arah.
        Secara default: garis Jakarta ~ 40% tinggi, garis Bandung ~ 60% tinggi.
        """
        self.line_y_jakarta = int(frame_height * float(jakarta_ratio))
        self.line_y_bandung = int(frame_height * float(bandung_ratio))
    
    def process_frame(self, frame, tracking=True):
        """
        Proses frame untuk deteksi dan penghitungan mobil
        
        Args:
            frame: Frame video dari OpenCV
            tracking (bool): Apakah menggunakan tracking atau tidak
            
        Returns:
            tuple: (frame_processed, counts)
        """
        # Set dua garis penghitungan jika belum diset
        if self.line_y_jakarta is None or self.line_y_bandung is None:
            self.set_counting_lines(frame.shape[0])
        
        # Deteksi mobil dengan YOLO
        if tracking:
            # Gunakan tracking untuk mendapatkan ID yang konsisten
            results = self.model.track(
                frame, 
                persist=True, 
                classes=[0],  # Hanya deteksi kelas mobil (index 0)
                tracker="bytetrack.yaml",
                conf=0.3,  # Confidence threshold lebih rendah agar sensitif
                iou=0.5    # IoU threshold
            )
        else:
            # Deteksi tanpa tracking
            results = self.model(
                frame,
                classes=[0],
                conf=0.3,
                iou=0.5
            )
        
        # Gambar dua garis penghitungan (Jakarta & Bandung)
        cv2.line(frame, (0, self.line_y_jakarta), (frame.shape[1], self.line_y_jakarta), (0, 200, 0), 3)
        cv2.putText(frame, 'Garis Jakarta (turun)', (10, self.line_y_jakarta - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.line(frame, (0, self.line_y_bandung), (frame.shape[1], self.line_y_bandung), (200, 0, 0), 3)
        cv2.putText(frame, 'Garis Bandung (naik)', (10, self.line_y_bandung - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
        
        # Proses hasil deteksi
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            if tracking and boxes.id is not None:
                # Mode tracking dengan ID
                box_coords = boxes.xyxy.cpu().numpy().astype(int)
                track_ids = boxes.id.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                
                for i, (box, track_id, conf) in enumerate(zip(box_coords, track_ids, confidences)):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Ambil posisi sebelumnya untuk deteksi arah
                    prev_cy = self.previous_centers.get(track_id)
                    self.previous_centers[track_id] = cy
                    
                    # Deteksi crossing ke arah Jakarta (bergerak turun melewati garis Jakarta)
                    if prev_cy is not None:
                        crossed_jakarta = (
                            prev_cy < (self.line_y_jakarta - self.tolerance) and
                            cy >= (self.line_y_jakarta + 0)
                        )
                        if crossed_jakarta and track_id not in self.counted_ids_jakarta:
                            self.counts['jakarta'] += 1
                            self.counted_ids_jakarta.add(track_id)
                            self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                            print(f"Arah JKT +1 -> JKT:{self.counts['jakarta']} | BDG:{self.counts['bandung']}")
                        
                        # Deteksi crossing ke arah Bandung (bergerak naik melewati garis Bandung)
                        crossed_bandung = (
                            prev_cy > (self.line_y_bandung + self.tolerance) and
                            cy <= (self.line_y_bandung - 0)
                        )
                        if crossed_bandung and track_id not in self.counted_ids_bandung:
                            self.counts['bandung'] += 1
                            self.counted_ids_bandung.add(track_id)
                            self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                            print(f"Arah BDG +1 -> JKT:{self.counts['jakarta']} | BDG:{self.counts['bandung']}")
                    
                    # Gambar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Label dengan ID dan confidence
                    label = f'ID:{track_id} ({conf:.2f})'
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 0, 0), 
                        2
                    )
                    
                    # Gambar titik pusat
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            else:
                # Mode deteksi tanpa ID tracking atau tracking.id None
                box_coords = boxes.xyxy.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                # Bersihkan event lama (> 1.0 detik)
                now = time.time()
                self.recent_cross_events = [e for e in self.recent_cross_events if now - e['t'] < 1.0]

                for i, (box, conf) in enumerate(zip(box_coords, confidences)):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Gambar bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Label dengan confidence
                    label = f'Mobil ({conf:.2f})'
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

                    # Gambar titik pusat
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    # Fallback counting: jika pusat dekat salah satu garis dan belum ada event duplikat terbaru
                    def is_duplicate(line_name, line_y):
                        for e in self.recent_cross_events:
                            if e['line'] == line_name and abs(e['x'] - cx) < 30 and abs(e['y'] - line_y) < self.tolerance:
                                return True
                        return False

                    # Cek Jakarta line (arah turun secara default)
                    if abs(cy - self.line_y_jakarta) <= self.tolerance and not is_duplicate('jakarta', self.line_y_jakarta):
                        self.counts['jakarta'] += 1
                        self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                        self.recent_cross_events.append({'t': now, 'x': cx, 'y': self.line_y_jakarta, 'line': 'jakarta'})

                    # Cek Bandung line (arah naik secara default)
                    if abs(cy - self.line_y_bandung) <= self.tolerance and not is_duplicate('bandung', self.line_y_bandung):
                        self.counts['bandung'] += 1
                        self.counts['mobil'] = self.counts['jakarta'] + self.counts['bandung']
                        self.recent_cross_events.append({'t': now, 'x': cx, 'y': self.line_y_bandung, 'line': 'bandung'})
        
        # Tambahkan informasi jumlah mobil di frame
        cv2.putText(
            frame, 
            f"Total: {self.counts['mobil']} | JKT: {self.counts['jakarta']} | BDG: {self.counts['bandung']}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255), 
            2
        )
        
        return frame, self.counts
    
    def reset_counter(self):
        """Reset counter dan ID yang sudah dihitung"""
        self.counted_ids_jakarta.clear()
        self.counted_ids_bandung.clear()
        self.previous_centers.clear()
        self.counts = {'mobil': 0, 'jakarta': 0, 'bandung': 0}
        print("Counter direset!")
    
    def get_count(self):
        """Mendapatkan jumlah mobil yang terdeteksi"""
        return self.counts['mobil']
