"""
Aplikasi GUI Clean untuk deteksi dan penghitungan mobil menggunakan YOLO
Versi Minimalistic dan User-Friendly - FIXED COUNTING
"""

import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, 
                             QMessageBox, QFileDialog, QCheckBox, QProgressBar,
                             QFrame)
from PyQt5.QtGui import QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import numpy as np
import time
from ultralytics import YOLO

class CarCounter:
    """
    Kelas untuk deteksi dan penghitungan mobil dalam video stream - FIXED
    """
    
    def __init__(self, model_path):
        """
        Inisialisasi CarCounter
        
        Args:
            model_path (str): Path ke model YOLO yang sudah dilatih
        """
        self.model = YOLO(model_path)
        
        # Counter dan tracking data
        self.counts = {'total': 0, 'up': 0, 'down': 0}
        self.tracked_objects = {}  # {id: {'last_y': y, 'counted': False, 'direction': None}}
        
        # Line counting setup
        self.counting_line_y = None
        self.line_thickness = 3
        self.detection_zone = 50  # Zona deteksi di sekitar garis
        
        # Debug info
        self.debug = True
        
    def set_counting_line(self, frame_height, line_ratio=0.5):
        """
        Set posisi garis penghitungan (default di tengah frame)
        
        Args:
            frame_height: Tinggi frame
            line_ratio: Rasio posisi garis (0.0-1.0)
        """
        self.counting_line_y = int(frame_height * line_ratio)
        if self.debug:
            print(f"Counting line set at y={self.counting_line_y}")
    
    def process_frame(self, frame, tracking=True):
        """
        Proses frame untuk deteksi dan penghitungan mobil
        
        Args:
            frame: Frame video dari OpenCV
            tracking (bool): Apakah menggunakan tracking atau tidak
            
        Returns:
            tuple: (frame_processed, counts)
        """
        # Set garis penghitungan jika belum diset
        if self.counting_line_y is None:
            self.set_counting_line(frame.shape[0])
        
        # Deteksi mobil dengan YOLO
        if tracking:
            results = self.model.track(
                frame, 
                persist=True, 
                classes=[0, 5, 7],  # Classes: car, bus, truck dalam COCO dataset
                tracker="bytetrack.yaml",
                conf=0.3,    # Confidence threshold
                iou=0.5      # IoU threshold
            )
        else:
            results = self.model(
                frame,
                classes=[0, 5, 7],  # Classes: car, bus, truck
                conf=0.3,
                iou=0.5
            )
        
        # Gambar garis penghitungan
        self.draw_counting_line(frame)
        
        # Proses deteksi
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            if tracking and boxes.id is not None:
                self._process_with_tracking(frame, boxes)
            else:
                self._process_without_tracking(frame, boxes)
        
        # Cleanup tracked objects yang hilang
        self._cleanup_tracked_objects()
        
        # Tambahkan info counter di frame
        self.draw_counter_info(frame)
        
        # Update counts untuk kompatibilitas dengan UI lama
        legacy_counts = {
            'mobil': self.counts['total'],
            'jakarta': self.counts['down'],  # Arah turun = Jakarta
            'bandung': self.counts['up']     # Arah naik = Bandung
        }
        
        return frame, legacy_counts
    
    def _process_with_tracking(self, frame, boxes):
        """Proses deteksi dengan tracking ID"""
        box_coords = boxes.xyxy.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []
        confidences = boxes.conf.cpu().numpy()
        
        current_ids = set()
        
        for i, (box, conf) in enumerate(zip(box_coords, confidences)):
            x1, y1, x2, y2 = box
            
            # Jika ada tracking ID, gunakan, jika tidak buat ID sementara
            track_id = track_ids[i] if i < len(track_ids) else -1
            
            # Hitung pusat bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Gambar bounding box dan info
            label = f'ID:{track_id} {conf:.2f}' if track_id != -1 else f'Car {conf:.2f}'
            self.draw_detection(frame, box, label, center_x, center_y)
            
            # Jika tidak ada tracking ID, skip processing lebih lanjut
            if track_id == -1:
                continue
                
            current_ids.add(track_id)
            
            # Update tracking data
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'last_y': center_y,
                    'counted': False,
                    'direction': None,
                    'last_seen': time.time()
                }
            else:
                # Cek apakah objek melewati garis penghitungan
                self._check_line_crossing(track_id, center_y)
                self.tracked_objects[track_id]['last_y'] = center_y
                self.tracked_objects[track_id]['last_seen'] = time.time()
    
    def _process_without_tracking(self, frame, boxes):
        """Proses deteksi tanpa tracking (fallback sederhana)"""
        box_coords = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        for box, conf in zip(box_coords, confidences):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Gambar bounding box
            self.draw_detection(frame, box, f'Car {conf:.2f}', center_x, center_y)
    
    def _check_line_crossing(self, track_id, current_y):
        """Cek apakah objek melewati garis penghitungan"""
        obj_data = self.tracked_objects[track_id]
        last_y = obj_data['last_y']
        
        # Jarak dari garis penghitungan
        line_distance = abs(current_y - self.counting_line_y)
        
        # Cek apakah objek melewati garis dan belum dihitung
        if not obj_data['counted'] and line_distance < self.detection_zone:
            
            # Tentukan arah gerakan
            if last_y < self.counting_line_y and current_y >= self.counting_line_y:
                # Bergerak ke bawah (turun)
                direction = 'down'
            elif last_y > self.counting_line_y and current_y <= self.counting_line_y:
                # Bergerak ke atas (naik)
                direction = 'up'
            else:
                direction = None
            
            if direction:
                # Hitung kendaraan
                self.counts[direction] += 1
                self.counts['total'] += 1
                obj_data['counted'] = True
                obj_data['direction'] = direction
                
                if self.debug:
                    print(f"Vehicle ID:{track_id} counted going {direction}. Total: {self.counts['total']}")
    
    def _cleanup_tracked_objects(self):
        """Bersihkan objek yang sudah tidak terdeteksi"""
        current_time = time.time()
        to_remove = []
        
        for track_id, data in self.tracked_objects.items():
            if current_time - data['last_seen'] > 2.0:  # 2 detik timeout
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracked_objects[track_id]
    
    def draw_counting_line(self, frame):
        """Gambar garis penghitungan"""
        height, width = frame.shape[:2]
        
        # Garis penghitungan utama
        cv2.line(frame, (0, self.counting_line_y), (width, self.counting_line_y), 
                (0, 255, 0), self.line_thickness)
        
        # Zona deteksi (opsional, untuk debug)
        if self.debug:
            # Garis atas zona
            cv2.line(frame, (0, self.counting_line_y - self.detection_zone), 
                    (width, self.counting_line_y - self.detection_zone), (0, 255, 0), 1)
            # Garis bawah zona  
            cv2.line(frame, (0, self.counting_line_y + self.detection_zone), 
                    (width, self.counting_line_y + self.detection_zone), (0, 255, 0), 1)
        
        # Label garis
        cv2.putText(frame, 'COUNTING LINE', (10, self.counting_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_detection(self, frame, box, label, center_x, center_y):
        """Gambar bounding box dan info deteksi"""
        x1, y1, x2, y2 = box
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Label
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Titik pusat
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    def draw_counter_info(self, frame):
        """Gambar informasi counter di frame"""
        # Background untuk text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Counter info
        cv2.putText(frame, f"Total Vehicles: {self.counts['total']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Going Up: {self.counts['up']}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Going Down: {self.counts['down']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def reset_counter(self):
        """Reset counter dan tracking data"""
        self.counts = {'total': 0, 'up': 0, 'down': 0}
        self.tracked_objects.clear()
        if self.debug:
            print("Counter reset!")
    
    def get_count(self):
        """Mendapatkan total jumlah kendaraan"""
        return self.counts['total']
    
    def set_debug(self, debug=True):
        """Enable/disable debug mode"""
        self.debug = debug