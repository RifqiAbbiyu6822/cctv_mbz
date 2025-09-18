"""
Aplikasi GUI Clean untuk deteksi dan penghitungan mobil menggunakan YOLO
Versi Minimalistic dan User-Friendly
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
from detector import CarCounter

class SimpleDropArea(QFrame):
    """Area sederhana untuk drag & drop video"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #666;
            }
            QFrame:hover {
                background-color: #e8f5e9;
                border-color: #2E7D32;
            }
        """)
        
        layout = QVBoxLayout(self)
        self.label = QLabel("📁 Drop video file here or click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: none; font-weight: bold;")
        layout.addWidget(self.label)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if self._is_video_file(file_path):
                event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if self._is_video_file(file_path):
                self.file_dropped.emit(file_path)
                self.label.setText(f"✓ {os.path.basename(file_path)}")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", 
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file_path:
                self.file_dropped.emit(file_path)
                self.label.setText(f"✓ {os.path.basename(file_path)}")
    
    def _is_video_file(self, file_path):
        return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
    
    def reset(self):
        self.label.setText("📁 Drop video file here or click to browse")

class VideoProcessor(QThread):
    """Thread untuk memproses video"""
    frame_ready = pyqtSignal(QImage)
    count_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal()
    
    def __init__(self, video_path, model_path):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.running = True
        self.paused = False
        self.car_counter = None
    
    def run(self):
        try:
            # Initialize detector
            self.car_counter = CarCounter(self.model_path)
            
            # Open video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {self.video_path}")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            frame_count = 0
            
            while self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame, counts = self.car_counter.process_frame(frame, tracking=True)
                    
                    # Convert to Qt format
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
                    
                    # Emit signals
                    self.frame_ready.emit(qt_image)
                    self.count_updated.emit(counts)
                    
                    # Update progress
                    frame_count += 1
                    progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                    self.progress_updated.emit(progress)
                    
                    # Control playback speed
                    self.msleep(int(1000 / fps))
            
            cap.release()
            self.finished_processing.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
    
    def stop(self):
        self.running = False
        self.wait()
    
    def pause(self):
        self.paused = not self.paused
    
    def reset_counter(self):
        if self.car_counter:
            self.car_counter.reset_counter()

class CarCounterApp(QWidget):
    """Main application - Clean and Simple"""
    
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_video = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup clean UI"""
        self.setWindowTitle('🚗 Vehicle Counter - YOLO Detection')
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px;
                color: #2196F3;
            }
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                border: none;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: #fafafa;
            }
            QLineEdit:focus {
                border-color: #2196F3;
                background-color: white;
            }
        """)
        
        main_layout = QHBoxLayout(self)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel - Video display
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, 2)
    
    def create_control_panel(self):
        """Create clean control panel"""
        group = QGroupBox("🎛️ Controls")
        layout = QVBoxLayout(group)
        
        # Video file selection
        self.drop_area = SimpleDropArea()
        self.drop_area.file_dropped.connect(self.on_video_selected)
        layout.addWidget(self.drop_area)
        
        # Model path
        model_label = QLabel("🧠 Model Path:")
        self.model_input = QLineEdit("weights/best.pt")
        layout.addWidget(model_label)
        layout.addWidget(self.model_input)
        
        # Enable tracking
        self.tracking_cb = QCheckBox("🎯 Enable Vehicle Tracking (Recommended)")
        self.tracking_cb.setChecked(True)
        layout.addWidget(self.tracking_cb)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶️ Start Processing")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.start_btn.clicked.connect(self.start_processing)
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #F44336; color: white; }")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("🔄 Reset Counter")
        self.reset_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        self.reset_btn.clicked.connect(self.reset_counter)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("📊 Status: Ready")
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        layout.addWidget(self.status_label)
        
        # Simple log
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(100)
        self.log_area.setPlaceholderText("Processing logs will appear here...")
        layout.addWidget(QLabel("📝 Log:"))
        layout.addWidget(self.log_area)
        
        return group
    
    def create_video_panel(self):
        """Create video display panel"""
        group = QGroupBox("📹 Video & Statistics")
        layout = QVBoxLayout(group)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel { 
                border: 2px solid #e0e0e0; 
                border-radius: 8px;
                background-color: #f5f5f5;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("🎬 Video preview will appear here")
        layout.addWidget(self.video_label)
        
        # Counter display
        counter_layout = QHBoxLayout()
        
        self.total_label = QLabel("📊 Total: 0")
        self.total_label.setStyleSheet("""
            QLabel { 
                font-size: 18px; 
                font-weight: bold; 
                color: #2196F3;
                padding: 10px;
                border: 2px solid #2196F3;
                border-radius: 8px;
                background-color: #E3F2FD;
            }
        """)
        
        self.up_label = QLabel("⬆️ Up: 0")
        self.up_label.setStyleSheet("""
            QLabel { 
                font-size: 16px; 
                color: #4CAF50;
                padding: 8px;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                background-color: #E8F5E8;
            }
        """)
        
        self.down_label = QLabel("⬇️ Down: 0")
        self.down_label.setStyleSheet("""
            QLabel { 
                font-size: 16px; 
                color: #FF5722;
                padding: 8px;
                border: 2px solid #FF5722;
                border-radius: 6px;
                background-color: #FFF3E0;
            }
        """)
        
        counter_layout.addWidget(self.total_label)
        counter_layout.addWidget(self.up_label)
        counter_layout.addWidget(self.down_label)
        layout.addLayout(counter_layout)
        
        return group
    
    def on_video_selected(self, file_path):
        """Handle video file selection"""
        self.current_video = file_path
        self.log(f"Video selected: {os.path.basename(file_path)}")
    
    def start_processing(self):
        """Start video processing"""
        if not self.current_video:
            QMessageBox.warning(self, "Warning", "Please select a video file first!")
            return
        
        model_path = self.model_input.text().strip()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {model_path}")
            return
        
        # Create and start processing thread
        self.video_thread = VideoProcessor(self.current_video, model_path)
        self.video_thread.frame_ready.connect(self.update_video)
        self.video_thread.count_updated.connect(self.update_counters)
        self.video_thread.progress_updated.connect(self.update_progress)
        self.video_thread.error_occurred.connect(self.show_error)
        self.video_thread.finished_processing.connect(self.on_processing_finished)
        
        self.video_thread.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText("📊 Status: Processing...")
        self.status_label.setStyleSheet("QLabel { color: #FF9800; font-weight: bold; }")
        
        self.log("Processing started")
    
    def pause_processing(self):
        """Pause/resume processing"""
        if self.video_thread:
            self.video_thread.pause()
            if self.video_thread.paused:
                self.pause_btn.setText("▶️ Resume")
                self.status_label.setText("📊 Status: Paused")
                self.status_label.setStyleSheet("QLabel { color: #9E9E9E; font-weight: bold; }")
            else:
                self.pause_btn.setText("⏸️ Pause")
                self.status_label.setText("📊 Status: Processing...")
                self.status_label.setStyleSheet("QLabel { color: #FF9800; font-weight: bold; }")
    
    def stop_processing(self):
        """Stop processing"""
        if self.video_thread:
            self.video_thread.stop()
        self.on_processing_finished()
    
    def reset_counter(self):
        """Reset vehicle counter"""
        if self.video_thread:
            self.video_thread.reset_counter()
            self.update_counters({'mobil': 0, 'jakarta': 0, 'bandung': 0})
            self.log("Counter reset")
        else:
            self.update_counters({'mobil': 0, 'jakarta': 0, 'bandung': 0})
            self.log("Counter reset (no active processing)")
    
    def update_video(self, qt_image):
        """Update video display"""
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_counters(self, counts):
        """Update counter displays"""
        total = counts.get('mobil', 0)
        up = counts.get('bandung', 0)  # bandung = up
        down = counts.get('jakarta', 0)  # jakarta = down
        
        self.total_label.setText(f"📊 Total: {total}")
        self.up_label.setText(f"⬆️ Up: {up}")
        self.down_label.setText(f"⬇️ Down: {down}")
    
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def on_processing_finished(self):
        """Handle processing finished"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸️ Pause")
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("📊 Status: Finished")
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        
        self.log("Processing finished")
        self.video_thread = None
    
    def show_error(self, error_msg):
        """Show error message"""
        QMessageBox.critical(self, "Error", error_msg)
        self.log(f"ERROR: {error_msg}")
        self.on_processing_finished()
    
    def log(self, message):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.video_thread:
            self.video_thread.stop()
        event.accept()

def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setApplicationName("Vehicle Counter")
    app.setApplicationVersion("2.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = CarCounterApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()