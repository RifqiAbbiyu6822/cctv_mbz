"""
Aplikasi GUI Clean untuk deteksi dan penghitungan mobil menggunakan YOLO
Versi Minimalistic dan User-Friendly
"""

import sys
import os
import cv2
import time
import requests
import threading
from urllib.parse import urlparse
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QGroupBox, 
                             QMessageBox, QFileDialog, QCheckBox, QProgressBar,
                             QFrame, QSlider, QSpinBox, QComboBox, QTabWidget,
                             QSplitter, QScrollArea)
from PyQt5.QtGui import QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from detector import CarCounter

class SimpleDropArea(QFrame):
    """Area sederhana untuk drag & drop video"""
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            QFrame {
                border: 1px dashed #d0d0d0;
                border-radius: 6px;
                background-color: #fafafa;
                color: #666;
            }
            QFrame:hover {
                background-color: #f0f0f0;
                border-color: #4CAF50;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        self.label = QLabel("Drop video file here or click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: none; font-size: 12px; color: #666;")
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
                self.label.setText(f"Selected: {os.path.basename(file_path)}")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", 
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
            )
            if file_path:
                self.file_dropped.emit(file_path)
                self.label.setText(f"Selected: {os.path.basename(file_path)}")
    
    def _is_video_file(self, file_path):
        return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
    
    def reset(self):
        self.label.setText("Drop video file here or click to browse")

class VideoProcessor(QThread):
    """Thread untuk memproses video"""
    frame_ready = pyqtSignal(QImage)
    count_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    finished_processing = pyqtSignal()
    
    def __init__(self, video_path, model_path, line_position=70, confidence=0.25, 
                 iou=0.45, detection_zone=50, frame_skip=1, device="auto"):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.line_position = line_position
        self.confidence = confidence
        self.iou = iou
        self.detection_zone = detection_zone
        self.frame_skip = frame_skip
        self.device = device
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
            
            # Read first frame to get dimensions
            ret, first_frame = cap.read()
            if not ret:
                self.error_occurred.emit("Cannot read first frame from video")
                return
            
            # Set counting line position based on frame dimensions
            self.car_counter.set_counting_line(first_frame.shape[0], self.line_position / 100.0)
            
            # Set detection zone
            self.car_counter.set_detection_zone(self.detection_zone)
            
            # Set debug mode
            self.car_counter.set_debug(False)  # Disable debug for performance
            
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_count = 0
            
            while self.running:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process frame with custom parameters
                    processed_frame, counts = self.car_counter.process_frame(
                        frame, 
                        tracking=True,  # Always use tracking for now
                        confidence=self.confidence,
                        iou=self.iou
                    )
                    
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
        self.frame_count = 0
        self.start_time = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup clean UI"""
        self.setWindowTitle('Vehicle Counter - YOLO Detection')
        self.setGeometry(100, 100, 1200, 800)
        
        # Minimal clean styling
        self.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 13px;
            }
            
            QGroupBox {
                font-weight: 600;
                border: none;
                background-color: #fafafa;
                border-radius: 8px;
                margin: 8px 0px;
                padding: 12px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0px 8px;
                color: #333;
                font-size: 14px;
                font-weight: 600;
            }
            
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
                border: none;
                font-size: 12px;
                min-height: 20px;
            }
            
            QLineEdit {
                padding: 6px 10px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
            }
            
            QLineEdit:focus {
                border-color: #4CAF50;
                outline: none;
            }
            
            QComboBox {
                padding: 4px 8px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                font-size: 12px;
                min-height: 16px;
            }
            
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background-color: #e0e0e0;
                border-radius: 2px;
            }
            
            QSlider::handle:horizontal {
                background-color: #4CAF50;
                border: none;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            
            QTabWidget::pane {
                border: none;
                background-color: white;
            }
            
            QTabBar::tab {
                background-color: #f5f5f5;
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 12px;
            }
            
            QTabBar::tab:selected {
                background-color: white;
                color: #4CAF50;
                font-weight: 500;
            }
            
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #e0e0e0;
                height: 6px;
            }
            
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #fafafa;
                font-size: 11px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        header_widget = self.create_header()
        main_layout.addWidget(header_widget)
        
        # Content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Left panel - Controls (narrower)
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)
        content_layout.addWidget(control_panel)
        
        # Right panel - Video display (wider)
        video_panel = self.create_video_panel()
        content_layout.addWidget(video_panel, 1)
        
        main_layout.addLayout(content_layout)
    
    def create_header(self):
        """Create minimal header"""
        header_container = QFrame()
        header_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border-bottom: 1px solid #e0e0e0;
                padding: 12px 0px;
            }
        """)
        
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 12)
        
        # Simple title
        title_label = QLabel("Vehicle Counter")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: 700;
                color: #333;
                margin: 0px;
                border: none;
            }
        """)
        
        subtitle = QLabel("YOLO Detection System")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #666;
                margin-left: 12px;
                border: none;
            }
        """)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle)
        header_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 16px;
                border: none;
                margin: 0px 8px;
            }
        """)
        header_layout.addWidget(self.status_indicator)
        
        return header_container
    
    def create_control_panel(self):
        """Create minimal control panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Video Source Section
        source_group = QGroupBox("Video Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)
        
        # Source type (simplified)
        self.source_type = QComboBox()
        self.source_type.addItems(["Local File", "CCTV Stream", "Webcam"])
        self.source_type.currentTextChanged.connect(self.on_source_type_changed)
        source_layout.addWidget(self.source_type)
        
        # File drop area
        self.drop_area = SimpleDropArea()
        self.drop_area.file_dropped.connect(self.on_video_selected)
        source_layout.addWidget(self.drop_area)
        
        # CCTV input (hidden by default)
        self.cctv_input = QLineEdit()
        self.cctv_input.setPlaceholderText("rtsp://ip:port/stream")
        self.cctv_input.setVisible(False)
        source_layout.addWidget(self.cctv_input)
        
        # Model path
        model_layout = QHBoxLayout()
        self.model_input = QLineEdit("weights/best.pt")
        self.model_input.setPlaceholderText("Model path")
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_input)
        source_layout.addLayout(model_layout)
        
        layout.addWidget(source_group)
        
        # Detection Settings (simplified)
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(8)
        
        # Confidence slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(10, 95)
        self.confidence_slider.setValue(25)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        self.confidence_label = QLabel("0.25")
        self.confidence_label.setMinimumWidth(35)
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        detection_layout.addLayout(conf_layout)
        
        # Line position slider
        line_layout = QHBoxLayout()
        line_layout.addWidget(QLabel("Line Position:"))
        self.line_position_slider = QSlider(Qt.Horizontal)
        self.line_position_slider.setRange(10, 90)
        self.line_position_slider.setValue(70)
        self.line_position_slider.valueChanged.connect(self.on_line_position_changed)
        self.line_position_label = QLabel("70%")
        self.line_position_label.setMinimumWidth(35)
        line_layout.addWidget(self.line_position_slider)
        line_layout.addWidget(self.line_position_label)
        detection_layout.addLayout(line_layout)
        
        layout.addWidget(detection_group)
        
        # Control buttons (simplified)
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout(button_group)
        button_layout.setSpacing(6)
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.start_btn.clicked.connect(self.start_processing)
        
        control_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        
        control_row.addWidget(self.pause_btn)
        control_row.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("Reset Counter")
        self.reset_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        self.reset_btn.clicked.connect(self.reset_counter)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addLayout(control_row)
        button_layout.addWidget(self.reset_btn)
        layout.addWidget(button_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: 500; font-size: 12px; }")
        layout.addWidget(self.status_label)
        
        # Minimal log
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(60)
        self.log_area.setPlaceholderText("Logs...")
        layout.addWidget(self.log_area)
        
        layout.addStretch()
        
        return container
    
    def create_video_panel(self):
        """Create minimal video display panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel { 
                border: 1px solid #e0e0e0; 
                border-radius: 8px;
                background-color: #fafafa;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Video preview will appear here")
        layout.addWidget(self.video_label)
        
        # Counter display (clean row)
        counter_container = QFrame()
        counter_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        counter_layout = QHBoxLayout(counter_container)
        counter_layout.setSpacing(24)
        
        self.total_label = QLabel("Total: 0")
        self.total_label.setStyleSheet("""
            QLabel { 
                font-size: 18px; 
                font-weight: 600; 
                color: #333;
                border: none;
            }
        """)
        
        self.up_label = QLabel("Up: 0")
        self.up_label.setStyleSheet("""
            QLabel { 
                font-size: 14px; 
                color: #4CAF50;
                border: none;
            }
        """)
        
        self.down_label = QLabel("Down: 0")
        self.down_label.setStyleSheet("""
            QLabel { 
                font-size: 14px; 
                color: #FF5722;
                border: none;
            }
        """)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("""
            QLabel { 
                font-size: 12px; 
                color: #666;
                border: none;
            }
        """)
        
        counter_layout.addWidget(self.total_label)
        counter_layout.addWidget(self.up_label)
        counter_layout.addWidget(self.down_label)
        counter_layout.addStretch()
        counter_layout.addWidget(self.fps_label)
        
        layout.addWidget(counter_container)
        
        return container
    
    def on_video_selected(self, file_path):
        """Handle video file selection"""
        try:
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(self, "Warning", "Invalid file selected!")
                return
            
            if not self._is_video_file(file_path):
                QMessageBox.warning(self, "Warning", "Please select a valid video file!")
                return
            
            self.current_video = file_path
            self.log(f"Video selected: {os.path.basename(file_path)}")
            self.status_label.setText("Video Ready")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error selecting video: {str(e)}")
            self.log(f"ERROR: {str(e)}")
    
    def _is_video_file(self, file_path):
        """Check if file is a valid video file"""
        if not file_path:
            return False
        return file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'))
    
    def on_source_type_changed(self, source_type):
        """Handle source type change"""
        self.drop_area.setVisible(source_type == "Local File")
        self.cctv_input.setVisible(source_type == "CCTV Stream")
        if source_type == "CCTV Stream":
            self.cctv_input.setPlaceholderText("Enter RTSP URL")
        elif source_type == "Webcam":
            self.current_video = 0  # Default webcam
            self.status_label.setText("Webcam Ready")
    
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        self.confidence_label.setText(f"{value/100:.2f}")
    
    def on_line_position_changed(self, value):
        """Handle line position change"""
        self.line_position_label.setText(f"{value}%")
    
    def start_processing(self):
        """Start video processing"""
        if not self.current_video:
            QMessageBox.warning(self, "Warning", "Please select a video source first!")
            return
        
        # Handle CCTV input
        if self.source_type.currentText() == "CCTV Stream":
            cctv_url = self.cctv_input.text().strip()
            if not cctv_url:
                QMessageBox.warning(self, "Warning", "Please enter CCTV URL!")
                return
            self.current_video = cctv_url
        
        model_path = self.model_input.text().strip()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {model_path}")
            return
        
        # Get settings
        line_position = self.line_position_slider.value()
        confidence = self.confidence_slider.value() / 100.0
        
        # Create processing thread
        self.video_thread = VideoProcessor(
            self.current_video, model_path, line_position, confidence
        )
        self.video_thread.frame_ready.connect(self.update_video)
        self.video_thread.count_updated.connect(self.update_counters)
        self.video_thread.progress_updated.connect(self.update_progress)
        self.video_thread.error_occurred.connect(self.show_error)
        self.video_thread.finished_processing.connect(self.on_processing_finished)
        
        self.video_thread.start()
        
        # Reset counters
        self.frame_count = 0
        self.start_time = None
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")
        self.status_indicator.setStyleSheet("QLabel { color: #FF9800; font-size: 16px; border: none; }")
        
        self.log(f"Processing started")
    
    def pause_processing(self):
        """Pause/resume processing"""
        if self.video_thread:
            self.video_thread.pause()
            if self.video_thread.paused:
                self.pause_btn.setText("Resume")
                self.status_label.setText("Paused")
                self.status_indicator.setStyleSheet("QLabel { color: #9E9E9E; font-size: 16px; border: none; }")
            else:
                self.pause_btn.setText("Pause")
                self.status_label.setText("Processing...")
                self.status_indicator.setStyleSheet("QLabel { color: #FF9800; font-size: 16px; border: none; }")
    
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
            self.log("Counter reset")
        
        self.frame_count = 0
        self.start_time = None
        self.fps_label.setText("FPS: 0")
    
    def update_video(self, qt_image):
        """Update video display"""
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_counters(self, counts):
        """Update counter displays"""
        total = counts.get('mobil', 0)
        up = counts.get('bandung', 0)
        down = counts.get('jakarta', 0)
        
        self.total_label.setText(f"Total: {total}")
        self.up_label.setText(f"Up: {up}")
        self.down_label.setText(f"Down: {down}")
        
        # Update FPS
        self.frame_count += 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def on_processing_finished(self):
        """Handle processing finished"""
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Finished")
        self.status_indicator.setStyleSheet("QLabel { color: #4CAF50; font-size: 16px; border: none; }")
        
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