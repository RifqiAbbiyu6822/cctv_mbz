"""
Aplikasi GUI untuk deteksi dan penghitungan mobil menggunakan YOLO
"""

import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QComboBox, 
                             QTextEdit, QGroupBox, QGridLayout, QMessageBox,
                             QFileDialog, QSpinBox, QCheckBox, QSlider,
                             QFrame, QTabWidget, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QUrl
from detector import CarCounter

class DragDropArea(QFrame):
    """
    Area untuk drag and drop file video
    """
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                min-height: 100px;
            }
            QFrame:hover {
                border-color: #007acc;
                background-color: #f0f8ff;
            }
        """)
        
        layout = QVBoxLayout(self)
        self.label = QLabel("📁 Drag & Drop file video di sini\natau klik untuk memilih file")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.label)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QFrame {
                            border: 2px dashed #007acc;
                            border-radius: 10px;
                            background-color: #e6f3ff;
                        }
                    """)
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if self.is_video_file(file_path):
                    self.file_dropped.emit(file_path)
                    self.label.setText(f"📹 {os.path.basename(file_path)}")
                    self.setStyleSheet("""
                        QFrame {
                            border: 2px solid #4CAF50;
                            border-radius: 10px;
                            background-color: #e8f5e8;
                        }
                    """)
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Pilih File Video", 
                "", 
                "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
            )
            if file_path:
                self.file_dropped.emit(file_path)
                self.label.setText(f"📹 {os.path.basename(file_path)}")
                self.setStyleSheet("""
                    QFrame {
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        background-color: #e8f5e8;
                    }
                """)
    
    def is_video_file(self, file_path):
        """Cek apakah file adalah video"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def reset(self):
        """Reset area ke kondisi awal"""
        self.label.setText("📁 Drag & Drop file video di sini\natau klik untuk memilih file")
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)

class VideoThread(QThread):
    """
    Thread untuk memproses video stream tanpa memblokir GUI
    """
    change_pixmap_signal = pyqtSignal(QImage)
    update_count_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    duration_signal = pyqtSignal(int)
    position_signal = pyqtSignal(int)
    fps_signal = pyqtSignal(float)
    
    def __init__(self, video_source, model_path, tracking_enabled=True, is_file=False):
        super().__init__()
        self.video_source = video_source
        self.model_path = model_path
        self.tracking_enabled = tracking_enabled
        self.is_file = is_file
        self.running = True
        self.paused = False
        self.car_counter = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self._last_time = None
        self._fps_smooth = 0.0
        
    def run(self):
        """
        Main thread function untuk memproses video
        """
        try:
            # Inisialisasi CarCounter
            self.car_counter = CarCounter(self.model_path)
            
            # Buka video stream
            self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                self.error_signal.emit(f"Gagal membuka: {self.video_source}")
                return
            
            # Set properti video
            if self.is_file:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Dapatkan total frame untuk file video
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration_signal.emit(self.total_frames)
                print(f"File video berhasil dibuka: {self.video_source}")
                print(f"Total frames: {self.total_frames}")
            else:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Stream berhasil dibuka: {self.video_source}")
            
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.is_file:
                            # Video file selesai
                            self.error_signal.emit("Video selesai diputar")
                        else:
                            # Stream error
                            self.error_signal.emit("Gagal membaca frame dari stream")
                        break
                    
                    # Update progress untuk file video
                    if self.is_file:
                        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        progress = int((self.current_frame / self.total_frames) * 100) if self.total_frames > 0 else 0
                        self.progress_signal.emit(progress)
                        self.position_signal.emit(self.current_frame)
                    
                    # Proses frame untuk deteksi mobil
                    frame_processed, counts = self.car_counter.process_frame(
                        frame, 
                        tracking=self.tracking_enabled
                    )

                    # Hitung FPS sederhana (exponential moving average)
                    now = cv2.getTickCount() / cv2.getTickFrequency()
                    if self._last_time is not None:
                        dt = max(now - self._last_time, 1e-6)
                        inst_fps = 1.0 / dt
                        if self._fps_smooth == 0.0:
                            self._fps_smooth = inst_fps
                        else:
                            self._fps_smooth = 0.9 * self._fps_smooth + 0.1 * inst_fps
                        self.fps_signal.emit(float(self._fps_smooth))
                    self._last_time = now
                    
                    # Konversi frame ke format yang bisa ditampilkan di Qt
                    rgb_image = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Emit signal untuk update GUI
                    self.change_pixmap_signal.emit(qt_image)
                    self.update_count_signal.emit(counts)
                
                # Delay untuk kontrol frame rate
                if self.is_file:
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        self.msleep(int(1000 / fps))
                
        except Exception as e:
            self.error_signal.emit(f"Error dalam video thread: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        self.wait()
    
    def pause(self):
        """Pause video"""
        self.paused = True
    
    def resume(self):
        """Resume video"""
        self.paused = False
    
    def seek(self, frame_number):
        """Seek to specific frame (for file video)"""
        if self.cap and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
    
    def reset_counter(self):
        """Reset counter mobil"""
        if self.car_counter:
            self.car_counter.reset_counter()

class App(QWidget):
    """
    Main application class
    """
    
    def __init__(self):
        super().__init__()
        self.thread = None
        self.current_video_file = None
        self.initUI()
        
    def initUI(self):
        """Inisialisasi User Interface"""
        self.setWindowTitle('Aplikasi Penghitung Mobil - YOLO Detection')
        self.setGeometry(100, 100, 1200, 800)
        
        # Set font
        font = QFont()
        font.setPointSize(10)
        self.setFont(font)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        
        # Left panel untuk kontrol
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel untuk video dan informasi
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, 2)
        
    def create_control_panel(self):
        """Buat panel kontrol dengan tab"""
        group = QGroupBox("Kontrol Aplikasi")
        layout = QVBoxLayout(group)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Tab 1: File Video
        file_tab = self.create_file_tab()
        tab_widget.addTab(file_tab, "📁 File Video")
        
        # Tab 2: CCTV Stream
        stream_tab = self.create_stream_tab()
        tab_widget.addTab(stream_tab, "📡 CCTV Stream")
        
        # Model path
        model_label = QLabel('Path Model:')
        self.model_input = QLineEdit('weights/best.pt')
        self.model_input.setPlaceholderText('Path ke model YOLO')
        
        # Tracking option
        self.tracking_checkbox = QCheckBox('Gunakan Tracking (Recommended)')
        self.tracking_checkbox.setChecked(True)
        
        # Control buttons
        self.start_button = QPushButton('Mulai Deteksi')
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.start_button.clicked.connect(self.start_stream)
        
        self.pause_button = QPushButton('Pause')
        self.pause_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; }")
        self.pause_button.clicked.connect(self.pause_stream)
        self.pause_button.setEnabled(False)
        
        self.stop_button = QPushButton('Berhenti')
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)
        
        self.reset_button = QPushButton('Reset Counter')
        self.reset_button.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        self.reset_button.clicked.connect(self.reset_counter)
        
        # Status
        self.status_label = QLabel('Status: Siap')
        self.status_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        
        # Log area
        log_label = QLabel('Log:')
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        
        # Add widgets to layout
        layout.addWidget(tab_widget)
        layout.addWidget(model_label)
        layout.addWidget(self.model_input)
        layout.addWidget(self.tracking_checkbox)
        layout.addWidget(self.start_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.status_label)
        layout.addWidget(log_label)
        layout.addWidget(self.log_text)
        
        return group
    
    def create_file_tab(self):
        """Buat tab untuk file video"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Drag and drop area
        self.drag_drop_area = DragDropArea()
        self.drag_drop_area.file_dropped.connect(self.on_file_dropped)
        layout.addWidget(self.drag_drop_area)
        
        # File info
        self.file_info_label = QLabel('Belum ada file dipilih')
        self.file_info_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(self.file_info_label)
        
        # Video controls (akan ditampilkan saat video diputar)
        self.video_controls = self.create_video_controls()
        self.video_controls.setVisible(False)
        layout.addWidget(self.video_controls)
        
        return widget
    
    def create_stream_tab(self):
        """Buat tab untuk CCTV stream"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # URL input
        url_label = QLabel('URL CCTV:')
        self.url_input = QLineEdit('rtsp://user:pass@ip_address:port/stream')
        self.url_input.setPlaceholderText('Masukkan URL CCTV')
        
        # Webcam option
        webcam_label = QLabel('Webcam:')
        self.webcam_input = QLineEdit('0')
        self.webcam_input.setPlaceholderText('0 untuk webcam default')
        
        # Add widgets
        layout.addWidget(url_label)
        layout.addWidget(self.url_input)
        layout.addWidget(webcam_label)
        layout.addWidget(self.webcam_input)
        layout.addStretch()
        
        return widget
    
    def create_video_controls(self):
        """Buat kontrol video untuk file"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Time info
        self.time_label = QLabel('00:00 / 00:00')
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)
        
        # Seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setVisible(False)
        self.seek_slider.sliderPressed.connect(self.on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self.on_seek_released)
        layout.addWidget(self.seek_slider)
        
        return widget
    
    def create_video_panel(self):
        """Buat panel video"""
        group = QGroupBox("Video Stream & Informasi")
        layout = QVBoxLayout(group)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { border: 2px solid gray; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Video akan muncul di sini")
        
        # Count display
        count_group = QGroupBox("Statistik Deteksi")
        count_layout = QGridLayout(count_group)
        
        self.count_label = QLabel('Jumlah Mobil (Total): 0')
        self.count_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; color: #2196F3; }")

        self.count_jakarta_label = QLabel('Arah Jakarta: 0')
        self.count_jakarta_label.setStyleSheet("QLabel { font-size: 16px; color: #2E7D32; }")

        self.count_bandung_label = QLabel('Arah Bandung: 0')
        self.count_bandung_label.setStyleSheet("QLabel { font-size: 16px; color: #AD1457; }")
        
        self.fps_label = QLabel('FPS: 0')
        self.fps_label.setStyleSheet("QLabel { font-size: 14px; color: #4CAF50; }")
        
        count_layout.addWidget(self.count_label, 0, 0, 1, 2)
        count_layout.addWidget(self.count_jakarta_label, 1, 0)
        count_layout.addWidget(self.count_bandung_label, 1, 1)
        count_layout.addWidget(self.fps_label, 2, 0)
        
        # Add to main layout
        layout.addWidget(self.video_label)
        layout.addWidget(count_group)
        
        return group
    
    def on_file_dropped(self, file_path):
        """Handle file drop event"""
        self.current_video_file = file_path
        self.file_info_label.setText(f"File: {os.path.basename(file_path)}")
        self.log_message(f"File video dipilih: {file_path}")
    
    def on_seek_pressed(self):
        """Handle seek slider pressed"""
        if self.thread and self.thread.is_file:
            self.thread.pause()
    
    def on_seek_released(self):
        """Handle seek slider released"""
        if self.thread and self.thread.is_file:
            frame_number = self.seek_slider.value()
            self.thread.seek(frame_number)
            self.thread.resume()
    
    def browse_video_file(self):
        """Browse untuk file video lokal"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Pilih File Video", 
            "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        if file_path:
            self.current_video_file = file_path
            self.file_info_label.setText(f"File: {os.path.basename(file_path)}")
            self.log_message(f"File video dipilih: {file_path}")
    
    def start_stream(self):
        """Mulai video stream"""
        if self.thread is None or not self.thread.isRunning():
            model_path = self.model_input.text().strip()
            
            if not model_path:
                QMessageBox.warning(self, "Warning", "Masukkan path model YOLO!")
                return
            
            # Cek apakah model file ada
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Warning", f"File model tidak ditemukan: {model_path}")
                return
            
            # Handle path untuk PyInstaller
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.abspath(".")
            
            full_model_path = os.path.join(base_path, model_path)
            
            # Tentukan sumber video
            video_source = None
            is_file = False
            
            # Cek apakah ada file video yang dipilih
            if self.current_video_file and os.path.exists(self.current_video_file):
                video_source = self.current_video_file
                is_file = True
                self.log_message(f"Memulai deteksi dari file: {video_source}")
            else:
                # Coba dari input URL atau webcam
                cctv_url = self.url_input.text().strip()
                webcam_input = self.webcam_input.text().strip()
                
                if cctv_url and cctv_url != 'rtsp://user:pass@ip_address:port/stream':
                    video_source = cctv_url
                    self.log_message(f"Memulai deteksi dari CCTV: {video_source}")
                elif webcam_input:
                    try:
                        video_source = int(webcam_input)
                        self.log_message(f"Memulai deteksi dari webcam: {video_source}")
                    except ValueError:
                        QMessageBox.warning(self, "Warning", "Input webcam harus berupa angka!")
                        return
                else:
                    QMessageBox.warning(self, "Warning", "Pilih file video atau masukkan URL CCTV/webcam!")
                    return
            
            # Buat thread baru
            self.thread = VideoThread(
                video_source, 
                full_model_path,
                self.tracking_checkbox.isChecked(),
                is_file
            )
            
            # Connect signals
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.update_count_signal.connect(self.update_count)
            self.thread.error_signal.connect(self.show_error)
            self.thread.fps_signal.connect(self.update_fps)
            
            if is_file:
                self.thread.progress_signal.connect(self.update_progress)
                self.thread.duration_signal.connect(self.set_duration)
                self.thread.position_signal.connect(self.update_position)
                
                # Show video controls
                self.video_controls.setVisible(True)
                self.progress_bar.setVisible(True)
                self.seek_slider.setVisible(True)
            
            # Start thread
            self.thread.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText('Status: Sedang berjalan...')
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            
            self.log_message(f"Deteksi dimulai: {video_source}")
    
    def pause_stream(self):
        """Pause/Resume video stream"""
        if self.thread:
            if self.thread.paused:
                self.thread.resume()
                self.pause_button.setText('Pause')
                self.status_label.setText('Status: Sedang berjalan...')
                self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
                self.log_message("Video dilanjutkan")
            else:
                self.thread.pause()
                self.pause_button.setText('Resume')
                self.status_label.setText('Status: Pause')
                self.status_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
                self.log_message("Video di-pause")
    
    def stop_stream(self):
        """Stop video stream"""
        if self.thread:
            self.thread.stop()
            self.thread = None
            
            # Update UI
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText('Pause')
            self.stop_button.setEnabled(False)
            self.status_label.setText('Status: Berhenti')
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            
            # Hide video controls
            self.video_controls.setVisible(False)
            self.progress_bar.setVisible(False)
            self.seek_slider.setVisible(False)
            
            # Clear video display
            self.video_label.clear()
            self.video_label.setText("Video akan muncul di sini")
            
            self.log_message("Stream dihentikan")
    
    def reset_counter(self):
        """Reset counter mobil"""
        if self.thread and self.thread.car_counter:
            self.thread.reset_counter()
            self.count_label.setText('Jumlah Mobil (Total): 0')
            self.count_jakarta_label.setText('Arah Jakarta: 0')
            self.count_bandung_label.setText('Arah Bandung: 0')
            self.log_message("Counter direset")
        else:
            QMessageBox.information(self, "Info", "Tidak ada stream yang aktif")
    
    def update_image(self, qt_image):
        """Update tampilan video"""
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def update_count(self, counts):
        """Update jumlah mobil yang terdeteksi"""
        total_count = counts.get('mobil', 0)
        jkt = counts.get('jakarta', 0)
        bdg = counts.get('bandung', 0)
        self.count_label.setText(f'Jumlah Mobil (Total): {total_count}')
        self.count_jakarta_label.setText(f'Arah Jakarta: {jkt}')
        self.count_bandung_label.setText(f'Arah Bandung: {bdg}')

    def update_fps(self, fps_value):
        """Update label FPS"""
        try:
            self.fps_label.setText(f'FPS: {fps_value:.1f}')
        except Exception:
            pass
    
    def update_progress(self, progress):
        """Update progress bar"""
        self.progress_bar.setValue(progress)
    
    def set_duration(self, total_frames):
        """Set total duration untuk video"""
        self.seek_slider.setMaximum(total_frames)
        self.total_frames = total_frames
    
    def update_position(self, current_frame):
        """Update current position"""
        if hasattr(self, 'total_frames') and self.total_frames > 0:
            # Update seek slider (jika tidak sedang di-drag)
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(current_frame)
            
            # Update time display
            fps = 30  # Default FPS, bisa diambil dari video
            current_time = current_frame / fps
            total_time = self.total_frames / fps
            
            current_min = int(current_time // 60)
            current_sec = int(current_time % 60)
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            
            self.time_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
    
    def show_error(self, error_message):
        """Tampilkan error message"""
        if "Video selesai diputar" in error_message:
            # Video selesai, bukan error
            self.log_message("Video selesai diputar")
            self.stop_stream()
        else:
            QMessageBox.critical(self, "Error", error_message)
            self.log_message(f"ERROR: {error_message}")
            self.stop_stream()
    
    def log_message(self, message):
        """Tambah pesan ke log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Handle aplikasi close"""
        self.stop_stream()
        event.accept()

def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Car Counter YOLO")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    ex = App()
    ex.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
