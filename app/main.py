import os
import sys
import platform
import ctypes
import warnings
from pathlib import Path

# === DLL FIX FOR WINDOWS ===
if platform.system() == "Windows":
    try:
        import site
        for site_pkg in site.getsitepackages():
            dll_path = os.path.join(site_pkg, "torch", "lib", "c10.dll")
            if os.path.exists(dll_path):
                print(f"Pre-loading DLL: {dll_path}")
                ctypes.CDLL(os.path.normpath(dll_path))
                print("✅ c10.dll loaded successfully")
                break
    except Exception as e:
        print(f"Pre-loading failed: {e}")

warnings.filterwarnings("ignore", category=UserWarning)

# === MAIN IMPORTS ===
import cv2
import time
import numpy as np
from collections import deque

# IMPORTANT: Import PyTorch BEFORE creating QApplication
from ultralytics import YOLO
import torch

# Now import PyQt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QGroupBox, QGridLayout, QFrame,
                             QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# === COLORS AND NAMES ===
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 255, 0),    # Green
    'fear': (255, 0, 0),       # Blue
    'happy': (0, 255, 255),    # Yellow
    'neutral': (255, 255, 255),# White
    'sad': (255, 0, 255),      # Purple
    'surprise': (0, 165, 255)  # Orange
}

EMOTION_NAMES_EN = {
    'angry': '😠 Angry',
    'disgust': '🤢 Disgust',
    'fear': '😨 Fear',
    'happy': '😊 Happy',
    'neutral': '😐 Neutral',
    'sad': '😢 Sad',
    'surprise': '😲 Surprise'
}

class InferenceThread(QThread):
    """Thread for video processing"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    stats_signal = pyqtSignal(dict)
    emotion_signal = pyqtSignal(str, float, str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.model = None
        self.camera_id = 0
        self.model_path = None
        
        # Performance optimization
        self.fps_values = deque(maxlen=30)
        self.confidence_values = deque(maxlen=30)
        self.current_emotions = deque(maxlen=10)
        self.skip_frames = 0
        self.process_every_n = 2
        
    def set_camera(self, camera_id):
        self.camera_id = camera_id
        
    def set_model(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.to('cuda')
            print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    
    def run(self):
        # Use DirectShow for Windows (faster)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            print("❌ ERROR: Camera not opened")
            return
        
        # Capture optimization
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        # FPS calculation
        fps_counter = 0
        fps_timer = time.time()
        display_fps = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            fps_counter += 1
            
            # Update FPS every second
            if time.time() - fps_timer >= 1.0:
                display_fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()
            
            # Skip frames for better performance
            self.skip_frames += 1
            if self.skip_frames % self.process_every_n != 0:
                # Send frame without processing
                stats = {
                    'fps': display_fps,
                    'confidence': sum(self.confidence_values) / len(self.confidence_values) if self.confidence_values else 0,
                    'detections': 0
                }
                self.stats_signal.emit(stats)
                
                # Add FPS to frame
                cv2.putText(frame, f"FPS: {display_fps}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.change_pixmap_signal.emit(frame)
                continue
            
            # Model inference
            results = self.model(frame, stream=True, conf=0.5, verbose=False, 
                                device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Process results
            for r in results:
                frame = r.plot()
                
                if len(r.boxes) > 0:
                    box = r.boxes[0]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    emotion = self.model.names[cls]
                    
                    self.confidence_values.append(conf)
                    self.current_emotions.append(emotion)
                    
                    color = EMOTION_COLORS.get(emotion.lower(), (128, 128, 128))
                    color_str = f"{color[2]},{color[1]},{color[0]}"
                    self.emotion_signal.emit(emotion, conf, color_str)
            
            # Statistics
            stats = {
                'fps': display_fps,
                'confidence': sum(self.confidence_values) / len(self.confidence_values) if self.confidence_values else 0,
                'detections': len(r.boxes) if 'r' in locals() and hasattr(r, 'boxes') else 0
            }
            
            # Add FPS to frame
            cv2.putText(frame, f"FPS: {display_fps}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            self.stats_signal.emit(stats)
            self.change_pixmap_signal.emit(frame)
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lavoris-Sec Emotion Detector v2.0")
        self.setMinimumSize(1400, 800)
        
        # Model paths
        self.model_path = self.find_model()
        
        self.model = None
        self.thread = None
        self.emotion_history = []
        
        self.init_ui()
        self.load_model()
    
    def find_model(self):
        """Find model file"""
        possible_paths = [
            r'models\Lavoris-Sec.pt',
            r'runs\detect\Emotions_v11_Final\weights\last.pt',
            r'best.pt',
            r'last.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Model found: {path}")
                return path
        
        print("❌ Model not found!")
        return possible_paths[0]
    
    def init_ui(self):
        """Create interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left panel - video
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        video_layout = QVBoxLayout(video_frame)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #2c3e50; color: white; font-size: 20px;")
        self.video_label.setText("⚡ WAITING FOR START ⚡")
        video_layout.addWidget(self.video_label)
        left_layout.addWidget(video_frame)
        
        # Right panel - information
        right_panel = QWidget()
        right_panel.setMaximumWidth(450)
        right_panel.setStyleSheet("background-color: #2b2b2b; color: white;")
        right_layout = QVBoxLayout(right_panel)
        
        # Title
        title = QLabel("EMOTION ANALYSIS")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #ffaa00; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)
        
        # Current emotion
        current_group = QGroupBox("CURRENT EMOTION")
        current_group.setStyleSheet("""
            QGroupBox { color: white; font-weight: bold; border: 2px solid #555; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        current_layout = QVBoxLayout()
        
        self.emotion_display = QLabel("⚡")
        self.emotion_display.setFont(QFont("Arial", 36, QFont.Bold))
        self.emotion_display.setAlignment(Qt.AlignCenter)
        self.emotion_display.setStyleSheet("""
            background-color: #333; color: #ffaa00; border: 3px solid #555;
            border-radius: 10px; padding: 20px; min-height: 100px;
        """)
        current_layout.addWidget(self.emotion_display)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("""
            QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 3px; }
        """)
        confidence_layout.addWidget(self.confidence_bar)
        current_layout.addLayout(confidence_layout)
        current_group.setLayout(current_layout)
        right_layout.addWidget(current_group)
        
        # Emotion table
        table_group = QGroupBox("REAL-TIME DETECTIONS")
        table_group.setStyleSheet(current_group.styleSheet())
        table_layout = QVBoxLayout()
        
        self.emotion_table = QTableWidget(0, 3)
        self.emotion_table.setHorizontalHeaderLabels(["Emotion", "Confidence", "Color"])
        self.emotion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.emotion_table.setStyleSheet("""
            QTableWidget { background-color: #333; color: white; gridline-color: #555; }
            QHeaderView::section { background-color: #444; color: white; padding: 5px; border: 1px solid #555; }
        """)
        table_layout.addWidget(self.emotion_table)
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)
        
        # Statistics
        stats_group = QGroupBox("STATISTICS")
        stats_group.setStyleSheet(current_group.styleSheet())
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0")
        self.fps_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.fps_label.setStyleSheet("color: #27ae60;")
        stats_layout.addWidget(self.fps_label, 0, 1)
        
        stats_layout.addWidget(QLabel("👤 Faces detected:"), 1, 0)
        self.detections_label = QLabel("0")
        self.detections_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(self.detections_label, 1, 1)
        
        stats_layout.addWidget(QLabel("🎯 Average confidence:"), 2, 0)
        self.avg_conf_label = QLabel("0%")
        self.avg_conf_label.setFont(QFont("Arial", 12, QFont.Bold))
        stats_layout.addWidget(self.avg_conf_label, 2, 1)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Controls
        control_group = QGroupBox("CONTROLS")
        control_group.setStyleSheet(current_group.styleSheet())
        control_layout = QVBoxLayout()
        
        cam_layout = QHBoxLayout()
        cam_layout.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.setStyleSheet("background-color: #444; color: white; padding: 5px;")
        
        # Camera detection
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cam_combo.addItem(f"Camera {i}")
                cap.release()
        cam_layout.addWidget(self.cam_combo)
        control_layout.addLayout(cam_layout)
        
        self.model_status = QLabel("⏳ Loading model...")
        self.model_status.setStyleSheet("color: #f39c12;")
        control_layout.addWidget(self.model_status)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 START")
        self.start_btn.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-weight: bold; 
                         font-size: 14px; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #2ecc71; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.start_btn.clicked.connect(self.start_recognition)
        
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color: #c0392b; color: white; font-weight: bold; 
                         font-size: 14px; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:disabled { background-color: #555; }
        """)
        self.stop_btn.clicked.connect(self.stop_recognition)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)
        
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)
    
    def load_model(self):
        """Load model"""
        try:
            self.model = YOLO(self.model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
                gpu_name = torch.cuda.get_device_name(0)
                self.model_status.setText(f"✅ GPU: {gpu_name[:30]}...")
            else:
                self.model_status.setText("✅ Model loaded (CPU)")
            self.model_status.setStyleSheet("color: #27ae60;")
        except Exception as e:
            self.model_status.setText(f"❌ Error: {str(e)[:20]}")
            self.model_status.setStyleSheet("color: #e74c3c;")
    
    def start_recognition(self):
        """Start recognition"""
        if self.model is None:
            return
        
        self.thread = InferenceThread()
        self.thread.set_camera(self.cam_combo.currentIndex())
        self.thread.set_model(self.model_path)
        
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.emotion_signal.connect(self.update_emotion_display)
        
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.cam_combo.setEnabled(False)
        self.model_status.setText("🟢 RECOGNITION ACTIVE")
    
    def stop_recognition(self):
        """Stop recognition"""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cam_combo.setEnabled(True)
        self.model_status.setText("✅ Model loaded")
        self.video_label.setText("⚡ RECOGNITION STOPPED ⚡")
    
    def update_image(self, cv_img):
        """Update image"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
    
    def update_stats(self, stats):
        """Update statistics"""
        self.fps_label.setText(f"{stats['fps']:.1f}")
        self.detections_label.setText(str(stats['detections']))
        self.avg_conf_label.setText(f"{stats['confidence']*100:.1f}%")
    
    def update_emotion_display(self, emotion, confidence, color_str):
        """Update emotion display"""
        r, g, b = map(int, color_str.split(','))
        
        en_name = EMOTION_NAMES_EN.get(emotion.lower(), emotion)
        self.emotion_display.setText(f"{en_name}\n{confidence*100:.1f}%")
        self.emotion_display.setStyleSheet(f"""
            background-color: rgb({r//4}, {g//4}, {b//4});
            color: rgb({r}, {g}, {b});
            border: 3px solid rgb({r}, {g}, {b});
            border-radius: 10px;
            padding: 20px;
            min-height: 100px;
            font-weight: bold;
        """)
        
        self.confidence_bar.setValue(int(confidence * 100))
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{ border: 2px solid #555; border-radius: 5px; text-align: center; color: white; }}
            QProgressBar::chunk {{ background-color: rgb({r}, {g}, {b}); border-radius: 3px; }}
        """)
        
        self.add_to_table(emotion, confidence, (r, g, b))
    
    def add_to_table(self, emotion, confidence, color):
        """Add to table"""
        self.emotion_table.insertRow(0)
        
        en_name = EMOTION_NAMES_EN.get(emotion.lower(), emotion)
        item = QTableWidgetItem(en_name)
        item.setForeground(QColor(*color))
        self.emotion_table.setItem(0, 0, item)
        
        conf_item = QTableWidgetItem(f"{confidence*100:.1f}%")
        conf_item.setForeground(QColor(*color))
        self.emotion_table.setItem(0, 1, conf_item)
        
        color_item = QTableWidgetItem("⬤")
        color_item.setForeground(QColor(*color))
        color_item.setTextAlignment(Qt.AlignCenter)
        self.emotion_table.setItem(0, 2, color_item)
        
        if self.emotion_table.rowCount() > 20:
            self.emotion_table.removeRow(20)
    
    def closeEvent(self, event):
        self.stop_recognition()
        event.accept()

if __name__ == "__main__":
    # Important: First import torch, then create QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())