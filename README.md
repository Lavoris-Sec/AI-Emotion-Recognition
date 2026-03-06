# AI Emotion Recognition (YOLOv11)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FF00?style=flat)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🇷🇺 Описание / Description
**RU:** Профессиональная система распознавания 7 базовых эмоций в реальном времени. Оптимизирована для работы через веб-камеру с использованием архитектуры YOLOv11 и графического интерфейса на PyQt5. 

**EN:** Professional real-time recognition system for 7 basic emotions. Optimized for webcam streams using YOLOv11 architecture and a custom PyQt5 GUI.

### ⚙️ Функционал / Features:
- 🚀 **Real-time Detection:** 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
- 📊 **Stats Monitoring:** FPS, confidence levels, and session logs.
- 🎨 **Smart UI:** Dynamic color indication for each emotion.
- 💻 **GPU Support:** Full CUDA/PyTorch integration for high performance.

### 📂 Состав проекта / Project Structure:
- `main.py` — Application entry point (GUI).
- `train_emotions.py` — Training module for the model.
- `models/Lavoris-Sec.pt` — Pre-trained YOLOv11 weights.
- `requirements.txt` — Project dependencies.

---

### 🛠️ Hardware Stack (My Lab):
![ESP32](https://img.shields.io/badge/ESP32-E7352C?style=for-the-badge&logo=espressif&logoColor=white)
![Arduino](https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=Arduino&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/-Raspberry_Pi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)

---

### 🔧 Setup / Установка:
```bash
git clone [https://github.com/Lavoris-Sec/AI-Emotion-Recognition.git](https://github.com/Lavoris-Sec/AI-Emotion-Recognition.git)
cd AI-Emotion-Recognition
pip install -r requirements.txt
python main.py
