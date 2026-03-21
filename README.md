# 🎥 Advanced Video Analytics for Automated Detection of Suspicious Activities

## 🚀 Overview
This project presents an **AI-powered intelligent surveillance system** capable of detecting suspicious activities in real time using **Computer Vision and Machine Learning**.

Traditional surveillance systems rely heavily on manual monitoring, which is inefficient and error-prone. This system automates the process by analyzing live video streams and identifying:

- Intrusion into restricted zones  
- Loitering behavior  
- Abnormal motion patterns  
- Object abandonment  
- Crowd congestion and anomalies  

---

## 🧠 Key Features
- 🔍 **Real-time Object Detection** using YOLOv8  
- 🧭 **Object Tracking** with ByteTrack  
- 📊 **Behavior Analysis** using rule-based + ML approaches  
- ⚠️ **Anomaly Detection** using Isolation Forest  
- 🚨 **Alert System** (visual + audio alerts)  
- 👥 **Crowd Monitoring & Flow Analysis**

---

## 🏗️ System Architecture
The system follows a modular pipeline:

1. **Video Input Module**
   - Captures live video (Webcam / CCTV / RTSP)

2. **Object Detection & Tracking**
   - Detects humans using YOLO
   - Tracks them using ByteTrack

3. **Feature Extraction**
   - Speed, trajectory, displacement, dwell time

4. **Behavior Analysis**
   - Rule-based (Intrusion, Loitering)
   - ML-based (Anomaly detection)

5. **Alert System**
   - Bounding box color changes
   - Audio alerts

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** YOLOv8  
- **Tracking:** ByteTrack  
- **Machine Learning:** Scikit-learn (Isolation Forest)  
- **Numerical Computing:** NumPy  
- **Audio Alerts:** Pygame  

---

## 📌 Working Flow
```text
Video Stream → Object Detection → Tracking → Feature Extraction → 
Behavior Analysis → Alert Generation → Display Output
