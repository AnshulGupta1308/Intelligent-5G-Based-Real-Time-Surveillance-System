# 🎥 Advanced Video Analytics for Automated Detection of Suspicious Activities

## 🚀 Project Overview

Advanced Video Analytics for Automated Detection of Suspicious Activities is an AI-powered surveillance system designed to automatically identify suspicious behavior in real-time video streams. By leveraging Computer Vision, Deep Learning, and Machine Learning techniques, the system reduces the need for continuous manual monitoring and enhances security monitoring efficiency.

The system processes live video feeds, detects and tracks individuals, analyzes movement patterns, and generates alerts when abnormal or predefined suspicious activities are observed.

---

## 🎯 Objectives

* Detect unauthorized entry into restricted areas.
* Identify loitering behavior and prolonged presence.
* Monitor abnormal movement patterns.
* Detect abandoned objects in monitored zones.
* Analyze crowd density and congestion.
* Generate real-time alerts for security personnel.

---

## ✨ Key Features

### 🔍 Real-Time Object Detection

Utilizes YOLOv8 to accurately detect people and relevant objects from live video streams.

### 🧭 Multi-Object Tracking

Implements ByteTrack for robust and efficient tracking of detected individuals across frames.

### 📊 Behavioral Analysis

Extracts movement-related features such as:

* Speed
* Trajectory
* Displacement
* Dwell Time

### ⚠️ Anomaly Detection

Uses Isolation Forest to identify unusual activities and motion patterns that deviate from normal behavior.

### 🚨 Alert Generation

Provides instant visual and audio alerts when suspicious activity is detected.

### 👥 Crowd Monitoring

Analyzes crowd density and movement flow to identify congestion and unusual gathering patterns.

---

## 🏗️ System Architecture

```text
Video Input
      │
      ▼
Object Detection (YOLOv8)
      │
      ▼
Object Tracking (ByteTrack)
      │
      ▼
Feature Extraction
(Speed, Trajectory, Dwell Time)
      │
      ▼
Behavior Analysis
(Rule-Based + ML-Based)
      │
      ▼
Anomaly Detection
(Isolation Forest)
      │
      ▼
Alert Generation
      │
      ▼
Output Visualization
```

---

## ⚙️ Technology Stack

| Category             | Technology                      |
| -------------------- | ------------------------------- |
| Programming Language | Python                          |
| Computer Vision      | OpenCV                          |
| Deep Learning        | YOLOv8                          |
| Object Tracking      | ByteTrack                       |
| Machine Learning     | Scikit-learn (Isolation Forest) |
| Numerical Computing  | NumPy                           |
| Alert System         | Pygame                          |

---

## 🔄 Workflow

1. Capture video from a webcam, CCTV camera, or RTSP stream.
2. Detect objects using YOLOv8.
3. Track detected individuals using ByteTrack.
4. Extract behavioral features such as movement speed and dwell time.
5. Analyze activities using rule-based logic and anomaly detection algorithms.
6. Generate alerts when suspicious behavior is identified.
7. Display processed video with annotations and warnings.

---

## 📈 Applications

* Smart Surveillance Systems
* Restricted Area Monitoring
* Public Safety & Security
* Smart City Infrastructure
* Transportation Hubs
* Corporate and Industrial Security