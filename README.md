# Gesture-Controlled Maze Game – ML Course Final Project

This project integrates computer vision, machine learning, and web technologies to control a browser-based maze game using real-time hand gestures via webcam. The system detects gestures using Mediapipe, classifies them using a trained ML model, and converts them to directional game commands.

---

## 📌 Branch Overview

This repository is organized into four branches:

| Branch       | Description |
|--------------|-------------|
| `research`   | Contains Jupyter notebooks and scripts for gesture data collection, feature extraction using Mediapipe, and model training/evaluation (SVM, Random Forest, CNN). |
| `back-end`   | Contains a Flask API that serves gesture classification predictions. Includes monitoring infrastructure using Prometheus and Grafana, and Docker support for containerization. |
| `front-end`  | Contains the complete maze game built in HTML/CSS/JavaScript. Captures webcam video, sends data to the back-end API, and moves the player based on predictions. |
| `master`     | Integrates the front-end, back-end, and trained model into a production-ready version of the full system. |

---

## 🧠 Gesture-to-Command Mapping

The following mapping is used to convert recognized gestures to directional commands:

```python
gesture_to_command = {
    "like": "up",
    "two_up": "down",
    "fist": "left",
    "palm": "right"
}



## 🚀 Quick Start

1. Install the Live Server extension in VS Code:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Live Server"
   - Install the extension by Ritwick Dey

2. Launch the project:
   - Right-click on `index.html`
   - Select "Open with Live Server"
   - The game should open in your default browser at `http://localhost:5500`

## 📁 Project Structure

- `index.html` - Main game interface
- `api-call.js` - ML model API integration
- `cam.js` - Webcam handling and gesture processing
- `keyboard.js` - Keyboard controls implementation
- `maze.js` - Maze game logic
- `mp.js` - Media processing utilities


## 🎮 Controls

The game can be controlled through:
- Hand gestures (via webcam)
- Keyboard arrows (as fallback)
