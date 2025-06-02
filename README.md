# ML Course Final Project - Gesture Control Game

This project implements a gesture-controlled game using machine learning for gesture recognition. Players can control the game using hand gestures captured through their webcam.

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



## 📊 custom metrics tracked

this application exposes additional prometheus metrics via `/metrics` to monitor model performance and input characteristics:

| metric name           | description                                               | type       |
|-----------------------|-----------------------------------------------------------|------------|
| `model_confidence`    | predicted class probability per request                   | histogram  |
| `input_feature1_mean` | mean value of first input feature across requests         | gauge      |
| `input_feature1_std`  | standard deviation of first input feature across requests | gauge      |

these metrics help detect:
- low model confidence → potential out-of-distribution inputs  
- feature drift → changing distribution in input sensor data

---

## 🛠️ monitoring stack

this project uses **prometheus** and **grafana** to monitor:

- 🔹 model confidence
- 🔹 input feature drift
- 🔹 api latency and request throughput


## 🔧 Important Implementation Note

In `api-call.js`, there is a TODO section that needs to be implemented:

```javascript
// TODO: Call your model's api here
// and return the predicted label
// Possible labels: "up", "down", "left", "right", null
// null means stop & wait for the next gesture
```

You need to replace the current random label generation with your actual ML model API call. The function should:
- Take the processed tensor (`processed_t`) as input
- Call your deployed ML model's API
- Return one of these labels: "up", "down", "left", "right", or null

## 🎮 Controls

The game can be controlled through:
- Hand gestures (via webcam)
- Keyboard arrows (as fallback)
