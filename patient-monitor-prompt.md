# Patient Monitoring System using Hand Gestures - Complete Project Prompt

## Project Overview
Build a **real-time hand gesture-based patient monitoring dashboard** using computer vision. The system detects hand gestures from webcam feed and maps them to patient states/actions on an interactive dashboard. No hardware required—runs entirely on laptop.

---

## Technical Requirements

### Technology Stack
- **Language:** Python 3.9+
- **Hand Detection:** MediaPipe
- **UI Framework:** Streamlit (for dashboard)
- **ML/CV:** OpenCV, NumPy, Pandas
- **Data Logging:** JSON/CSV
- **Visualization:** Plotly (for analytics)

### Gesture Vocabulary (6 core gestures)
| Gesture | Detection | Patient State | Icon |
|---------|-----------|---------------|------|
| **Thumbs Up** | Thumb extended upward, other fingers closed | "Patient OK / Feeling Good" | ✅ |
| **Thumbs Down** | Thumb pointed downward, fingers closed | "Patient Alert / In Pain" | ⚠️ |
| **Open Palm** | All 5 fingers extended, palm facing camera | "Call for Help / Stop" | 🆘 |
| **Peace Sign** | Index + middle extended, others closed | "Request Vitals Check" | 📊 |
| **Closed Fist** | All fingers folded into palm | "Patient Resting / Sleep" | 😴 |
| **Point Gesture** | Index finger extended only | "Indicate Direction / Severity" | 👉 |

---

## Project Structure to Generate

```
patient-monitoring-vision/
│
├── main.py                          # Streamlit app entry point
├── gesture_detector.py              # MediaPipe hand detection module
├── gesture_classifier.py            # Gesture classification logic
├── patient_monitor.py               # Patient state management
├── utils.py                         # Helper functions
├── config.py                        # Configuration constants
│
├── data/
│   ├── gesture_logs.json            # Real-time gesture logs
│   ├── patient_history.csv          # Patient state history
│   └── gesture_samples/             # Directory for training data (if collecting)
│
├── models/
│   └── gesture_model.tflite         # (Optional) TensorFlow Lite model
│
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file

```

---

## Detailed Specifications

### 1. `config.py` - Configuration Constants
```
CONFIDENCE_THRESHOLD = 0.7
GESTURE_HOLD_FRAMES = 5  # Frames to hold gesture before logging
VIDEO_FPS = 30
HAND_DETECTION_CONFIDENCE = 0.7
ALERT_COOLDOWN_SECONDS = 2
GESTURE_HISTORY_LIMIT = 100
```

### 2. `gesture_detector.py` - Hand Detection Module
**Functions needed:**
- `initialize_hand_detector()` → Returns MediaPipe hand detector
- `detect_hands(frame)` → Returns hand landmarks + handedness (left/right)
- `draw_hand_landmarks(frame, landmarks)` → Draws skeleton on frame
- `get_hand_coordinates(landmarks)` → Extracts (x,y,z) for all 21 points
- `calculate_hand_orientation(landmarks)` → Returns palm normal vector

**Output:** Hand landmarks in normalized coordinates (0-1 range)

### 3. `gesture_classifier.py` - Gesture Recognition Engine
**Core Logic:**
- Implement gesture detection using **hand landmark angles** (NOT ML, rule-based):
  - **Thumbs Up/Down:** Thumb position relative to palm + other fingers closed
  - **Open Palm:** All fingers extended from palm center (distance threshold)
  - **Peace Sign:** Index + middle extended, ring + pinky folded
  - **Closed Fist:** All fingers distance < threshold from palm
  - **Point:** Index extended, others < threshold distance from palm

**Functions needed:**
- `calculate_finger_angles(landmarks)` → Returns angles for each finger
- `is_finger_extended(landmark, palm_center, threshold)` → Boolean
- `classify_gesture(landmarks)` → Returns (gesture_name, confidence)
- `smooth_gesture(gesture_history)` → Smooth noisy detections using history buffer

**Output:** Gesture name + confidence score (0-1)

### 4. `patient_monitor.py` - State Management
**Patient State Machine:**
```
States: OK, ALERT, CALL_NURSE, VITALS_CHECK, RESTING, UNKNOWN

Transitions:
- Thumbs Up → OK
- Thumbs Down → ALERT
- Open Palm → CALL_NURSE
- Peace Sign → VITALS_CHECK
- Closed Fist → RESTING
- Point → Indicate severity level (1-5 based on point direction)
```

**Functions needed:**
- `__init__()` → Initialize patient state + history
- `update_state(gesture, confidence)` → Update patient state based on gesture
- `log_gesture(gesture, timestamp, confidence)` → Add to history
- `get_current_state()` → Return current state + last gesture
- `get_gesture_history(limit=20)` → Return last N gestures
- `detect_pattern(window=10)` → Detect anomaly patterns (e.g., 3+ alerts in 10s)
- `generate_alert(reason)` → Trigger alert if needed

### 5. `main.py` - Streamlit Dashboard
**Layout:**
```
┌─────────────────────────────────────┐
│  🏥 Patient Monitoring System        │
├─────────────────────┬───────────────┤
│                     │               │
│  Live Video Feed    │  Patient      │
│  (640x480)          │  Status       │
│                     │  - Current    │
│  [Hand Overlay]     │  - Gesture    │
│                     │  - Alert      │
├─────────────────────┴───────────────┤
│  Gesture History (Last 10)          │
│  [Timeline view]                    │
├─────────────────────────────────────┤
│  Analytics                          │
│  - Gesture Frequency Chart          │
│  - State Timeline                   │
│  - Confidence Trend                 │
└─────────────────────────────────────┘
```

**Features:**
- Real-time webcam feed with hand pose overlay
- Current gesture + confidence display
- Patient state indicator (colored badge: Green/Yellow/Red)
- Gesture log (timestamp, gesture, confidence)
- Alert notification system
- Analytics: gesture frequency, state distribution over time
- Settings sidebar (confidence threshold, video FPS, history limit)
- Export button (download gesture log as CSV)

### 6. `utils.py` - Helper Functions
- `load_gesture_logs()` → Read gesture history from JSON
- `save_gesture_logs(logs)` → Save to JSON
- `get_current_timestamp()` → ISO format timestamp
- `calculate_distance(point1, point2)` → Euclidean distance
- `normalize_landmarks(landmarks)` → Normalize to 0-1 range
- `convert_to_rgb(frame)` → BGR to RGB conversion
- `log_to_csv(gesture, state, confidence)` → Append to CSV history

---

## Implementation Guidelines

### Gesture Classification Algorithm (Pseudo-code example)
```
def classify_gesture(landmarks):
    palm_center = (landmarks[0].x + landmarks[9].x) / 2, ...
    
    # Check if thumbs up/down
    if is_finger_extended(landmarks[4], palm_center, 0.05):
        if landmarks[4].y < landmarks[2].y:
            return ("Thumbs Up", 0.95)
        else:
            return ("Thumbs Down", 0.95)
    
    # Check if all fingers extended (open palm)
    extended_count = sum(is_finger_extended(landmarks[i], palm_center, 0.08) for i in [8,12,16,20])
    if extended_count >= 4:
        return ("Open Palm", 0.92)
    
    # Check if peace sign
    if is_finger_extended(landmarks[8], ...) and is_finger_extended(landmarks[12], ...):
        if not is_finger_extended(landmarks[16], ...) and not is_finger_extended(landmarks[20], ...):
            return ("Peace Sign", 0.90)
    
    # Check if fist
    if not any(is_finger_extended(...) for i in [8,12,16,20]):
        return ("Closed Fist", 0.93)
    
    # Check if pointing
    if is_finger_extended(landmarks[8], ...) and not is_finger_extended(landmarks[12], ...):
        return ("Point", 0.88)
    
    return ("Unknown", 0.5)
```

### Data Storage Format (JSON)
```json
{
  "session_id": "2026-08-08_14-30-45",
  "gestures": [
    {
      "timestamp": "2026-08-08T14:30:45.123Z",
      "gesture": "Thumbs Up",
      "confidence": 0.95,
      "patient_state": "OK",
      "hand": "right"
    },
    {
      "timestamp": "2026-08-08T14:30:48.456Z",
      "gesture": "Thumbs Down",
      "confidence": 0.89,
      "patient_state": "ALERT",
      "hand": "left"
    }
  ]
}
```

---

## Streamlit Features to Implement

### Main Dashboard
- **Video feed**: Live camera with hand skeleton + gesture label overlay
- **Status card**: Large, colored display of current patient state
- **Confidence meter**: Real-time confidence of current gesture (progress bar)
- **Alert box**: Pop-up warnings if patient in distress or anomaly detected

### Gesture History Panel
- **Table**: Last 10 gestures with timestamp, name, confidence, state
- **Timeline**: Visual timeline showing state changes over last 5 minutes
- **Gesture counter**: How many times each gesture detected in session

### Analytics Tab
- **Gesture frequency chart**: Bar chart (Thumbs Up count, Thumbs Down count, etc.)
- **State timeline**: Line chart showing state over time (OK=0, ALERT=1, etc.)
- **Confidence trend**: Confidence scores over time
- **Heatmap**: Which times of day see most alerts (if session long enough)

### Settings Sidebar
- Confidence threshold slider (0.5-1.0)
- Video FPS selector (15-60 FPS)
- Gesture history limit (50-500 entries)
- Toggle hand landmark display
- Toggle gesture smoothing
- Export button (download as CSV)
- Clear history button

---

## Testing Requirements

### Unit Tests (Create `test_gesture_classifier.py`)
- Test each gesture detection (5+ test cases per gesture)
- Test edge cases (hand partially visible, low confidence)
- Test gesture smoothing (noisy input → clean output)

### Integration Tests
- Webcam capture + detection pipeline (end-to-end)
- State transition logic (gesture sequence → correct state)
- Data logging (gestures saved correctly to JSON/CSV)

### Demo Scenarios
- Record 2-3 minute video showing all 6 gestures
- Demonstrate alert detection (3+ alerts in short time)
- Show export + analytics on recorded session

---

## Deliverables Checklist

- [x] Complete Python package with all modules
- [x] Streamlit dashboard (fully functional)
- [x] README with setup instructions + usage guide
- [x] requirements.txt with all dependencies
- [x] Sample gesture logs (JSON + CSV)
- [x] Gesture detection accuracy documentation
- [x] Demo video or screenshot walkthrough
- [x] GitHub-ready repo structure
- [x] Comments + docstrings in all functions
- [x] .gitignore file

---

## Notes for AI Agent

1. **Use MediaPipe, not custom ML**: Gesture detection should be rule-based using hand landmarks, NOT training a custom model. This is faster and more interpretable.

2. **Streamlit, not Flask**: Use Streamlit for UI—it's simpler and perfect for data/ML dashboards.

3. **Real-time performance**: Keep detection at 30 FPS on laptop (optimize with threading if needed).

4. **Logging everything**: All gestures, timestamps, states must be saved for audit trail.

5. **Clean, modular code**: Each module should be independent; no spaghetti code.

6. **Professional quality**: This goes in GitHub portfolio—comments, type hints, error handling matter.

---

## Success Criteria

✅ Detect all 6 gestures with >85% accuracy  
✅ Dashboard updates in real-time (<100ms latency)  
✅ Gestures logged to persistent storage (JSON/CSV)  
✅ Alert system works (detects patterns)  
✅ Code is clean, documented, GitHub-ready  
✅ Can run with single command: `streamlit run main.py`  
✅ Works on laptop webcam (no additional hardware)

---


