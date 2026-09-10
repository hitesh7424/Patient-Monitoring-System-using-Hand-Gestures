# 🏥 Patient Monitoring System using Hand Gestures

A real-time, touchless bedside patient monitoring dashboard powered by computer vision. The system captures hand gestures from a webcam (or built-in simulated demo feed) using MediaPipe and rule-based landmark geometric classification, mapping them to real-time patient states, logging events, and presenting live analytics.

---

## 🌟 Key Features

- **6 Core Gesture Recognition**: Rule-based hand pose detection using landmark vector math and angle calculation.
- **Real-Time Patient State Machine**: Automatically transitions states (OK, ALERT, CALL_NURSE, VITALS_CHECK, RESTING, POINTING).
- **Distress Anomaly Detection**: Detects distress patterns (e.g. 3+ alerts or nurse calls within 15 seconds) and triggers pop-up warnings.
- **Interactive Streamlit Dashboard**: Dark mode UI with live video stream, landmark skeleton overlay, status badges, metrics, and Plotly analytics.
- **Persistent Data Logging**: Real-time logging of gestures and patient history to `data/gesture_logs.json` and `data/patient_history.csv`.
- **CSV Data Export**: Single-click export of patient history logs directly from the sidebar.
- **Simulated Demo Mode**: Embedded synthetic gesture stream allows full testing and evaluation without requiring physical webcam hardware.

---

## 🖐️ Gesture Vocabulary & Patient State Mapping

| Gesture | Landmark Vector Criteria | Patient State | Icon | Description |
| :--- | :--- | :--- | :---: | :--- |
| **Thumbs Up** | Thumb tip extended upward, other 4 fingers folded | `OK` | ✅ | Patient OK / Feeling Good |
| **Thumbs Down** | Thumb tip pointed downward, other 4 fingers folded | `ALERT` | ⚠️ | Patient Alert / In Pain |
| **Open Palm** | All 5 fingers extended outward | `CALL_NURSE` | 🆘 | Call for Help / Stop |
| **Peace Sign** | Index + Middle extended, Ring + Pinky folded | `VITALS_CHECK` | 📊 | Request Vitals Check |
| **Closed Fist** | All fingers folded into palm | `RESTING` | 😴 | Patient Resting / Sleep |
| **Point Gesture** | Index finger extended only, others folded | `POINTING` | 👉 | Indicate Direction / Severity |

---

## 📁 Project Architecture

```
pmuhg/
├── main.py                          # Streamlit application entry point
├── gesture_detector.py              # MediaPipe hand detection module & landmarks
├── gesture_classifier.py            # Geometric rule-based classification & smoothing
├── patient_monitor.py               # Patient state machine & pattern anomaly alerts
├── utils.py                         # File I/O (JSON/CSV), math & image conversion
├── config.py                        # System constants, paths & color configurations
│
├── test_gesture_classifier.py       # PyTest suite for classifiers & state machine
│
├── data/
│   ├── gesture_logs.json            # JSON log storage
│   └── patient_history.csv          # CSV log storage
│
├── requirements.txt                 # Project dependencies
├── README.md                        # Setup and usage documentation
└── .gitignore                       # Git ignore configuration
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.9 or higher installed on Linux / macOS / Windows.

### 2. Set Up Virtual Environment & Dependencies

```bash
# Navigate to project directory
cd /path/to/pmuhg

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit dashboard using the following command:

```bash
streamlit run main.py
```

Open your browser at `http://localhost:8501`.

### Operating Modes:
1. **Simulated Demo Feed (Default)**: Uses synthetic landmark generators so you can test all 6 gestures without webcam hardware.
2. **Live Webcam Feed**: Connects to your laptop/USB camera (`cv2.VideoCapture(0)`), rendering real-time pose skeletons and tracking gestures.

---

## 🧪 Running Unit Tests

Run the PyTest test suite to verify classification accuracy, temporal smoothing, and patient monitor state transitions:

```bash
pytest test_gesture_classifier.py -v
```

---

## 📊 Analytics & Reporting

The dashboard features dedicated analytics powered by **Plotly**:
- **Gesture Frequency Bar Chart**: Distribution of gestures captured over the session.
- **State Distribution Pie Chart**: Percentage of time spent in each state.
- **State Progression Line Chart**: Chronological timeline of patient states.
- **Confidence Trend Graph**: Real-time confidence scores over time.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
