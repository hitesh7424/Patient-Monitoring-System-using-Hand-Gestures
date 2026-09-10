"""
Patient Monitoring System - Configuration Constants
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

GESTURE_LOGS_JSON = DATA_DIR / "gesture_logs.json"
PATIENT_HISTORY_CSV = DATA_DIR / "patient_history.csv"

# Detection & System Constants
CONFIDENCE_THRESHOLD = 0.7
GESTURE_HOLD_FRAMES = 5  # Frames to hold gesture before state change logging
VIDEO_FPS = 30
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
ALERT_COOLDOWN_SECONDS = 2
GESTURE_HISTORY_LIMIT = 100

# Gesture Vocabulary & Mappings
GESTURES = {
    "Thumbs Up": {
        "state": "OK",
        "label": "Patient OK / Feeling Good",
        "icon": "✅",
        "badge_color": "#28a745",
        "severity": 1,
    },
    "Thumbs Down": {
        "state": "ALERT",
        "label": "Patient Alert / In Pain",
        "icon": "⚠️",
        "badge_color": "#dc3545",
        "severity": 4,
    },
    "Open Palm": {
        "state": "CALL_NURSE",
        "label": "Call for Help / Stop",
        "icon": "🆘",
        "badge_color": "#fd7e14",
        "severity": 5,
    },
    "Peace Sign": {
        "state": "VITALS_CHECK",
        "label": "Request Vitals Check",
        "icon": "📊",
        "badge_color": "#17a2b8",
        "severity": 2,
    },
    "Closed Fist": {
        "state": "RESTING",
        "label": "Patient Resting / Sleep",
        "icon": "😴",
        "badge_color": "#6c757d",
        "severity": 1,
    },
    "Point": {
        "state": "POINTING",
        "label": "Indicate Direction / Severity",
        "icon": "👉",
        "badge_color": "#007bff",
        "severity": 3,
    },
    "Unknown": {
        "state": "UNKNOWN",
        "label": "No/Unrecognized Gesture",
        "icon": "❓",
        "badge_color": "#6c757d",
        "severity": 0,
    },
}

PATIENT_STATES = ["OK", "ALERT", "CALL_NURSE", "VITALS_CHECK", "RESTING", "POINTING", "UNKNOWN"]
