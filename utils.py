"""
Patient Monitoring System - Helper Utilities
"""

import json
import csv
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from config import GESTURE_LOGS_JSON, PATIENT_HISTORY_CSV, BASE_DIR


def get_current_timestamp() -> str:
    """Return ISO format timestamp string."""
    return datetime.now().isoformat()


def calculate_distance(p1, p2) -> float:
    """
    Calculate Euclidean distance between two points (2D or 3D).
    Supports tuples/lists or objects with x, y, z attributes.
    """
    x1 = getattr(p1, 'x', p1[0] if isinstance(p1, (list, tuple)) else 0)
    y1 = getattr(p1, 'y', p1[1] if isinstance(p1, (list, tuple)) else 0)
    z1 = getattr(p1, 'z', p1[2] if isinstance(p1, (list, tuple)) and len(p1) > 2 else 0)

    x2 = getattr(p2, 'x', p2[0] if isinstance(p2, (list, tuple)) else 0)
    y2 = getattr(p2, 'y', p2[1] if isinstance(p2, (list, tuple)) else 0)
    z2 = getattr(p2, 'z', p2[2] if isinstance(p2, (list, tuple)) and len(p2) > 2 else 0)

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def calculate_angle(a, b, c) -> float:
    """
    Calculate angle in degrees at vertex b formed by vectors ba and bc.
    a, b, c: landmark objects or tuples/lists.
    """
    ax = getattr(a, 'x', a[0] if isinstance(a, (list, tuple)) else 0)
    ay = getattr(a, 'y', a[1] if isinstance(a, (list, tuple)) else 0)
    
    bx = getattr(b, 'x', b[0] if isinstance(b, (list, tuple)) else 0)
    by = getattr(b, 'y', b[1] if isinstance(b, (list, tuple)) else 0)
    
    cx = getattr(c, 'x', c[0] if isinstance(c, (list, tuple)) else 0)
    cy = getattr(c, 'y', c[1] if isinstance(c, (list, tuple)) else 0)

    v_ba = np.array([ax - bx, ay - by])
    v_bc = np.array([cx - bx, cy - by])

    norm_ba = np.linalg.norm(v_ba)
    norm_bc = np.linalg.norm(v_bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.dot(v_ba, v_bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def normalize_landmarks(landmarks):
    """
    Normalize list of landmark objects to zero-centered wrist point.
    Returns list of dicts [{'x': x, 'y': y, 'z': z}, ...]
    """
    if not landmarks:
        return []
    
    wrist = landmarks[0]
    wx = getattr(wrist, 'x', 0)
    wy = getattr(wrist, 'y', 0)
    wz = getattr(wrist, 'z', 0)

    normalized = []
    for lm in landmarks:
        lx = getattr(lm, 'x', 0)
        ly = getattr(lm, 'y', 0)
        lz = getattr(lm, 'z', 0)
        normalized.append({'x': lx - wx, 'y': ly - wy, 'z': lz - wz})
    return normalized


def convert_to_rgb(frame):
    """Convert OpenCV BGR image to RGB."""
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_gesture_logs(filepath: Path = GESTURE_LOGS_JSON) -> dict:
    """Load gesture logs from JSON file."""
    if not filepath.exists():
        return {"session_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "gestures": []}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"session_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "gestures": []}


def save_gesture_logs(logs: dict, filepath: Path = GESTURE_LOGS_JSON) -> bool:
    """Save gesture logs dictionary to JSON file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON logs: {e}")
        return False


def log_to_csv(gesture: str, state: str, confidence: float, hand: str = "Right", filepath: Path = PATIENT_HISTORY_CSV):
    """Append a gesture record entry to CSV patient history."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "gesture", "patient_state", "confidence", "hand"])
            writer.writerow([get_current_timestamp(), gesture, state, f"{confidence:.3f}", hand])
    except Exception as e:
        print(f"Error logging to CSV: {e}")


def load_patient_history_df(filepath: Path = PATIENT_HISTORY_CSV) -> pd.DataFrame:
    """Load patient history CSV as Pandas DataFrame."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "gesture", "patient_state", "confidence", "hand"])
    try:
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "gesture", "patient_state", "confidence", "hand"])
