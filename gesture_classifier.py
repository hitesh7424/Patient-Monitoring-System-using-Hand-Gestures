"""
Patient Monitoring System - Rule-based Gesture Classifier
"""

from typing import List, Tuple, Dict, Any, Optional
from collections import Counter, deque
import math
import numpy as np

from utils import calculate_distance, calculate_angle
from config import CONFIDENCE_THRESHOLD, GESTURE_HOLD_FRAMES


class LandmarkPoint:
    """Wrapper class for landmark points with x, y, z attributes."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, idx):
        if idx == 0: return self.x
        if idx == 1: return self.y
        if idx == 2: return self.z
        raise IndexError


def _convert_landmarks(raw_landmarks) -> List[LandmarkPoint]:
    """Convert MediaPipe landmark output or list of tuples into LandmarkPoint objects."""
    if raw_landmarks is None:
        return []
    
    if hasattr(raw_landmarks, 'landmark'):
        lm_list = raw_landmarks.landmark
    elif isinstance(raw_landmarks, list):
        lm_list = raw_landmarks
    else:
        return []

    result = []
    for lm in lm_list:
        if isinstance(lm, (tuple, list)):
            x = lm[0]
            y = lm[1]
            z = lm[2] if len(lm) > 2 else 0.0
        elif hasattr(lm, 'x'):
            x = lm.x
            y = lm.y
            z = getattr(lm, 'z', 0.0)
        else:
            x, y, z = 0.0, 0.0, 0.0
        result.append(LandmarkPoint(x, y, z))
    return result


def calculate_finger_angles(landmarks: List[LandmarkPoint]) -> Dict[str, float]:
    """
    Calculate extension angle for each of the 5 fingers.
    Fingers: Thumb, Index, Middle, Ring, Pinky
    """
    if len(landmarks) < 21:
        return {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}

    # Thumb: angle at 2 (MCP) between 1 (CMC) and 4 (Tip)
    thumb_angle = calculate_angle(landmarks[1], landmarks[2], landmarks[4])
    # Index: angle at 6 (PIP) between 5 (MCP) and 8 (TIP)
    index_angle = calculate_angle(landmarks[5], landmarks[6], landmarks[8])
    # Middle: angle at 10 (PIP) between 9 (MCP) and 12 (TIP)
    middle_angle = calculate_angle(landmarks[9], landmarks[10], landmarks[12])
    # Ring: angle at 14 (PIP) between 13 (MCP) and 16 (TIP)
    ring_angle = calculate_angle(landmarks[13], landmarks[14], landmarks[16])
    # Pinky: angle at 18 (PIP) between 17 (MCP) and 20 (TIP)
    pinky_angle = calculate_angle(landmarks[17], landmarks[18], landmarks[20])

    return {
        "thumb": thumb_angle,
        "index": index_angle,
        "middle": middle_angle,
        "ring": ring_angle,
        "pinky": pinky_angle,
    }


def is_finger_extended(landmarks: List[LandmarkPoint], finger_idx: int) -> bool:
    """
    Check whether a specific finger is extended based on joint distances and coordinates.
    finger_idx: 1=Thumb, 2=Index, 3=Middle, 4=Ring, 5=Pinky
    """
    if len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    
    if finger_idx == 1:  # Thumb
        # Compare distance from thumb tip (4) to MCP (2) vs distance to wrist (0)
        dist_tip_wrist = calculate_distance(landmarks[4], wrist)
        dist_mcp_wrist = calculate_distance(landmarks[2], wrist)
        return dist_tip_wrist > dist_mcp_wrist * 1.2
    
    # Four main fingers (Index, Middle, Ring, Pinky)
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    mcp_ids = [5, 9, 13, 17]

    tip = landmarks[tip_ids[finger_idx - 2]]
    pip = landmarks[pip_ids[finger_idx - 2]]
    mcp = landmarks[mcp_ids[finger_idx - 2]]

    # A finger is extended if tip is further from wrist than PIP & MCP
    dist_tip = calculate_distance(tip, wrist)
    dist_pip = calculate_distance(pip, wrist)

    return dist_tip > dist_pip * 1.1


def classify_gesture(raw_landmarks) -> Tuple[str, float]:
    """
    Classify hand landmark positions into one of 6 patient gestures:
    - Thumbs Up
    - Thumbs Down
    - Open Palm
    - Peace Sign
    - Closed Fist
    - Point
    Returns: (gesture_name, confidence_score)
    """
    landmarks = _convert_landmarks(raw_landmarks)
    if len(landmarks) < 21:
        return ("Unknown", 0.0)

    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]

    index_tip = landmarks[8]
    index_pip = landmarks[6]
    
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    # Evaluate finger extension status
    index_ext = is_finger_extended(landmarks, 2)
    middle_ext = is_finger_extended(landmarks, 3)
    ring_ext = is_finger_extended(landmarks, 4)
    pinky_ext = is_finger_extended(landmarks, 5)
    thumb_ext = is_finger_extended(landmarks, 1)

    four_fingers_folded = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)
    all_five_extended = index_ext and middle_ext and ring_ext and pinky_ext

    # 1. Thumbs Up
    # Thumb tip is well above (lower y in screen coords) thumb IP and MCP while other 4 fingers folded
    if four_fingers_folded and thumb_tip.y < thumb_ip.y and thumb_tip.y < thumb_mcp.y:
        return ("Thumbs Up", 0.95)

    # 2. Thumbs Down
    # Thumb tip is below (higher y in screen coords) thumb IP and MCP while other 4 fingers folded
    if four_fingers_folded and thumb_tip.y > thumb_ip.y and thumb_tip.y > thumb_mcp.y:
        return ("Thumbs Down", 0.94)

    # 3. Open Palm
    # All 5 fingers extended outward
    if all_five_extended:
        return ("Open Palm", 0.96)

    # 4. Peace Sign
    # Index & Middle extended, Ring & Pinky folded
    if index_ext and middle_ext and (not ring_ext) and (not pinky_ext):
        return ("Peace Sign", 0.92)

    # 5. Point Gesture
    # Index extended ONLY, Middle, Ring & Pinky folded
    if index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext):
        return ("Point", 0.90)

    # 6. Closed Fist
    # All fingers folded into palm
    if four_fingers_folded and (not thumb_ext):
        return ("Closed Fist", 0.93)

    # Secondary check for Closed Fist if thumb is near fingers
    if four_fingers_folded:
        return ("Closed Fist", 0.85)

    # Fallback to Open Palm if 3 out of 4 main fingers extended
    ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])
    if ext_count >= 3:
        return ("Open Palm", 0.75)

    return ("Unknown", 0.50)


class GestureSmoother:
    """Buffer window gesture smoother to reduce detection jitter."""

    def __init__(self, window_size: int = GESTURE_HOLD_FRAMES):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def add_prediction(self, gesture: str, confidence: float) -> Tuple[str, float]:
        """Add prediction to queue and return smoothed (gesture, average_confidence)."""
        self.history.append((gesture, confidence))
        
        if len(self.history) == 0:
            return ("Unknown", 0.0)

        gestures_only = [g for g, c in self.history]
        counts = Counter(gestures_only)
        most_common_gesture, max_count = counts.most_common(1)[0]

        # Calculate average confidence for the dominant gesture
        matching_confidences = [c for g, c in self.history if g == most_common_gesture]
        avg_conf = float(np.mean(matching_confidences)) if matching_confidences else 0.0

        # Require majority occurrence in window
        if max_count >= math.ceil(self.window_size / 2):
            return (most_common_gesture, avg_conf)
        
        return (self.history[-1][0], self.history[-1][1])

    def clear(self):
        """Reset history buffer."""
        self.history.clear()


def smooth_gesture(history_buffer: List[Tuple[str, float]], window_size: int = GESTURE_HOLD_FRAMES) -> Tuple[str, float]:
    """Helper function to smooth a list of recent (gesture, confidence) tuples."""
    if not history_buffer:
        return ("Unknown", 0.0)
    
    recent = history_buffer[-window_size:]
    gestures = [g for g, c in recent]
    counts = Counter(gestures)
    top_gesture, _ = counts.most_common(1)[0]
    
    confidences = [c for g, c in recent if g == top_gesture]
    avg_conf = float(np.mean(confidences))
    return (top_gesture, avg_conf)


def create_mock_hand(gesture_type: str) -> List[LandmarkPoint]:
    """
    Generate synthetic 21 hand landmarks matching gesture layout.
    Normalized screen space (x: 0..1, y: 0..1, y=0 top).
    """
    landmarks = [LandmarkPoint(0.5, 0.8, 0.0) for _ in range(21)]
    
    # Set MCP joint positions
    landmarks[1] = LandmarkPoint(0.45, 0.70, 0.0) # Thumb CMC
    landmarks[2] = LandmarkPoint(0.40, 0.65, 0.0) # Thumb MCP
    landmarks[3] = LandmarkPoint(0.38, 0.60, 0.0) # Thumb IP
    
    landmarks[5] = LandmarkPoint(0.42, 0.50, 0.0) # Index MCP
    landmarks[6] = LandmarkPoint(0.42, 0.40, 0.0) # Index PIP
    landmarks[7] = LandmarkPoint(0.42, 0.35, 0.0) # Index DIP
    
    landmarks[9] = LandmarkPoint(0.50, 0.50, 0.0) # Middle MCP
    landmarks[10] = LandmarkPoint(0.50, 0.40, 0.0) # Middle PIP
    landmarks[11] = LandmarkPoint(0.50, 0.35, 0.0) # Middle DIP
    
    landmarks[13] = LandmarkPoint(0.58, 0.50, 0.0) # Ring MCP
    landmarks[14] = LandmarkPoint(0.58, 0.40, 0.0) # Ring PIP
    landmarks[15] = LandmarkPoint(0.58, 0.35, 0.0) # Ring DIP

    landmarks[17] = LandmarkPoint(0.65, 0.52, 0.0) # Pinky MCP
    landmarks[18] = LandmarkPoint(0.65, 0.44, 0.0) # Pinky PIP
    landmarks[19] = LandmarkPoint(0.65, 0.40, 0.0) # Pinky DIP

    if gesture_type == "Thumbs Up":
        landmarks[4] = LandmarkPoint(0.35, 0.30, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Thumbs Down":
        landmarks[4] = LandmarkPoint(0.35, 0.85, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Open Palm":
        landmarks[4] = LandmarkPoint(0.25, 0.45, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.18, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.22, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.25, 0.0)

    elif gesture_type == "Peace Sign":
        landmarks[4] = LandmarkPoint(0.45, 0.60, 0.0)
        landmarks[8] = LandmarkPoint(0.40, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.52, 0.20, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Point":
        landmarks[4] = LandmarkPoint(0.45, 0.60, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Closed Fist":
        landmarks[4] = LandmarkPoint(0.48, 0.62, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    return landmarks

