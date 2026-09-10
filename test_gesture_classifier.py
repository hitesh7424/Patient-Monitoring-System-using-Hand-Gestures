"""
Unit Tests for Gesture Classification, Smoothing, and Patient Monitor Engine
"""

import pytest
import numpy as np

from gesture_classifier import (
    LandmarkPoint,
    classify_gesture,
    is_finger_extended,
    calculate_finger_angles,
    GestureSmoother,
    smooth_gesture,
)
from patient_monitor import PatientMonitor
from config import GESTURES


def create_mock_hand(gesture_type: str):
    """
    Generate synthetic 21 hand landmarks matching gesture layout.
    Normalized screen space (x: 0..1, y: 0..1, y=0 top).
    """
    # Base wrist at center bottom
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
        # Thumb pointing UP (y < IP/MCP y)
        landmarks[4] = LandmarkPoint(0.35, 0.30, 0.0)
        # Fold other fingers (tips y > PIP y)
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Thumbs Down":
        # Thumb pointing DOWN (y > IP/MCP y)
        landmarks[4] = LandmarkPoint(0.35, 0.85, 0.0)
        # Fold other fingers
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Open Palm":
        # All tips extended far above MCP/PIP
        landmarks[4] = LandmarkPoint(0.25, 0.45, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.18, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.22, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.25, 0.0)

    elif gesture_type == "Peace Sign":
        # Index & Middle extended, Ring & Pinky folded
        landmarks[4] = LandmarkPoint(0.45, 0.60, 0.0)
        landmarks[8] = LandmarkPoint(0.40, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.52, 0.20, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Point":
        # Index extended ONLY, others folded
        landmarks[4] = LandmarkPoint(0.45, 0.60, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.20, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    elif gesture_type == "Closed Fist":
        # All tips folded close to wrist/palm
        landmarks[4] = LandmarkPoint(0.48, 0.62, 0.0)
        landmarks[8] = LandmarkPoint(0.42, 0.60, 0.0)
        landmarks[12] = LandmarkPoint(0.50, 0.60, 0.0)
        landmarks[16] = LandmarkPoint(0.58, 0.60, 0.0)
        landmarks[20] = LandmarkPoint(0.65, 0.60, 0.0)

    return landmarks


def test_thumbs_up_classification():
    hand = create_mock_hand("Thumbs Up")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Thumbs Up"
    assert confidence > 0.8


def test_thumbs_down_classification():
    hand = create_mock_hand("Thumbs Down")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Thumbs Down"
    assert confidence > 0.8


def test_open_palm_classification():
    hand = create_mock_hand("Open Palm")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Open Palm"
    assert confidence > 0.8


def test_peace_sign_classification():
    hand = create_mock_hand("Peace Sign")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Peace Sign"
    assert confidence > 0.8


def test_point_classification():
    hand = create_mock_hand("Point")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Point"
    assert confidence > 0.8


def test_closed_fist_classification():
    hand = create_mock_hand("Closed Fist")
    gesture, confidence = classify_gesture(hand)
    assert gesture == "Closed Fist"
    assert confidence > 0.8


def test_unknown_classification():
    gesture, confidence = classify_gesture([])
    assert gesture == "Unknown"
    assert confidence == 0.0


def test_gesture_smoother_filtering():
    smoother = GestureSmoother(window_size=5)
    
    # Push noisy sequence with 4 Thumbs Up and 1 glitch
    smoother.add_prediction("Thumbs Up", 0.90)
    smoother.add_prediction("Thumbs Up", 0.92)
    smoother.add_prediction("Closed Fist", 0.70) # noise
    smoother.add_prediction("Thumbs Up", 0.95)
    smoothed_gesture, avg_conf = smoother.add_prediction("Thumbs Up", 0.91)

    assert smoothed_gesture == "Thumbs Up"
    assert avg_conf > 0.85


def test_patient_monitor_state_machine():
    monitor = PatientMonitor(confidence_threshold=0.7)
    
    # Send Thumbs Up held for 5 frames
    for _ in range(5):
        state, changed, alert = monitor.update_state("Thumbs Up", 0.95)

    assert monitor.current_state == "OK"
    assert state == "OK"

    # Send Thumbs Down held for 5 frames
    for _ in range(5):
        state, changed, alert = monitor.update_state("Thumbs Down", 0.92)

    assert monitor.current_state == "ALERT"
    assert state == "ALERT"
