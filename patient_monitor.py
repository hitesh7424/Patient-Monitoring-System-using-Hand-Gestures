"""
Patient Monitoring System - Patient State Management & Anomaly Detection
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import time

from config import (
    CONFIDENCE_THRESHOLD,
    GESTURE_HOLD_FRAMES,
    ALERT_COOLDOWN_SECONDS,
    GESTURE_HISTORY_LIMIT,
    GESTURES,
    PATIENT_STATES,
)
from utils import (
    get_current_timestamp,
    load_gesture_logs,
    save_gesture_logs,
    log_to_csv,
)


class PatientMonitor:
    """State management engine for patient monitoring."""

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.current_state = "OK"
        self.last_gesture = "Unknown"
        self.last_confidence = 0.0
        self.last_update_time = get_current_timestamp()
        
        self.gesture_hold_counter = 0
        self.pending_gesture = "Unknown"

        self.last_alert_time: Optional[datetime] = None
        self.active_alerts: List[Dict[str, Any]] = []

        # In-memory history buffers
        self.history: deque = deque(maxlen=GESTURE_HISTORY_LIMIT)
        self.raw_log: List[Dict[str, Any]] = []

        # Load existing JSON log if available
        existing = load_gesture_logs()
        self.session_id = existing.get("session_id", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.raw_log = existing.get("gestures", [])
        for entry in self.raw_log[-GESTURE_HISTORY_LIMIT:]:
            self.history.append(entry)

    def update_state(
        self, gesture: str, confidence: float, hand: str = "Right"
    ) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Process incoming detected gesture & confidence.
        Returns tuple: (current_patient_state, state_changed_bool, alert_dict_if_triggered)
        """
        self.last_gesture = gesture
        self.last_confidence = confidence

        if confidence < self.confidence_threshold or gesture == "Unknown":
            self.gesture_hold_counter = 0
            self.pending_gesture = "Unknown"
            return (self.current_state, False, None)

        # Multi-frame hold verification logic
        if gesture == self.pending_gesture:
            self.gesture_hold_counter += 1
        else:
            self.pending_gesture = gesture
            self.gesture_hold_counter = 1

        # Check if held for minimum frames
        if self.gesture_hold_counter < GESTURE_HOLD_FRAMES:
            return (self.current_state, False, None)

        # Gesture mapped state
        gesture_info = GESTURES.get(gesture, GESTURES["Unknown"])
        new_state = gesture_info["state"]

        state_changed = False
        if new_state != self.current_state and new_state != "UNKNOWN":
            self.current_state = new_state
            self.last_update_time = get_current_timestamp()
            state_changed = True

        # Log entry
        log_entry = {
            "timestamp": get_current_timestamp(),
            "gesture": gesture,
            "confidence": round(confidence, 3),
            "patient_state": self.current_state,
            "hand": hand,
            "icon": gesture_info.get("icon", "❓"),
        }

        # Avoid redundant consecutive logging if state & gesture unchanged
        if not self.history or self.history[-1]["gesture"] != gesture or state_changed:
            self.history.append(log_entry)
            self.raw_log.append(log_entry)
            
            # Persist to disk
            log_to_csv(gesture, self.current_state, confidence, hand)
            save_gesture_logs({"session_id": self.session_id, "gestures": self.raw_log})

        # Pattern anomaly detection
        alert = self.detect_pattern()

        return (self.current_state, state_changed, alert)

    def detect_pattern(self, window_seconds: int = 15) -> Optional[Dict[str, Any]]:
        """
        Detect anomaly patterns (e.g. 3+ ALERT or CALL_NURSE gestures within window).
        Returns alert payload if triggered, else None.
        """
        now = datetime.now()
        
        # Check cooldown
        if self.last_alert_time and (now - self.last_alert_time).total_seconds() < ALERT_COOLDOWN_SECONDS:
            return None

        # Filter recent entries within window
        cutoff = now - timedelta(seconds=window_seconds)
        recent_distress = 0
        
        for item in list(self.history)[::-1]:
            try:
                item_time = datetime.fromisoformat(item["timestamp"])
                if item_time < cutoff:
                    break
                if item["patient_state"] in ["ALERT", "CALL_NURSE"]:
                    recent_distress += 1
            except Exception:
                continue

        if recent_distress >= 3 or self.current_state == "CALL_NURSE":
            alert_payload = self.generate_alert(
                reason=f"Patient distress detected! State: {self.current_state} ({recent_distress} alerts in {window_seconds}s)"
            )
            self.last_alert_time = now
            return alert_payload

        return None

    def generate_alert(self, reason: str) -> Dict[str, Any]:
        """Trigger alert object."""
        alert_item = {
            "timestamp": get_current_timestamp(),
            "state": self.current_state,
            "gesture": self.last_gesture,
            "reason": reason,
            "severity": GESTURES.get(self.last_gesture, {}).get("severity", 5),
        }
        self.active_alerts.append(alert_item)
        return alert_item

    def get_current_state(self) -> Dict[str, Any]:
        """Return dict with full current patient status summary."""
        info = GESTURES.get(self.last_gesture, GESTURES["Unknown"])
        return {
            "state": self.current_state,
            "last_gesture": self.last_gesture,
            "confidence": self.last_confidence,
            "timestamp": self.last_update_time,
            "icon": info["icon"],
            "badge_color": info["badge_color"],
            "label": info["label"],
        }

    def get_gesture_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return last N logged gestures."""
        return list(self.history)[-limit:]

    def clear_history(self):
        """Clear session logs."""
        self.history.clear()
        self.raw_log.clear()
        self.active_alerts.clear()
        self.current_state = "OK"
        save_gesture_logs({"session_id": self.session_id, "gestures": []})
