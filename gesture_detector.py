"""
Patient Monitoring System - MediaPipe Hand Detection Module
Supports both MediaPipe Tasks HandLandmarker (MediaPipe 1.0+) and Legacy Solutions.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Any
import cv2
import numpy as np

from config import HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE, MODELS_DIR

# Import mediapipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    mp = None

# Connection pairs for drawing hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]


class LandmarkResultsWrapper:
    """Standardized wrapper for hand landmarks to keep compatibility."""
    def __init__(self, multi_hand_landmarks: List[Any], multi_handedness: Optional[List[Any]] = None):
        self.multi_hand_landmarks = multi_hand_landmarks
        self.multi_handedness = multi_handedness


class HandDetector:
    """Wrapper class for MediaPipe Hand Detection supporting Tasks API and Legacy API."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = HAND_TRACKING_CONFIDENCE,
        model_path: Optional[str] = None
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_tasks_api = False
        self.landmarker = None
        self.legacy_hands = None


        if not MP_AVAILABLE:
            print("Warning: MediaPipe package not found.")
            return

        # Attempt to load MediaPipe Tasks HandLandmarker
        if model_path is None:
            model_path = str(MODELS_DIR / "hand_landmarker.task")

        if os.path.exists(model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_tracking_confidence,
                )
                self.landmarker = vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api = True
            except Exception as e:
                print(f"MediaPipe Tasks initialization fallback: {e}")

        # Fallback to legacy solutions if available
        if not self.use_tasks_api and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.legacy_hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            except Exception as e:
                print(f"Legacy MediaPipe initialization error: {e}")

    def detect_hands(self, frame: np.ndarray) -> Tuple[Optional[LandmarkResultsWrapper], Optional[List[str]]]:
        """
        Detect hands in BGR or RGB frame.
        Returns tuple: (landmark_results_wrapper, list_of_handedness_labels)
        """
        if frame is None or not MP_AVAILABLE:
            return None, None

        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Tasks API mode
        if self.use_tasks_api and self.landmarker is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.landmarker.detect(mp_image)

                if result and result.hand_landmarks:
                    handedness_labels = []
                    if result.handedness:
                        for h_list in result.handedness:
                            if h_list:
                                raw_label = h_list[0].category_name
                                # Invert label for mirrored camera frame
                                label = "Left" if raw_label == "Right" else ("Right" if raw_label == "Left" else raw_label)
                                handedness_labels.append(label)
                            else:
                                handedness_labels.append("Right")
                    else:
                        handedness_labels = ["Right"] * len(result.hand_landmarks)


                    wrapper = LandmarkResultsWrapper(
                        multi_hand_landmarks=result.hand_landmarks,
                        multi_handedness=result.handedness
                    )
                    return wrapper, handedness_labels
                return None, None
            except Exception as e:
                print(f"Error in Tasks detect_hands: {e}")
                return None, None

        # Legacy Solutions API mode
        if self.legacy_hands is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
                results = self.legacy_hands.process(rgb_frame)
                handedness = []
                if results and results.multi_handedness:
                    for hand_info in results.multi_handedness:
                        handedness.append(hand_info.classification[0].label)
                if results and results.multi_hand_landmarks:
                    wrapper = LandmarkResultsWrapper(
                        multi_hand_landmarks=results.multi_hand_landmarks,
                        multi_handedness=results.multi_handedness
                    )
                    return wrapper, handedness
                return None, None
            except Exception as e:
                print(f"Error in Legacy detect_hands: {e}")
                return None, None

        return None, None

    def draw_hand_landmarks(
        self,
        frame: np.ndarray,
        results: Any,
        draw_connections: bool = True
    ) -> np.ndarray:
        """Draw skeleton and landmark points on OpenCV frame."""
        if frame is None or results is None:
            return frame

        landmarks_list = getattr(results, "multi_hand_landmarks", None)
        if not landmarks_list:
            return frame

        output_frame = frame.copy()
        h, w, _ = output_frame.shape

        for hand_lms in landmarks_list:
            lm_nodes = getattr(hand_lms, "landmark", hand_lms)
            pts = []
            for lm in lm_nodes:
                px = int(getattr(lm, "x", 0.0) * w)
                py = int(getattr(lm, "y", 0.0) * h)
                pts.append((px, py))

            # Draw skeleton connections
            if draw_connections:
                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(pts) and end_idx < len(pts):
                        cv2.line(output_frame, pts[start_idx], pts[end_idx], (0, 255, 127), 2)

            # Draw joints
            for idx, pt in enumerate(pts):
                color = (0, 215, 255) if idx in [4, 8, 12, 16, 20] else (255, 100, 0)
                radius = 6 if idx in [4, 8, 12, 16, 20] else 4
                cv2.circle(output_frame, pt, radius, color, -1)
                cv2.circle(output_frame, pt, radius + 1, (255, 255, 255), 1)

        return output_frame

    def close(self):
        """Release landmarker resources."""
        if self.use_tasks_api and self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
        elif self.legacy_hands:
            try:
                self.legacy_hands.close()
            except Exception:
                pass


def initialize_hand_detector(
    min_detection_confidence: float = HAND_DETECTION_CONFIDENCE,
    min_tracking_confidence: float = HAND_TRACKING_CONFIDENCE,
) -> HandDetector:
    """Helper function to create HandDetector instance."""
    return HandDetector(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def detect_hands(
    frame: np.ndarray, hand_detector: Optional[HandDetector] = None
) -> Tuple[Optional[Any], Optional[List[str]]]:
    """Standalone function to detect hands from frame."""
    if hand_detector is None:
        hand_detector = initialize_hand_detector()
    return hand_detector.detect_hands(frame)


def get_hand_coordinates(landmarks: Any) -> List[Tuple[float, float, float]]:
    """Extract list of (x, y, z) 3D coordinates for all 21 hand landmarks."""
    if landmarks is None:
        return []
    
    lm_list = getattr(landmarks, 'landmark', landmarks)
    coords = []
    for lm in lm_list:
        coords.append((getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)))
    return coords


def calculate_hand_orientation(landmarks: Any) -> Tuple[float, float, float]:
    """Calculate hand palm normal vector based on Wrist (0), Index MCP (5), Pinky MCP (17)."""
    coords = get_hand_coordinates(landmarks)
    if len(coords) < 18:
        return (0.0, 0.0, 1.0)

    p0 = np.array(coords[0])
    p5 = np.array(coords[5])
    p17 = np.array(coords[17])

    v1 = p5 - p0
    v2 = p17 - p0

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return (0.0, 0.0, 1.0)
    normal = normal / norm
    return (float(normal[0]), float(normal[1]), float(normal[2]))
