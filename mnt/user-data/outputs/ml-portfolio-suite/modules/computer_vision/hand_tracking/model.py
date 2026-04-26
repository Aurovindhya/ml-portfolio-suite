"""
modules/computer_vision/hand_tracking/model.py

Real-time hand tracking and gesture recognition using OpenCV + MediaPipe.

Features:
  - 21-landmark detection per hand (up to 2 hands)
  - Finger counting (0–5 per hand)
  - Gesture classification: open, closed fist, thumbs up, peace sign, pointing
  - Works on webcam feed or static image input
  - Outputs annotated frame bytes + landmark JSON

Usage (static image):
    from modules.computer_vision.hand_tracking.model import analyze_image
    result = analyze_image(image_bytes)

Usage (webcam, run as script):
    python -m modules.computer_vision.hand_tracking.webcam
"""

import io
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# MediaPipe landmark indices for fingertips and knuckles
FINGER_TIPS   = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
FINGER_KNUCK  = [3, 6, 10, 14, 18]   # proximal knuckles


def _count_fingers(landmarks) -> int:
    """Count raised fingers from MediaPipe hand landmarks."""
    count = 0
    # Thumb: tip x < knuckle x (for right hand; flipped for left)
    if landmarks[4].x < landmarks[3].x:
        count += 1
    # Other fingers: tip y < knuckle y (finger is raised when tip is above knuckle)
    for tip, knuck in zip(FINGER_TIPS[1:], FINGER_KNUCK[1:]):
        if landmarks[tip].y < landmarks[knuck].y:
            count += 1
    return count


def _classify_gesture(finger_count: int, landmarks) -> str:
    """Rule-based gesture classification from finger count + landmark geometry."""
    if finger_count == 0:
        return "closed_fist"
    if finger_count == 5:
        return "open_hand"
    if finger_count == 1:
        # Only index raised → pointing; only thumb → thumbs up
        index_up = landmarks[8].y < landmarks[6].y
        thumb_up = landmarks[4].x < landmarks[3].x
        if index_up:
            return "pointing"
        if thumb_up:
            return "thumbs_up"
    if finger_count == 2:
        index_up  = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        if index_up and middle_up:
            return "peace_sign"
    return f"{finger_count}_fingers"


def analyze_image(image_bytes: bytes) -> Dict:
    """
    Detect hands in a static image.

    Returns:
        {
          "hands_detected": int,
          "hands": [{"finger_count": int, "gesture": str, "landmarks": [...]}],
          "annotated_image_bytes": bytes,  # JPEG bytes of annotated frame
          "inference_ms": float,
        }
    """
    if not CV2_AVAILABLE:
        raise ImportError("opencv-python not installed. Run: pip install opencv-python")
    if not MP_AVAILABLE:
        raise ImportError("mediapipe not installed. Run: pip install mediapipe")

    mp_hands  = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(rgb)
    ms = (time.time() - t0) * 1000

    hand_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            fingers = _count_fingers(lm)
            gesture = _classify_gesture(fingers, lm)

            hand_data.append({
                "finger_count": fingers,
                "gesture": gesture,
                "landmarks": [
                    {"id": i, "x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)}
                    for i, p in enumerate(lm)
                ],
            })

    # Encode annotated frame
    _, buffer = cv2.imencode(".jpg", frame)
    annotated_bytes = buffer.tobytes()

    return {
        "hands_detected":       len(hand_data),
        "hands":                hand_data,
        "annotated_image_bytes": annotated_bytes,
        "inference_ms":          round(ms, 2),
    }
