from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def count(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])


class SignLanguageDetector:
    """Original static sign detector built on hand landmark geometry."""

    def __init__(self) -> None:
        self.supported_signs = [
            "Open Palm / Stop",
            "Fist / Yes",
            "Peace",
            "Point One",
            "I Love You",
            "Thumbs Up",
            "OK",
            "Call Me",
            "Unknown",
        ]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def predict(self, frame_bgr: np.ndarray) -> dict:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return {
                "label": "No hand detected",
                "confidence": 0.0,
                "landmarks": [],
                "fingerState": {},
            }

        hand_landmarks = results.multi_hand_landmarks[0]
        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label

        normalized = self._normalize(points)
        fingers = self._finger_state(normalized, handedness)
        label, confidence = self._classify(normalized, fingers)

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "handedness": handedness,
            "landmarks": [
                {"x": round(float(point[0]), 4), "y": round(float(point[1]), 4)}
                for point in points
            ],
            "fingerState": {
                "thumb": fingers.thumb,
                "index": fingers.index,
                "middle": fingers.middle,
                "ring": fingers.ring,
                "pinky": fingers.pinky,
            },
        }

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        wrist = points[0].copy()
        normalized = points - wrist
        palm_size = self._distance(normalized[0], normalized[9])
        if palm_size > 0:
            normalized = normalized / palm_size
        return normalized

    def _finger_state(self, points: np.ndarray, handedness: str) -> FingerState:
        index = self._is_finger_extended(points, tip=8, pip=6, mcp=5)
        middle = self._is_finger_extended(points, tip=12, pip=10, mcp=9)
        ring = self._is_finger_extended(points, tip=16, pip=14, mcp=13)
        pinky = self._is_finger_extended(points, tip=20, pip=18, mcp=17)

        thumb_tip = points[4]
        thumb_ip = points[3]
        index_mcp = points[5]
        pinky_mcp = points[17]
        palm_width = self._distance(index_mcp, pinky_mcp)
        thumb_reach = self._distance(thumb_tip, index_mcp)
        thumb_sideways = abs(thumb_tip[0] - thumb_ip[0]) > 0.18
        thumb = thumb_sideways and thumb_reach > palm_width * 0.55

        return FingerState(thumb=thumb, index=index, middle=middle, ring=ring, pinky=pinky)

    def _is_finger_extended(self, points: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
        tip_to_wrist = self._distance(points[tip], points[0])
        pip_to_wrist = self._distance(points[pip], points[0])
        tip_above_pip = points[tip][1] < points[pip][1] - 0.08
        straight_enough = tip_to_wrist > pip_to_wrist * 1.08
        raised_from_knuckle = self._distance(points[tip], points[mcp]) > 0.78
        return bool((tip_above_pip and straight_enough) or raised_from_knuckle)

    def _classify(self, points: np.ndarray, fingers: FingerState) -> tuple[str, float]:
        thumb_index_gap = self._distance(points[4], points[8])
        thumb_middle_gap = self._distance(points[4], points[12])
        index_middle_gap = self._distance(points[8], points[12])
        ring_pinky_gap = self._distance(points[16], points[20])

        if fingers.count == 5:
            return "Open Palm / Stop", 0.93

        if fingers.thumb and fingers.index and fingers.pinky and not fingers.middle and not fingers.ring:
            return "I Love You", 0.92

        if fingers.index and fingers.middle and not fingers.ring and not fingers.pinky:
            confidence = 0.88 if index_middle_gap > 0.24 else 0.76
            return "Peace", confidence

        if fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky and not fingers.thumb:
            return "Point One", 0.9

        if fingers.thumb and not fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky:
            return "Thumbs Up", 0.86

        if fingers.thumb and fingers.pinky and not fingers.index and not fingers.middle and not fingers.ring:
            return "Call Me", 0.88

        if fingers.middle and fingers.ring and fingers.pinky and thumb_index_gap < 0.42:
            return "OK", 0.84

        if fingers.count == 0 or (
            not fingers.index and not fingers.middle and not fingers.ring and not fingers.pinky
        ):
            return "Fist / Yes", 0.87

        if thumb_index_gap < 0.34 and thumb_middle_gap > 0.55:
            return "OK", 0.76

        if fingers.index and fingers.middle and ring_pinky_gap < 0.25:
            return "Peace", 0.68

        return "Unknown", 0.45

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        return float(hypot(first[0] - second[0], first[1] - second[1]))
