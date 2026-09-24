from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np


FACE_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATHS = [
    str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"),
    str(Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"),
]

FACE_ROI_SIZE = (160, 160)  # for motion scoring


@dataclass
class FaceTrackSample:
    frame_index: int
    face_bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    face_area: int
    eyes_open: bool
    face_crop_gray: np.ndarray  # resized grayscale face crop


def _load_face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)


def _load_eye_cascade() -> List[cv2.CascadeClassifier]:
    cascades: List[cv2.CascadeClassifier] = []
    for p in EYE_CASCADE_PATHS:
        casc = cv2.CascadeClassifier(p)
        if not casc.empty():
            cascades.append(casc)
    return cascades


def _detect_largest_face_bbox(
    gray: np.ndarray, face_cascade: cv2.CascadeClassifier
) -> Optional[Tuple[int, int, int, int]]:
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _eyes_open_from_face_roi(face_gray: np.ndarray, eye_cascades: List[cv2.CascadeClassifier]) -> bool:
    """
    Approximate "eye open" by using OpenCV Haar eye cascades inside the top
    half of the face ROI.
    """
    h, w = face_gray.shape[:2]
    if h < 30 or w < 30:
        return False

    eye_band = face_gray[0 : int(h * 0.55), :]
    band_h, band_w = eye_band.shape[:2]
    half_w = band_w // 2
    left_roi = eye_band[:, 0:half_w]
    right_roi = eye_band[:, half_w:band_w]

    # Minimum eye area scales with face ROI size.
    min_eye_area = max(20, int(0.0008 * (h * w)))

    best_left_area = 0.0
    best_right_area = 0.0

    for casc in eye_cascades:
        if casc.empty():
            continue

        left_eyes = casc.detectMultiScale(
            left_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
        )
        right_eyes = casc.detectMultiScale(
            right_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(10, 10),
        )

        for (ex, ey, ew, eh) in left_eyes:
            best_left_area = max(best_left_area, float(ew * eh))
        for (ex, ey, ew, eh) in right_eyes:
            best_right_area = max(best_right_area, float(ew * eh))

    # If we can find at least one reasonable eye region, treat it as open.
    total_area = best_left_area + best_right_area
    return total_area >= float(min_eye_area)


def _score_live_motion(crops_gray: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Score "live" motion using simple pixel diffs and optical-flow magnitude.
    """
    if len(crops_gray) < 2:
        return False, 0.0

    diff_scores: List[float] = []
    flow_scores: List[float] = []

    # Farneback parameters chosen for small grayscale crops.
    for i in range(1, len(crops_gray)):
        prev = crops_gray[i - 1]
        curr = crops_gray[i]

        abs_diff = cv2.absdiff(prev, curr)
        diff_norm = float(np.mean(abs_diff)) / 255.0
        diff_scores.append(diff_norm)

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_norm = float(np.mean(mag)) / 10.0  # normalize magnitude scale
        flow_scores.append(flow_norm)

    diff_max = max(diff_scores) if diff_scores else 0.0
    flow_max = max(flow_scores) if flow_scores else 0.0

    live_motion_ok = diff_max >= 0.012 or flow_max >= 0.028
    motion_score = 0.6 * diff_max + 0.4 * flow_max
    return live_motion_ok, float(motion_score)


def _count_blinks(eyes_open_seq: List[bool]) -> int:
    """
    Count blinks by counting open->closed->open transitions.
    """
    blink_count = 0
    closed_run = 0

    for is_open in eyes_open_seq:
        if is_open:
            if 1 <= closed_run <= 4:
                blink_count += 1
            closed_run = 0
        else:
            closed_run += 1
            # Prevent long "closed" streaks from being counted as repeated blinks.
            if closed_run > 10:
                closed_run = 10

    return blink_count


def check_liveness(frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
    """
    Anti-spoof heuristic checks over a short frame sequence.

    Returns a dict with:
      - passed: bool
      - blink_count: int
      - head_movement_ratio: float
      - live_motion_score: float
      - frames_used: int
      - best_frame_index: int (index into input list)
    """
    if not frames_bgr:
        return {
            "passed": False,
            "blink_count": 0,
            "head_movement_ratio": 0.0,
            "live_motion_score": 0.0,
            "frames_used": 0,
            "best_frame_index": -1,
        }

    eye_cascades = _load_eye_cascade()
    if not eye_cascades:
        # If OpenCV's eye cascade isn't available, we can still do motion + head movement.
        eye_cascades = []

    # Track face bbox + simple features for each frame.
    samples: List[FaceTrackSample] = []
    base_h, base_w = frames_bgr[0].shape[:2]
    face_cascade = _load_face_cascade()

    for i, frame in enumerate(frames_bgr):
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_bbox = _detect_largest_face_bbox(gray, face_cascade)
        if face_bbox is None:
            continue

        x, y, w, h = face_bbox
        face_area = int(w * h)
        centroid = (x + w / 2.0, y + h / 2.0)
        face_roi = gray[y : y + h, x : x + w]
        face_crop_gray = cv2.resize(face_roi, FACE_ROI_SIZE, interpolation=cv2.INTER_AREA)

        eyes_open = False
        if eye_cascades:
            eyes_open = _eyes_open_from_face_roi(face_roi, eye_cascades)

        samples.append(
            FaceTrackSample(
                frame_index=i,
                face_bbox=face_bbox,
                centroid=centroid,
                face_area=face_area,
                eyes_open=eyes_open,
                face_crop_gray=face_crop_gray,
            )
        )

    if len(samples) < 2:
        return {
            "passed": False,
            "blink_count": 0,
            "head_movement_ratio": 0.0,
            "live_motion_score": 0.0,
            "frames_used": len(samples),
            "best_frame_index": samples[0].frame_index if samples else -1,
        }

    best_sample = max(samples, key=lambda s: s.face_area)
    best_frame_index = int(best_sample.frame_index)

    # Blink scoring
    eyes_open_seq = [s.eyes_open for s in samples]
    blink_count = _count_blinks(eyes_open_seq)

    # Head movement scoring (normalized centroid displacement)
    c0x, c0y = samples[0].centroid
    max_disp = 0.0
    for s in samples:
        dx = abs(s.centroid[0] - c0x) / max(1.0, float(base_w))
        dy = abs(s.centroid[1] - c0y) / max(1.0, float(base_h))
        disp = float(np.sqrt(dx * dx + dy * dy))
        if disp > max_disp:
            max_disp = disp

    head_movement_ratio = max_disp
    head_movement_ok = head_movement_ratio >= 0.012

    crops_gray = [s.face_crop_gray for s in samples]
    live_motion_ok, motion_score = _score_live_motion(crops_gray)

    blink_ok = blink_count >= 1

    area_std = float(np.std([s.face_area for s in samples]))
    face_scale_change_ok = area_std >= 80.0

    passed = blink_ok or head_movement_ok or live_motion_ok or face_scale_change_ok

    return {
        "passed": bool(passed),
        "blink_count": int(blink_count),
        "head_movement_ratio": float(head_movement_ratio),
        "live_motion_score": float(motion_score),
        "frames_used": int(len(samples)),
        "best_frame_index": int(best_frame_index),
    }

