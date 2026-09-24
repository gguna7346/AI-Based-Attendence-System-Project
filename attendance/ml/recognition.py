from __future__ import annotations



"""

OpenCV-based face embedding and matching utilities.



Uses Haar cascades with eye alignment, CLAHE normalization, and a combined

pixel + HOG descriptor for more stable recognition than raw grayscale pixels.

"""



from dataclasses import dataclass

from pathlib import Path

from typing import Dict, Iterable, List, Optional, Sequence, Tuple



import cv2  # type: ignore[import]

import numpy as np





HAAR_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

HAAR_CASCADE_ALT_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml")

EYE_CASCADE_PATH = str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml")



FACE_SIZE = (128, 128)

PIXEL_GRID = (64, 64)



# Tuned for combined pixel+HOG embeddings (re-train after changing these).

MATCH_THRESHOLD = 0.78

MATCH_MARGIN = 0.04

PER_FRAME_VOTE_THRESHOLD = 0.72

MIN_FRAME_CONSENSUS_COUNT = 2



_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None

_FACE_CASCADE_ALT: Optional[cv2.CascadeClassifier] = None

_EYE_CASCADE: Optional[cv2.CascadeClassifier] = None

_HOG: Optional[cv2.HOGDescriptor] = None





@dataclass

class LabeledEmbedding:

    student_id: int

    vector: np.ndarray





@dataclass

class MatchResult:

    student_id: Optional[int]

    best_score: float

    second_best_score: float

    margin: float





def _get_face_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:

    global _FACE_CASCADE, _FACE_CASCADE_ALT

    if _FACE_CASCADE is None:

        _FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    if _FACE_CASCADE_ALT is None:

        _FACE_CASCADE_ALT = cv2.CascadeClassifier(HAAR_CASCADE_ALT_PATH)

    return _FACE_CASCADE, _FACE_CASCADE_ALT





def _get_eye_cascade() -> cv2.CascadeClassifier:

    global _EYE_CASCADE

    if _EYE_CASCADE is None:

        _EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_PATH)

    return _EYE_CASCADE





def _get_hog() -> cv2.HOGDescriptor:

    global _HOG

    if _HOG is None:

        _HOG = cv2.HOGDescriptor(

            (64, 64),

            (16, 16),

            (8, 8),

            (8, 8),

            9,

        )

    return _HOG





def _detect_largest_face_bbox(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:

    """Detect the largest face using primary and fallback Haar cascades."""

    for cascade in _get_face_cascades():

        if cascade.empty():

            continue

        faces = cascade.detectMultiScale(

            gray,

            scaleFactor=1.08,

            minNeighbors=4,

            minSize=(50, 50),

            flags=cv2.CASCADE_SCALE_IMAGE,

        )

        if len(faces) > 0:

            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            return int(x), int(y), int(w), int(h)

    return None





def _expand_bbox(

    x: int, y: int, w: int, h: int, img_w: int, img_h: int, padding: float = 0.15

) -> Tuple[int, int, int, int]:

    """Add padding around the detected face for a more consistent crop."""

    pad_w = int(w * padding)

    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)

    y1 = max(0, y - pad_h)

    x2 = min(img_w, x + w + pad_w)

    y2 = min(img_h, y + h + pad_h)

    return x1, y1, x2 - x1, y2 - y1





def _align_face_by_eyes(face_gray: np.ndarray) -> np.ndarray:

    """Rotate the face so both eyes sit on a horizontal line."""

    h, w = face_gray.shape[:2]

    if h < 40 or w < 40:

        return face_gray



    eye_cascade = _get_eye_cascade()

    if eye_cascade.empty():

        return face_gray



    eye_band = face_gray[0 : int(h * 0.6), :]

    eyes = eye_cascade.detectMultiScale(

        eye_band,

        scaleFactor=1.1,

        minNeighbors=3,

        minSize=(12, 12),

    )

    if len(eyes) < 2:

        return face_gray



    eyes = sorted(eyes, key=lambda e: e[0])

    left = eyes[0]

    right = eyes[-1]



    left_center = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)

    right_center = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)



    dy = right_center[1] - left_center[1]

    dx = right_center[0] - left_center[0]

    if abs(dx) < 1.0:

        return face_gray



    angle = float(np.degrees(np.arctan2(dy, dx)))

    if abs(angle) < 1.5:

        return face_gray



    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(

        face_gray,

        matrix,

        (w, h),

        flags=cv2.INTER_LINEAR,

        borderMode=cv2.BORDER_REPLICATE,

    )





def _preprocess_face_roi(gray_roi: np.ndarray) -> np.ndarray:

    """Align, denoise, resize, and normalize lighting."""

    aligned = _align_face_by_eyes(gray_roi)

    denoised = cv2.bilateralFilter(aligned, d=5, sigmaColor=50, sigmaSpace=50)

    face_resized = cv2.resize(denoised, FACE_SIZE, interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    return clahe.apply(face_resized)





def _vectorize_face(face_gray: np.ndarray) -> np.ndarray:

    """Build a combined pixel + HOG embedding and L2-normalize it."""

    small = cv2.resize(face_gray, PIXEL_GRID, interpolation=cv2.INTER_AREA)

    pixel_vec = small.astype("float32").flatten()

    pixel_norm = np.linalg.norm(pixel_vec)

    if pixel_norm > 0:

        pixel_vec /= pixel_norm



    hog_input = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)

    hog_vec = _get_hog().compute(hog_input).astype("float32").flatten()

    hog_norm = np.linalg.norm(hog_vec)

    if hog_norm > 0:

        hog_vec /= hog_norm



    combined = np.concatenate([pixel_vec * 0.55, hog_vec * 0.45])

    combined_norm = np.linalg.norm(combined)

    if combined_norm > 0:

        combined /= combined_norm

    return combined





def extract_face_embedding_from_bgr(img: np.ndarray) -> Optional[np.ndarray]:

    """

    Detect a face in-memory and return a normalized embedding vector.

    """

    if img is None:

        return None



    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_bbox = _detect_largest_face_bbox(gray)

    if face_bbox is None:

        return None



    img_h, img_w = gray.shape[:2]

    x, y, w, h = _expand_bbox(*face_bbox, img_w, img_h)

    face_roi = gray[y : y + h, x : x + w]

    face_processed = _preprocess_face_roi(face_roi)

    return _vectorize_face(face_processed)





def extract_face_embedding(image_path: str) -> Optional[np.ndarray]:

    """

    Load an image from disk, detect a face, and return a normalized embedding.

    Returns None if no face is detected.

    """

    img = cv2.imread(image_path)

    return extract_face_embedding_from_bgr(img) if img is not None else None





def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))





def build_student_centroids(labeled_vectors: Sequence[LabeledEmbedding]) -> Dict[int, np.ndarray]:

    """Average each student's training embeddings into one normalized centroid."""

    by_student: Dict[int, List[np.ndarray]] = {}

    for item in labeled_vectors:

        by_student.setdefault(item.student_id, []).append(item.vector)



    centroids: Dict[int, np.ndarray] = {}

    for student_id, vectors in by_student.items():

        mean = np.mean(vectors, axis=0)

        norm = np.linalg.norm(mean)

        if norm > 0:

            mean /= norm

        centroids[student_id] = mean

    return centroids





def _best_individual_scores(

    query_vec: np.ndarray, labeled_vectors: Sequence[LabeledEmbedding]

) -> Dict[int, float]:

    """Best cosine score per student across all individual training embeddings."""

    scores: Dict[int, float] = {}

    for item in labeled_vectors:

        score = cosine_similarity(query_vec, item.vector)

        scores[item.student_id] = max(scores.get(item.student_id, -1.0), score)

    return scores





def _score_students(

    query_vec: np.ndarray,

    centroids: Dict[int, np.ndarray],

    labeled_vectors: Sequence[LabeledEmbedding],

) -> List[Tuple[int, float]]:

    individual = _best_individual_scores(query_vec, labeled_vectors)

    scores = []

    for student_id, centroid in centroids.items():

        centroid_score = cosine_similarity(query_vec, centroid)

        individual_score = individual.get(student_id, -1.0)

        scores.append((student_id, max(centroid_score, individual_score)))

    scores.sort(key=lambda item: item[1], reverse=True)

    return scores





def find_best_match_centroid(

    query_vec: np.ndarray,

    labeled_vectors: Sequence[LabeledEmbedding],

    threshold: float = MATCH_THRESHOLD,

    margin: float = MATCH_MARGIN,

) -> MatchResult:

    """

    Match against per-student centroids and individual embeddings.

    """

    centroids = build_student_centroids(labeled_vectors)

    if not centroids:

        return MatchResult(None, -1.0, -1.0, 0.0)



    scores = _score_students(query_vec, centroids, labeled_vectors)

    best_id, best_score = scores[0]

    second_score = scores[1][1] if len(scores) > 1 else -1.0

    margin_val = best_score - second_score



    if best_score >= threshold and margin_val >= margin:

        return MatchResult(best_id, best_score, second_score, margin_val)

    return MatchResult(None, best_score, second_score, margin_val)





def recognize_face(

    query_vectors: Sequence[np.ndarray],

    labeled_vectors: Sequence[LabeledEmbedding],

    threshold: float = MATCH_THRESHOLD,

    margin: float = MATCH_MARGIN,

) -> MatchResult:

    """

    Recognize a face using multi-frame consensus and per-student centroids.

    """

    if not query_vectors or not labeled_vectors:

        return MatchResult(None, -1.0, -1.0, 0.0)



    centroids = build_student_centroids(labeled_vectors)

    if not centroids:

        return MatchResult(None, -1.0, -1.0, 0.0)



    votes: Dict[int, int] = {}

    for vec in query_vectors:

        scores = _score_students(vec, centroids, labeled_vectors)

        if scores and scores[0][1] >= PER_FRAME_VOTE_THRESHOLD:

            winner_id = scores[0][0]

            votes[winner_id] = votes.get(winner_id, 0) + 1



    if not votes:

        return MatchResult(None, -1.0, -1.0, 0.0)



    voted_student = max(votes, key=lambda sid: votes[sid])

    min_votes = max(MIN_FRAME_CONSENSUS_COUNT, len(query_vectors) // 3)

    if votes[voted_student] < min_votes:

        return MatchResult(None, -1.0, -1.0, 0.0)



    avg_query = np.mean(query_vectors, axis=0)

    norm = np.linalg.norm(avg_query)

    if norm > 0:

        avg_query /= norm



    scores = _score_students(avg_query, centroids, labeled_vectors)

    best_id, best_score = scores[0]

    second_score = scores[1][1] if len(scores) > 1 else -1.0

    margin_val = best_score - second_score



    if (

        best_id == voted_student

        and best_score >= threshold

        and margin_val >= margin

    ):

        return MatchResult(voted_student, best_score, second_score, margin_val)

    return MatchResult(None, best_score, second_score, margin_val)





def find_best_match(

    query_vec: np.ndarray, labeled_vectors: Sequence[LabeledEmbedding], threshold: float = 0.75

) -> Optional[int]:

    """Legacy single-vector matcher. Prefer `recognize_face` for attendance."""

    result = find_best_match_centroid(query_vec, labeled_vectors, threshold=threshold, margin=0.0)

    return result.student_id





def json_to_vector(data: Iterable[float]) -> np.ndarray:

    arr = np.asarray(list(data), dtype="float32")

    norm = np.linalg.norm(arr)

    if norm > 0:

        arr /= norm

    return arr


