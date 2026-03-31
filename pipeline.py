"""
Cherry Sorting Pipeline

Real-time detection, tracking, and classification pipeline for cherry sorting.
Runs a YOLO detector (ONNX) with ByteTrack multi-object tracking and a timm
classifier (ONNX) to decide PERFECT / IMPERFECT / NO_STEM for each tracked cherry.

Usage:
    python pipeline.py --detector models/detect.onnx --classifier models/cls.onnx --source 0
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from hyperparams import (
    ARUCO_CALIB_FRAMES,
    ARUCO_REAL_SIZE_CM,
    CENTER_WEIGHT_SIGMA,
    CROP_BUFFER_PX,
    DETECTION_CONF_THRESH,
    IMPERFECT_THRESHOLD,
    MAX_SAMPLES_PER_TRACK,
    MIN_SAMPLES_FOR_DECISION,
    SAMPLE_EVERY_N_FRAMES,
    TRACK_BUFFER,
)
from log import logger

# ---------------------------------------------------------------------------
# Class index constants (YOLO classifier sorts classes alphabetically)
# Folder names: "cherry" (0=perfect), "cherry-imperfect" (1=imperfect)
# ---------------------------------------------------------------------------

CHERRY_CLASS: int = 0  # detector class index for cherry body
STEM_CLASS: int = 1  # detector class index for stem
PERFECT_CLASS: int = 0  # classifier index for perfect cherry
IMPERFECT_CLASS: int = 1  # classifier index for imperfect cherry

MAX_CHERRIES: int = 3  # max cherries kept per frame (top-N by confidence)
MAX_STEMS: int = 3  # max stems kept per frame (top-N by confidence)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrackState(Enum):
    ACTIVE = auto()
    LOST = auto()
    REMOVED = auto()


class Decision(Enum):
    PENDING = auto()
    PERFECT = auto()
    IMPERFECT = auto()
    NO_STEM = auto()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassificationSample:
    crop: np.ndarray
    frame_id: int
    detection_conf: float
    center_offset: float
    classifier_conf: float | None = None
    label: int | None = None  # PERFECT_CLASS or IMPERFECT_CLASS


@dataclass
class OnnxClassifier:
    session: onnxruntime.InferenceSession
    input_name: str
    input_size: tuple[int, int]  # (H, W)
    fixed_batch: bool  # True when the model only accepts batch_size=1


@dataclass
class Track:
    track_id: int
    state: TrackState = TrackState.ACTIVE
    age: int = 0
    frames_since_seen: int = 0
    bbox_history: list[np.ndarray] = field(default_factory=list)
    samples: list[ClassificationSample] = field(default_factory=list)
    stem_detections: list[bool] = field(default_factory=list)
    decision: Decision = Decision.PENDING
    decided: bool = False


# ---------------------------------------------------------------------------
# ArUco calibration
# ---------------------------------------------------------------------------


def _compute_marker_size_px(corners: np.ndarray) -> float:
    """Average side length of a 4-corner marker polygon in pixels.

    Args:
        corners: Array of shape (4, 2) with [TL, TR, BR, BL] corner coordinates.

    Returns:
        Mean side length in pixels.
    """
    side_lengths = [
        float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])) for i in range(4)
    ]
    return float(np.mean(side_lengths))


def calibrate_aruco(
    cap: cv2.VideoCapture,
    marker_real_size_cm: float = ARUCO_REAL_SIZE_CM,
    n_frames: int = ARUCO_CALIB_FRAMES,
    show: bool = False,
    calib_seconds: float = 10.0,
) -> float | None:
    """Estimate pixels-per-cm from ArUco marker detections.

    When *show* is False, reads up to *n_frames* frames silently.
    When *show* is True, runs for *calib_seconds* seconds, displays a live
    window with detected marker outlines, and prints a terminal countdown.

    Args:
        cap: An already-opened cv2.VideoCapture (position is NOT reset).
        marker_real_size_cm: Known physical side length of the ArUco marker.
        n_frames: Frames to sample when show=False.
        show: If True, display a live calibration window with countdown.
        calib_seconds: Duration of the calibration window when show=True.

    Returns:
        Median px/cm ratio, or None if no markers detected.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    readings: list[float] = []

    if show:
        deadline = time.monotonic() + calib_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ret, frame = cap.read()
            if not ret:
                break
            corners, ids, _ = aruco_detector.detectMarkers(frame)
            for c in corners:
                readings.append(_compute_marker_size_px(c[0]) / marker_real_size_cm)
            vis = frame.copy()
            if corners:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.putText(
                vis,
                f"Calibrating... {remaining:.1f}s",
                (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.imshow("ArUco Calibration", vis)
            cv2.waitKey(1)
            print(f"\rCalibrating ArUco… {remaining:.1f}s remaining", end="", flush=True)
        print()  # newline after countdown finishes
        cv2.destroyWindow("ArUco Calibration")
    else:
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            corners, _, _ = aruco_detector.detectMarkers(frame)
            for c in corners:
                readings.append(_compute_marker_size_px(c[0]) / marker_real_size_cm)

    if not readings:
        return None
    return float(np.median(readings))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def gaussian(x: float, sigma: float) -> float:
    """Gaussian weight: 1.0 at x=0, decays with sigma."""
    return math.exp(-0.5 * (x / sigma) ** 2)


def compute_center_offset(bbox: np.ndarray, frame_width: int) -> float:
    """Normalised horizontal distance of cherry center from frame center.

    Returns:
        0.0 when cherry center is at the frame center, 1.0 at the frame edge.
    """
    cherry_cx = (bbox[0] + bbox[2]) / 2.0
    return abs(cherry_cx - frame_width / 2.0) / (frame_width / 2.0)


def clip_with_buffer(frame: np.ndarray, bbox: np.ndarray, buffer_px: int) -> np.ndarray:
    """Crop *frame* to *bbox* expanded by *buffer_px*, clamped to frame bounds.

    Args:
        frame: Full BGR frame (H x W x 3).
        bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates.
        buffer_px: Extra pixels to add on every side.

    Returns:
        Cropped BGR image (copy).
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - buffer_px)
    y1 = max(0, int(bbox[1]) - buffer_px)
    x2 = min(w, int(bbox[2]) + buffer_px)
    y2 = min(h, int(bbox[3]) + buffer_px)
    return frame[y1:y2, x1:x2].copy()


def _top_k_per_class(
    cls_ids: np.ndarray,
    confs: np.ndarray,
    target_class: int,
    k: int,
) -> np.ndarray:
    """Return a boolean mask keeping only the top-*k* detections of *target_class*.

    Detections of other classes are not included in the returned mask.
    """
    indices = np.where(cls_ids == target_class)[0]
    if len(indices) > k:
        indices = indices[np.argsort(confs[indices])[::-1][:k]]
    mask = np.zeros(len(cls_ids), dtype=bool)
    mask[indices] = True
    return mask


def any_stem_overlaps(
    cherry_bbox: np.ndarray,
    stem_boxes: np.ndarray,
    iou_thresh: float = 0.1,
) -> bool:
    """Return True if any stem box overlaps with *cherry_bbox*.

    Two checks are performed:
    1. IoU > *iou_thresh*
    2. Stem centroid falls inside *cherry_bbox* (looser spatial check)

    Args:
        cherry_bbox: [x1, y1, x2, y2] for the cherry.
        stem_boxes: Array of shape (N, 4) with stem bounding boxes.
        iou_thresh: Minimum IoU to count as an overlap.

    Returns:
        True if at least one stem overlaps.
    """
    if len(stem_boxes) == 0:
        return False

    # IoU check (vectorised via torchvision)
    cherry_t = torch.tensor(cherry_bbox[:4][None], dtype=torch.float32)
    stems_t = torch.tensor(stem_boxes[:, :4], dtype=torch.float32)
    ious = box_iou(cherry_t, stems_t)[0]
    if float(ious.max().item()) > iou_thresh:
        return True

    # Centroid containment fallback
    cx1, cy1, cx2, cy2 = cherry_bbox[:4]
    for stem in stem_boxes:
        stem_cx = (stem[0] + stem[2]) / 2.0
        stem_cy = (stem[1] + stem[3]) / 2.0
        if cx1 <= stem_cx <= cx2 and cy1 <= stem_cy <= cy2:
            return True

    return False


# ---------------------------------------------------------------------------
# Decision aggregation
# ---------------------------------------------------------------------------


def aggregate_decision(
    track: Track,
    classified_samples: list[ClassificationSample],
) -> tuple[Decision, float]:
    """Compute a weighted sorting decision from all classified samples.

    Each sample is weighted by detection confidence × Gaussian center weight.
    A higher weight means the cherry was well-centred and confidently detected.

    Returns:
        (decision, weighted_imperfect_score) where score ∈ [0, 1].
    """
    weighted_imperfect_sum = 0.0
    weight_total = 0.0

    for sample in classified_samples:
        assert sample.label is not None and sample.classifier_conf is not None
        center_weight = gaussian(sample.center_offset, CENTER_WEIGHT_SIGMA)
        w = sample.detection_conf * center_weight

        # classifier_conf is confidence for the top-1 predicted class;
        # convert to P(imperfect) regardless of which class was top-1.
        if sample.label == IMPERFECT_CLASS:
            p_imperfect = sample.classifier_conf
        else:
            p_imperfect = 1.0 - sample.classifier_conf

        weighted_imperfect_sum += w * p_imperfect
        weight_total += w

    if weight_total == 0.0:
        return Decision.IMPERFECT, 1.0  # fail-safe: reject

    weighted_score = weighted_imperfect_sum / weight_total

    # Stem check: reject if stem was absent in the majority of frames
    stem_ratio = sum(track.stem_detections) / max(len(track.stem_detections), 1)
    if stem_ratio <= 0.5:
        return Decision.NO_STEM, weighted_score

    if weighted_score > IMPERFECT_THRESHOLD:
        return Decision.IMPERFECT, weighted_score
    return Decision.PERFECT, weighted_score


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    source: int | str | Path,
    detector_path: str | Path,
    classifier_path: str | Path,
    show: bool = False,
) -> Generator[tuple[Track, float, int], None, None]:
    """Run the cherry sorting pipeline and yield decisions as they are made.

    Performs ArUco calibration on the first ARUCO_CALIB_FRAMES frames, then
    enters a per-frame loop that detects cherries, tracks them with ByteTrack,
    collects crops, classifies them in batch, and aggregates a decision.

    Yields (Track, score, decision_frame_id) each time a track reaches a final
    decision (either early — MIN_SAMPLES_FOR_DECISION reached — or when the
    track is lost after TRACK_BUFFER frames). The decision itself is stored in
    ``track.decision``.

    Args:
        source: Camera index (int), video file path, or RTSP URL string.
        detector_path: Path to the YOLO detection model (.pt).
        classifier_path: Path to the YOLO classification model (.pt).
        show: If True, display an annotated live feed via cv2.imshow.

    Raises:
        RuntimeError: If the video source cannot be opened or ArUco
                      calibration fails (no marker found).
    """
    detector = _load_model(detector_path)
    classifier = _load_onnx_classifier(classifier_path)

    cap_source: int | str = int(source) if isinstance(source, int) else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    try:
        # ── Calibration ───────────────────────────────────────────────────
        logger.info("Starting ArUco calibration…")
        px_per_cm = calibrate_aruco(cap, show=show)
        if px_per_cm is None:
            raise RuntimeError(
                "ArUco marker not detected — cannot calibrate. Aborting."
            )
        logger.info(f"Calibration complete: {px_per_cm:.2f} px/cm")

        active_tracks: dict[int, Track] = {}
        completed_queue: deque[tuple[Track, float, int]] = deque()
        frame_id = 0

        # ── Per-frame loop ────────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_width: int = frame.shape[1]

            # 1 & 2. DETECT + TRACK (ByteTrack built into ultralytics)
            results = detector.track(
                frame,
                persist=True,
                conf=DETECTION_CONF_THRESH,
                verbose=False,
                tracker="bytetrack.yaml",
            )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                _increment_lost(active_tracks, set(), completed_queue, frame_id)
                frame_id += 1
                yield from _drain(completed_queue)
                if show:
                    _show_detection_frame(frame, np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int))
                    _show_otsu_frame(frame, {})
                    _show_frame(frame, active_tracks)
                continue

            xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = boxes.conf.cpu().numpy()  # (N,)
            cls_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)
            ids_tensor = boxes.id
            track_ids = (
                ids_tensor.cpu().numpy().astype(int) if ids_tensor is not None else None
            )

            # Cap detections to top-N per class by confidence
            keep = _top_k_per_class(cls_ids, confs, CHERRY_CLASS, MAX_CHERRIES)
            keep |= _top_k_per_class(cls_ids, confs, STEM_CLASS, MAX_STEMS)
            xyxy = xyxy[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep]
            if track_ids is not None:
                track_ids = track_ids[keep]

            stem_mask = cls_ids == STEM_CLASS
            stem_boxes = xyxy[stem_mask] if stem_mask.any() else np.empty((0, 4))

            # 3. SYNC active_tracks with ByteTrack output
            # Build per-frame cherry track info for use in sampling step
            current_ids: set[int] = set()
            cherry_track_info: dict[int, tuple[np.ndarray, float]] = {}

            if track_ids is not None:
                for bbox, conf, cls_id, tid in zip(xyxy, confs, cls_ids, track_ids):
                    if cls_id != CHERRY_CLASS or tid <= 0:
                        continue
                    tid = int(tid)
                    current_ids.add(tid)
                    cherry_track_info[tid] = (bbox, float(conf))

                    if tid not in active_tracks:
                        active_tracks[tid] = Track(track_id=tid)

                    track = active_tracks[tid]
                    track.state = TrackState.ACTIVE
                    track.age += 1
                    track.frames_since_seen = 0
                    track.bbox_history.append(bbox)
                    track.stem_detections.append(
                        any_stem_overlaps(bbox, stem_boxes, iou_thresh=0.1)
                    )

            # Mark non-current tracks as LOST and increment counter
            _increment_lost(active_tracks, current_ids, completed_queue, frame_id)

            # 4. SAMPLE — collect crops from ACTIVE, undecided tracks
            for tid, (bbox, conf) in cherry_track_info.items():
                track = active_tracks[tid]
                if track.decided:
                    continue
                if len(track.samples) >= MAX_SAMPLES_PER_TRACK:
                    continue
                if track.age % SAMPLE_EVERY_N_FRAMES != 0:
                    continue

                crop = clip_with_buffer(frame, bbox, CROP_BUFFER_PX)
                if crop.size == 0:
                    continue
                crop = _erase_stem_regions(crop, bbox, stem_boxes)

                track.samples.append(
                    ClassificationSample(
                        crop=crop,
                        frame_id=frame_id,
                        detection_conf=conf,
                        center_offset=compute_center_offset(bbox, frame_width),
                    )
                )

            # 5. CLASSIFY — batch classify all unclassified crops
            pending: list[tuple[int, ClassificationSample]] = [
                (tid, s)
                for tid, track in active_tracks.items()
                for s in track.samples
                if s.classifier_conf is None and not track.decided
            ]

            if pending:
                crops_pre = _preprocess_crops(
                    [s.crop for _, s in pending], classifier.input_size
                )
                if classifier.fixed_batch:
                    logits = np.concatenate(
                        [
                            classifier.session.run(
                                None, {classifier.input_name: crops_pre[i : i + 1]}
                            )[0]
                            for i in range(len(crops_pre))
                        ],
                        axis=0,
                    )
                else:
                    logits = classifier.session.run(
                        None, {classifier.input_name: crops_pre}
                    )[0]
                probs = _softmax(logits)
                top1_labels = probs.argmax(axis=1)
                top1_confs = probs.max(axis=1)
                for i, (_, sample) in enumerate(pending):
                    sample.label = int(top1_labels[i])
                    sample.classifier_conf = float(top1_confs[i])

            # 6. EARLY DECISION — decide as soon as MIN_SAMPLES_FOR_DECISION reached
            for tid, track in list(active_tracks.items()):
                if track.decided:
                    continue
                classified = [s for s in track.samples if s.classifier_conf is not None]
                if len(classified) >= MIN_SAMPLES_FOR_DECISION:
                    decision, score = aggregate_decision(track, classified)
                    track.decision = decision
                    track.decided = True
                    completed_queue.append((track, score, frame_id))
                    logger.info(
                        f"Early decision — track {tid}: {decision.name}"
                        f" (score={score:.3f}, samples={len(classified)})"
                    )

            frame_id += 1

            if show:
                _show_detection_frame(frame, xyxy, confs, cls_ids)
                _show_otsu_frame(frame, cherry_track_info)
                _show_frame(frame, active_tracks)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield from _drain(completed_queue)

        # ── End of stream: decide any remaining undecided tracks ──────────
        for tid, track in active_tracks.items():
            if track.decided:
                continue
            classified = [s for s in track.samples if s.classifier_conf is not None]
            if classified:
                decision, score = aggregate_decision(track, classified)
            else:
                decision, score = Decision.IMPERFECT, 1.0  # fail-safe
            track.decision = decision
            track.decided = True
            completed_queue.append((track, score, frame_id))

        yield from _drain(completed_queue)

    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _assert_ultralytics_onnx(path: Path) -> None:
    """Raise ValueError if *path* was not exported by Ultralytics.

    Ultralytics always writes ``author: "Ultralytics"`` into the ONNX
    ``metadata_props`` at export time, so this check is reliable.
    """
    proto = onnx.load(str(path))
    metadata = {p.key: p.value for p in proto.metadata_props}
    if metadata.get("author") != "Ultralytics":
        raise ValueError(
            f"ONNX model at '{path}' does not appear to be an Ultralytics"
            " export (missing or wrong 'author' metadata)."
        )


def _load_model(path: str | Path) -> YOLO:
    """Load and validate a YOLO detector from an .onnx file.

    Verifies that the file was produced by Ultralytics before loading.

    Args:
        path: Path to the detector weights file (.onnx).

    Returns:
        Loaded and validated YOLO model.

    Raises:
        ValueError: If the file is not an ONNX file or not an Ultralytics export.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if path.suffix != ".onnx":
        raise ValueError(
            f"Detector must be an ONNX file, got '{path.suffix}'."
        )
    if not path.exists():
        raise FileNotFoundError(f"Detector model not found: {path}")

    _assert_ultralytics_onnx(path)
    return YOLO(str(path), task="detect")


# ---------------------------------------------------------------------------
# ONNX classifier loader
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for a 2-D logits array."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _preprocess_crops(
    crops: list[np.ndarray], input_size: tuple[int, int]
) -> np.ndarray:
    """Resize, normalise, and batch BGR crops for ONNX classifier inference.

    Args:
        crops: List of BGR numpy arrays (variable size).
        input_size: (H, W) expected by the model.

    Returns:
        Float32 array of shape (N, 3, H, W) with ImageNet normalisation.
    """
    h, w = input_size
    batch = []
    for crop in crops:
        img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (w, h))
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        batch.append(img.transpose(2, 0, 1))  # HWC → CHW
    return np.stack(batch)


def _load_onnx_classifier(path: str | Path) -> OnnxClassifier:
    """Load a timm model exported to ONNX for classification inference.

    Args:
        path: Path to the .onnx classifier file.

    Returns:
        OnnxClassifier with an onnxruntime session and input metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not an ONNX file or input shape is dynamic.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier model not found: {path}")
    if path.suffix != ".onnx":
        raise ValueError(
            f"Classifier must be an ONNX file, got '{path.suffix}'."
        )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if onnxruntime.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    session = onnxruntime.InferenceSession(str(path), providers=providers)

    input_meta = session.get_inputs()[0]
    input_name: str = input_meta.name
    shape = input_meta.shape  # expected: [batch, 3, H, W]
    if not (len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int)):
        raise ValueError(
            f"Cannot infer input size from '{path}': "
            f"expected static spatial dims, got shape {shape}."
        )

    fixed_batch = isinstance(shape[0], int) and shape[0] == 1

    return OnnxClassifier(
        session=session,
        input_name=input_name,
        input_size=(int(shape[2]), int(shape[3])),
        fixed_batch=fixed_batch,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _erase_stem_regions(
    crop: np.ndarray,
    cherry_bbox: np.ndarray,
    stem_boxes: np.ndarray,
) -> np.ndarray:
    """Black out stem bounding box regions from a cherry crop.

    Translates each stem box into crop-local coordinates (relative to the
    cherry bbox origin) and zeroes out those pixels so the classifier never
    sees stem content.

    Args:
        crop: BGR cherry crop produced by clip_with_buffer (H x W x 3).
        cherry_bbox: [x1, y1, x2, y2] of the cherry in frame coordinates.
        stem_boxes: Array of shape (N, 4) with stem bounding boxes in frame
                    coordinates.

    Returns:
        A copy of *crop* with stem regions zeroed out.
    """
    if len(stem_boxes) == 0:
        return crop

    result = crop.copy()
    cx1, cy1 = (
        int(cherry_bbox[0]) - CROP_BUFFER_PX,
        int(cherry_bbox[1]) - CROP_BUFFER_PX,
    )
    h, w = result.shape[:2]

    for stem in stem_boxes:
        sx1 = max(0, int(stem[0]) - cx1)
        sy1 = max(0, int(stem[1]) - cy1)
        sx2 = min(w, int(stem[2]) - cx1)
        sy2 = min(h, int(stem[3]) - cy1)
        if sx2 > sx1 and sy2 > sy1:
            result[sy1:sy2, sx1:sx2] = 0

    return result


def _increment_lost(
    active_tracks: dict[int, Track],
    current_ids: set[int],
    completed_queue: deque[tuple[Track, float, int]],
    frame_id: int,
) -> None:
    """Increment frames_since_seen for non-current tracks; evict stale ones."""
    stale_tids = []
    for tid, track in active_tracks.items():
        if tid in current_ids:
            continue
        track.state = TrackState.LOST
        track.frames_since_seen += 1
        if track.frames_since_seen > TRACK_BUFFER:
            stale_tids.append(tid)

    for tid in stale_tids:
        track = active_tracks.pop(tid)
        if track.decided:
            continue
        classified = [s for s in track.samples if s.classifier_conf is not None]
        if classified:
            decision, score = aggregate_decision(track, classified)
        else:
            decision, score = Decision.IMPERFECT, 1.0  # fail-safe: no samples
        track.decision = decision
        track.decided = True
        completed_queue.append((track, score, frame_id))
        logger.info(
            f"Track {tid} lost — final decision: {decision.name}"
            f" (score={score:.3f}, samples={len(classified)})"
        )


def _drain(
    queue: deque[tuple[Track, float, int]],
) -> Generator[tuple[Track, float, int], None, None]:
    """Yield and remove all items from *queue*."""
    while queue:
        yield queue.popleft()


# ---------------------------------------------------------------------------
# Optional live display
# ---------------------------------------------------------------------------

_DECISION_COLORS: dict[Decision, tuple[int, int, int]] = {
    Decision.PENDING: (200, 200, 200),
    Decision.PERFECT: (0, 255, 0),
    Decision.IMPERFECT: (0, 0, 255),
    Decision.NO_STEM: (0, 165, 255),
}


def _show_frame(frame: np.ndarray, active_tracks: dict[int, Track]) -> None:
    """Draw bounding boxes and track decisions onto *frame* and display it."""
    vis = frame.copy()
    for track in active_tracks.values():
        if not track.bbox_history:
            continue
        bbox = track.bbox_history[-1]
        color = _DECISION_COLORS[track.decision]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"T{track.track_id} {track.decision.name}"
        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imshow("Cherry Pipeline", vis)


_DET_COLORS: dict[int, tuple[int, int, int]] = {
    CHERRY_CLASS: (0, 255, 0),
    STEM_CLASS: (255, 180, 0),
}


def _show_detection_frame(
    frame: np.ndarray,
    xyxy: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
) -> None:
    """Draw raw YOLO detection boxes (no tracking) and display them."""
    vis = frame.copy()
    for bbox, conf, cls_id in zip(xyxy, confs, cls_ids):
        color = _DET_COLORS.get(int(cls_id), (200, 200, 200))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{'cherry' if cls_id == CHERRY_CLASS else 'stem'} {conf:.2f}"
        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imshow("Detection", vis)


_OTSU_TILE_SIZE: int = 160  # each cherry crop is resized to this square


def _otsu_crop(crop: np.ndarray) -> np.ndarray:
    """Apply R-G channel difference + Otsu threshold to a BGR crop.

    Returns a 3-channel BGR image: the crop masked to the Otsu region,
    resized to _OTSU_TILE_SIZE x _OTSU_TILE_SIZE.
    """
    r, g = crop[:, :, 2].astype(np.int16), crop[:, :, 1].astype(np.int16)
    diff = np.clip(r - g, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.bitwise_and(crop, crop, mask=mask)
    return cv2.resize(result, (_OTSU_TILE_SIZE, _OTSU_TILE_SIZE))


def _show_otsu_frame(
    frame: np.ndarray,
    cherry_track_info: dict[int, tuple[np.ndarray, float]],
) -> None:
    """Show Otsu R-G segmentation tiles for all currently visible cherries."""
    tiles: list[np.ndarray] = []
    for tid, (bbox, _) in cherry_track_info.items():
        crop = clip_with_buffer(frame, bbox, CROP_BUFFER_PX)
        if crop.size == 0:
            continue
        tile = _otsu_crop(crop)
        label = f"T{tid}"
        cv2.putText(tile, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(tile)

    if not tiles:
        canvas = np.zeros((_OTSU_TILE_SIZE, _OTSU_TILE_SIZE, 3), dtype=np.uint8)
    else:
        canvas = np.concatenate(tiles, axis=1)
    cv2.imshow("Otsu Classification", canvas)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cherry sorting pipeline — detect, track, classify, decide."
    )
    parser.add_argument(
        "--detector",
        type=Path,
        required=True,
        help="Path to the YOLO detection model (.onnx).",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        required=True,
        help="Path to the timm classifier checkpoint (.pth).",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (int) or file/RTSP path (default: 0).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated video feed (press Q to quit).",
    )
    args = parser.parse_args()

    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    for track, score, decision_frame in run_pipeline(
        source=source,
        detector_path=args.detector,
        classifier_path=args.classifier,
        show=args.show,
    ):
        logger.info(
            f"[GATE DECISION] track={track.track_id} decision={track.decision.name}"
            f" score={score:.3f} samples={len(track.samples)}"
        )


if __name__ == "__main__":
    main()
