"""
detector.py — YOLOv11 player detection with performance optimisations

Changes from original:
  1. Frame skipping — only process every FRAME_SKIP frames (3x speed boost)
  2. Resize before inference — 640px wide input to YOLO (2-4x speed boost)
  3. Use YOLOv11n (nano) model by default — fastest, still accurate
  4. Cache last detections for skipped frames (smooth output)
  5. Returns PlayerDetection objects compatible with new team_detector
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple
import logging

from .team_detector import PlayerDetection

logger = logging.getLogger(__name__)

# ── Performance settings ──────────────────────────────────────────────────────
FRAME_SKIP = 3          # Process 1 in every N frames  (3 = ~3x faster)
INFERENCE_WIDTH = 640   # Resize frame to this width before YOLO (keeps aspect ratio)
MIN_CONFIDENCE = 0.35   # Minimum detection confidence to keep
PERSON_CLASS_ID = 0     # COCO class ID for 'person'

# ── Model path ────────────────────────────────────────────────────────────────
# Use yolo11n.pt (nano) — fastest. Upgrade to yolo11s.pt for slightly better accuracy.
# Download happens automatically on first run via ultralytics.
DEFAULT_MODEL = "yolo11n.pt"


class PlayerDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()  # Fuse Conv+BN layers for faster CPU inference

        self._frame_count = 0
        self._cached_detections: List[PlayerDetection] = []
        self._cached_track_data = []

        logger.info("PlayerDetector ready")

    def detect(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None
    ) -> Tuple[List[PlayerDetection], np.ndarray]:
        """
        Detect players in a frame.

        Returns:
            (detections, annotated_frame)
        """
        self._frame_count += 1
        current_frame = frame_number if frame_number is not None else self._frame_count

        # ── Frame skipping ────────────────────────────────────────────────────
        if current_frame % FRAME_SKIP != 0 and self._cached_detections:
            # Return cached result for skipped frames
            return self._cached_detections, self._draw_cached(frame)

        # ── Resize for faster inference ───────────────────────────────────────
        orig_h, orig_w = frame.shape[:2]
        scale = INFERENCE_WIDTH / orig_w
        inference_h = int(orig_h * scale)
        inference_frame = cv2.resize(frame, (INFERENCE_WIDTH, inference_h))

        # ── YOLO inference ────────────────────────────────────────────────────
        results = self.model.track(
            inference_frame,
            persist=True,           # ByteTrack — maintains track IDs across frames
            classes=[PERSON_CLASS_ID],
            conf=MIN_CONFIDENCE,
            iou=0.5,
            verbose=False,
            tracker="bytetrack.yaml"
        )

        # ── Parse detections and scale back to original frame size ───────────
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                # Scale bbox back to original frame dimensions
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = x1 / scale
                y1 = y1 / scale
                x2 = x2 / scale
                y2 = y2 / scale

                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else i

                detection = PlayerDetection(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf
                )
                detections.append(detection)

        # ── Cache results ─────────────────────────────────────────────────────
        self._cached_detections = detections
        self._cached_bbox_data = [
            (int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3]))
            for d in detections
        ]

        # ── Annotate frame ────────────────────────────────────────────────────
        annotated = self._draw_detections(frame, detections)

        logger.debug(f"Frame {current_frame}: {len(detections)} players detected")
        return detections, annotated

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[PlayerDetection]
    ) -> np.ndarray:
        """Draw bounding boxes on frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{det.track_id} {det.confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1
            )
        return annotated

    def _draw_cached(self, frame: np.ndarray) -> np.ndarray:
        """Draw cached detections on a new frame (for skipped frames)."""
        return self._draw_detections(frame, self._cached_detections)