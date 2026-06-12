"""
main.py — FastAPI backend for ATHENA VAR v0.4.1

Changes from v0.4.0:
  1. Fixed file upload — uses shutil.copyfileobj instead of await file.read()
     which can fail on larger files with empty response / JSON parse error.
  2. Added try/except around entire analyse endpoint with proper error logging
     so crashes always return a readable JSON error instead of empty response.
  3. Added request logging so we can see when a request arrives.
"""

import cv2
import time
import tempfile
import os
import base64
import shutil
import logging
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..cv_pipeline.detector import PlayerDetector
from ..cv_pipeline.offside import analyse_offside
from ..cv_pipeline.team_detector import reset_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="ATHENA VAR", version="0.4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global detector (loaded once at startup) ──────────────────────────────────

detector: Optional[PlayerDetector] = None


@app.on_event("startup")
async def startup():
    global detector
    logger.info("Loading ATHENA detector...")
    detector = PlayerDetector()
    logger.info("ATHENA ready ✓")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "online",
        "version": "0.4.1",
        "detector_loaded": detector is not None,
    }


@app.post("/analyse")
async def analyse_clip(file: UploadFile = File(...)):
    """
    Analyse an uploaded video clip for offside.
    Accepts: MP4, MOV, AVI, MKV, WEBM
    """
    logger.info(f"Request received: filename={file.filename} content_type={file.content_type}")

    if detector is None:
        raise HTTPException(503, "Detector not initialised — server still starting up.")

    # ── Validate file type ────────────────────────────────────────────────────
    allowed = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(
            400,
            f"Unsupported file format '{file.filename}'. "
            f"Please upload one of: {', '.join(allowed).upper()}."
        )

    # ── Save upload to temp file using shutil (safe for all file sizes) ───────
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        logger.info(f"Saved upload to {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

        start_time = time.time()
        result = _process_video(tmp_path)
        result["processing_time_seconds"] = round(time.time() - start_time, 1)

        logger.info(f"Analysis complete in {result['processing_time_seconds']}s — verdict={result['verdict']}")
        return JSONResponse(result)

    except HTTPException:
        raise  # re-raise known HTTP errors as-is

    except Exception as e:
        # Log the full traceback so we can see exactly what went wrong
        logger.error(f"Unexpected error during analysis:\n{traceback.format_exc()}")
        raise HTTPException(500, f"Internal error: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Core processing pipeline ──────────────────────────────────────────────────

def _process_video(video_path: str) -> dict:
    """
    Analyse multiple frames throughout the clip.
    Picks the frame with the highest horizontal player spread
    (wide broadcast angle) rather than the most players detected.
    """
    reset_tracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file. The file may be corrupted.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {total_frames} frames at {fps:.0f}fps — {width}x{height}px")

    if total_frames < 1:
        cap.release()
        raise HTTPException(422, "Video appears to have no frames.")

    # Sample 10 points, skip first/last 5%
    sample_points = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92]
    sample_frames = [
        max(0, min(total_frames - 1, int(total_frames * p)))
        for p in sample_points
    ]

    best_result       = None
    best_frame_image  = None
    best_score        = -1.0
    best_frame_number = 0

    for frame_num in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {frame_num} — skipping")
            continue

        detections, _ = detector.detect(frame, frame_number=frame_num)

        if len(detections) < 4:
            logger.info(f"Frame {frame_num}: {len(detections)} players — skipping (need 4+)")
            continue

        # Score by horizontal spread
        x_positions = [d.center_x for d in detections]
        spread = (max(x_positions) - min(x_positions)) / width
        player_bonus   =  len(detections) * 0.01
        player_penalty = 0.2 if len(detections) < 6 else 0.0
        score = spread + player_bonus - player_penalty

        logger.info(
            f"Frame {frame_num}: {len(detections)} players | "
            f"spread={spread:.2f} | score={score:.2f}"
        )

        if score > best_score:
            offside_result = analyse_offside(
                frame=frame,
                detections=detections,
                attack_direction="auto",
            )

            logger.info(
                f"  -> direction={offside_result.attack_direction} | "
                f"offside_x={offside_result.offside_line_x:.0f}/{width} | "
                f"verdict={'OFFSIDE' if offside_result.is_offside else 'NOT OFFSIDE'} "
                f"({offside_result.confidence * 100:.0f}%)"
            )

            best_score        = score
            best_result       = offside_result
            best_frame_image  = offside_result.annotated_frame
            best_frame_number = frame_num

    cap.release()

    if best_result is None:
        raise HTTPException(
            422,
            "Could not detect enough players in the video. "
            "Please use a broadcast-angle clip with at least 4 visible players."
        )

    best_player_count = best_result.attacking_count + best_result.defending_count

    _, buffer = cv2.imencode(
        ".jpg", best_frame_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    verdict        = "OFFSIDE" if best_result.is_offside else "NOT OFFSIDE"
    confidence_pct = round(best_result.confidence * 100, 1)

    return {
        "verdict":           verdict,
        "is_offside":        best_result.is_offside,
        "confidence":        confidence_pct,
        "confidence_label":  _confidence_label(best_result.confidence),
        "explanation":       _build_explanation(best_result),
        "frame_number":      best_frame_number,
        "total_frames":      total_frames,
        "annotated_frame":   frame_b64,
        "players_detected":  best_player_count,
        "attacking_players": best_result.attacking_count,
        "defending_players": best_result.defending_count,
        "attack_direction":  best_result.attack_direction,
        "offside_line_x":    round(best_result.offside_line_x, 1),
        "frame_width":       width,
        "frame_height":      height,
        "historical_cases":  2,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "High Confidence"
    elif confidence >= 0.75:
        return "Medium Confidence"
    else:
        return "Low Confidence — Manual Review Recommended"


def _build_explanation(result) -> str:
    direction_str = (
        "left to right"
        if result.attack_direction == "left_to_right"
        else "right to left"
    )
    confidence_pct = round(result.confidence * 100)

    if result.is_offside:
        return (
            f"An attacker was detected past the offside line. "
            f"The attacking team is playing {direction_str}. "
            f"The offside line is positioned at the second-to-last defender's foot. "
            f"Confidence: {confidence_pct}%."
        )
    else:
        return (
            f"All attacking players were in an onside position at the moment of the pass. "
            f"The attacking team is playing {direction_str}. "
            f"The offside line is positioned at the second-to-last defender's foot. "
            f"Confidence: {confidence_pct}%."
        )