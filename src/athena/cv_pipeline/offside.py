"""
offside.py — Offside line computation and rendering

Changes from original:
  1. Uses pitch-position team detection (replaces KMeans colour)
  2. Offside line drawn at second-to-last DEFENDER foot position (correct rule)
  3. Line drawn as dashed yellow (VAR broadcast style)
  4. Draws attacker markers to show who is being checked
  5. Returns structured OffsideResult for engine.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from .team_detector import (
    PlayerDetection,
    TeamAssignment,
    detect_teams_by_position,
    check_offside,
)

logger = logging.getLogger(__name__)


@dataclass
class OffsideResult:
    is_offside: bool
    confidence: float
    offside_line_x: float
    offside_player: Optional[PlayerDetection]
    attacking_count: int
    defending_count: int
    attack_direction: str
    annotated_frame: Optional[np.ndarray] = None


def analyse_offside(
    frame: np.ndarray,
    detections: List[PlayerDetection],
    attack_direction: str = "auto"
) -> OffsideResult:
    """
    Full offside analysis pipeline for a single frame.
    
    Args:
        frame: The video frame as numpy array
        detections: Player detections from detector.py
        attack_direction: 'left_to_right', 'right_to_left', or 'auto'
    
    Returns:
        OffsideResult with verdict, confidence, and annotated frame
    """
    h, w = frame.shape[:2]

    # ── Step 1: Assign teams using pitch position ─────────────────────────────
    assignment = detect_teams_by_position(
        players=detections,
        frame_width=w,
        frame_height=h,
        attack_direction=attack_direction
    )

    # ── Step 2: Check for offside ─────────────────────────────────────────────
    is_offside, offside_player, confidence = check_offside(
        attacking_players=assignment.attacking_team,
        offside_line_x=assignment.offside_line_x,
        attack_direction=assignment.attack_direction,
        frame_width=w
    )

    # ── Step 3: Annotate frame ────────────────────────────────────────────────
    annotated = _draw_offside_overlay(
        frame=frame,
        assignment=assignment,
        is_offside=is_offside,
        offside_player=offside_player
    )

    return OffsideResult(
        is_offside=is_offside,
        confidence=confidence,
        offside_line_x=assignment.offside_line_x,
        offside_player=offside_player,
        attacking_count=len(assignment.attacking_team),
        defending_count=len(assignment.defending_team),
        attack_direction=assignment.attack_direction,
        annotated_frame=annotated
    )


def _draw_offside_overlay(
    frame: np.ndarray,
    assignment: TeamAssignment,
    is_offside: bool,
    offside_player: Optional[PlayerDetection]
) -> np.ndarray:
    """
    Draw the VAR-style offside overlay on the frame.
    
    Draws:
      - Yellow dashed offside line at second-to-last defender
      - Red boxes on attacking players
      - Blue boxes on defending players  
      - Orange highlight on the offside player (if any)
      - Small dot at second-to-last defender's foot
    """
    img = frame.copy()
    h, w = img.shape[:2]
    line_x = int(assignment.offside_line_x)

    # ── Draw team bounding boxes ──────────────────────────────────────────────
    # Defending team — blue
    for player in assignment.defending_team:
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 80, 0), 2)   # Blue

    # Attacking team — red
    for player in assignment.attacking_team:
        x1, y1, x2, y2 = [int(v) for v in player.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 80, 255), 2)   # Red

    # Offside player — bright orange highlight
    if offside_player:
        x1, y1, x2, y2 = [int(v) for v in offside_player.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange
        cv2.putText(
            img, "OFFSIDE", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
        )

    # ── Draw second-to-last defender marker ──────────────────────────────────
    if assignment.second_last_defender:
        fx = int(assignment.second_last_defender.foot_x)
        fy = int(assignment.second_last_defender.foot_y)
        cv2.circle(img, (fx, fy), 6, (0, 255, 255), -1)  # Cyan dot at foot

    # ── Draw dashed offside line ──────────────────────────────────────────────
    line_color = (0, 255, 255) if not is_offside else (0, 100, 255)  # Yellow / Red-orange
    _draw_dashed_line(img, (line_x, 0), (line_x, h), line_color, thickness=2, dash_length=15)

    # ── Draw ATHENA watermark ─────────────────────────────────────────────────
    cv2.putText(
        img, f"ATHENA — Offside Line",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )

    return img


def _draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 15
):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    draw = True
    pos = 0
    while pos < length:
        end_pos = min(pos + dash_length, length)
        if draw:
            sx = int(x1 + dx * pos)
            sy = int(y1 + dy * pos)
            ex = int(x1 + dx * end_pos)
            ey = int(y1 + dy * end_pos)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        pos = end_pos
        draw = not draw