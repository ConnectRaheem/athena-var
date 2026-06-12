"""
team_detector.py — Pitch-position-based team detection
Replaces unreliable KMeans colour clustering.

Core idea:
  - Split players into LEFT half vs RIGHT half by median X position.
  - Infer attack direction by comparing cluster centroids across multiple
    frames (temporal smoothing) — NOT single outlier positions.
  - Offside line = second-to-last defender's foot X position.

Key fix over previous version:
  - _infer_attack_direction now uses CLUSTER CENTROIDS not extreme outliers.
  - Added TemporalDirectionTracker to smooth direction across frames
    (prevents flipping when a GK wanders or ball boy gets detected).
  - Goalkeeper excluded from offside line calculation (they are the
    LAST defender — offside line is the SECOND-to-last).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import deque


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlayerDetection:
    """Single player detection from YOLO."""
    track_id: int
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    confidence: float
    center_x: float = field(init=False)
    center_y: float = field(init=False)
    foot_x: float = field(init=False)         # bottom-centre of bbox
    foot_y: float = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2
        self.foot_x   = (x1 + x2) / 2
        self.foot_y   = y2   # bottom of bounding box ≈ foot position


@dataclass
class TeamAssignment:
    attacking_team: List[PlayerDetection]
    defending_team: List[PlayerDetection]
    last_defender: Optional[PlayerDetection]
    second_last_defender: Optional[PlayerDetection]
    offside_line_x: float
    attack_direction: str   # 'left_to_right' or 'right_to_left'


# ── Temporal direction smoother ───────────────────────────────────────────────

class TemporalDirectionTracker:
    """
    Smooths attack direction across frames to prevent rapid flipping.

    Keeps a rolling vote window. Direction only changes when the new
    direction wins a clear majority (>= FLIP_THRESHOLD) of recent frames.

    Usage:
        tracker = TemporalDirectionTracker()
        direction = tracker.update("left_to_right")   # call every frame
    """
    WINDOW = 12          # frames to consider
    FLIP_THRESHOLD = 0.7 # fraction of window needed to flip direction

    def __init__(self):
        self._votes: deque = deque(maxlen=self.WINDOW)
        self._current: str = "left_to_right"

    def update(self, raw_direction: str) -> str:
        self._votes.append(raw_direction)
        if len(self._votes) < 3:
            self._current = raw_direction
            return self._current

        ltr_fraction = self._votes.count("left_to_right") / len(self._votes)
        if ltr_fraction >= self.FLIP_THRESHOLD:
            self._current = "left_to_right"
        elif ltr_fraction <= (1 - self.FLIP_THRESHOLD):
            self._current = "right_to_left"
        # else: keep current direction (not enough votes to flip)
        return self._current

    @property
    def current(self) -> str:
        return self._current


# Module-level tracker — shared across all frames of a single video
_direction_tracker = TemporalDirectionTracker()


def reset_tracker():
    """Call this at the start of each new video to clear direction history."""
    global _direction_tracker
    _direction_tracker = TemporalDirectionTracker()


# ── Main team detection ───────────────────────────────────────────────────────

def detect_teams_by_position(
    players: List[PlayerDetection],
    frame_width: int,
    frame_height: int,
    attack_direction: str = "auto"
) -> TeamAssignment:
    """
    Split players into attacking and defending teams using pitch position.

    Args:
        players:          Detected players this frame.
        frame_width:      Frame width in pixels.
        frame_height:     Frame height in pixels.
        attack_direction: 'left_to_right', 'right_to_left', or 'auto'.

    Returns:
        TeamAssignment with offside line position.
    """
    # ── Fallback: not enough players ─────────────────────────────────────────
    if len(players) < 4:
        return TeamAssignment(
            attacking_team=[],
            defending_team=players,
            last_defender=None,
            second_last_defender=None,
            offside_line_x=frame_width / 2,
            attack_direction=_direction_tracker.current,
        )

    # ── Step 1: Remove outliers near frame edges (ball boys, staff) ──────────
    # Keep only players whose feet are in the central 90% of frame height
    # and whose centre X is in the central 94% of frame width.
    margin_y_top   = frame_height * 0.05
    margin_y_bot   = frame_height * 0.95
    margin_x_left  = frame_width  * 0.03
    margin_x_right = frame_width  * 0.97

    field_players = [
        p for p in players
        if (margin_y_top < p.center_y < margin_y_bot)
        and (margin_x_left < p.center_x < margin_x_right)
    ]

    # Fall back if filter was too aggressive
    if len(field_players) < 4:
        field_players = players

    # ── Step 2: Split into LEFT half / RIGHT half by median X ────────────────
    x_positions = [p.center_x for p in field_players]
    median_x = np.median(x_positions)

    left_players  = [p for p in field_players if p.center_x <  median_x]
    right_players = [p for p in field_players if p.center_x >= median_x]

    # Need at least 2 players in each half for a reliable split
    if len(left_players) < 2 or len(right_players) < 2:
        # Fallback: just use the previous known direction
        left_players  = [p for p in field_players if p.center_x <  frame_width / 2]
        right_players = [p for p in field_players if p.center_x >= frame_width / 2]

    # ── Step 3: Infer or use provided attack direction ────────────────────────
    if attack_direction == "auto":
        raw_direction = _infer_attack_direction_by_centroid(
            left_players, right_players, frame_width
        )
        attack_direction = _direction_tracker.update(raw_direction)
    else:
        # Manual override — still update tracker so it converges correctly
        _direction_tracker.update(attack_direction)

    # ── Step 4: Assign attacking vs defending ─────────────────────────────────
    if attack_direction == "left_to_right":
        # Attacking team pushed into right half
        attacking_team = right_players
        defending_team = left_players
    else:
        # Attacking team pushed into left half
        attacking_team = left_players
        defending_team = right_players

    # ── Step 5: Find last two defenders for offside line ─────────────────────
    # Sort defenders by how far they are from their own goal (most advanced first)
    if attack_direction == "left_to_right":
        # Defending goal is LEFT side; most advanced defender = highest X
        sorted_defenders = sorted(defending_team, key=lambda p: p.foot_x, reverse=True)
    else:
        # Defending goal is RIGHT side; most advanced defender = lowest X
        sorted_defenders = sorted(defending_team, key=lambda p: p.foot_x, reverse=False)

    # Identify goalkeeper — the defender closest to their own goal line
    if attack_direction == "left_to_right":
        gk = min(defending_team, key=lambda p: p.foot_x) if defending_team else None
    else:
        gk = max(defending_team, key=lambda p: p.foot_x) if defending_team else None

    # Exclude goalkeeper from the sorted list for offside line calculation
    # (offside line = second-to-last OUTFIELD defender, not the GK)
    outfield_defenders = [p for p in sorted_defenders if gk is None or p.track_id != gk.track_id]

    last_defender        = outfield_defenders[0] if len(outfield_defenders) > 0 else sorted_defenders[0] if sorted_defenders else None
    second_last_defender = outfield_defenders[1] if len(outfield_defenders) > 1 else None

    # ── Step 6: Calculate offside line X ─────────────────────────────────────
    # Offside line = second-to-last defender's most advanced foot
    if second_last_defender:
        offside_line_x = second_last_defender.foot_x
    elif last_defender:
        offside_line_x = last_defender.foot_x
    else:
        offside_line_x = frame_width / 2

    # Clamp: offside line must be inside the pitch (not on touchlines)
    offside_line_x = max(frame_width * 0.05, min(frame_width * 0.95, offside_line_x))

    return TeamAssignment(
        attacking_team=attacking_team,
        defending_team=defending_team,
        last_defender=last_defender,
        second_last_defender=second_last_defender,
        offside_line_x=offside_line_x,
        attack_direction=attack_direction,
    )


# ── Direction inference ───────────────────────────────────────────────────────

def _infer_attack_direction_by_centroid(
    left_players: List[PlayerDetection],
    right_players: List[PlayerDetection],
    frame_width: int,
) -> str:
    """
    Infer attack direction by comparing how far each cluster has pushed
    into the opponent's half — using centroid positions, NOT outliers.

    Key insight:
      - The ATTACKING team's centroid will be displaced toward the opponent's goal.
      - The DEFENDING team's centroid will be displaced toward their own goal.
      - So: compare left_centroid vs right_centroid relative to frame centre.
        If left_centroid > frame_centre → left team pushed right → left attacks right.
        If right_centroid < frame_centre → right team pushed left → right attacks left.

    We also weight by the most advanced player in each group (the striker),
    not just the centroid, to catch cases where only one player is in the box.
    """
    if not left_players or not right_players:
        return "left_to_right"

    frame_centre = frame_width / 2

    # Centroid of each half
    left_centroid  = np.mean([p.center_x for p in left_players])
    right_centroid = np.mean([p.center_x for p in right_players])

    # Most advanced player in each half
    left_most_advanced  = max(p.foot_x for p in left_players)   # highest X = pushed right
    right_most_advanced = min(p.foot_x for p in right_players)  # lowest X = pushed left

    # Score: how much has each group pushed INTO the opponent's half?
    # Positive score = that group is attacking left_to_right
    # Negative score = that group is attacking right_to_left

    # Left group score: how far right of centre is their centroid?
    left_push_score  = left_centroid - frame_centre   # positive if left group is in right half
    # Right group score: how far left of centre is their centroid?
    right_push_score = frame_centre - right_centroid  # positive if right group is in left half

    # Bonus: add weight from the most advanced attacker in each group
    left_attacker_bonus  = max(0, left_most_advanced  - frame_centre) * 0.3
    right_attacker_bonus = max(0, frame_centre - right_most_advanced) * 0.3

    left_total  = left_push_score  + left_attacker_bonus
    right_total = right_push_score + right_attacker_bonus

    # The group with the higher total score is the attacking team.
    # If left group is attacking → they're pushing right → left_to_right.
    # If right group is attacking → they're pushing left → right_to_left.
    if left_total >= right_total:
        return "left_to_right"
    else:
        return "right_to_left"


def _find_goalkeeper(
    players: List[PlayerDetection],
    goal_line_x: float
) -> Optional[PlayerDetection]:
    """Find the player closest to the given goal line X position."""
    if not players:
        return None
    return min(players, key=lambda p: abs(p.center_x - goal_line_x))


# ── Offside check ─────────────────────────────────────────────────────────────

def check_offside(
    attacking_players: List[PlayerDetection],
    offside_line_x: float,
    attack_direction: str,
    frame_width: int,
) -> Tuple[bool, Optional[PlayerDetection], float]:
    """
    Check if any attacker is in an offside position.

    Uses the player's LEADING EDGE (the body part furthest toward the goal)
    which matches the FIFA Laws of the Game definition.

    Returns:
        (is_offside, most_offside_player, confidence_0_to_1)
    """
    if not attacking_players:
        return False, None, 0.95

    offside_players = []

    for player in attacking_players:
        if attack_direction == "left_to_right":
            # Leading edge = right side of bbox (x2)
            leading_edge = player.bbox[2]
            if leading_edge > offside_line_x:
                margin_px = leading_edge - offside_line_x
                offside_players.append((player, margin_px))
        else:
            # Leading edge = left side of bbox (x1)
            leading_edge = player.bbox[0]
            if leading_edge < offside_line_x:
                margin_px = offside_line_x - leading_edge
                offside_players.append((player, margin_px))

    if not offside_players:
        # Confidence: scales with how safely onside the most advanced attacker is
        if attack_direction == "left_to_right":
            most_advanced_edge = max(p.bbox[2] for p in attacking_players)
            margin_px = offside_line_x - most_advanced_edge
        else:
            most_advanced_edge = min(p.bbox[0] for p in attacking_players)
            margin_px = most_advanced_edge - offside_line_x

        # Larger margin → more confident NOT OFFSIDE
        # Clamp between 0.65 and 0.99
        confidence = float(np.clip(0.70 + (margin_px / frame_width) * 1.5, 0.65, 0.99))
        return False, None, confidence

    # Return the most offside player (largest pixel margin past the line)
    offside_players.sort(key=lambda x: x[1], reverse=True)
    most_offside_player, margin_px = offside_players[0]

    # Larger margin → more confident it IS offside
    confidence = float(np.clip(0.70 + (margin_px / frame_width) * 2.0, 0.65, 0.99))

    return True, most_offside_player, confidence