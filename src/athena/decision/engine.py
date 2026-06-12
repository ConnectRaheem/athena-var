from dataclasses import dataclass
from typing import Optional

@dataclass
class AthenaDecision:
    """Represents ATHENA's decision output."""
    situation: str          # offside, foul, corner, throw_in, goal_kick
    verdict: str            # OFFSIDE / NOT OFFSIDE / FOUL / NO FOUL etc
    confidence: float       # 0.0 to 1.0
    confidence_label: str   # Low / Medium / High
    explanation: str        # Human readable explanation
    similar_cases: int      # How many similar historical cases found
    result_image: Optional[str] = None  # base64 image


class AthenaDecisionEngine:
    """
    ATHENA Decision Engine
    Takes CV pipeline output and produces
    structured decision with confidence scores.
    """

    # Historical decision database
    # In Phase 2 this connects to real SoccerNet data
    HISTORICAL_DECISIONS = {
        "offside": [
            {"verdict": "OFFSIDE", "confidence": 0.94, "description": "Clear offside — attacker past last defender"},
            {"verdict": "OFFSIDE", "confidence": 0.87, "description": "Shoulder past defensive line at pass moment"},
            {"verdict": "NOT OFFSIDE", "confidence": 0.91, "description": "Attacker level with last defender — onside"},
            {"verdict": "OFFSIDE", "confidence": 0.78, "description": "Marginal offside — arm past defensive line"},
            {"verdict": "NOT OFFSIDE", "confidence": 0.85, "description": "Two defenders ahead of attacker"},
        ],
        "foul": [
            {"verdict": "FOUL", "confidence": 0.82, "description": "Reckless challenge from behind"},
            {"verdict": "NO FOUL", "confidence": 0.76, "description": "Player won the ball cleanly"},
            {"verdict": "FOUL", "confidence": 0.91, "description": "Excessive force in the challenge"},
            {"verdict": "NO FOUL", "confidence": 0.68, "description": "Minimal contact — play on"},
        ],
        "corner": [
            {"verdict": "CORNER", "confidence": 0.97, "description": "Ball crossed goal line off defender"},
            {"verdict": "GOAL KICK", "confidence": 0.94, "description": "Ball crossed goal line off attacker"},
        ],
        "throw_in": [
            {"verdict": "THROW IN", "confidence": 0.96, "description": "Ball crossed touchline"},
        ]
    }

    def compute_offside_decision(
        self,
        offside_results: list,
        pixels_over: float = 0,
        result_image: str = None
    ) -> AthenaDecision:
        """
        Compute final offside decision with confidence score.
        """
        any_offside = any(r["is_offside"] for r in offside_results)

        # Base confidence on how far over the line
        if any_offside:
            if pixels_over > 100:
                confidence = 0.94
            elif pixels_over > 50:
                confidence = 0.87
            elif pixels_over > 20:
                confidence = 0.76
            else:
                confidence = 0.65  # marginal
            verdict = "OFFSIDE"
        else:
            confidence = 0.89
            verdict = "NOT OFFSIDE"

        # Find similar historical cases
        similar = self._find_similar_cases("offside", verdict, confidence)

        # Confidence label
        if confidence >= 0.85:
            conf_label = "High"
        elif confidence >= 0.70:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        # Generate explanation
        if any_offside:
            max_over = max(
                (r["pixels_over"] for r in offside_results if r["is_offside"]),
                default=0
            )
            explanation = (
                f"Player was detected past the offside line by "
                f"approximately {max_over:.0f}px at the moment of the pass. "
                f"The system identified the 2nd-last defender position and "
                f"confirmed the attacker's body part crossed the line."
            )
        else:
            explanation = (
                "No attacker was detected past the offside line "
                "at the moment of the pass. All attacking players "
                "were in an onside position."
            )

        return AthenaDecision(
            situation="offside",
            verdict=verdict,
            confidence=confidence,
            confidence_label=conf_label,
            explanation=explanation,
            similar_cases=similar,
            result_image=result_image
        )

    def _find_similar_cases(
        self,
        situation: str,
        verdict: str,
        confidence: float
    ) -> int:
        """Find number of similar historical cases."""
        cases = self.HISTORICAL_DECISIONS.get(situation, [])
        similar = [
            c for c in cases
            if c["verdict"] == verdict
            and abs(c["confidence"] - confidence) < 0.15
        ]
        return len(similar)