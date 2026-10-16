"""Synthetic broadcast scenes with exact ground truth, for geometry validation.

Nothing here is evidence about real footage; it only verifies that the math is correct
and characterizes estimator behavior under controlled noise.
"""

from nbatrack.synthetic.scene import (
    BroadcastScenario,
    SyntheticFrame,
    default_rig,
    generate_sequence,
    observe_landmarks,
    sample_players,
)

__all__ = [
    "BroadcastScenario", "SyntheticFrame", "default_rig", "generate_sequence",
    "observe_landmarks", "sample_players",
]
