"""Class-incremental stage definitions.

Plan §4 (Incremental expansion roadmap). For the base stage only stage 0
is used. Stages 1-4 are placeholders documented here so the train loop
knows the planned schedule.
"""
from typing import Tuple, List


def stage_classes(stage_id: int) -> Tuple[int, ...]:
    """Return the cumulative class set for a stage.

    Stage 0: {0, 1}    (base)
    Stage 1: {0,1,2}
    Stage 2: {0,1,2,3}
    Stage 3: {0,1,2,3,4}
    Stage 4: {0..9}    (jump to full MNIST per plan §4)
    """
    schedule: List[Tuple[int, ...]] = [
        (0, 1),
        (0, 1, 2),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4),
        tuple(range(10)),
    ]
    if not (0 <= stage_id < len(schedule)):
        raise ValueError(f"unknown stage_id {stage_id}; valid range 0..{len(schedule)-1}")
    return schedule[stage_id]
