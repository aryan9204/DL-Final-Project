from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TrackRow:
    frame: int
    pedestrian: int
    x: float
    y: float
    prediction_number: Optional[int] = None
    scene_id: Optional[int] = None


@dataclass(frozen=True)
class SceneRow:
    scene: int
    pedestrian: int
    start: int
    end: int
    fps: float
    tag: List = field(default_factory=list)
