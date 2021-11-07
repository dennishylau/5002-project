from dataclasses import dataclass
from typing import Optional


@dataclass
class Anomaly:
    idx: int
    confidence: Optional[float]
