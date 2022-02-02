from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass
class HitAtRanks:
    rank_to_hit: Dict[int, Union[int, float]]


@dataclass
class PrecisionAtRankMetrics:
    precision: float
    url: int
    section: int


@dataclass
class IRMetrics:
    rank_to_p_metrics: Dict[int, PrecisionAtRankMetrics]
    section_hit_scores_rank: Optional[int] = None
    article_hit_scores_rank: Optional[int] = None
