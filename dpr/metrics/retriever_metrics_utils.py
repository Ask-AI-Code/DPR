from typing import List, Set

import numpy as np

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.metrics.data_classes import IRMetrics, PrecisionAtRankMetrics, HitAtRanks


def get_hit_at_scores(hit_rank: int) -> HitAtRanks:
    """
    returns hit@2,5,30 given the list of scores at every index
    """
    if hit_rank is None:
        return HitAtRanks(rank_to_hit={1: 0, 2: 0, 5: 0, 30: 0})
    hit_at1 = 1 if hit_rank < 1 else 0
    hit_at2 = 1 if hit_rank < 2 else 0
    hit_at5 = 1 if hit_rank < 5 else 0
    hit_at30 = 1 if hit_rank < 30 else 0
    return HitAtRanks(rank_to_hit={1: hit_at1, 2: hit_at2, 5: hit_at5, 30: hit_at30})


def merge_chunk_and_url_scores(chunk_score: int, url_score: int) -> float:
    """
    merge between the chunk level and url level scores to get one metric
    we give the url strength of 0.5 that of the chunks
    """
    if chunk_score > 0:
        return chunk_score
    else:
        return url_score * 0.5


def merge_chunks_url_scores(chunks_scores: HitAtRanks, article_scores: HitAtRanks) -> HitAtRanks:
    p1 = merge_chunk_and_url_scores(chunks_scores.rank_to_hit[1], article_scores.rank_to_hit[1])
    p2 = merge_chunk_and_url_scores(chunks_scores.rank_to_hit[2], article_scores.rank_to_hit[2])
    p5 = merge_chunk_and_url_scores(chunks_scores.rank_to_hit[5], article_scores.rank_to_hit[5])
    p30 = merge_chunk_and_url_scores(chunks_scores.rank_to_hit[30], article_scores.rank_to_hit[30])

    return HitAtRanks(rank_to_hit={1: p1, 2: p2, 5: p5, 30: p30})


def get_url_no_anchor(chunk_url):
    return chunk_url.split("#")[0]


def calculate_ir_scores(
    gold_passages: List[BiEncoderPassage],
    predicted_passages: List[BiEncoderPassage]
) -> IRMetrics:
    gold_sections: Set[int] = {passage.chunk_index for passage in gold_passages}

    section_hit_scores = [
        1 if pred.chunk_index in gold_sections else 0
        for pred in predicted_passages
    ]

    gold_article_urls: Set[set] = {get_url_no_anchor(passage.url) for passage in gold_passages}

    article_hit_scores = [
        1 if get_url_no_anchor(pred.url) in gold_article_urls else 0
        for pred in predicted_passages
    ]

    section_hit_scores_rank, article_hit_scores_rank = None, None
    section_hit_scores_argmax = np.argmax(section_hit_scores)
    article_hit_scores_argmax = np.argmax(article_hit_scores)

    if section_hit_scores[section_hit_scores_argmax] == 1:
        section_hit_scores_rank = int(section_hit_scores_argmax)
    if article_hit_scores[article_hit_scores_argmax] == 1:
        article_hit_scores_rank = int(article_hit_scores_argmax)

    article_hits: HitAtRanks = get_hit_at_scores(article_hit_scores_rank)
    sections_hits: HitAtRanks = get_hit_at_scores(section_hit_scores_rank)
    p_at_ranks: HitAtRanks = merge_chunks_url_scores(chunks_scores=sections_hits, article_scores=article_hits)

    ir_metrics: IRMetrics = IRMetrics(
        rank_to_p_metrics={
            1: PrecisionAtRankMetrics(
                p_at_ranks.rank_to_hit[1], url=article_hits.rank_to_hit[1], section=sections_hits.rank_to_hit[1]
            ),
            2: PrecisionAtRankMetrics(
                p_at_ranks.rank_to_hit[2], url=article_hits.rank_to_hit[2], section=sections_hits.rank_to_hit[2]
            ),
            5: PrecisionAtRankMetrics(
                p_at_ranks.rank_to_hit[5], url=article_hits.rank_to_hit[5], section=sections_hits.rank_to_hit[5]
            ),
            30: PrecisionAtRankMetrics(
                p_at_ranks.rank_to_hit[30], url=article_hits.rank_to_hit[30], section=sections_hits.rank_to_hit[30]
            ),
        },
        section_hit_scores_rank=section_hit_scores_rank,
        article_hit_scores_rank=article_hit_scores_rank,
    )

    return ir_metrics
