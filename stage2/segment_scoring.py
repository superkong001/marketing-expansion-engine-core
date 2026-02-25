"""
客群评分与筛选模块

对候选客群进行统一评分和过滤
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

try:
    from .stage2_config import Stage2Config
    from .rule_combination import SegmentRule
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule

logger = logging.getLogger(__name__)


def score_segment_rule(
    rule: SegmentRule,
    config: Stage2Config
) -> Dict[str, float]:
    """
    计算单个规则的综合得分（Coverage-free版本）
    
    计算公式：score = w₁ · divergence_score + w₂ · stability_score + w₃ · diversity_score
    
    权重可配置，默认值：
    - w_divergence = 0.5（差异评分权重）
    - w_stability = 0.3（稳定性评分权重）
    - w_diversity = 0.2（多样性评分权重）
    
    Args:
        rule: SegmentRule对象（已包含divergence_score, stability_score, diversity_score）
        config: 阶段2配置对象
    
    Returns:
        包含得分和详细指标的字典
    """
    # 获取各项评分（已在rule_combination中计算）
    divergence_score = rule.divergence_score
    stability_score = rule.stability_score
    diversity_score = rule.diversity_score
    
    # 使用配置中的权重（从config中读取）
    w_div = config.beam_w_divergence
    w_stab = config.beam_w_stability
    w_divs = config.beam_w_diversity
    
    # 归一化权重（确保权重和为1）
    total_weight = w_div + w_stab + w_divs
    if total_weight > 0:
        w_div = w_div / total_weight
        w_stab = w_stab / total_weight
        w_divs = w_divs / total_weight
    
    # 计算综合得分
    score = (
        w_div * divergence_score +
        w_stab * stability_score +
        w_divs * diversity_score
    )
    
    return {
        'score': score,
        'divergence_score': divergence_score,
        'stability_score': stability_score,
        'diversity_score': diversity_score,
        'w_divergence': w_div,
        'w_stability': w_stab,
        'w_diversity': w_divs
    }


def filter_candidate_segments(
    candidate_rules: List[SegmentRule],
    config: Stage2Config
) -> List[SegmentRule]:
    """
    筛选候选客群（Coverage-free版本）
    
    Args:
        candidate_rules: 候选规则列表
        config: 阶段2配置对象
    
    Returns:
        筛选后的候选客群列表
    """
    filtered_rules = []
    
    for rule in candidate_rules:
        # 过滤条件：基于差异评分和稳定性评分
        if rule.divergence_score < config.min_divergence_score:
            continue
        
        if rule.stability_score < config.min_stability_score:
            continue
        
        # 计算综合得分
        score_info = score_segment_rule(rule, config)
        rule.score = score_info['score']
        
        filtered_rules.append(rule)
    
    # 按得分排序，保留Top N
    filtered_rules.sort(key=lambda x: x.score, reverse=True)
    filtered_rules = filtered_rules[:config.top_n_candidates]
    
    logger.info(f"筛选完成，保留 {len(filtered_rules)} 个候选规则（从 {len(candidate_rules)} 个中筛选）")
    
    return filtered_rules

