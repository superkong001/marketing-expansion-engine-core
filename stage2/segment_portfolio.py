"""
多客群组合与去重模块

从候选客群中选择互相区分度高的客群组合
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from .stage2_config import Stage2Config
    from .rule_combination import SegmentRule
    from .segment_diversity import (
        calculate_rule_similarity,
        calculate_rule_structure_distance,
        filter_similar_rules
    )
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule
    from segment_diversity import (
        calculate_rule_similarity,
        calculate_rule_structure_distance,
        filter_similar_rules
    )

logger = logging.getLogger(__name__)


@dataclass
class SegmentPortfolio:
    """客群组合方案数据结构"""
    segments: List[SegmentRule]  # 客群1～客群N
    portfolio_metrics: Dict  # 组合整体指标


def calculate_rule_divergence_strength(rule: SegmentRule, divergence_scores: dict) -> float:
    """
    计算规则差异强度
    
    公式：D(rule) = ∑ᵢ S(featureᵢ)
    
    Args:
        rule: SegmentRule对象
        divergence_scores: 差异评分字典（column_id -> score）
    
    Returns:
        规则差异强度
    """
    total_strength = 0.0
    for fr in rule.feature_rules:
        col_id = fr['column_id']
        if col_id in divergence_scores:
            total_strength += divergence_scores[col_id]
        else:
            # 如果找不到，使用rule中的divergence_score
            total_strength += rule.divergence_score / len(rule.feature_rules) if len(rule.feature_rules) > 0 else 0.0
    
    return total_strength


def estimate_overlap_rate(rule1: SegmentRule, rule2: SegmentRule, config: Stage2Config) -> float:
    """
    估算两个规则的相似度（用于差异最大化）
    
    基于规则结构距离计算相似度：similarity = 1 - structure_distance
    
    Args:
        rule1: 规则1
        rule2: 规则2
        config: 阶段2配置对象
    
    Returns:
        相似度（0-1之间），越高表示越相似
    """
    # 使用规则结构距离计算相似度
    structure_distance = calculate_rule_structure_distance(rule1, rule2)
    similarity = 1.0 - structure_distance
    return similarity


def build_segment_portfolio(
    candidate_rules: List[SegmentRule],
    config: Stage2Config,
    divergence_scores: Optional[dict] = None
) -> SegmentPortfolio:
    """
    构建客群组合方案（差异最大化原则）
    
    使用规则结构距离（字段集合 + 区间重叠）进行多客群差异最大化选择
    
    算法：
    1. 按得分从高到低排序候选规则
    2. 使用贪心算法，选择与已选规则结构距离足够大的规则
    3. 基于规则结构距离计算相似度，过滤相似规则
    
    Args:
        candidate_rules: 候选规则列表（应已按得分排序）
        config: 阶段2配置对象
        divergence_scores: 差异评分字典（可选，用于计算规则差异强度）
    
    Returns:
        客群组合方案（SegmentPortfolio对象）
    """
    if not candidate_rules:
        logger.warning("候选规则列表为空，无法构建客群组合")
        return SegmentPortfolio(segments=[], portfolio_metrics={})
    
    # require_exact_k 时仅保留规则数严格为 k 的候选
    min_rules = getattr(config, 'min_rules_per_segment', 2)
    if getattr(config, 'require_exact_k', False):
        candidate_rules = [r for r in candidate_rules if len(r.feature_rules) == min_rules]
        if not candidate_rules:
            logger.warning(
                "无规则数严格为 k=%d 的候选，可能因阈值过宽或 combo 约束过严导致；portfolio 为空",
                min_rules
            )
            return SegmentPortfolio(segments=[], portfolio_metrics={})
    
    # 构建差异评分字典（如果未提供）
    if divergence_scores is None:
        divergence_scores = {}
        for rule in candidate_rules:
            for fr in rule.feature_rules:
                col_id = fr['column_id']
                if col_id not in divergence_scores:
                    # 使用rule的divergence_score作为近似
                    divergence_scores[col_id] = rule.divergence_score / len(rule.feature_rules) if len(rule.feature_rules) > 0 else 0.0
    
    # 使用基于规则结构距离的去重函数
    selected_segments = filter_similar_rules(
        candidate_rules,
        similarity_threshold=config.similarity_threshold,
        max_rules=config.max_segments
    )
    
    # 最终输出硬约束：仅保留规则数 >= min_rules_per_segment 的客群（require_exact_k 时已在入口过滤为 == min_rules）
    n_before = len(selected_segments)
    selected_segments = [r for r in selected_segments if len(r.feature_rules) >= min_rules]
    if n_before > len(selected_segments):
        logger.info(f"过滤掉 {n_before - len(selected_segments)} 个规则数不足的客群（min_rules_per_segment={min_rules}）")
    
    # 记录选择详情
    for i, rule in enumerate(selected_segments):
        div_strength = calculate_rule_divergence_strength(rule, divergence_scores)
        logger.debug(f"选择规则 {rule.rule_id}（排名 {i+1}，差异强度: {div_strength:.3f}）")
    
    # 计算组合整体指标
    portfolio_metrics = {
        'total_segments': len(selected_segments),
        'avg_divergence_score': sum(s.divergence_score for s in selected_segments) / len(selected_segments) if selected_segments else 0.0,
        'avg_stability_score': sum(s.stability_score for s in selected_segments) / len(selected_segments) if selected_segments else 0.0,
        'avg_diversity_score': sum(s.diversity_score for s in selected_segments) / len(selected_segments) if selected_segments else 0.0,
        'avg_score': sum(s.score for s in selected_segments) / len(selected_segments) if selected_segments else 0.0,
        'min_structure_distance': min(
            [calculate_rule_structure_distance(selected_segments[i], selected_segments[j])
             for i in range(len(selected_segments))
             for j in range(i+1, len(selected_segments))]
        ) if len(selected_segments) > 1 else 1.0
    }
    
    logger.info(f"构建客群组合完成，选择 {len(selected_segments)} 个客群（基于规则结构距离的差异最大化原则）")
    
    return SegmentPortfolio(
        segments=selected_segments,
        portfolio_metrics=portfolio_metrics
    )

