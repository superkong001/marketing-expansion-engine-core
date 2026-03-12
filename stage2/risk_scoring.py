"""
连续风险分接口

为每条规则赋予权重（strong/medium/weak 或 rule_score），提供无监督接口：
max_rule_score、top2_rule_score_sum、weighted_sum_of_hit_rules，
输入为「规则是否命中」dict + 规则权重，供外部在明细上算连续预测值并计算 AUC。
"""
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# 默认档位权重（按 rule_score 分位数分为 strong/medium/weak）
DEFAULT_TIER_WEIGHTS = {"strong": 3, "medium": 2, "weak": 1}


def assign_tier_weights(
    rule_scores: Dict[str, float],
    tier_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """
    按 rule_score 分位数将每条规则归为 strong/medium/weak，返回 rule_id -> tier。
    """
    if not rule_scores:
        return {}
    tier_weights = tier_weights or DEFAULT_TIER_WEIGHTS
    scores = sorted(rule_scores.values(), reverse=True)
    n = len(scores)
    if n == 0:
        return {}
    q66 = scores[int(0.34 * n)] if n >= 3 else scores[0]
    q33 = scores[int(0.67 * n)] if n >= 3 else scores[-1]
    out = {}
    for rid, s in rule_scores.items():
        if s >= q66:
            out[rid] = "strong"
        elif s >= q33:
            out[rid] = "medium"
        else:
            out[rid] = "weak"
    return out


def tier_to_weight(tier: str, tier_weights: Optional[Dict[str, float]] = None) -> float:
    """档位 -> 数值权重。"""
    w = (tier_weights or DEFAULT_TIER_WEIGHTS).get(tier, 1.0)
    return float(w)


def max_rule_score(
    hits: Dict[str, bool],
    rule_scores: Dict[str, float],
    use_tier_weights: bool = False,
    tier_weights: Optional[Dict[str, float]] = None,
    rule_tiers: Optional[Dict[str, str]] = None,
) -> float:
    """
    命中规则中取 rule_score（或档位权重）的最大值作为连续风险分。
    hits: rule_id -> 是否命中；rule_scores: rule_id -> rule_score。
    """
    if use_tier_weights and rule_tiers:
        weights = {rid: tier_to_weight(rule_tiers[rid], tier_weights) for rid in rule_tiers}
    else:
        weights = rule_scores
    hit_scores = [weights.get(rid, 0.0) for rid, h in hits.items() if h]
    return max(hit_scores) if hit_scores else 0.0


def top2_rule_score_sum(
    hits: Dict[str, bool],
    rule_scores: Dict[str, float],
    use_tier_weights: bool = False,
    tier_weights: Optional[Dict[str, float]] = None,
    rule_tiers: Optional[Dict[str, str]] = None,
) -> float:
    """命中规则中取 score/权重 的 top2 之和。"""
    if use_tier_weights and rule_tiers:
        weights = {rid: tier_to_weight(rule_tiers[rid], tier_weights) for rid in rule_tiers}
    else:
        weights = rule_scores
    hit_scores = sorted([weights.get(rid, 0.0) for rid, h in hits.items() if h], reverse=True)
    return sum(hit_scores[:2])


def weighted_sum_of_hit_rules(
    hits: Dict[str, bool],
    rule_scores: Dict[str, float],
    use_tier_weights: bool = False,
    tier_weights: Optional[Dict[str, float]] = None,
    rule_tiers: Optional[Dict[str, str]] = None,
) -> float:
    """命中规则的 score/权重 之和。"""
    if use_tier_weights and rule_tiers:
        weights = {rid: tier_to_weight(rule_tiers[rid], tier_weights) for rid in rule_tiers}
    else:
        weights = rule_scores
    return sum(weights.get(rid, 0.0) for rid, h in hits.items() if h)


def build_risk_score_weights_from_portfolio(
    recommended_rules: List[Any],
    use_rule_score_directly: bool = True,
    tier_weights: Optional[Dict[str, float]] = None,
) -> tuple:
    """
    从 recommended 规则列表构建 rule_id -> score 与（可选）rule_id -> tier。
    recommended_rules: 元素需有 rule_id 与 rule_score（或 score）。
    Returns:
        (rule_scores: Dict[str, float], rule_tiers: Optional[Dict[str, str]])
    """
    rule_scores = {}
    for r in recommended_rules:
        rid = getattr(r, 'rule_id', None) or getattr(r, 'rule_id', id(r))
        if rid is None:
            continue
        rid = str(rid)
        s = getattr(r, 'rule_score', None) or getattr(r, 'score', 0.0)
        rule_scores[rid] = float(s) if s is not None else 0.0
    if use_rule_score_directly:
        return rule_scores, None
    rule_tiers = assign_tier_weights(rule_scores, tier_weights)
    return rule_scores, rule_tiers
