"""
统一评分模块

score_final = w1*precision_est + w2*lift_est - w3*cov_penalty - w4*instability_penalty
              - w5*redundancy_penalty - w6*proxy_penalty + w7*diversity_bonus
高 base_cov、pair 缺失、proxy 仅扣分不删规则（宽进严排）。
"""
import logging
from typing import Any, Dict, List, Optional

try:
    from .stage2_config import Stage2Config
    from .rule_combination import SegmentRule
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule

logger = logging.getLogger(__name__)


def _cov_penalty(all_cov: Optional[float], threshold: float) -> float:
    """base_cov 超过阈值时施加惩罚（线性或阶梯）。"""
    if all_cov is None or threshold <= 0:
        return 0.0
    try:
        c = float(all_cov)
    except (TypeError, ValueError):
        return 0.0
    if c <= threshold:
        return 0.0
    return min(1.0, (c - threshold) / (1.0 - threshold))


def _instability_penalty(stability_score: Optional[float], floor: float) -> float:
    """稳定性低于 floor 时施加惩罚。"""
    if stability_score is None or floor <= 0:
        return 0.0
    try:
        s = float(stability_score)
    except (TypeError, ValueError):
        return 1.0
    if s >= floor:
        return 0.0
    return min(1.0, (floor - s) / floor)


def _rule_feature_key(rule: Any) -> tuple:
    """规则特征集合的规范 key（用于冗余/多样性）。"""
    ids = getattr(rule, 'rule_feature_ids', None)
    if ids is not None:
        return tuple(sorted(ids))
    frs = getattr(rule, 'feature_rules', None)
    if not frs:
        return tuple()
    try:
        return tuple(sorted(fr.get('column_id', '') for fr in frs if isinstance(fr, dict)))
    except Exception:
        return tuple()


def _redundancy_penalty(rule: Any, selected_list: Optional[List[Any]], similarity_threshold: float) -> float:
    """与已选规则结构相似（同特征集）则施加惩罚。"""
    if not selected_list or similarity_threshold <= 0:
        return 0.0
    key = _rule_feature_key(rule)
    if not key:
        return 0.0
    for other in selected_list:
        if _rule_feature_key(other) == key:
            return 1.0
    return 0.0


def _diversity_bonus(rule: Any, selected_list: Optional[List[Any]]) -> float:
    """相对已选规则带来新特征则加分。"""
    if not selected_list:
        return 0.0
    keys = set(_rule_feature_key(rule))
    if not keys:
        return 0.0
    existing = set()
    for other in selected_list:
        existing.update(_rule_feature_key(other))
    new_count = len(keys - existing)
    if new_count <= 0:
        return 0.0
    return min(1.0, new_count * 0.3)


def compute_score_final(
    rule: SegmentRule,
    config: Stage2Config,
    selected_list: Optional[List[SegmentRule]] = None,
) -> float:
    """
    统一最终得分：score_final = w1*precision + w2*lift - w3*cov_penalty - w4*instability
                               - w5*redundancy - w6*proxy + w7*diversity_bonus
    将 score_breakdown 写入 rule（若 rule 支持动态属性）。
    """
    w1 = getattr(config, 'w1_precision', 0.35)
    w2 = getattr(config, 'w2_lift', 0.25)
    w3 = getattr(config, 'w3_all_cov_penalty', 0.15)
    w4 = getattr(config, 'w4_instability_penalty', 0.10)
    w5 = getattr(config, 'w5_redundancy_penalty', 0.05)
    w6 = getattr(config, 'w6_proxy_penalty', 0.05)
    w7 = getattr(config, 'w7_diversity_bonus', 0.05)
    cov_thr = getattr(config, 'cov_penalty_threshold', 0.35)
    stab_floor = getattr(config, 'stability_floor', 0.2)
    red_thr = getattr(config, 'redundancy_similarity_threshold', 0.9)

    prec = getattr(rule, 'combo_precision_est', None) or getattr(rule, 'combo_precision_lb', 0.0)
    lift = getattr(rule, 'combo_lift_est', None) or 0.0
    all_cov = getattr(rule, 'combo_all_cov_est', None)
    stability = getattr(rule, 'stability_score', 0.0)
    try:
        prec_n = max(0.0, min(1.0, float(prec))) if prec is not None else 0.0
    except (TypeError, ValueError):
        prec_n = 0.0
    try:
        lift_n = max(0.0, min(1.0, float(lift) / 10.0)) if lift is not None else 0.0
    except (TypeError, ValueError):
        lift_n = 0.0

    cov_pen = _cov_penalty(all_cov, cov_thr)
    inst_pen = _instability_penalty(stability, stab_floor)
    red_pen = _redundancy_penalty(rule, selected_list, red_thr)
    proxy_pen = 0.0  # 无 feature_family 时 proxy 惩罚为 0，可由后续扩展
    div_bonus = _diversity_bonus(rule, selected_list)

    score_final = (
        w1 * prec_n + w2 * lift_n
        - w3 * cov_pen - w4 * inst_pen - w5 * red_pen - w6 * proxy_pen
        + w7 * div_bonus
    )
    breakdown = {
        'precision_term': w1 * prec_n,
        'lift_term': w2 * lift_n,
        'cov_penalty': -w3 * cov_pen,
        'instability_penalty': -w4 * inst_pen,
        'redundancy_penalty': -w5 * red_pen,
        'proxy_penalty': -w6 * proxy_pen,
        'diversity_bonus': w7 * div_bonus,
    }
    try:
        rule.score_breakdown = breakdown
        rule.score_final = score_final
    except Exception:
        pass
    return score_final
