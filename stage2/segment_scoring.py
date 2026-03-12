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
    from .unified_scoring import compute_score_final
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule
    from unified_scoring import compute_score_final

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
    # PairAssoc 缺失惩罚：score_penalty = min(lambda * missing_pair_ratio, missing_pair_penalty_cap)，带上限
    k = len(rule.feature_rules)
    pair_total = (k * (k - 1)) // 2 if k >= 2 else 0
    missing_pair_count = getattr(rule, 'missing_pair_count', None)
    if missing_pair_count is None:
        missing_pair_count = 0
    else:
        missing_pair_count = int(missing_pair_count)
    score_penalty = 0.0
    if pair_total > 0 and missing_pair_count > 0:
        missing_pair_ratio = missing_pair_count / pair_total
        lam = getattr(config, 'missing_pair_penalty_lambda', 0.1)
        cap = getattr(config, 'missing_pair_penalty_cap', 0.05)
        score_penalty = min(lam * missing_pair_ratio, cap)
        score = score - score_penalty

    # 统一 rule_score（与原子规则同公式：precision+lift+coverage+divergence+stability）
    w1 = getattr(config, 'rule_score_w_precision', 0.25)
    w2 = getattr(config, 'rule_score_w_lift', 0.25)
    w3 = getattr(config, 'rule_score_w_coverage', 0.1)
    w4 = getattr(config, 'rule_score_w_divergence', 0.2)
    w5 = getattr(config, 'rule_score_w_stability', 0.2)
    total_w = w1 + w2 + w3 + w4 + w5 or 1.0
    w1, w2, w3, w4, w5 = w1 / total_w, w2 / total_w, w3 / total_w, w4 / total_w, w5 / total_w
    prec = getattr(rule, 'combo_precision_est', None)
    lift = getattr(rule, 'combo_lift_est', None)
    cov = getattr(rule, 'combo_all_cov_est', None)
    prec_n = min(1.0, max(0.0, float(prec))) if prec is not None and not (isinstance(prec, float) and (np.isnan(prec) or prec < 0)) else 0.0
    lift_n = min(1.0, max(0.0, float(lift) / 10.0)) if lift is not None and not (isinstance(lift, float) and (np.isnan(lift) or lift <= 0)) else 0.0
    cov_n = min(1.0, max(0.0, float(cov))) if cov is not None and not (isinstance(cov, float) and np.isnan(cov)) else 0.0
    div_n = max(0.0, min(1.0, divergence_score))
    stab_n = max(0.0, min(1.0, stability_score))
    rule_score_val = w1 * prec_n + w2 * lift_n + w3 * cov_n + w4 * div_n + w5 * stab_n
    rule.rule_score = rule_score_val

    if getattr(config, 'use_unified_scoring', False):
        score_final = compute_score_final(rule, config, selected_list=None)
        rule.score_final = score_final
        score = score_final  # 宽进严排：用 score_final 作为主排序分

    return {
        'score': score,
        'rule_score': rule_score_val,
        'divergence_score': divergence_score,
        'stability_score': stability_score,
        'diversity_score': diversity_score,
        'w_divergence': w_div,
        'w_stability': w_stab,
        'w_diversity': w_divs,
        'score_penalty': score_penalty,
        'missing_pair_count': missing_pair_count,
    }


def _recall_est(precision_est: float, all_cov_est: float, prior_pi: float) -> Optional[float]:
    """目标召回率估计：recall = P(segment|target) = precision * all_cov / prior_pi（all_cov = P(rule|all)）。"""
    if not (0 < prior_pi < 1 and precision_est is not None and all_cov_est is not None):
        return None
    return (precision_est * all_cov_est) / prior_pi


def _f1_est(precision_est: float, recall_est: float) -> Optional[float]:
    """F1 = 2*P*R/(P+R)，平衡准确率与覆盖率。"""
    if precision_est is None or recall_est is None or (precision_est + recall_est) <= 0:
        return None
    return 2.0 * precision_est * recall_est / (precision_est + recall_est)


def filter_candidate_segments(
    candidate_rules: List[SegmentRule],
    config: Stage2Config
) -> List[SegmentRule]:
    """
    筛选候选客群（Coverage-free + 目标准确率/覆盖率约束 + 可选 F1 平衡排序）
    
    - 过滤：差异/稳定性阈值；若配置了 target_precision_min/target_coverage_min 且估计值存在，则剔除不达标候选。
    - 排序：默认在 use_f1_balance 时按 F1 与 score 的混合排序，避免单边极端。
    - 兜底：若严格过滤后无候选，则仅按差异/稳定性再筛一轮；若仍无则仅按得分取 top N，避免有 k=3 候选却输出全空。
    """
    target_precision_min = getattr(config, 'target_precision_min', 0.0)
    target_coverage_min = getattr(config, 'target_coverage_min', 0.0)
    use_f1_balance = getattr(config, 'use_f1_balance', True)
    prior_pi = getattr(config, 'expected_cohort_ratio', None)

    def _apply_filter(rules: List[SegmentRule], apply_target: bool) -> List[SegmentRule]:
        out = []
        for rule in rules:
            if rule.divergence_score < config.min_divergence_score:
                continue
            if rule.stability_score < config.min_stability_score:
                continue
            if apply_target:
                prec = getattr(rule, 'combo_precision_lb', None)
                all_cov = getattr(rule, 'combo_all_cov_est', None) or getattr(rule, 'combo_all_cov_ub', None)
                if prec is not None and prec < target_precision_min:
                    continue
                recall = _recall_est(prec, all_cov, prior_pi) if (prior_pi and 0 < prior_pi < 1 and all_cov is not None) else None
                if recall is not None and recall < target_coverage_min:
                    continue
            score_info = score_segment_rule(rule, config)
            rule.score = score_info['score']
            if score_info.get('score_penalty', 0) > 0:
                logger.info(
                    "候选 rule_id=%s missing_pair_count=%d score_penalty=%.4f score_final=%.4f",
                    rule.rule_id, score_info.get('missing_pair_count', 0),
                    score_info['score_penalty'], rule.score,
                )
            prec = getattr(rule, 'combo_precision_lb', None)
            all_cov = getattr(rule, 'combo_all_cov_est', None) or getattr(rule, 'combo_all_cov_ub', None)
            recall = _recall_est(prec, all_cov, prior_pi) if (prior_pi and 0 < prior_pi < 1 and all_cov is not None) else None
            if recall is not None and prec is not None:
                setattr(rule, '_f1_est', _f1_est(prec, recall))
                setattr(rule, '_recall_est', recall)
            else:
                setattr(rule, '_f1_est', None)
                setattr(rule, '_recall_est', None)
            out.append(rule)
        return out

    filtered_rules = _apply_filter(candidate_rules, apply_target=True)
    if not filtered_rules and candidate_rules:
        logger.warning("目标准确率/覆盖率过滤后无候选，兜底：仅按差异/稳定性筛选")
        filtered_rules = _apply_filter(candidate_rules, apply_target=False)
    if not filtered_rules and candidate_rules:
        logger.warning("差异/稳定性过滤后仍无候选，兜底：仅按得分取 top_n，保证有输出")
        for rule in candidate_rules:
            score_info = score_segment_rule(rule, config)
            rule.score = score_info['score']
            setattr(rule, '_f1_est', None)
            setattr(rule, '_recall_est', None)
        candidate_rules.sort(key=lambda r: r.score, reverse=True)
        filtered_rules = candidate_rules[:config.top_n_candidates]

    if not filtered_rules:
        logger.info("筛选完成，无候选规则通过（差异/稳定性/目标准确率与覆盖率约束）")
        return filtered_rules

    max_score = max(r.score for r in filtered_rules)
    if max_score <= 0:
        max_score = 1.0

    def sort_key(r: SegmentRule) -> float:
        f1 = getattr(r, '_f1_est', None)
        if use_f1_balance and f1 is not None:
            return 0.5 * (r.score / max_score) + 0.5 * f1
        return r.score / max_score

    filtered_rules.sort(key=sort_key, reverse=True)
    filtered_rules = filtered_rules[:config.top_n_candidates]

    logger.info(
        "筛选完成，保留 %d 个候选规则（从 %d 个中筛选；目标准确率>=%.2f、覆盖率>=%.2f%s）",
        len(filtered_rules), len(candidate_rules), target_precision_min, target_coverage_min,
        "，F1 平衡排序" if use_f1_balance else ""
    )
    return filtered_rules

