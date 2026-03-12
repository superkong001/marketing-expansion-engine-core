"""
多客群组合与去重模块

从候选客群中选择互相区分度高的客群组合
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    from .stage2_config import Stage2Config
    from .rule_combination import SegmentRule
    from .segment_diversity import (
        calculate_rule_similarity,
        calculate_rule_structure_distance,
        filter_similar_rules,
        get_rule_overlap_stats,
    )
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule
    from segment_diversity import (
        calculate_rule_similarity,
        calculate_rule_structure_distance,
        filter_similar_rules,
        get_rule_overlap_stats,
    )

logger = logging.getLogger(__name__)


def _feature_set_key(rule: SegmentRule) -> tuple:
    """同特征集去冗余用：特征集合的规范 key（仅 column_id，不含阈值/类别取值）。"""
    ids = getattr(rule, 'rule_feature_ids', None)
    if ids is not None:
        return tuple(sorted(ids))
    return tuple(sorted(fr.get('column_id', '') for fr in rule.feature_rules if isinstance(fr, dict)))


def dedup_same_structure_candidates(
    rules: List[SegmentRule],
    config: Stage2Config,
) -> List[SegmentRule]:
    """
    同特征集去冗余：特征集合完全相同、仅阈值/类别不同的规则只保留 score_final（或 score）最高的一条，
    并写入 redundancy_group_id。
    """
    if not getattr(config, 'same_structure_at_candidate', True):
        return rules
    if not rules:
        return rules
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rules:
        by_key[_feature_set_key(r)].append(r)
    out = []
    for key, group in by_key.items():
        best = max(
            group,
            key=lambda x: (
                getattr(x, 'score_final', None) or getattr(x, 'score', 0.0) or 0.0,
                getattr(x, 'rule_score', 0.0) or 0.0,
            ),
        )
        try:
            best.redundancy_group_id = id(best)
        except Exception:
            pass
        out.append(best)
    return out


def dedup_same_structure_portfolio(
    recommended: List[SegmentRule],
    config: Stage2Config,
) -> List[SegmentRule]:
    """Portfolio 推荐确定后再做一次同特征集去冗余，只保留 score_final 最高的一条。"""
    if not getattr(config, 'same_structure_at_portfolio', True):
        return recommended
    if not recommended:
        return recommended
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in recommended:
        by_key[_feature_set_key(r)].append(r)
    out = []
    for key, group in by_key.items():
        best = max(
            group,
            key=lambda x: (
                getattr(x, 'score_final', None) or getattr(x, 'score', 0.0) or 0.0,
                getattr(x, 'rule_score', 0.0) or 0.0,
            ),
        )
        out.append(best)
    return out


def _anchor_or_family_key(rule: SegmentRule, cluster_by: str) -> str:
    """分簇用 key：anchor_feature 或 feature_family 聚合。"""
    if cluster_by == 'anchor_feature':
        a = getattr(rule, 'anchor_feature', None)
        if a:
            return str(a)
        rids = getattr(rule, 'rule_feature_ids', None)
        if rids:
            return str(rids[0]) if rids else ''
        if rule.feature_rules:
            return str(rule.feature_rules[0].get('column_id', ''))
        return ''
    if cluster_by == 'feature_family':
        families = sorted(set(fr.get('feature_family', 'other') for fr in rule.feature_rules if isinstance(fr, dict)))
        return '|'.join(families) if families else 'other'
    return ''


def cluster_then_fill(
    candidate_rules: List[SegmentRule],
    config: Stage2Config,
) -> List[SegmentRule]:
    """
    多簇选择：按 anchor_feature（或 feature_family）分簇，先每簇取 top1（按 score_final），
    再按 score_final 全局补足到 target_segment_count。
    """
    cluster_by = getattr(config, 'cluster_by', None) or 'anchor_feature'
    max_per_cluster = getattr(config, 'max_per_cluster_first', 1)
    min_clusters = getattr(config, 'min_clusters_represented', 2)
    target_n = getattr(config, 'target_segment_count', None) or getattr(config, 'max_segments', 5)

    from collections import defaultdict
    by_key = defaultdict(list)
    for r in candidate_rules:
        by_key[_anchor_or_family_key(r, cluster_by)].append(r)

    selected = []
    for key, group in by_key.items():
        if not key:
            continue
        sorted_group = sorted(
            group,
            key=lambda x: (getattr(x, 'score_final', None) or getattr(x, 'score', 0.0) or 0.0),
            reverse=True,
        )
        selected.extend(sorted_group[:max_per_cluster])
    selected_ids = {id(r) for r in selected}
    remaining = [r for r in candidate_rules if id(r) not in selected_ids]
    remaining.sort(
        key=lambda x: (getattr(x, 'score_final', None) or getattr(x, 'score', 0.0) or 0.0),
        reverse=True,
    )
    while len(selected) < target_n and remaining:
        selected.append(remaining.pop(0))
    if len(by_key) >= min_clusters or not by_key:
        logger.debug("[stage2] clusters count=%d per_cluster_top1 then fill to %d", len(by_key), len(selected))
    return selected


def _has_discrete_rule(rule: SegmentRule) -> bool:
    """是否至少包含一条离散规则。"""
    return any(fr.get('type') == 'categorical' for fr in rule.feature_rules)


def assign_tiers(
    segments: List[SegmentRule],
    config: Stage2Config
) -> Dict[str, List[SegmentRule]]:
    """
    v2.0: 将候选客群后分层为 elite / standard / expand。
    每个 segment 只归入第一个满足的档位；未满足任何档位的 segment 不进入 tiers。
    """
    elite_cov = getattr(config, 'elite_combo_cov_max', 0.02)
    elite_prec = getattr(config, 'elite_combo_precision_min', 0.35)
    standard_cov = getattr(config, 'standard_combo_cov_max', 0.05)
    standard_prec = getattr(config, 'standard_combo_precision_min', 0.20)
    expand_cov = getattr(config, 'expand_combo_cov_max', 0.10)
    expand_require_disc = getattr(config, 'expand_require_discrete', True)
    expand_allow_unknown = getattr(config, 'expand_allow_cov_unknown', True)

    elite: List[SegmentRule] = []
    standard: List[SegmentRule] = []
    expand: List[SegmentRule] = []

    # 方案2：若全量候选中无任何离散规则，则 expand 不要求离散，避免纯连续型数据三档全空
    any_has_disc = any(_has_discrete_rule(seg) for seg in segments)

    for seg in segments:
        # 全量覆盖率口径：combo_all_cov_ub (P(rule|all))
        all_cov_ub = getattr(seg, 'combo_all_cov_ub', None)
        prec = getattr(seg, 'combo_precision_lb', None)
        cov_unk = getattr(seg, 'combo_cov_unknown', True)
        has_disc = _has_discrete_rule(seg)

        if not cov_unk and all_cov_ub is not None and all_cov_ub <= elite_cov and (prec is not None and prec >= elite_prec) and has_disc:
            elite.append(seg)
            continue
        if not cov_unk and all_cov_ub is not None and all_cov_ub <= standard_cov and (prec is not None and prec >= standard_prec) and has_disc:
            standard.append(seg)
            continue
        if (expand_allow_unknown or not cov_unk) and all_cov_ub is not None and all_cov_ub <= expand_cov:
            if not expand_require_disc or has_disc or (not any_has_disc):
                expand.append(seg)

    if not elite and expand:
        elite_prec_relaxed = getattr(config, 'elite_fallback_precision_min', 0.25)
        elite_cov_relaxed = getattr(config, 'elite_fallback_cov_max', 0.03)
        relaxed_elite = [
            s for s in expand
            if not getattr(s, 'combo_cov_unknown', True)
            and getattr(s, 'combo_all_cov_ub', None) is not None
            and getattr(s, 'combo_all_cov_ub', 1.0) <= elite_cov_relaxed
            and (getattr(s, 'combo_precision_lb', None) is not None and getattr(s, 'combo_precision_lb', 0) >= elite_prec_relaxed)
            and _has_discrete_rule(s)
        ]
        if relaxed_elite:
            best = max(relaxed_elite, key=lambda s: (getattr(s, 'score', 0.0) or 0.0, getattr(s, 'combo_precision_est', -1) or -1))
            elite = [best]
            expand = [s for s in expand if s is not best]
            logger.info(
                "elite 降级：已放宽阈值（elite_combo_precision_min=%.2f -> %.2f, elite_combo_cov_max=%.2f -> %.2f），elite 数=1，rule_id=%s",
                elite_prec, elite_prec_relaxed, elite_cov, elite_cov_relaxed, best.rule_id,
            )
        elif getattr(config, 'elite_fallback_use_top1_by_score', True):
            top1 = max(expand, key=lambda s: (getattr(s, 'score', 0.0) or 0.0, getattr(s, 'combo_precision_est', -1) or -1))
            elite = [top1]
            expand = [s for s in expand if s is not top1]
            logger.info("elite 降级：使用 expand 按 score 取 top1 作为 elite，rule_id=%s", top1.rule_id)
    if not elite:
        logger.warning("精英档为空，请检查 combo 阈值或原子规则质量")
    if not elite and not standard and not expand:
        logger.warning("无候选满足任意档位条件（combo_cov/combo_precision/cov_unknown/离散规则）")
    elif not elite and (standard or expand):
        max_prec = max(
            (getattr(s, 'combo_precision_lb', None) for s in (standard + expand)),
            default=None,
        )
        min_cov = min(
            (getattr(s, 'combo_all_cov_ub', None) for s in (standard + expand) if getattr(s, 'combo_all_cov_ub', None) is not None),
            default=None,
        )
        logger.warning(
            "elite 不可达：无候选满足 combo_all_cov_ub<=%.2f 且 combo_precision_lb>=%.2f；当前最高 precision_lb=%s，最小 all_cov_ub=%s",
            elite_cov, elite_prec, max_prec, min_cov,
        )
    else:
        logger.info(f"分层统计: elite=%d, standard=%d, expand=%d", len(elite), len(standard), len(expand))

    return {"elite": elite, "standard": standard, "expand": expand}


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
    
    # 使用基于规则结构距离的去重函数；预期输出客群数优先用 target_segment_count；预选不足时逐步放宽相似度阈值
    max_rules = getattr(config, 'target_segment_count', None) or config.max_segments
    target_n = max_rules
    cluster_by = getattr(config, 'cluster_by', None)
    if cluster_by:
        selected_segments = cluster_then_fill(candidate_rules, config)
        logger.info("[stage2] clusters count=%d per_cluster_top1 then fill, selected=%d", len(set(_anchor_or_family_key(r, cluster_by) for r in candidate_rules)), len(selected_segments))
    else:
        fallback_thresholds = getattr(config, 'similarity_fallback_thresholds', (0.72, 0.5, 0.3))
        if not isinstance(fallback_thresholds, (list, tuple)):
            fallback_thresholds = (config.similarity_threshold, 0.5, 0.3)
        thresholds_to_try = [t for t in fallback_thresholds if isinstance(t, (int, float))]
        if not thresholds_to_try:
            thresholds_to_try = [config.similarity_threshold]
        selected_segments = []
        for i, th in enumerate(thresholds_to_try):
            selected_segments = filter_similar_rules(
                candidate_rules,
                similarity_threshold=float(th),
                max_rules=max_rules,
            )
            if len(selected_segments) >= target_n:
                break
            if len(candidate_rules) >= target_n and i + 1 < len(thresholds_to_try):
                logger.info(
                    "filter_similar_rules fallback: similarity_threshold %.2f -> %.2f，预选数 %d",
                    th, thresholds_to_try[i + 1], len(selected_segments),
                )
        min_preselect = min(3, target_n) if target_n else 3
        if len(selected_segments) < min_preselect and len(candidate_rules) >= min_preselect:
            selected_ids = {id(r) for r in selected_segments}
            remaining_by_score = sorted(
                [r for r in candidate_rules if id(r) not in selected_ids],
                key=lambda r: getattr(r, 'score', 0.0),
                reverse=True,
            )
            need = min_preselect - len(selected_segments)
            for r in remaining_by_score[:need]:
                selected_segments.append(r)
            logger.info("预选保底: 按 score 补足 %d 条，预选总数 %d", need, len(selected_segments))
    
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
    
    # 同特征集去冗余：recommended 确定后再做一次，只保留 score_final 最高的一条
    n_port_before = len(selected_segments)
    selected_segments = dedup_same_structure_portfolio(selected_segments, config)
    n_port_after = len(selected_segments)
    if n_port_before > n_port_after:
        logger.info(
            "[stage2] portfolio_dedup before=%d after=%d removed=%d",
            n_port_before, n_port_after, n_port_before - n_port_after,
        )

    logger.info(f"构建客群组合完成，选择 {len(selected_segments)} 个客群（基于规则结构距离的差异最大化原则）")

    return SegmentPortfolio(
        segments=selected_segments,
        portfolio_metrics=portfolio_metrics
    )


def select_final_segments(
    segments: List[SegmentRule],
    tiers: Dict[str, List[SegmentRule]],
    config: Stage2Config,
) -> Tuple[List[SegmentRule], List[dict], bool]:
    """
    两阶段选择最终推荐客群：elite 非空时强制选 1 个 elite；剩余名额按结构距离+综合得分。
    Returns:
        (selected_list, selection_steps, elite_forced)
    """
    target_n = getattr(config, 'target_segment_count', None) or getattr(config, 'max_segments', 5)
    w_dist = getattr(config, 'selection_w_dist', 0.6)
    w_quality = getattr(config, 'selection_w_quality', 0.4)
    min_score = getattr(config, 'selection_min_score', None)
    shared_max = getattr(config, 'selection_overlap_shared_fields_max', 1)
    jaccard_min = getattr(config, 'selection_jaccard_dist_min', 0.5)

    selected_list: List[SegmentRule] = []
    selection_steps: List[dict] = []
    elite_forced = False

    elite_pool = tiers.get('elite') or []
    segment_set = set(id(s) for s in segments)

    # Phase 1: 若 elite 非空，强制选 1 个
    if elite_pool:
        # tie-break: score 高 -> combo_precision_est 高 -> combo_lift_est 高 -> N_all_est 小
        n_total = getattr(config, 'total_population', None) or 1

        def _elite_sort_key(s: SegmentRule):
            prec = getattr(s, 'combo_precision_est', None)
            lift = getattr(s, 'combo_lift_est', None)
            all_cov = getattr(s, 'combo_all_cov_est', None)
            n_all = (all_cov * n_total) if (all_cov is not None and n_total) else float('inf')
            return (
                -(getattr(s, 'score', 0.0) or 0.0),
                -(prec if prec is not None else -1.0),
                -(lift if lift is not None else -1.0),
                n_all,
            )

        best_elite = min(elite_pool, key=lambda s: _elite_sort_key(s))
        # 确保 best_elite 在 segments 中（同一对象或可匹配）
        for seg in segments:
            if seg.rule_id == best_elite.rule_id:
                best_elite = seg
                break
        selected_list.append(best_elite)
        elite_forced = True
        selection_steps.append({
            "step": 1,
            "rule_id": best_elite.rule_id,
            "reason": "force_elite",
            "tier": "elite",
        })

    # Phase 2: 剩余名额从 segments 中选，结构距离最大化 + 综合得分；锚点限流 + family 覆盖软约束
    max_per_anchor = getattr(config, 'max_segments_per_anchor', 2)
    min_family = getattr(config, 'min_family_coverage', 2)
    family_bonus = 0.05

    def _anchor(rule: SegmentRule):
        anchor = getattr(rule, 'anchor_feature', None)
        if anchor:
            return anchor
        rids = getattr(rule, 'rule_feature_ids', None)
        if rids:
            return rids[0]
        if rule.feature_rules:
            return rule.feature_rules[0].get('column_id', '')
        return ''

    def _families(rule: SegmentRule):
        return set(fr.get('feature_family', 'other') for fr in rule.feature_rules)

    remaining = [s for s in segments if s not in selected_list]
    if not remaining or len(selected_list) >= target_n:
        return selected_list, selection_steps, elite_forced

    # recommended 可信度：missing_pair_count 超过阈值的候选不参与组合选择（仅可进 menu 并标 low_confidence）
    max_missing = getattr(config, 'recommended_max_missing_pair_count', 1)
    use_rule_score = getattr(config, 'portfolio_use_rule_score', True)
    sim_penalty_lambda = getattr(config, 'similarity_penalty_lambda', 0.3)
    scores_raw = [
        (getattr(s, 'rule_score', 0.0) or getattr(s, 'score', 0.0) or 0.0) if use_rule_score else (getattr(s, 'score', 0.0) or 0.0)
        for s in remaining
    ]
    smin, smax = min(scores_raw), max(scores_raw)
    score_range = (smax - smin) if (smax > smin) else 1.0

    # 同结构限制：与已选规则同结构（min_struct_dist<=eps）的候选，仅当该结构在 recommended 中已达上限时才剔除
    STRUCT_DIST_EPS = 1e-9
    max_same_structure = getattr(config, 'max_same_structure_in_recommended', 1)
    pruned_by_struct_dist_zero = 0

    step_idx = len(selection_steps) + 1
    while len(selected_list) < target_n and remaining:
        selected_anchor_counts: Dict[str, int] = {}
        for s in selected_list:
            a = _anchor(s)
            selected_anchor_counts[a] = selected_anchor_counts.get(a, 0) + 1
        selected_families = set()
        for s in selected_list:
            selected_families |= _families(s)

        best_candidate = None
        best_score_sel = -1.0
        best_min_dist = -1.0
        best_quality = -1.0
        best_max_sim = -1.0
        best_score_effective = -1.0

        for cand in remaining:
            missing_cnt = getattr(cand, 'missing_pair_count', None)
            if missing_cnt is not None and int(missing_cnt) > max_missing:
                continue
            if max_per_anchor > 0 and selected_anchor_counts.get(_anchor(cand), 0) >= max_per_anchor:
                continue
            if selected_list:
                struct_dists = [calculate_rule_structure_distance(cand, sel) for sel in selected_list]
                min_struct_dist = min(struct_dists)
                if min_struct_dist <= STRUCT_DIST_EPS:
                    same_structure_count = sum(1 for d in struct_dists if d <= STRUCT_DIST_EPS)
                    if same_structure_count >= max_same_structure:
                        pruned_by_struct_dist_zero += 1
                        continue
                    # 同结构但未达上限：允许进入
                max_similarity = max(calculate_rule_similarity(cand, sel) for sel in selected_list)
            else:
                min_struct_dist = 1.0
                max_similarity = 0.0
            # 高重叠：共享字段 >= 2 或 jaccard_dist < 0.5
            high_overlap = False
            for sel in selected_list:
                shared, jaccard_d = get_rule_overlap_stats(cand, sel)
                if shared > shared_max or (jaccard_d < jaccard_min):
                    high_overlap = True
                    break
            quality_raw = getattr(cand, 'rule_score', None) if use_rule_score else None
            if quality_raw is None:
                quality_raw = getattr(cand, 'score', 0.0) or 0.0
            else:
                quality_raw = quality_raw or getattr(cand, 'score', 0.0) or 0.0
            if min_score is not None and quality_raw < min_score:
                continue
            # Greedy + 相似惩罚：score_effective = rule_score - lambda * max_similarity
            score_effective = quality_raw - sim_penalty_lambda * max_similarity
            normalized_quality = (quality_raw - smin) / score_range if score_range > 0 else 0.5
            if high_overlap:
                normalized_quality *= 0.5
            score_selection = score_effective + w_dist * min_struct_dist + w_quality * normalized_quality
            if min_family > 0 and len(selected_families) < min_family:
                cand_families = _families(cand)
                if cand_families - selected_families:
                    score_selection += family_bonus
            if score_selection > best_score_sel:
                best_score_sel = score_selection
                best_candidate = cand
                best_min_dist = min_struct_dist
                best_quality = quality_raw
                best_max_sim = max_similarity if selected_list else 0.0
                best_score_effective = score_effective

        if best_candidate is None:
            break
        selected_list.append(best_candidate)
        remaining.remove(best_candidate)
        selection_steps.append({
            "step": step_idx,
            "rule_id": best_candidate.rule_id,
            "reason": "score_distance",
            "min_struct_dist": round(best_min_dist, 4),
            "score": round(best_quality, 4),
            "score_selection": round(best_score_sel, 4),
            "rule_score": round(best_quality, 4),
            "max_similarity": round(best_max_sim, 4),
            "score_effective": round(best_score_effective, 4),
        })
        logger.info(
            "组合选择阶段2：选中 rule_id=%s, min_struct_dist=%.4f, rule_score=%.4f, max_similarity=%.4f, score_effective=%.4f, score_selection=%.4f",
            best_candidate.rule_id, best_min_dist, best_quality, best_max_sim, best_score_effective, best_score_sel,
        )
        if best_min_dist <= STRUCT_DIST_EPS and len(selected_list) > 1:
            logger.warning("min_struct_dist=0（与已选某条规则结构完全相同），rule_id=%s", best_candidate.rule_id)
        step_idx += 1

    if pruned_by_struct_dist_zero > 0:
        logger.info("组合选择阶段2：因 min_struct_dist<=eps 跳过候选数 pruned_by_struct_dist_zero=%d", pruned_by_struct_dist_zero)
    return selected_list, selection_steps, elite_forced

