"""
规则组合搜索模块

从原子规则库中搜索多特征组合规则
"""
import re
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

try:
    from .stage2_config import Stage2Config
    from .segment_diversity import calculate_rule_diversity_score
    from .rule_conflict_checker import check_rule_conflicts, check_numeric_conflict, check_categorical_conflict
    from .pair_assoc import PairAssocIndex
    from .pair_assoc_loader import should_prune_pair
    from .combo_cov_adjust import compute_combo_metrics
except ImportError:
    from stage2_config import Stage2Config
    from segment_diversity import calculate_rule_diversity_score
    from rule_conflict_checker import check_rule_conflicts, check_numeric_conflict, check_categorical_conflict
    try:
        from pair_assoc import PairAssocIndex
    except ImportError:
        PairAssocIndex = None
    try:
        from pair_assoc_loader import should_prune_pair
    except ImportError:
        def should_prune_pair(*args, **kwargs):
            return False
    try:
        from combo_cov_adjust import compute_combo_metrics
    except ImportError:
        compute_combo_metrics = None

logger = logging.getLogger(__name__)


def _norm_cov(v: Any) -> Optional[float]:
    """将 None/NaN 视为无覆盖率，统一为 None；rare_anchor/combo_cov 仅在有有效 cov 时生效。"""
    if v is None:
        return None
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _frechet_precision_from_lift(lift: Optional[float], pi: Optional[float]) -> Optional[float]:
    """precision = (pi * lift) / (pi * lift + (1 - pi))；pi 未配置或 lift 为 None 则返回 None。"""
    if lift is None or pi is None or not (0 < pi < 1):
        return None
    denom = pi * lift + (1 - pi)
    return (pi * lift) / denom if denom else None


def get_feature_family(column_id: str, demographic_regex_list: Optional[list] = None) -> str:
    """v2.0: 按 regex 将 column_id 归类为 fee_family / recency_family / identity_family / demographic_family / other。"""
    import re
    cid = (column_id or '').strip()
    cid_upper = cid.upper()
    if re.search(r'FEE|ARPU|总费用|含税|不含税|收入|GPRS费|GPRS_FEE|MON_FEE|PLAN_FEE|TAX_FEE|ZH_FEE|XZ_FEE|PERMONTH|OUT_PLAN|CALL_FEE|SMS_FEE|ACY_FEE|OTH_FEE|ISMP_FEE|PLAN_SPL|ONNET|GRP_PAY|GRP_PROD', cid_upper, re.I):
        return 'fee_family'
    if re.search(r'LAST_.*_DATE|_TIME_SEG|TIME_SEG', cid_upper):
        return 'recency_family'
    if re.search(r'^IS_|^ORG_', cid_upper):
        return 'identity_family'
    if demographic_regex_list:
        for pat in demographic_regex_list:
            try:
                if re.search(pat, cid, re.I):
                    return 'demographic_family'
            except re.error:
                continue
    return 'other'


@dataclass
class SegmentRule:
    """客群规则数据结构（Coverage-free版本）；组合覆盖率与精度使用 Fréchet 区间。"""
    rule_id: str
    feature_rules: List[Dict]  # [{"column_id": "age", "type": "numeric", "low": 25, "high": 40, "base_cov", "sub_cov"}, ...]
    divergence_score: float = 0.0  # 平均差异评分
    stability_score: float = 0.0  # 平均稳定性评分
    diversity_score: float = 0.0  # 多样性评分
    score: float = 0.0  # 综合得分，后续会被评分模块更新
    rule_score: float = 0.0  # 统一规则得分（precision+lift+coverage+divergence+stability），用于导出与 portfolio 排序
    anchor_feature: Optional[str] = None  # 锚点特征 column_id（首特征或 rare 最小 base_cov）
    # 区间估计（Fréchet bounds）；语义：non_sub=P(rule|non-sub), sub=P(rule|sub), all=P(rule|all)
    combo_non_sub_cov_lb: Optional[float] = None
    combo_non_sub_cov_ub: Optional[float] = None
    combo_sub_cov_lb: Optional[float] = None
    combo_sub_cov_ub: Optional[float] = None
    combo_all_cov_lb: Optional[float] = None
    combo_all_cov_ub: Optional[float] = None
    combo_precision_lb: Optional[float] = None
    combo_precision_ub: Optional[float] = None
    # 点估计：有 cov 时用 ind_est/adj_est 填充，优先 adj
    combo_non_sub_cov_est: Optional[float] = None
    combo_sub_cov_est: Optional[float] = None
    combo_all_cov_est: Optional[float] = None
    combo_lift_est: Optional[float] = None
    combo_precision_est: Optional[float] = None
    fp_rate_est: Optional[float] = None
    combo_cov_unknown: Optional[bool] = None  # v2.0: True 时仅允许进入扩展档
    # 独立性点估计与相关性修正
    combo_non_sub_cov_ind_est: Optional[float] = None
    combo_sub_cov_ind_est: Optional[float] = None
    combo_non_sub_cov_adj_est: Optional[float] = None
    combo_sub_cov_adj_est: Optional[float] = None
    combo_pi_missing: Optional[bool] = None  # True 时 precision 不可算（缺 expected_cohort_ratio）
    # 相关性驱动 precision_ub 收紧
    precision_ub_shrunk_by_corr: Optional[bool] = None
    r_max: Optional[float] = None
    # 点估计来源：adj=相关性修正，ind=独立性估计（供导出说明）
    combo_cov_est_source: Optional[str] = None  # "adj" | "ind"
    # 病态度量/条件数稳定性（仅基于字段对表）
    kappa: Optional[float] = None  # 相关矩阵条件数 σ_max/σ_min
    missing_pair_count: Optional[int] = None  # 字段对表中缺失的对数
    # 区间上界收紧（pair 联合上界 + 缺失惩罚）
    combo_all_cov_ub_shrunk: Optional[float] = None
    combo_sub_cov_ub_shrunk: Optional[float] = None
    # 结构距离与去重：column_id 列表与唯一签名（与 rule_output.segment_canonical_key 一致）
    rule_feature_ids: Optional[List[str]] = None
    rule_signature: Optional[str] = None


def _build_rule_signature(feature_rules: List[Dict]) -> str:
    """生成与 rule_output.segment_canonical_key 一致的唯一键，用于去重与结构距离。"""
    parts = []
    for fr in sorted(feature_rules, key=lambda x: x.get('column_id', '')):
        cid = fr.get('column_id', '')
        if fr.get('type') == 'numeric':
            low, high = fr.get('low'), fr.get('high')
            direction = fr.get('direction')
            if low is not None and high is not None:
                summary = f"{low}_{high}"
            else:
                summary = str(direction) if direction else "numeric"
            parts.append(f"{cid}:{summary}")
        elif fr.get('type') == 'categorical':
            cats = fr.get('categories') or []
            summary = ",".join(sorted(str(c) for c in cats))
            parts.append(f"{cid}:{summary}")
        else:
            parts.append(cid)
    return "|".join(parts)


def calculate_rule_divergence_score(rule: SegmentRule, divergence_scores: pd.Series) -> float:
    """
    计算规则的平均差异评分
    
    Args:
        rule: SegmentRule对象
        divergence_scores: 差异评分Series，index为column_id
    
    Returns:
        平均差异评分
    """
    feature_scores = []
    for fr in rule.feature_rules:
        col_id = fr['column_id']
        if col_id in divergence_scores.index:
            feature_scores.append(divergence_scores[col_id])
    
    if feature_scores:
        return np.mean(feature_scores)
    else:
        return 0.0


def calculate_rule_stability(rule: SegmentRule, stability_scores: pd.Series) -> float:
    """
    计算规则的平均稳定性
    
    Args:
        rule: SegmentRule对象
        stability_scores: 稳定性评分Series，index为column_id
    
    Returns:
        平均稳定性评分
    """
    feature_stabilities = []
    for fr in rule.feature_rules:
        col_id = fr['column_id']
        if col_id in stability_scores.index:
            feature_stabilities.append(stability_scores[col_id])
    
    if feature_stabilities:
        return np.mean(feature_stabilities)
    else:
        return 0.5  # 默认稳定性


def check_business_group_constraint(
    rule: SegmentRule,
    business_groups: Dict[str, str],
    max_per_group: int = 2
) -> bool:
    """
    检查业务分组约束：每个segment最多选N个同业务组字段
    
    Args:
        rule: SegmentRule对象
        business_groups: 业务分组字典（column_id -> business_group）
        max_per_group: 每个业务组最多允许的字段数（默认2）
    
    Returns:
        是否满足约束
    """
    if not business_groups:
        return True  # 如果没有业务分组信息，不进行约束
    
    # 统计每个业务组的字段数
    group_counts = {}
    for fr in rule.feature_rules:
        col_id = fr['column_id']
        group = business_groups.get(col_id, 'unknown')
        group_counts[group] = group_counts.get(group, 0) + 1
    
    # 检查是否有业务组超过限制
    for group, count in group_counts.items():
        if group != 'unknown' and count > max_per_group:
            return False
    
    return True


def check_internal_rule_conflict(rule: SegmentRule) -> Tuple[bool, str]:
    """
    检查规则内部的冲突（同字段多区间、区间互斥、离散集合为空等）
    
    Args:
        rule: SegmentRule对象
    
    Returns:
        (是否有冲突, 冲突描述)
    """
    # 检查是否有重复字段
    column_ids = [fr['column_id'] for fr in rule.feature_rules]
    if len(column_ids) != len(set(column_ids)):
        return True, "存在重复字段"
    
    # 检查同字段的数值区间是否互斥
    numeric_rules_by_col = {}
    for fr in rule.feature_rules:
        if fr.get('type') == 'numeric':
            col_id = fr['column_id']
            if col_id not in numeric_rules_by_col:
                numeric_rules_by_col[col_id] = []
            numeric_rules_by_col[col_id].append(fr)
    
    for col_id, rules_list in numeric_rules_by_col.items():
        if len(rules_list) > 1:
            # 检查多个区间是否互斥
            for i in range(len(rules_list)):
                for j in range(i + 1, len(rules_list)):
                    r1 = rules_list[i]
                    r2 = rules_list[j]
                    low1 = r1.get('low', float('-inf'))
                    high1 = r1.get('high', float('inf'))
                    low2 = r2.get('low', float('-inf'))
                    high2 = r2.get('high', float('inf'))
                    
                    # 检查区间是否完全分离（无重叠）
                    if high1 <= low2 or high2 <= low1:
                        return True, f"字段 {col_id} 的区间完全分离: [{low1}, {high1}) 和 [{low2}, {high2})"
    
    # 检查离散字段的类别集合是否为空
    for fr in rule.feature_rules:
        if fr.get('type') == 'categorical':
            categories = fr.get('categories', [])
            if not categories or len(categories) == 0:
                return True, f"字段 {fr['column_id']} 的类别集合为空"
    
    return False, ""


# 字段对关联统计表列名（可选；表缺失时退化为结构约束）
PAIR_COL_A = 'column_id_a'
PAIR_COL_B = 'column_id_b'
PAIR_CORR_STRENGTH = 'corr_strength'
PAIR_CORR_TYPE = 'corr_type'
PAIR_CORR_SIGN = 'corr_sign'
REDUNDANT_CORR_THR = 0.7
REDUNDANT_CORR_TYPES = ('cont_cont_linear', 'cont_cont_rank')
CONFLICT_CORR_THR = 0.8
CONFLICT_CORR_SIGN = -1


def _pair_assoc_skip(
    pair_assoc_df: Optional[pd.DataFrame],
    used_columns: set,
    new_col_id: str,
    config: Optional[Any] = None
) -> Tuple[bool, str]:
    """
    基于字段对关联表判断是否应跳过该组合（冗余或冲突）。
    若表缺失或无对应行，返回 (False, '') 即不跳过。
    阈值优先使用 config 的 pair_* 参数，无则用模块常量。
    """
    if pair_assoc_df is None or len(pair_assoc_df) == 0:
        return False, ''
    cols = pair_assoc_df.columns
    need = {PAIR_COL_A, PAIR_COL_B, PAIR_CORR_STRENGTH}
    if not need.issubset(set(str(c) for c in cols)):
        return False, ''
    redundant_thr = getattr(config, 'pair_redundant_thr', REDUNDANT_CORR_THR) if config is not None else REDUNDANT_CORR_THR
    conflict_thr = getattr(config, 'pair_conflict_thr', CONFLICT_CORR_THR) if config is not None else CONFLICT_CORR_THR
    conflict_sign = getattr(config, 'pair_conflict_sign', CONFLICT_CORR_SIGN) if config is not None else CONFLICT_CORR_SIGN
    redundant_types = getattr(config, 'pair_redundant_types', REDUNDANT_CORR_TYPES) if config is not None else REDUNDANT_CORR_TYPES
    if isinstance(redundant_types, list):
        redundant_types = tuple(redundant_types)
    type_col = PAIR_CORR_TYPE if PAIR_CORR_TYPE in cols else None
    sign_col = PAIR_CORR_SIGN if PAIR_CORR_SIGN in cols else None
    for uc in used_columns:
        mask = (
            ((pair_assoc_df[PAIR_COL_A].astype(str) == str(uc)) & (pair_assoc_df[PAIR_COL_B].astype(str) == str(new_col_id))) |
            ((pair_assoc_df[PAIR_COL_A].astype(str) == str(new_col_id)) & (pair_assoc_df[PAIR_COL_B].astype(str) == str(uc)))
        )
        sub = pair_assoc_df.loc[mask]
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            try:
                strength = float(row[PAIR_CORR_STRENGTH])
            except (TypeError, ValueError):
                continue
            if strength >= conflict_thr and sign_col is not None:
                try:
                    sign = int(row[sign_col])
                    if sign == conflict_sign:
                        return True, 'conflict'
                except (TypeError, ValueError):
                    pass
            if strength >= redundant_thr and type_col is not None:
                try:
                    ct = str(row[type_col]).strip().lower()
                    if ct in [t.lower() for t in redundant_types]:
                        return True, 'redundant'
                except (TypeError, ValueError):
                    pass
    return False, ''


def combine_rules_beam_search(
    atomic_rules_df: pd.DataFrame,
    numeric_diff_df: pd.DataFrame,
    categorical_diff_df: pd.DataFrame,
    config: Stage2Config,
    divergence_scores: Optional[pd.Series] = None,
    stability_scores: Optional[pd.Series] = None,
    business_groups: Optional[Dict[str, str]] = None,
    max_fields_per_business_group: int = 2,
    pair_assoc_df: Optional[pd.DataFrame] = None,
    pair_assoc_index: Optional[Any] = None,
    prior_pi: Optional[float] = None,
) -> Tuple[List[SegmentRule], Dict[str, int]]:
    """
    Coverage-free Beam Search组合搜索
    
    若提供 pair_assoc_index（或 pair_assoc_df 且 enable_pair_assoc_pruning），则启用相关性剪枝：
    强相关硬剪枝、中相关软惩罚、可选冲突（light）检测。返回 (候选列表, beam_stats)。
    """
    beam_stats = {'pruned_high_corr': 0, 'penalized_mid_corr': 0, 'pruned_conflict': 0, 'pruned_demographic': 0, 'pruned_height': 0, 'pruned_kappa': 0, 'height_as_third_count': 0}
    if len(atomic_rules_df) == 0:
        logger.warning("原子规则库为空，无法进行组合搜索")
        return [], beam_stats
    
    # 从atomic_rules_df中提取差异评分和稳定性评分
    if divergence_scores is None:
        divergence_scores = atomic_rules_df.set_index('column_id')['divergence_score'] if 'divergence_score' in atomic_rules_df.columns else pd.Series(dtype=float)
    if stability_scores is None:
        stability_scores = atomic_rules_df.set_index('column_id')['stability_score'] if 'stability_score' in atomic_rules_df.columns else pd.Series(dtype=float)
    
    # 按特征类型分组原子规则
    numeric_rules = atomic_rules_df[atomic_rules_df['rule_type_feature'] == 'numeric'].copy()
    categorical_rules = atomic_rules_df[atomic_rules_df['rule_type_feature'] == 'categorical'].copy()
    
    # 第一层：以单特征规则为种子
    candidates = []
    
    # 处理连续特征单规则（v2.0: feature_family, combo_cov_unknown）
    for _, rule_row in numeric_rules.iterrows():
        col_id = rule_row['column_id']
        feature_rule = {
            'column_id': col_id,
            'column_name': rule_row.get('column_name', col_id),
            'type': 'numeric',
            'low': rule_row.get('rule_low', np.nan),
            'high': rule_row.get('rule_high', np.nan),
            'direction': rule_row.get('direction', 'high'),
            'base_cov': _norm_cov(rule_row.get('base_cov')),
            'sub_cov': _norm_cov(rule_row.get('sub_cov')),
            'feature_family': get_feature_family(col_id, getattr(config, 'demographic_family_regex_list', None)),
        }
        div_score = rule_row.get('divergence_score', divergence_scores.get(col_id, 0.0) if col_id in divergence_scores.index else 0.0)
        stab_score = rule_row.get('stability_score', stability_scores.get(col_id, 0.5) if col_id in stability_scores.index else 0.5)
        rule_id = f"rule_{col_id}_1"
        segment_rule = SegmentRule(
            rule_id=rule_id,
            feature_rules=[feature_rule],
            divergence_score=div_score,
            stability_score=stab_score,
            diversity_score=1.0,
            score=div_score
        )
        segment_rule.rule_feature_ids = [col_id]
        segment_rule.anchor_feature = col_id
        segment_rule.rule_signature = _build_rule_signature([feature_rule])
        bc, sc = _norm_cov(rule_row.get('base_cov')), _norm_cov(rule_row.get('sub_cov'))
        if bc is not None and sc is not None:
            segment_rule.combo_non_sub_cov_lb = segment_rule.combo_non_sub_cov_ub = float(bc)
            segment_rule.combo_sub_cov_lb = segment_rule.combo_sub_cov_ub = float(sc)
            lift_val = float(sc) / float(bc) if bc else None
            pi = getattr(config, 'expected_cohort_ratio', None)
            segment_rule.combo_precision_lb = segment_rule.combo_precision_ub = _frechet_precision_from_lift(lift_val, pi)
            segment_rule.combo_cov_unknown = False
            segment_rule.combo_non_sub_cov_ind_est = float(bc)
            segment_rule.combo_sub_cov_ind_est = float(sc)
            segment_rule.combo_non_sub_cov_adj_est = segment_rule.combo_sub_cov_adj_est = None
            segment_rule.combo_non_sub_cov_est = float(bc)
            segment_rule.combo_sub_cov_est = float(sc)
            if pi is not None and 0 < pi < 1:
                segment_rule.combo_all_cov_lb = segment_rule.combo_all_cov_ub = segment_rule.combo_all_cov_est = float(pi * sc + (1 - pi) * bc)
            else:
                segment_rule.combo_all_cov_lb = segment_rule.combo_all_cov_ub = segment_rule.combo_all_cov_est = None
            segment_rule.combo_lift_est = lift_val
            segment_rule.combo_pi_missing = (pi is None or not (0 < pi < 1))
            segment_rule.combo_precision_est = _frechet_precision_from_lift(lift_val, pi) if not segment_rule.combo_pi_missing else None
            segment_rule.fp_rate_est = float(bc) * (1 - pi) if not segment_rule.combo_pi_missing and pi else None
        else:
            segment_rule.combo_cov_unknown = True
            segment_rule.combo_pi_missing = True
        candidates.append(segment_rule)
    
    # 处理离散特征单规则（v2.0: feature_family, combo_cov_unknown）
    for _, rule_row in categorical_rules.iterrows():
        col_id = rule_row['column_id']
        categories_str = rule_row.get('rule_categories', '')
        categories = [c.strip() for c in categories_str.split(',') if c.strip()] if categories_str else []
        feature_rule = {
            'column_id': col_id,
            'column_name': rule_row.get('column_name', col_id),
            'type': 'categorical',
            'categories': categories,
            'base_cov': _norm_cov(rule_row.get('base_cov')),
            'sub_cov': _norm_cov(rule_row.get('sub_cov')),
            'feature_family': get_feature_family(col_id, getattr(config, 'demographic_family_regex_list', None)),
            'direction': None,
        }
        div_score = rule_row.get('divergence_score', divergence_scores.get(col_id, 0.0) if col_id in divergence_scores.index else 0.0)
        stab_score = rule_row.get('stability_score', stability_scores.get(col_id, 0.5) if col_id in stability_scores.index else 0.5)
        rule_id = f"rule_{col_id}_1"
        segment_rule = SegmentRule(
            rule_id=rule_id,
            feature_rules=[feature_rule],
            divergence_score=div_score,
            stability_score=stab_score,
            diversity_score=1.0,
            score=div_score
        )
        segment_rule.rule_feature_ids = [col_id]
        segment_rule.anchor_feature = col_id
        segment_rule.rule_signature = _build_rule_signature([feature_rule])
        bc, sc = _norm_cov(rule_row.get('base_cov')), _norm_cov(rule_row.get('sub_cov'))
        if bc is not None and sc is not None:
            segment_rule.combo_non_sub_cov_lb = segment_rule.combo_non_sub_cov_ub = float(bc)
            segment_rule.combo_sub_cov_lb = segment_rule.combo_sub_cov_ub = float(sc)
            lift_val = float(sc) / float(bc) if bc else None
            pi = getattr(config, 'expected_cohort_ratio', None)
            segment_rule.combo_precision_lb = segment_rule.combo_precision_ub = _frechet_precision_from_lift(lift_val, pi)
            segment_rule.combo_cov_unknown = False
            segment_rule.combo_non_sub_cov_ind_est = float(bc)
            segment_rule.combo_sub_cov_ind_est = float(sc)
            segment_rule.combo_non_sub_cov_adj_est = segment_rule.combo_sub_cov_adj_est = None
            segment_rule.combo_non_sub_cov_est = float(bc)
            segment_rule.combo_sub_cov_est = float(sc)
            if pi is not None and 0 < pi < 1:
                segment_rule.combo_all_cov_lb = segment_rule.combo_all_cov_ub = segment_rule.combo_all_cov_est = float(pi * sc + (1 - pi) * bc)
            else:
                segment_rule.combo_all_cov_lb = segment_rule.combo_all_cov_ub = segment_rule.combo_all_cov_est = None
            segment_rule.combo_lift_est = lift_val
            segment_rule.combo_pi_missing = (pi is None or not (0 < pi < 1))
            segment_rule.combo_precision_est = _frechet_precision_from_lift(lift_val, pi) if not segment_rule.combo_pi_missing else None
            segment_rule.fp_rate_est = float(bc) * (1 - pi) if not segment_rule.combo_pi_missing and pi else None
        else:
            segment_rule.combo_cov_unknown = True
            segment_rule.combo_pi_missing = True
        candidates.append(segment_rule)
    
    logger.info(f"第一层种子规则数: {len(candidates)}")
    
    # 按得分排序，选择Top-K作为初始候选
    candidates.sort(key=lambda x: x.score, reverse=True)
    candidates = candidates[:config.top_k_features]
    
    # 第二层及以后：使用Coverage-free Beam Search扩展
    all_candidates = candidates.copy()  # 保存所有候选（包括单特征）
    
    for depth in range(2, config.max_features_per_segment + 1):
        if len(candidates) == 0:
            break
        
        new_candidates = []
        
        # 根据深度调整权重（深度越深，稳定性权重增加）
        if depth == 2:
            w_div = 0.5
            w_stab = 0.3
            w_divs = 0.2
        elif depth == 3:
            w_div = 0.4
            w_stab = 0.4
            w_divs = 0.2
        else:  # depth >= 4
            w_div = 0.3
            w_stab = 0.5
            w_divs = 0.2
        
        skip_reasons = {'used_columns': 0, 'pair_assoc': 0, 'pruned_high_corr': 0, 'penalized_mid_corr': 0, 'pruned_conflict': 0, 'combo_precision': 0, 'internal_conflict': 0, 'business_group': 0, 'rule_conflicts': 0, 'combo_cov': 0, 'rare_anchor': 0, 'diversity_family': 0, 'pruned_demographic': 0, 'pruned_height': 0, 'pruned_kappa': 0}
        for candidate in candidates:
            # 获取当前候选已使用的特征ID
            used_columns = {fr['column_id'] for fr in candidate.feature_rules}
            
            # 尝试与其他特征的原子规则进行AND组合
            for _, rule_row in atomic_rules_df.iterrows():
                height_as_third = False
                col_id = rule_row['column_id']
                
                # 跳过已使用的特征
                if col_id in used_columns:
                    skip_reasons['used_columns'] += 1
                    continue
                
                # height：禁止作 anchor（第一层）；允许作第 3 特征时施加 height_as_third_penalty
                height_as_third = False
                height_prune_list = getattr(config, 'height_prune_regex_list', None) or []
                if height_prune_list and col_id:
                    col_lower = str(col_id).strip().lower()
                    matched_height = False
                    for pat in height_prune_list:
                        try:
                            if re.search(pat, col_lower, re.IGNORECASE):
                                matched_height = True
                                break
                        except re.error:
                            if pat.lower() in col_lower:
                                matched_height = True
                                break
                    if matched_height:
                        n_feat = len(candidate.feature_rules)
                        if n_feat < 2:
                            skip_reasons['pruned_height'] += 1
                            beam_stats['pruned_height'] += 1
                            logger.debug(f"跳过规则（height 禁止作 anchor/第2 特征）: {col_id}")
                            continue
                        if n_feat == 2:
                            height_as_third = True
                
                # 相关性剪枝（v2.0 统计口径）：abs(assoc) >= high 直接剪枝；mid <= abs(assoc) < high 降权
                soft_penalty_applied = False
                corr_high = getattr(config, 'corr_prune_threshold_high', None) or getattr(config, 'pair_assoc_hard_thr', None) or getattr(config, 'corr_high', 0.85)
                corr_mid = getattr(config, 'corr_penalty_threshold_mid', None) or getattr(config, 'corr_mid', 0.60)
                corr_penalty_weight = getattr(config, 'corr_penalty_weight', None) or getattr(config, 'pair_assoc_soft_penalty', 0.2)
                if pair_assoc_index is not None and getattr(config, 'enable_pair_assoc_pruning', True):
                    skip_by_corr = False
                    for old_fr in candidate.feature_rules:
                        old_col = old_fr.get('column_id')
                        if not old_col:
                            continue
                        strength = pair_assoc_index.get_strength(old_col, col_id)
                        if strength is None:
                            continue
                        abs_strength = abs(float(strength))
                        if abs_strength >= corr_high:
                            beam_stats['pruned_high_corr'] += 1
                            skip_reasons['pruned_high_corr'] += 1
                            skip_by_corr = True
                            break
                        if corr_mid <= abs_strength < corr_high:
                            beam_stats['penalized_mid_corr'] += 1
                            skip_reasons['penalized_mid_corr'] += 1
                            soft_penalty_applied = True
                    if skip_by_corr:
                        continue
                    conflict_mode = getattr(config, 'pair_assoc_conflict_mode', 'off')
                    if conflict_mode == 'light':
                        for old_fr in candidate.feature_rules:
                            old_col = old_fr.get('column_id')
                            if not old_col:
                                continue
                            strength = pair_assoc_index.get_strength(old_col, col_id)
                            if strength is None:
                                continue
                            if float(strength) >= corr_high:
                                ct = pair_assoc_index.get_corr_type(old_col, col_id) or ''
                                sign = pair_assoc_index.get_sign(old_col, col_id)
                                if 'cont_cont' in ct and (sign or 0) < 0:
                                    old_dir = old_fr.get('direction')
                                    new_dir = rule_row.get('direction', 'high')
                                    if old_fr.get('type') == 'numeric' and rule_row.get('rule_type_feature') == 'numeric' and old_dir == new_dir:
                                        beam_stats['pruned_conflict'] += 1
                                        skip_reasons['pruned_conflict'] += 1
                                        skip_by_corr = True
                                        break
                        if skip_by_corr:
                            continue
                else:
                    skip, reason = _pair_assoc_skip(pair_assoc_df, used_columns, col_id, config)
                    if skip:
                        skip_reasons['pair_assoc'] += 1
                        logger.debug(f"跳过规则（字段对关联 {reason}）: {col_id}")
                        continue
                
                # 构建新的特征规则（v2.0: 增加 feature_family, direction 供导出与分层）
                fam = get_feature_family(col_id, getattr(config, 'demographic_family_regex_list', None))
                if rule_row['rule_type_feature'] == 'numeric':
                    new_feature_rule = {
                        'column_id': col_id,
                        'column_name': rule_row.get('column_name', col_id),
                        'type': 'numeric',
                        'low': rule_row.get('rule_low', np.nan),
                        'high': rule_row.get('rule_high', np.nan),
                        'direction': rule_row.get('direction', 'high'),
                        'base_cov': _norm_cov(rule_row.get('base_cov')),
                        'sub_cov': _norm_cov(rule_row.get('sub_cov')),
                        'feature_family': fam,
                    }
                else:  # categorical
                    categories_str = rule_row.get('rule_categories', '')
                    categories = [c.strip() for c in categories_str.split(',') if c.strip()] if categories_str else []
                    new_feature_rule = {
                        'column_id': col_id,
                        'column_name': rule_row.get('column_name', col_id),
                        'type': 'categorical',
                        'categories': categories,
                        'base_cov': _norm_cov(rule_row.get('base_cov')),
                        'sub_cov': _norm_cov(rule_row.get('sub_cov')),
                        'feature_family': fam,
                        'direction': None,
                    }
                
                # 获取新特征的差异评分和稳定性评分
                new_div_score = rule_row.get('divergence_score', divergence_scores.get(col_id, 0.0) if col_id in divergence_scores.index else 0.0)
                new_stab_score = rule_row.get('stability_score', stability_scores.get(col_id, 0.5) if col_id in stability_scores.index else 0.5)
                
                # 计算组合后的平均差异评分和稳定性评分
                combined_div_score = (candidate.divergence_score * len(candidate.feature_rules) + new_div_score) / (len(candidate.feature_rules) + 1)
                combined_stab_score = (candidate.stability_score * len(candidate.feature_rules) + new_stab_score) / (len(candidate.feature_rules) + 1)
                
                # 计算多样性评分（相对于现有候选）
                new_feature_rules = candidate.feature_rules + [new_feature_rule]
                temp_rule = SegmentRule(
                    rule_id="temp",
                    feature_rules=new_feature_rules,
                    divergence_score=combined_div_score,
                    stability_score=combined_stab_score
                )
                diversity_score = calculate_rule_diversity_score(temp_rule, all_candidates)
                
                # 计算综合得分：score = w₁ · mean(S(feature)) + w₂ · mean(stability) + w₃ · diversity
                combined_score = (
                    w_div * combined_div_score +
                    w_stab * combined_stab_score +
                    w_divs * diversity_score
                )
                if soft_penalty_applied:
                    combined_score -= corr_penalty_weight
                
                # 创建新的组合规则
                rule_id = f"rule_{depth}feat_{len(all_candidates) + len(new_candidates)}"
                
                new_candidate = SegmentRule(
                    rule_id=rule_id,
                    feature_rules=new_feature_rules,
                    divergence_score=combined_div_score,
                    stability_score=combined_stab_score,
                    diversity_score=diversity_score,
                    score=combined_score
                )
                new_candidate.rule_feature_ids = sorted([fr['column_id'] for fr in new_feature_rules])
                new_candidate.anchor_feature = (new_candidate.rule_feature_ids[0]) if new_candidate.rule_feature_ids else None
                new_candidate.rule_signature = _build_rule_signature(new_feature_rules)
                # 组合覆盖率与精度：Fréchet 边界 + ind_est + 相关性校正 adj_est（combo_cov_adjust）
                all_frs = new_candidate.feature_rules
                base_covs_norm = [_norm_cov(fr.get('base_cov')) for fr in all_frs]
                sub_covs_norm = [_norm_cov(fr.get('sub_cov')) for fr in all_frs]
                if all(b is not None for b in base_covs_norm) and all(s is not None for s in sub_covs_norm):
                    base_vals = [float(b) for b in base_covs_norm]
                    sub_vals = [float(s) for s in sub_covs_norm]
                    column_ids = [fr.get('column_id') for fr in all_frs]
                    pi = prior_pi if prior_pi is not None else getattr(config, 'expected_cohort_ratio', None)
                    gamma = getattr(config, 'combo_cov_gamma', 1.0)
                    pairwise_alpha = getattr(config, 'combo_pairwise_alpha', 0.5)
                    inflation_cap = getattr(config, 'inflation_cap', 2.0)
                    cov_shrunk_penalty = getattr(config, 'cov_shrunk_missing_penalty', 0.8)
                    min_cov_lb_floor = getattr(config, 'min_cov_lb_floor', 1e-9)
                    precision_ub_cap = getattr(config, 'precision_ub_cap', 0.99)
                    if compute_combo_metrics is not None:
                        m = compute_combo_metrics(
                            base_vals, sub_vals, column_ids, pi, pair_assoc_index,
                            gamma, pairwise_alpha, inflation_cap, cov_shrunk_penalty,
                            min_cov_lb_floor=min_cov_lb_floor, precision_ub_cap=precision_ub_cap,
                        )
                    else:
                        non_sub_lb, non_sub_ub = max(0.0, sum(base_vals) - (len(base_vals) - 1)), min(base_vals)
                        sub_lb, sub_ub = max(0.0, sum(sub_vals) - (len(sub_vals) - 1)), min(sub_vals)
                        ind_non_sub, ind_sub = float(np.prod(base_vals)), float(np.prod(sub_vals))
                        if ind_non_sub > 0 and (non_sub_lb <= 0 or non_sub_lb < min_cov_lb_floor):
                            non_sub_lb = max(min_cov_lb_floor, min(ind_non_sub / inflation_cap, non_sub_ub) if non_sub_ub > 0 else min_cov_lb_floor)
                        if ind_sub > 0 and (sub_lb <= 0 or sub_lb < min_cov_lb_floor):
                            sub_lb = max(min_cov_lb_floor, min(ind_sub / inflation_cap, sub_ub) if sub_ub > 0 else min_cov_lb_floor)
                        lift_lb = sub_lb / non_sub_ub if non_sub_ub > 0 else None
                        lift_ub = sub_ub / non_sub_lb if non_sub_lb > 0 else None
                        lift_est = ind_sub / ind_non_sub if ind_non_sub > 0 else None
                        pi_miss = pi is None or not (0 < pi < 1)
                        _pub = _frechet_precision_from_lift(lift_ub, pi)
                        if _pub is not None and precision_ub_cap is not None:
                            _pub = min(_pub, precision_ub_cap)
                        m = {
                            'non_sub_lb': non_sub_lb, 'non_sub_ub': non_sub_ub, 'sub_lb': sub_lb, 'sub_ub': sub_ub,
                            'all_lb': (pi * sub_lb + (1 - pi) * non_sub_lb) if not pi_miss else None,
                            'all_ub': (pi * sub_ub + (1 - pi) * non_sub_ub) if not pi_miss else None,
                            'all_est': (pi * ind_sub + (1 - pi) * ind_non_sub) if not pi_miss else None,
                            'ind_non_sub': ind_non_sub, 'ind_sub': ind_sub, 'adj_non_sub': None, 'adj_sub': None,
                            'cov_est_non_sub': ind_non_sub, 'cov_est_sub': ind_sub,
                            'lift_lb': lift_lb, 'lift_ub': lift_ub, 'lift_est': lift_est,
                            'precision_lb': _frechet_precision_from_lift(lift_lb, pi),
                            'precision_ub': _pub,
                            'precision_est': _frechet_precision_from_lift(lift_est, pi) if (pi and 0 < pi < 1 and lift_est) else None,
                            'pi_missing': pi_miss,
                        }
                    new_candidate.combo_non_sub_cov_lb = m['non_sub_lb']
                    new_candidate.combo_non_sub_cov_ub = m['non_sub_ub']
                    new_candidate.combo_sub_cov_lb = m['sub_lb']
                    new_candidate.combo_sub_cov_ub = m['sub_ub']
                    new_candidate.combo_all_cov_lb = m.get('all_lb')
                    new_candidate.combo_all_cov_ub = m.get('all_ub')
                    new_candidate.combo_all_cov_est = m.get('all_est')
                    new_candidate.combo_all_cov_ub_shrunk = m.get('all_ub_shrunk')
                    new_candidate.combo_sub_cov_ub_shrunk = m.get('sub_ub_shrunk')
                    _missing = m.get('missing_pair_count')
                    if _missing is not None:
                        new_candidate.missing_pair_count = _missing
                    new_candidate.combo_non_sub_cov_ind_est = m['ind_non_sub']
                    new_candidate.combo_sub_cov_ind_est = m['ind_sub']
                    new_candidate.combo_non_sub_cov_adj_est = m['adj_non_sub']
                    new_candidate.combo_sub_cov_adj_est = m['adj_sub']
                    new_candidate.combo_non_sub_cov_est = m['cov_est_non_sub']
                    new_candidate.combo_sub_cov_est = m['cov_est_sub']
                    new_candidate.combo_lift_est = m['lift_est']
                    new_candidate.combo_precision_lb = m['precision_lb']
                    # precision_ub：先由 shrunk cov 推导；(pi*sub_ub_shrunk)/all_ub_shrunk，再与原有 precision_ub 取 min
                    all_ub_s = m.get('all_ub_shrunk')
                    sub_ub_s = m.get('sub_ub_shrunk')
                    if not m['pi_missing'] and pi is not None and all_ub_s is not None and all_ub_s > 0 and sub_ub_s is not None:
                        precision_ub_from_shrunk = min(1.0, (pi * sub_ub_s) / all_ub_s)
                        new_candidate.combo_precision_ub = min(m['precision_ub'], precision_ub_from_shrunk) if m.get('precision_ub') is not None else precision_ub_from_shrunk
                    else:
                        new_candidate.combo_precision_ub = m['precision_ub']
                    new_candidate.combo_precision_est = m['precision_est']
                    new_candidate.combo_pi_missing = m['pi_missing']
                    new_candidate.combo_cov_unknown = False
                    new_candidate.combo_cov_est_source = "adj" if m.get('adj_non_sub') is not None else "ind"
                    if not m['pi_missing'] and m['cov_est_non_sub'] is not None:
                        new_candidate.fp_rate_est = m['cov_est_non_sub'] * (1 - pi)
                    else:
                        new_candidate.fp_rate_est = None
                    min_combo_precision = getattr(config, 'min_combo_precision', None)
                    if min_combo_precision is not None:
                        if new_candidate.combo_precision_est is not None and new_candidate.combo_precision_est < min_combo_precision:
                            skip_reasons['combo_precision'] += 1
                            logger.debug(f"跳过规则 {rule_id}（precision_est %.4f < min_combo_precision %.4f）", new_candidate.combo_precision_est, min_combo_precision)
                            continue
                        # 仅当无 precision_est 时才用 precision_lb 过滤，避免弱下界过保守导致误杀
                        if new_candidate.combo_precision_est is None and new_candidate.combo_precision_lb is not None and new_candidate.combo_precision_lb < min_combo_precision and m['non_sub_lb'] > 0 and m['sub_lb'] > 0:
                            skip_reasons['combo_precision'] += 1
                            logger.debug(f"跳过规则 {rule_id}（precision_lb %.4f < min_combo_precision %.4f）", new_candidate.combo_precision_lb, min_combo_precision)
                            continue
                    # 相关性驱动 precision_ub 收紧：r_max = max pair corr_strength, shrink = clamp(1 - beta*r_max, 0.5, 1.0)
                    column_ids_combo = [fr.get('column_id') for fr in new_feature_rules]
                    if pair_assoc_index is not None and len(column_ids_combo) >= 2:
                        r_max_val = None
                        for ii in range(len(column_ids_combo)):
                            for jj in range(ii + 1, len(column_ids_combo)):
                                s = pair_assoc_index.get_strength(column_ids_combo[ii], column_ids_combo[jj])
                                if s is not None:
                                    r_max_val = max(r_max_val, float(s)) if r_max_val is not None else float(s)
                        if r_max_val is not None:
                            beta = getattr(config, 'precision_ub_shrink_beta', 0.8)
                            shrink = max(0.5, min(1.0, 1.0 - beta * r_max_val))
                            pub_old = new_candidate.combo_precision_ub
                            pest = new_candidate.combo_precision_est
                            if pub_old is not None and pest is not None:
                                new_candidate.combo_precision_ub = min(pub_old, pest + shrink * (pub_old - pest))
                            new_candidate.precision_ub_shrunk_by_corr = True
                            new_candidate.r_max = r_max_val
                        else:
                            new_candidate.precision_ub_shrunk_by_corr = False
                            new_candidate.r_max = None
                    else:
                        new_candidate.precision_ub_shrunk_by_corr = False
                        new_candidate.r_max = None
                else:
                    new_candidate.combo_non_sub_cov_lb = new_candidate.combo_non_sub_cov_ub = None
                    new_candidate.combo_sub_cov_lb = new_candidate.combo_sub_cov_ub = None
                    new_candidate.combo_all_cov_lb = new_candidate.combo_all_cov_ub = new_candidate.combo_all_cov_est = None
                    new_candidate.combo_precision_lb = new_candidate.combo_precision_ub = None
                    new_candidate.combo_non_sub_cov_est = new_candidate.combo_sub_cov_est = None
                    new_candidate.combo_lift_est = new_candidate.combo_precision_est = new_candidate.fp_rate_est = None
                    new_candidate.combo_non_sub_cov_ind_est = new_candidate.combo_sub_cov_ind_est = None
                    new_candidate.combo_non_sub_cov_adj_est = new_candidate.combo_sub_cov_adj_est = None
                    new_candidate.combo_pi_missing = True
                    new_candidate.combo_cov_unknown = True
                    new_candidate.precision_ub_shrunk_by_corr = False
                    new_candidate.r_max = None
                    new_candidate.combo_cov_est_source = None
                
                # 病态度量/条件数稳定性：基于字段对表构造 R，kappa>kappa_prune 剪枝，否则惩罚并写入 kappa/stability_score/missing_pair_count
                col_ids_for_kappa = [fr.get('column_id') for fr in new_candidate.feature_rules]
                if getattr(config, 'enable_kappa_stability', True) and pair_assoc_index is not None and len(col_ids_for_kappa) >= 2:
                    try:
                        from .kappa_stability import compute_kappa_stability, kappa_to_stability_score, kappa_penalty
                    except ImportError:
                        from kappa_stability import compute_kappa_stability, kappa_to_stability_score, kappa_penalty
                    lambda_diag = getattr(config, 'lambda_diag', 0.02)
                    kappa_val, missing_pairs = compute_kappa_stability(col_ids_for_kappa, pair_assoc_index, lambda_diag)
                    if kappa_val is not None:
                        new_candidate.kappa = kappa_val
                        new_candidate.missing_pair_count = missing_pairs
                        kappa_prune_thr = getattr(config, 'kappa_prune', 100.0)
                        if kappa_val > kappa_prune_thr:
                            skip_reasons['pruned_kappa'] += 1
                            beam_stats['pruned_kappa'] += 1
                            logger.debug(f"跳过规则 {rule_id}（kappa %.2f > %.2f）", kappa_val, kappa_prune_thr)
                            continue
                        new_candidate.stability_score = kappa_to_stability_score(kappa_val)
                        kappa_alpha = getattr(config, 'kappa_penalty_alpha', 0.1)
                        new_candidate.score -= kappa_penalty(kappa_val, kappa_alpha)
                    else:
                        new_candidate.kappa = None
                        new_candidate.missing_pair_count = None
                else:
                    new_candidate.kappa = None
                    new_candidate.missing_pair_count = None
                
                # v2.0: 多样性约束 fee_family≤1, recency_family≤1, identity_family≤2；demographic_family 按 max_rules_per_demographic_family 剪枝
                if getattr(config, 'enable_diversity_family_constraint', True):
                    counts = {'fee_family': 0, 'recency_family': 0, 'identity_family': 0, 'demographic_family': 0}
                    for fr in new_feature_rules:
                        fam = fr.get('feature_family', 'other')
                        if fam in counts:
                            counts[fam] += 1
                    max_demographic = getattr(config, 'max_rules_per_demographic_family', 0)
                    if max_demographic == 0 and counts['demographic_family'] > 0:
                        skip_reasons['pruned_demographic'] += 1
                        beam_stats['pruned_demographic'] += 1
                        logger.debug(f"跳过规则 {rule_id}（demographic_family 限制 0，当前 %d 条）", counts['demographic_family'])
                        continue
                    if max_demographic == 1 and counts['demographic_family'] > 1:
                        skip_reasons['pruned_demographic'] += 1
                        beam_stats['pruned_demographic'] += 1
                        logger.debug(f"跳过规则 {rule_id}（demographic_family 最多 1 条，当前 %d 条）", counts['demographic_family'])
                        continue
                    if counts['fee_family'] > 1 or counts['recency_family'] > 1 or counts['identity_family'] > 2:
                        skip_reasons['diversity_family'] = skip_reasons.get('diversity_family', 0) + 1
                        logger.debug(f"跳过规则 {rule_id}（多样性约束: fee=%d recency=%d identity=%d）", counts['fee_family'], counts['recency_family'], counts['identity_family'])
                        continue
                
                # 检查规则内部冲突
                has_conflict, conflict_msg = check_internal_rule_conflict(new_candidate)
                if has_conflict:
                    skip_reasons['internal_conflict'] += 1
                    logger.debug(f"跳过规则 {rule_id}（内部冲突: {conflict_msg}）")
                    continue
                
                # 检查业务分组约束
                if business_groups is not None:
                    if not check_business_group_constraint(new_candidate, business_groups, max_fields_per_business_group):
                        skip_reasons['business_group'] += 1
                        logger.debug(f"跳过规则 {rule_id}（业务分组约束：同业务组字段过多）")
                        continue
                
                # 检查与现有候选的冲突
                conflicts = check_rule_conflicts(new_candidate, all_candidates)
                if height_as_third:
                    penalty = getattr(config, 'height_as_third_penalty', 0.8)
                    new_candidate.score *= penalty
                    beam_stats['height_as_third_count'] = beam_stats.get('height_as_third_count', 0) + 1
                    logger.debug("height 作为第 3 特征加入并惩罚: rule_id=%s, score*=%.2f", rule_id, penalty)
                if conflicts:
                    skip_reasons['rule_conflicts'] += 1
                    logger.debug(f"跳过规则 {rule_id}（与现有候选冲突: {conflicts}）")
                    continue
                
                # 组合覆盖率与稀有锚点约束：仅当所有 base_cov 均为有效数值时才检查（None/NaN 视为无 cov，不误杀 tail）
                all_frs = new_candidate.feature_rules
                base_covs = [fr.get('base_cov') for fr in all_frs]
                base_covs_norm = [_norm_cov(b) for b in base_covs]
                if all(b is not None for b in base_covs_norm):
                    max_combo_cov = getattr(config, 'max_combo_cov_est', 0.03)
                    cov_est_all = 1.0
                    for b in base_covs_norm:
                        cov_est_all *= float(b)
                    if cov_est_all > max_combo_cov:
                        skip_reasons['combo_cov'] += 1
                        logger.debug(f"跳过规则 {rule_id}（组合覆盖率估计 %.4f > %.4f）", cov_est_all, max_combo_cov)
                        continue
                    # 首层种子数过少时放宽稀有锚点，避免小数据/候选族 base_cov 普遍 0.3 时无法形成任何组合
                    require_rare = getattr(config, 'require_rare_anchor', True) and not (depth == 2 and len(candidates) < 12)
                    if require_rare:
                        rare_anchor_cov = getattr(config, 'rare_anchor_base_cov', 0.10)
                        if not any(float(b) <= rare_anchor_cov for b in base_covs_norm):
                            skip_reasons['rare_anchor'] += 1
                            logger.debug(f"跳过规则 {rule_id}（无稀有锚点：无规则 base_cov <= %.2f）", rare_anchor_cov)
                            continue
                
                new_candidates.append(new_candidate)
        
        # 按得分排序，保留Top beam_size个候选
        new_candidates.sort(key=lambda x: x.score, reverse=True)
        new_candidates = new_candidates[:config.beam_size]
        
        if not new_candidates:
            logger.info(f"第 {depth} 层未生成新候选，停止扩展；跳过原因统计: %s", skip_reasons)
            break
        
        all_candidates.extend(new_candidates)
        candidates = new_candidates
        
        logger.info(f"第 {depth} 层扩展完成，生成 {len(new_candidates)} 个新候选（权重: div={w_div:.2f}, stab={w_stab:.2f}, divs={w_divs:.2f}）；本层跳过原因: %s", skip_reasons)
    
    logger.info(f"Coverage-free Beam Search完成，总候选规则数: {len(all_candidates)}")
    kappas = [getattr(c, "kappa", None) for c in all_candidates if getattr(c, "kappa", None) is not None]
    if kappas:
        logger.info(
            "kappa 分布: min=%.2f max=%.2f mean=%.2f, 带 kappa 的候选数=%d",
            min(kappas), max(kappas), sum(kappas) / len(kappas), len(kappas),
        )
    if beam_stats.get("pruned_kappa", 0) > 0:
        logger.info("病态度量剪枝 pruned_kappa=%d", beam_stats["pruned_kappa"])
    if beam_stats.get("pruned_height", 0) > 0 or beam_stats.get("pruned_demographic", 0) > 0 or beam_stats.get("height_as_third_count", 0) > 0:
        logger.info("pruned_height=%d, pruned_demographic=%d, height_as_third_count=%d", beam_stats.get("pruned_height", 0), beam_stats.get("pruned_demographic", 0), beam_stats.get("height_as_third_count", 0))
    if getattr(config, 'require_exact_k', False):
        k = getattr(config, 'max_features_per_segment', 3)
        all_candidates = [c for c in all_candidates if len(c.feature_rules) == k]
        logger.info(f"require_exact_k=true，仅保留规则数=%d 的候选: %d 条", k, len(all_candidates))
    return all_candidates, beam_stats

