"""
规则组合搜索模块

从原子规则库中搜索多特征组合规则
"""
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

try:
    from .stage2_config import Stage2Config
    from .segment_diversity import calculate_rule_diversity_score
    from .rule_conflict_checker import check_rule_conflicts, check_numeric_conflict, check_categorical_conflict
except ImportError:
    from stage2_config import Stage2Config
    from segment_diversity import calculate_rule_diversity_score
    from rule_conflict_checker import check_rule_conflicts, check_numeric_conflict, check_categorical_conflict

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


@dataclass
class SegmentRule:
    """客群规则数据结构（Coverage-free版本）"""
    rule_id: str
    feature_rules: List[Dict]  # [{"column_id": "age", "type": "numeric", "low": 25, "high": 40, "base_cov", "sub_cov"}, ...]
    divergence_score: float = 0.0  # 平均差异评分
    stability_score: float = 0.0  # 平均稳定性评分
    diversity_score: float = 0.0  # 多样性评分
    score: float = 0.0  # 综合得分，后续会被评分模块更新
    combo_base_cov_est: Optional[float] = None
    combo_sub_cov_est: Optional[float] = None
    combo_lift_est: Optional[float] = None
    combo_precision_est: Optional[float] = None
    fp_rate_est: Optional[float] = None


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
    pair_assoc_df: Optional[pd.DataFrame] = None
) -> List[SegmentRule]:
    """
    Coverage-free Beam Search组合搜索
    
    核心创新：不依赖覆盖率估算，使用差异评分、稳定性评分和多样性评分。
    若提供 pair_assoc_df（字段对关联统计表），则启用相关性剪枝：冗余（高相关不同时选）、冲突（强负相关禁止组合）；表缺失时退化为仅结构约束。
    
    Args:
        atomic_rules_df: 原子规则库DataFrame（包含divergence_score, stability_score）
        numeric_diff_df: 阶段1连续特征差异结果（用于获取差异评分）
        categorical_diff_df: 阶段1离散特征差异结果（用于获取差异评分）
        config: 阶段2配置对象
        divergence_scores: 预计算的差异评分Series（可选）
        stability_scores: 预计算的稳定性评分Series（可选）
        business_groups: 业务分组（可选）
        max_fields_per_business_group: 每业务组最多字段数
        pair_assoc_df: 字段对关联统计表（可选）；列含 column_id_a, column_id_b, corr_strength, corr_type, corr_sign；缺失则不报错、仅做结构约束
    
    Returns:
        候选客群规则列表（SegmentRule对象列表）
    """
    if len(atomic_rules_df) == 0:
        logger.warning("原子规则库为空，无法进行组合搜索")
        return []
    
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
    
    # 处理连续特征单规则
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
        }
        
        # 获取差异评分和稳定性评分
        div_score = rule_row.get('divergence_score', divergence_scores.get(col_id, 0.0) if col_id in divergence_scores.index else 0.0)
        stab_score = rule_row.get('stability_score', stability_scores.get(col_id, 0.5) if col_id in stability_scores.index else 0.5)
        
        rule_id = f"rule_{col_id}_1"
        segment_rule = SegmentRule(
            rule_id=rule_id,
            feature_rules=[feature_rule],
            divergence_score=div_score,
            stability_score=stab_score,
            diversity_score=1.0,  # 单特征规则，多样性最高
            score=div_score  # 初始得分（仅基于差异评分）
        )
        bc, sc = _norm_cov(rule_row.get('base_cov')), _norm_cov(rule_row.get('sub_cov'))
        if bc is not None and sc is not None:
            segment_rule.combo_base_cov_est = float(bc)
            segment_rule.combo_sub_cov_est = float(sc)
            segment_rule.combo_lift_est = float(sc) / float(bc) if bc else None
            pi = getattr(config, 'expected_cohort_ratio', None)
            if pi is not None and 0 < pi < 1 and segment_rule.combo_lift_est is not None:
                denom = pi * segment_rule.combo_lift_est + (1 - pi)
                segment_rule.combo_precision_est = (pi * segment_rule.combo_lift_est) / denom if denom else None
                segment_rule.fp_rate_est = float(bc) * (1 - pi)
            else:
                segment_rule.combo_precision_est = None
                segment_rule.fp_rate_est = None
        candidates.append(segment_rule)
    
    # 处理离散特征单规则
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
        }
        
        # 获取差异评分和稳定性评分
        div_score = rule_row.get('divergence_score', divergence_scores.get(col_id, 0.0) if col_id in divergence_scores.index else 0.0)
        stab_score = rule_row.get('stability_score', stability_scores.get(col_id, 0.5) if col_id in stability_scores.index else 0.5)
        
        rule_id = f"rule_{col_id}_1"
        segment_rule = SegmentRule(
            rule_id=rule_id,
            feature_rules=[feature_rule],
            divergence_score=div_score,
            stability_score=stab_score,
            diversity_score=1.0,  # 单特征规则，多样性最高
            score=div_score  # 初始得分
        )
        bc, sc = _norm_cov(rule_row.get('base_cov')), _norm_cov(rule_row.get('sub_cov'))
        if bc is not None and sc is not None:
            segment_rule.combo_base_cov_est = float(bc)
            segment_rule.combo_sub_cov_est = float(sc)
            segment_rule.combo_lift_est = float(sc) / float(bc) if bc else None
            pi = getattr(config, 'expected_cohort_ratio', None)
            if pi is not None and 0 < pi < 1 and segment_rule.combo_lift_est is not None:
                denom = pi * segment_rule.combo_lift_est + (1 - pi)
                segment_rule.combo_precision_est = (pi * segment_rule.combo_lift_est) / denom if denom else None
                segment_rule.fp_rate_est = float(bc) * (1 - pi)
            else:
                segment_rule.combo_precision_est = None
                segment_rule.fp_rate_est = None
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
        
        skip_reasons = {'used_columns': 0, 'pair_assoc': 0, 'combo_precision': 0, 'internal_conflict': 0, 'business_group': 0, 'rule_conflicts': 0, 'combo_cov': 0, 'rare_anchor': 0}
        for candidate in candidates:
            # 获取当前候选已使用的特征ID
            used_columns = {fr['column_id'] for fr in candidate.feature_rules}
            
            # 尝试与其他特征的原子规则进行AND组合
            for _, rule_row in atomic_rules_df.iterrows():
                col_id = rule_row['column_id']
                
                # 跳过已使用的特征
                if col_id in used_columns:
                    skip_reasons['used_columns'] += 1
                    continue
                
                # 可选：基于字段对关联表的冗余/冲突剪枝（表缺失时不做；阈值来自 config.pair_*）
                skip, reason = _pair_assoc_skip(pair_assoc_df, used_columns, col_id, config)
                if skip:
                    skip_reasons['pair_assoc'] += 1
                    logger.debug(f"跳过规则（字段对关联 {reason}）: {col_id}")
                    continue
                
                # 构建新的特征规则
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
                
                # 组合覆盖率与 combo_precision / fp 计算（用于约束与导出）
                all_frs = new_candidate.feature_rules
                base_covs = [fr.get('base_cov') for fr in all_frs]
                sub_covs = [fr.get('sub_cov') for fr in all_frs]
                base_covs_norm = [_norm_cov(b) for b in base_covs]
                sub_covs_norm = [_norm_cov(s) for s in sub_covs]
                if all(b is not None for b in base_covs_norm) and all(s is not None for s in sub_covs_norm):
                    combo_base_cov_est = 1.0
                    for b in base_covs_norm:
                        combo_base_cov_est *= float(b)
                    combo_sub_cov_est = 1.0
                    for s in sub_covs_norm:
                        combo_sub_cov_est *= float(s)
                    combo_lift_est = combo_sub_cov_est / combo_base_cov_est if combo_base_cov_est and combo_base_cov_est > 0 else None
                    pi = getattr(config, 'expected_cohort_ratio', None)
                    if pi is not None and 0 < pi < 1 and combo_lift_est is not None:
                        denom = pi * combo_lift_est + (1 - pi)
                        combo_precision_est = (pi * combo_lift_est) / denom if denom else None
                        fp_rate_est = combo_base_cov_est * (1 - pi)
                    else:
                        combo_precision_est = None
                        fp_rate_est = None
                    new_candidate.combo_base_cov_est = combo_base_cov_est
                    new_candidate.combo_sub_cov_est = combo_sub_cov_est
                    new_candidate.combo_lift_est = combo_lift_est
                    new_candidate.combo_precision_est = combo_precision_est
                    new_candidate.fp_rate_est = fp_rate_est
                    min_combo_precision = getattr(config, 'min_combo_precision', None)
                    if min_combo_precision is not None and combo_precision_est is not None and combo_precision_est < min_combo_precision:
                        skip_reasons['combo_precision'] += 1
                        logger.debug(f"跳过规则 {rule_id}（组合精度 %.4f < min_combo_precision %.4f）", combo_precision_est, min_combo_precision)
                        continue
                else:
                    new_candidate.combo_base_cov_est = None
                    new_candidate.combo_sub_cov_est = None
                    new_candidate.combo_lift_est = None
                    new_candidate.combo_precision_est = None
                    new_candidate.fp_rate_est = None
                
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
                    if getattr(config, 'require_rare_anchor', True):
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
    if getattr(config, 'require_exact_k', False):
        k = getattr(config, 'max_features_per_segment', 3)
        all_candidates = [c for c in all_candidates if len(c.feature_rules) == k]
        logger.info(f"require_exact_k=true，仅保留规则数=%d 的候选: %d 条", k, len(all_candidates))
    return all_candidates

