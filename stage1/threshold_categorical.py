"""
离散特征推荐类别集合模块

基于频率差异，为离散特征推荐最优类别集合。
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Any

# 配置日志
logger = logging.getLogger(__name__)


def _coverage_for_categories(
    subset: pd.DataFrame,
    categories: List[Any],
    ratio_col: str = 'val_ratio',
) -> float:
    """给定类别列表，在 subset（index 含 val_enum）上求覆盖率之和。"""
    if not categories or subset.empty:
        return 0.0
    if 'stat_date' in subset.index.names or 'column_id' in subset.index.names:
        sub = subset.reset_index(level=[n for n in subset.index.names if n in ('stat_date', 'column_id')], drop=True)
    else:
        sub = subset
    if ratio_col not in sub.columns:
        return 0.0
    total = 0.0
    for v in categories:
        if v in sub.index:
            total += float(sub.loc[v, ratio_col])
    return total


def build_candidate_category_family(
    full_subset: pd.DataFrame,
    cohort_subset: pd.DataFrame,
    min_delta: float = 0.01,
    bad_tokens: Optional[set] = None,
    max_category_combos: int = 3,
) -> List[Dict[str, Any]]:
    """
    生成候选类别族：单类别、Top2 组合、Top3 组合（至多 max_category_combos 档）；
    每档含 base_cov, sub_cov, lift, precision（用 full/cohort 统计）。
    """
    full_subset_reset = full_subset.reset_index(level=['stat_date', 'column_id'], drop=True) if 'stat_date' in full_subset.index.names else full_subset
    cohort_subset_reset = cohort_subset.reset_index(level=['stat_date', 'column_id'], drop=True) if 'stat_date' in cohort_subset.index.names else cohort_subset
    merged = pd.merge(
        full_subset_reset[['val_ratio']].rename(columns={'val_ratio': 'val_ratio_full'}),
        cohort_subset_reset[['val_ratio']].rename(columns={'val_ratio': 'val_ratio_base'}),
        left_index=True, right_index=True, how='outer'
    )
    merged['val_ratio_full'] = merged['val_ratio_full'].fillna(0.0)
    merged['val_ratio_base'] = merged['val_ratio_base'].fillna(0.0)
    merged['delta'] = merged['val_ratio_base'] - merged['val_ratio_full']
    cand = merged[merged['delta'] > min_delta].sort_values('delta', ascending=False)
    if bad_tokens is None:
        bad_tokens = {"__NULL__", "__OTHER__"}
    cand = cand[~cand.index.isin(bad_tokens)]
    if len(cand) == 0:
        return []

    def base_cov(cats: List[Any]) -> float:
        return sum(cand.loc[v, 'val_ratio_full'] for v in cats if v in cand.index)
    def sub_cov(cats: List[Any]) -> float:
        return sum(cand.loc[v, 'val_ratio_base'] for v in cats if v in cand.index)

    family: List[Dict[str, Any]] = []
    top_vals = list(cand.index[: max_category_combos])
    for k in range(1, min(len(top_vals) + 1, max_category_combos + 1)):
        cats = top_vals[:k]
        bc = base_cov(cats)
        sc = sub_cov(cats)
        if bc <= 0:
            continue
        lift = sc / bc
        precision = sc  # 无 prior 时用 sub_cov 作为 precision 代理
        family.append({
            'categories': cats,
            'base_cov': bc,
            'sub_cov': sc,
            'lift': lift,
            'precision': precision,
        })
    return family


def _dedupe_category_candidates_by_overlap(
    candidates: List[Dict[str, Any]],
    full_subset: pd.DataFrame,
    overlap_threshold: float,
) -> List[Dict[str, Any]]:
    """覆盖率重叠 > overlap_threshold 时只保留 precision 更优的一条。重叠 = 交集在 full 上的覆盖率 / min(base_cov1, base_cov2)。"""
    if not candidates or overlap_threshold <= 0:
        return candidates
    if full_subset.empty:
        return candidates
    if 'stat_date' in full_subset.index.names or 'column_id' in full_subset.index.names:
        full_flat = full_subset.reset_index(level=[n for n in full_subset.index.names if n in ('stat_date', 'column_id')], drop=True)
    else:
        full_flat = full_subset
    if 'val_ratio' not in full_flat.columns:
        return candidates

    def cov_intersection(cats1: List, cats2: List) -> float:
        inter = set(cats1) & set(cats2)
        return sum(float(full_flat.loc[v, 'val_ratio']) for v in inter if v in full_flat.index)

    sorted_c = sorted(candidates, key=lambda x: (-(x.get('precision') or 0), -(x.get('lift') or 0)))
    keep: List[Dict[str, Any]] = []
    for c in sorted_c:
        overlap_any = False
        bc = c.get('base_cov') or 0
        for k in keep:
            bk = k.get('base_cov') or 0
            denom = min(bc, bk)
            if denom <= 0:
                continue
            inter_cov = cov_intersection(c.get('categories', []), k.get('categories', []))
            if inter_cov / denom >= overlap_threshold:
                overlap_any = True
                break
        if not overlap_any:
            keep.append(c)
    return keep


def recommend_categorical_threshold(
    full_subset: pd.DataFrame,
    cohort_subset: pd.DataFrame,
    min_delta: float = 0.01,
    min_cov: float = 0.1,
    min_increment: float = 0.01,
    bad_tokens: Optional[set] = None
) -> Optional[Dict]:
    """
    为单个离散特征推荐类别集合
    
    基于频率差异，选择圈定明显更偏好的类别，贪心构建类别集合直到满足覆盖率阈值。
    
    Args:
        full_subset: 全量数据子集，index为val_enum
        cohort_subset: 圈定数据子集，index为val_enum
        min_delta: 最小差异阈值（默认0.01，放宽以增加推荐）
        min_cov: 最小覆盖率阈值（默认0.1，放宽以增加推荐）
        min_increment: 边际增量阈值（默认0.01），每加一个类别至少提升的覆盖率
        bad_tokens: 需要过滤的类别集合（默认过滤 "__NULL__", "__OTHER__"）
    
    Returns:
        推荐结果字典，包含：
        - rec_categories: 推荐类别列表
        - cohort_coverage: 圈定中命中率
        - full_coverage: 全量中命中率
        - lift: 覆盖增幅
        - full_hit_count: 全量命中人数（估算）
        - cohort_hit_count: 圈定命中人数（估算）
        - rec_category_count: 推荐类别个数
        如果无法生成推荐，返回None
    """
    # 重置索引，只保留val_enum作为索引
    if 'stat_date' in full_subset.index.names:
        full_subset_reset = full_subset.reset_index(level=['stat_date', 'column_id'], drop=True)
    else:
        full_subset_reset = full_subset
    
    if 'stat_date' in cohort_subset.index.names:
        cohort_subset_reset = cohort_subset.reset_index(level=['stat_date', 'column_id'], drop=True)
    else:
        cohort_subset_reset = cohort_subset
    
    # 合并数据（复用compute_categorical_diffs的逻辑）
    merged = pd.merge(
        full_subset_reset[['val_ratio', 'column_name', 'entropy', 'gini']],
        cohort_subset_reset[['val_ratio', 'column_name', 'entropy', 'gini']],
        left_index=True,
        right_index=True,
        how='outer',
        suffixes=('_full', '_base')
    )
    
    # 缺失的val_ratio填0
    merged['val_ratio_full'] = merged['val_ratio_full'].fillna(0.0)
    merged['val_ratio_base'] = merged['val_ratio_base'].fillna(0.0)
    
    # 计算delta
    merged['delta'] = merged['val_ratio_base'] - merged['val_ratio_full']
    
    # 只保留圈定更偏好的类别（delta > min_delta）
    cand = merged[merged['delta'] > min_delta].sort_values('delta', ascending=False)
    
    # 过滤"NULL / OTHER"这类类别
    if bad_tokens is None:
        bad_tokens = {"__NULL__", "__OTHER__"}
    
    cand = cand[~cand.index.isin(bad_tokens)]
    
    if len(cand) == 0:
        return None
    
    # 贪心构建推荐类别集合（注意边际增量阈值）
    rec_cats = []
    cov_cohort = 0.0
    cov_full = 0.0
    
    for val_enum, row in cand.iterrows():
        # 计算添加该类别后的新覆盖率
        new_cov_cohort = cov_cohort + row['val_ratio_base']
        increment = new_cov_cohort - cov_cohort
        
        # 边际增量阈值：每加一个类别至少提升 min_increment 的覆盖率
        if increment < min_increment:
            continue  # 跳过增益太小的类别
        
        rec_cats.append(val_enum)
        cov_cohort = new_cov_cohort
        cov_full += row['val_ratio_full']
        
        # 当圈定覆盖率达到阈值时停止
        if cov_cohort >= min_cov:
            break
    
    if not rec_cats:
        return None
    
    # 计算lift
    lift = cov_cohort / max(cov_full, 1e-6)
    
    # 获取总人数（用于估算命中人数）
    full_total_count = full_subset['total_count'].iloc[0] if 'total_count' in full_subset.columns and len(full_subset) > 0 else 0
    cohort_total_count = cohort_subset['total_count'].iloc[0] if 'total_count' in cohort_subset.columns and len(cohort_subset) > 0 else 0
    
    # 估算命中人数
    full_hit_count = cov_full * full_total_count if full_total_count > 0 else 0
    cohort_hit_count = cov_cohort * cohort_total_count if cohort_total_count > 0 else 0
    
    return {
        'rec_categories': rec_cats,
        'cohort_coverage': cov_cohort,
        'full_coverage': cov_full,
        'lift': lift,
        'full_hit_count': full_hit_count,
        'cohort_hit_count': cohort_hit_count,
        'rec_category_count': len(rec_cats)
    }


def recommend_categorical_thresholds(
    categorical_diff_df: pd.DataFrame,
    full_categorical_df: pd.DataFrame,
    cohort_categorical_df: pd.DataFrame,
    min_delta: float = 0.01,
    min_cov: float = 0.1,
    min_increment: float = 0.01,
    bad_tokens: Optional[set] = None,
    *,
    output_candidate_family: bool = True,
    max_category_combos: int = 3,
    dedup_overlap_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    为离散特征推荐类别集合
    
    对每个字段，基于频率差异选择推荐类别，贪心构建类别集合直到满足覆盖率阈值。
    
    Args:
        categorical_diff_df: 差异计算结果DataFrame，index为column_id
        full_categorical_df: 全量统计DataFrame，index=['stat_date', 'column_id', 'val_enum']
        cohort_categorical_df: 圈定统计DataFrame，index=['stat_date', 'column_id', 'val_enum']
        min_delta: 最小差异阈值（默认0.01）
        min_cov: 最小覆盖率阈值（默认0.1）
        min_increment: 边际增量阈值（默认0.01），每加一个类别至少提升的覆盖率
        bad_tokens: 需要过滤的类别集合（默认过滤 "__NULL__", "__OTHER__"）
    
    Returns:
        DataFrame，index为column_id，包含推荐类别集合信息：
        - rec_categories: 推荐类别集合（逗号分隔）
        - cohort_coverage: 圈定中命中率
        - full_coverage: 全量中命中率
        - lift: 覆盖增幅
        - full_hit_count: 全量命中人数（估算）
        - cohort_hit_count: 圈定命中人数（估算）
        - rec_category_count: 推荐类别个数
        - rule_desc: 规则描述
    """
    logger.info("开始推荐离散特征类别集合...")

    total_candidate_categories = 0

    # 获取共同的(stat_date, column_id)组合
    full_keys = set(zip(
        full_categorical_df.index.get_level_values('stat_date'),
        full_categorical_df.index.get_level_values('column_id')
    ))
    cohort_keys = set(zip(
        cohort_categorical_df.index.get_level_values('stat_date'),
        cohort_categorical_df.index.get_level_values('column_id')
    ))
    common_keys = full_keys & cohort_keys
    
    if not common_keys:
        logger.warning("没有共同的(stat_date, column_id)组合，无法生成推荐")
        return pd.DataFrame()
    
    # 存储推荐结果
    recommendations = []
    
    # 针对每个(stat_date, column_id)生成推荐
    for stat_date, column_id in sorted(common_keys):
        try:
            # 检查是否在差异结果中
            if column_id not in categorical_diff_df.index:
                continue
            
            diff_row = categorical_diff_df.loc[column_id]
            
            # 获取该字段的全量和圈定数据
            full_mask = (
                (full_categorical_df.index.get_level_values('stat_date') == stat_date) &
                (full_categorical_df.index.get_level_values('column_id') == column_id)
            )
            cohort_mask = (
                (cohort_categorical_df.index.get_level_values('stat_date') == stat_date) &
                (cohort_categorical_df.index.get_level_values('column_id') == column_id)
            )
            
            full_subset = full_categorical_df[full_mask]
            cohort_subset = cohort_categorical_df[cohort_mask]
            
            if len(full_subset) == 0 or len(cohort_subset) == 0:
                continue

            candidate_category_family: Optional[List[Dict[str, Any]]] = None
            if output_candidate_family:
                family_raw = build_candidate_category_family(
                    full_subset, cohort_subset, min_delta, bad_tokens, max_category_combos
                )
                candidate_category_family = _dedupe_category_candidates_by_overlap(
                    family_raw, full_subset, dedup_overlap_threshold
                )
                total_candidate_categories += len(candidate_category_family)
                logger.debug(
                    "[stage1] column_id=%s candidate_category_count=%d after_dedup=%d",
                    column_id, len(family_raw), len(candidate_category_family),
                )

            # 生成推荐（保留 recommended_categories 兼容）
            recommendation = recommend_categorical_threshold(
                full_subset, cohort_subset, min_delta, min_cov, min_increment, bad_tokens
            )

            if recommendation is None:
                column_name = diff_row.get('column_name', column_id)
                row_dict = {
                    'column_id': column_id,
                    'rec_categories': None,
                    'cohort_coverage': np.nan,
                    'full_coverage': np.nan,
                    'lift': np.nan,
                    'full_hit_count': np.nan,
                    'cohort_hit_count': np.nan,
                    'rec_category_count': 0,
                    'rule_desc': None
                }
                if candidate_category_family is not None:
                    row_dict['candidate_category_family'] = candidate_category_family
                recommendations.append(row_dict)
            else:
                column_name = diff_row.get('column_name', column_id)
                rec_cats_str = ",".join(map(str, recommendation['rec_categories']))
                rule_desc = f"{column_name} in {{{rec_cats_str}}}"
                row_dict = {
                    'column_id': column_id,
                    'rec_categories': rec_cats_str,
                    'cohort_coverage': recommendation['cohort_coverage'],
                    'full_coverage': recommendation['full_coverage'],
                    'lift': recommendation['lift'],
                    'full_hit_count': recommendation['full_hit_count'],
                    'cohort_hit_count': recommendation['cohort_hit_count'],
                    'rec_category_count': recommendation['rec_category_count'],
                    'rule_desc': rule_desc
                }
                if candidate_category_family is not None:
                    row_dict['candidate_category_family'] = candidate_category_family
                recommendations.append(row_dict)
                
        except Exception as e:
            logger.warning(f"处理字段 (stat_date={stat_date}, column_id={column_id}) 时出错: {e}")
            continue
    
    if not recommendations:
        logger.warning("未生成任何推荐类别集合")
        return pd.DataFrame()
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(recommendations)
    result_df = result_df.set_index('column_id')
    if output_candidate_family and total_candidate_categories > 0:
        logger.info("[stage1] total_candidate_categories=%d", total_candidate_categories)
    logger.info(f"类别集合推荐完成，共 {len(result_df)} 个字段")

    return result_df

