"""
离散特征推荐类别集合模块

基于频率差异，为离散特征推荐最优类别集合。
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List

# 配置日志
logger = logging.getLogger(__name__)


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
    bad_tokens: Optional[set] = None
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
            
            # 生成推荐
            recommendation = recommend_categorical_threshold(
                full_subset, cohort_subset, min_delta, min_cov, min_increment, bad_tokens
            )
            
            if recommendation is None:
                # 没有找到合适的类别集合
                column_name = diff_row.get('column_name', column_id)
                recommendations.append({
                    'column_id': column_id,
                    'rec_categories': None,
                    'cohort_coverage': np.nan,
                    'full_coverage': np.nan,
                    'lift': np.nan,
                    'full_hit_count': np.nan,
                    'cohort_hit_count': np.nan,
                    'rec_category_count': 0,
                    'rule_desc': None
                })
            else:
                # 生成规则描述
                column_name = diff_row.get('column_name', column_id)
                rec_cats_str = ",".join(map(str, recommendation['rec_categories']))
                rule_desc = f"{column_name} in {{{rec_cats_str}}}"
                
                recommendations.append({
                    'column_id': column_id,
                    'rec_categories': rec_cats_str,
                    'cohort_coverage': recommendation['cohort_coverage'],
                    'full_coverage': recommendation['full_coverage'],
                    'lift': recommendation['lift'],
                    'full_hit_count': recommendation['full_hit_count'],
                    'cohort_hit_count': recommendation['cohort_hit_count'],
                    'rec_category_count': recommendation['rec_category_count'],
                    'rule_desc': rule_desc
                })
                
        except Exception as e:
            logger.warning(f"处理字段 (stat_date={stat_date}, column_id={column_id}) 时出错: {e}")
            continue
    
    if not recommendations:
        logger.warning("未生成任何推荐类别集合")
        return pd.DataFrame()
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(recommendations)
    result_df = result_df.set_index('column_id')
    
    logger.info(f"类别集合推荐完成，共 {len(result_df)} 个字段")
    
    return result_df

