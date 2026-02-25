"""
离散特征差异度计算模块

计算全量客群vs圈定客群的离散特征差异，包括频率差异、最大差异、Top差异类别等指标。
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

# 配置日志
logger = logging.getLogger(__name__)


def compute_categorical_diffs(
    full_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    top_k: int = 3
) -> pd.DataFrame:
    """
    计算离散型特征的差异度指标
    
    针对每个(stat_date, column_id)，以val_enum为键对齐全量和圈定的分布，
    计算频率差异并聚合为字段级指标。
    
    Args:
        full_df: 全量离散特征统计DataFrame，index=['stat_date', 'column_id', 'val_enum']
        cohort_df: 圈定离散特征统计DataFrame，index=['stat_date', 'column_id', 'val_enum']
        top_k: 返回前k个差异最大的类别（默认3）
    
    Returns:
        DataFrame，index为column_id，包含以下列：
        - column_id: 字段ID
        - column_name: 字段名称（取cohort或full中某一行的column_name）
        - stat_date: 统计日期
        - sum_abs_diff: 绝对频率差异之和
        - max_abs_diff: 最大绝对频率差异
        - top_diff_categories: 前top_k个差异最大的类别，格式"类别1(+0.25); 类别2(+0.12); 类别3(-0.08)"
        - recommended_categories: 推荐类别集合（圈定>全量的类别，逗号分隔）
        - entropy_diff: 熵差异（Δimb 分量，圈定减全量）
        - gini_diff: Gini 差异（Δimb 分量，圈定减全量）
        - diff_score: 综合差异分数（考虑样本量权重）
    
    Note:
        - 对缺少某字段的情形做鲁棒处理（跳过并记录warning）
        - 默认假设full与cohort是同一stat_date
        - 缺失的val_ratio填0
        - diff_score会乘以圈定客群占比，让"覆盖多+差异大"的标签更靠前
    
    Example:
        >>> full_df = load_categorical_stats("categorical_stats_full_202510.csv")
        >>> cohort_df = load_categorical_stats("categorical_stats_PRODUCT_A_202510.csv")
        >>> result = compute_categorical_diffs(full_df, cohort_df, top_k=5)
        >>> print(result.head())
    """
    # 复制DataFrame避免修改原始数据
    full_df = full_df.copy()
    cohort_df = cohort_df.copy()
    
    logger.info(f"开始计算离散特征差异，全量记录数: {len(full_df)}, 圈定记录数: {len(cohort_df)}")
    
    # 获取共同的(stat_date, column_id)组合
    full_keys = set(zip(
        full_df.index.get_level_values('stat_date'),
        full_df.index.get_level_values('column_id')
    ))
    cohort_keys = set(zip(
        cohort_df.index.get_level_values('stat_date'),
        cohort_df.index.get_level_values('column_id')
    ))
    common_keys = full_keys & cohort_keys
    only_full = full_keys - cohort_keys
    only_cohort = cohort_keys - full_keys
    
    if only_full:
        logger.warning(f"以下(stat_date, column_id)只在全量中出现，将被跳过: {len(only_full)}个")
    if only_cohort:
        logger.warning(f"以下(stat_date, column_id)只在圈定中出现，将被跳过: {len(only_cohort)}个")
    
    if not common_keys:
        logger.error("没有共同的(stat_date, column_id)组合，无法计算差异")
        return pd.DataFrame()
    
    logger.info(f"共同(stat_date, column_id)组合数: {len(common_keys)}")
    
    # 存储结果
    results = []
    
    # 针对每个(stat_date, column_id)计算差异
    for stat_date, column_id in sorted(common_keys):
        try:
            # 获取该字段的全量和圈定数据
            # 使用布尔索引筛选，因为索引是['stat_date', 'column_id', 'val_enum']
            full_mask = (
                (full_df.index.get_level_values('stat_date') == stat_date) &
                (full_df.index.get_level_values('column_id') == column_id)
            )
            cohort_mask = (
                (cohort_df.index.get_level_values('stat_date') == stat_date) &
                (cohort_df.index.get_level_values('column_id') == column_id)
            )
            
            full_subset = full_df[full_mask]
            cohort_subset = cohort_df[cohort_mask]
            
            if len(full_subset) == 0:
                logger.warning(f"全量数据中未找到 (stat_date={stat_date}, column_id={column_id})")
                continue
            
            if len(cohort_subset) == 0:
                logger.warning(f"圈定数据中未找到 (stat_date={stat_date}, column_id={column_id})")
                continue
            
            # 重置索引，只保留val_enum作为索引
            full_subset_reset = full_subset.reset_index(level=['stat_date', 'column_id'], drop=True)
            cohort_subset_reset = cohort_subset.reset_index(level=['stat_date', 'column_id'], drop=True)
            
            # 合并数据
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
            
            # 计算频率差异
            ratio_diff = merged['val_ratio_base'] - merged['val_ratio_full']
            abs_ratio_diff = np.abs(ratio_diff)
            
            # 聚合字段级指标
            sum_abs_diff = abs_ratio_diff.sum()
            max_abs_diff = abs_ratio_diff.max()
            
            # 获取前top_k个差异最大的类别
            top_diff_indices = abs_ratio_diff.nlargest(top_k).index
            top_diff_categories = []
            for idx in top_diff_indices:
                val_enum = idx  # val_enum是索引
                diff_val = ratio_diff.loc[idx]
                sign = '+' if diff_val >= 0 else ''
                top_diff_categories.append(f"{val_enum}({sign}{diff_val:.4f})")
            
            top_diff_categories_str = "; ".join(top_diff_categories) if top_diff_categories else ""
            
            # 推荐类别集合（圈定>全量的类别）
            good = merged[ratio_diff > 0]  # 圈定>全量
            recommended = good.index.tolist()
            recommended_str = ",".join(map(str, recommended)) if recommended else ""
            
            # 计算entropy差异（从第一行获取，因为entropy是字段级指标）
            entropy_full = merged['entropy_full'].iloc[0] if len(merged) > 0 and not pd.isna(merged['entropy_full'].iloc[0]) else 0.0
            entropy_base = merged['entropy_base'].iloc[0] if len(merged) > 0 and not pd.isna(merged['entropy_base'].iloc[0]) else 0.0
            entropy_diff = entropy_base - entropy_full
            
            # 计算 Gini 差异（Δimb，从第一行获取，字段级指标）
            gini_full = merged['gini_full'].iloc[0] if len(merged) > 0 and 'gini_full' in merged.columns and not pd.isna(merged['gini_full'].iloc[0]) else np.nan
            gini_base = merged['gini_base'].iloc[0] if len(merged) > 0 and 'gini_base' in merged.columns and not pd.isna(merged['gini_base'].iloc[0]) else np.nan
            gini_diff = (gini_base - gini_full) if not (np.isnan(gini_base) or np.isnan(gini_full)) else np.nan
            
            # 计算综合差异分数（考虑总体样本量）
            # 获取样本量（从原始subset获取，因为total_count是字段级指标）
            cohort_size = cohort_subset['total_count'].iloc[0] if 'total_count' in cohort_subset.columns and len(cohort_subset) > 0 else 0
            full_size = full_subset['total_count'].iloc[0] if 'total_count' in full_subset.columns and len(full_subset) > 0 else 1
            weight = cohort_size / max(full_size, 1)  # 圈定客群占比
            
            # diff_score = (sum_abs_diff + 0.5 * max_abs_diff) * weight
            # 让"覆盖多 + 差异大"的标签更靠前
            diff_score = (sum_abs_diff + 0.5 * max_abs_diff) * weight
            
            # 获取column_name（优先取cohort的，若为空取full的）
            column_name = (
                cohort_subset_reset['column_name'].iloc[0]
                if len(cohort_subset_reset) > 0 and 'column_name' in cohort_subset_reset.columns
                else (
                    full_subset_reset['column_name'].iloc[0]
                    if len(full_subset_reset) > 0 and 'column_name' in full_subset_reset.columns
                    else ''
                )
            )
            
            # 添加到结果列表
            results.append({
                'column_id': column_id,
                'column_name': column_name,
                'stat_date': stat_date,
                'sum_abs_diff': sum_abs_diff,
                'max_abs_diff': max_abs_diff,
                'top_diff_categories': top_diff_categories_str,
                'recommended_categories': recommended_str,
                'entropy_diff': entropy_diff,
                'gini_diff': gini_diff,
                'diff_score': diff_score
            })
            
        except Exception as e:
            logger.warning(f"处理字段 (stat_date={stat_date}, column_id={column_id}) 时出错: {e}")
    
    # 构建结果DataFrame
    if not results:
        logger.warning("未生成任何差异计算结果")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index('column_id')
    
    # 按diff_score降序排序
    result_df = result_df.sort_values('diff_score', ascending=False)
    
    logger.info(f"差异计算完成，共 {len(result_df)} 个字段")
    
    return result_df