"""
连续特征差异度计算模块

计算全量客群vs圈定客群的连续特征差异，包括均值差异、效应量、分位数差异等指标。
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

# 配置日志
logger = logging.getLogger(__name__)


def compute_numeric_diffs(
    full_df: pd.DataFrame,
    cohort_df: pd.DataFrame
) -> pd.DataFrame:
    """
    计算连续型特征的差异度指标
    
    通过index对齐full_df与cohort_df，匹配相同stat_date和column_id的记录，
    计算多种差异指标并生成综合差异分数。
    
    Args:
        full_df: 全量连续特征统计DataFrame，index=['stat_date', 'column_id']
        cohort_df: 圈定连续特征统计DataFrame，index=['stat_date', 'column_id']
    
    Returns:
        DataFrame，index为column_id，按diff_score降序排序，包含以下列：
        - column_id: 字段ID
        - column_name: 字段名称（优先取cohort_df的，缺失时回填full_df的）
        - stat_date: 统计日期
        - mean_full, mean_base: 全量和圈定的均值
        - mean_diff: 均值差异
        - mean_diff_ratio: 均值差异比例
        - effect_size: 效应量（Cohen's d）
        - delta_median: 中位数差异
        - delta_p95: P95分位数差异
        - delta_IQR: 四分位距差异
        - delta_CV: 变异系数差异（Δdisp 分量，圈定减全量）
        - diff_score: 综合差异分数
        - is_significant: 是否显著差异（abs(effect_size) >= 0.2）
    
    Note:
        - 对只在full或cohort出现的字段，会记录warning并跳过
        - 不修改输入DataFrame（内部使用copy）
    
    Example:
        >>> full_df = load_numeric_stats("numeric_stats_full_202510.csv")
        >>> cohort_df = load_numeric_stats("numeric_stats_PRODUCT_A_202510.csv")
        >>> result = compute_numeric_diffs(full_df, cohort_df)
        >>> print(result.head())
    """
    # 复制DataFrame避免修改原始数据
    full_df = full_df.copy()
    cohort_df = cohort_df.copy()
    
    logger.info(f"开始计算连续特征差异，全量记录数: {len(full_df)}, 圈定记录数: {len(cohort_df)}")
    
    # 列名安全检查
    required_cols = ["mean", "std", "median", "p95", "IQR", "column_name"]
    missing_full = [c for c in required_cols if c not in full_df.columns]
    missing_cohort = [c for c in required_cols if c not in cohort_df.columns]
    
    if missing_full:
        logger.error(f"全量数据缺少必需列: {missing_full}")
        raise ValueError(f"全量数据缺少必需列: {missing_full}")
    
    if missing_cohort:
        logger.error(f"圈定数据缺少必需列: {missing_cohort}")
        raise ValueError(f"圈定数据缺少必需列: {missing_cohort}")
    
    # 获取共同的索引（stat_date, column_id）
    full_index = set(full_df.index)
    cohort_index = set(cohort_df.index)
    common_index = full_index & cohort_index
    only_full = full_index - cohort_index
    only_cohort = cohort_index - full_index
    
    if only_full:
        logger.warning(f"以下字段只在全量中出现，将被跳过: {len(only_full)}个")
    if only_cohort:
        logger.warning(f"以下字段只在圈定中出现，将被跳过: {len(only_cohort)}个")
    
    if not common_index:
        logger.error("没有共同的字段，无法计算差异")
        return pd.DataFrame()
    
    logger.info(f"共同字段数: {len(common_index)}")
    
    # 对齐数据
    full_aligned = full_df.loc[list(common_index)]
    cohort_aligned = cohort_df.loc[list(common_index)]
    
    # 确保索引顺序一致
    full_aligned = full_aligned.sort_index()
    cohort_aligned = cohort_aligned.sort_index()
    
    # 提取统计指标
    # 全量指标
    mean_full = full_aligned['mean'].values
    std_full = full_aligned['std'].values
    median_full = full_aligned['median'].values
    p95_full = full_aligned['p95'].values
    iqr_full = full_aligned['IQR'].values
    
    # 圈定指标（base表示基础客群，即圈定客群）
    mean_base = cohort_aligned['mean'].values
    std_base = cohort_aligned['std'].values
    median_base = cohort_aligned['median'].values
    p95_base = cohort_aligned['p95'].values
    iqr_base = cohort_aligned['IQR'].values
    
    # CV 可选：存在则计算 delta_CV（Δdisp）
    if 'CV' in full_aligned.columns and 'CV' in cohort_aligned.columns:
        cv_full = full_aligned['CV'].values
        cv_base = cohort_aligned['CV'].values
    else:
        cv_full = np.full(len(full_aligned), np.nan)
        cv_base = np.full(len(cohort_aligned), np.nan)
    delta_CV = cv_base - cv_full
    
    # 计算差异指标
    # 均值差异
    mean_diff = mean_base - mean_full
    mean_diff_ratio = mean_diff / np.maximum(np.abs(mean_full), 1e-6)
    
    # 合并标准差（pooled standard deviation）
    pooled_std = np.sqrt((std_full ** 2 + std_base ** 2) / 2)
    # 处理无效值
    pooled_std = np.where(np.isnan(pooled_std) | (pooled_std <= 0), 1e-6, pooled_std)
    
    # 效应量（Cohen's d）
    effect_size = (mean_base - mean_full) / pooled_std
    
    # 分位数差异
    delta_median = median_base - median_full
    delta_p95 = p95_base - p95_full
    delta_IQR = iqr_base - iqr_full
    
    # 计算综合差异分数
    # diff_score = |effect_size| + 0.5 * |delta_median / (std_full + 1e-6)| + 0.3 * |delta_p95 / (std_full + 1e-6)|
    std_full_safe = np.where(np.isnan(std_full) | (std_full <= 0), 1e-6, std_full)
    diff_score = (
        np.abs(effect_size) +
        0.5 * np.abs(delta_median / std_full_safe) +
        0.3 * np.abs(delta_p95 / std_full_safe)
    )
    
    # 构建结果DataFrame
    result_data = {
        'column_id': full_aligned.index.get_level_values('column_id'),
        'column_name': cohort_aligned['column_name'].values,  # 优先取cohort的
        'stat_date': full_aligned.index.get_level_values('stat_date'),
        'mean_full': mean_full,
        'mean_base': mean_base,
        'mean_diff': mean_diff,
        'mean_diff_ratio': mean_diff_ratio,
        'effect_size': effect_size,
        'delta_median': delta_median,
        'delta_p95': delta_p95,
        'delta_IQR': delta_IQR,
        'delta_CV': delta_CV,
        'diff_score': diff_score,
        'is_significant': np.abs(effect_size) >= 0.2  # 效应量阈值内联
    }
    
    result_df = pd.DataFrame(result_data)
    
    # 设置索引为column_id（先设置索引，方便后续操作）
    result_df = result_df.set_index('column_id')
    
    # column_name 回填逻辑（简化版）
    # 如果cohort的column_name为空，使用full的
    mask_missing = result_df['column_name'].isna() | (result_df['column_name'] == '')
    if mask_missing.any():
        # 创建映射Series，使用column_id作为键
        full_name_map = pd.Series(
            full_aligned['column_name'].values,
            index=full_aligned.index.get_level_values('column_id')
        )
        result_df.loc[mask_missing, 'column_name'] = result_df.loc[mask_missing].index.map(full_name_map)
    
    # 返回时附上排序（按diff_score降序）
    result_df = result_df.sort_values('diff_score', ascending=False)
    
    logger.info(f"差异计算完成，共 {len(result_df)} 个字段")
    
    return result_df

