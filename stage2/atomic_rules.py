"""
原子规则生成模块

将阶段1的推荐结果转化为原子规则库
"""
import json
import re
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any

try:
    from .stage2_config import Stage2Config
except ImportError:
    from stage2_config import Stage2Config

logger = logging.getLogger(__name__)


def calculate_categorical_divergence_score(
    categorical_diff_df: pd.DataFrame,
    config: Stage2Config = None
) -> pd.Series:
    """
    计算离散特征的多尺度统计差异评分
    
    公式：S(feature) = w₁ · delta_ratio + w₂ · significance
    
    Args:
        categorical_diff_df: 阶段1的离散特征差异结果（包含delta_ratio, is_significant等）
        config: 阶段2配置对象
    
    Returns:
        差异评分 Series，index为column_id
    """
    if config is None:
        from stage2_config import Stage2Config
        config = Stage2Config()
    
    scores = []
    indices = []
    
    # 计算全局最大值用于归一化
    max_delta_ratio = categorical_diff_df['delta_ratio'].abs().max() if len(categorical_diff_df) > 0 and 'delta_ratio' in categorical_diff_df.columns else 1.0
    
    for idx, row in categorical_diff_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        indices.append(column_id)
        
        # 提取各项指标
        delta_ratio = abs(row.get('delta_ratio', 0)) if pd.notna(row.get('delta_ratio')) else 0
        is_significant = 1.0 if row.get('is_significant', False) else 0.0
        
        # 归一化各项指标
        normalized_delta_ratio = delta_ratio / max_delta_ratio if max_delta_ratio > 0 else 0.0
        
        # 计算综合差异评分（离散特征主要看delta_ratio）
        divergence_score = (
            config.w_delta_mean * normalized_delta_ratio +  # 复用w_delta_mean权重
            config.w_significance * is_significant
        )
        
        scores.append(divergence_score)
    
    return pd.Series(scores, index=indices, name='divergence_score')


def calculate_divergence_score(
    numeric_diff_df: pd.DataFrame,
    full_numeric_df: Optional[pd.DataFrame] = None,
    cohort_numeric_df: Optional[pd.DataFrame] = None,
    config: Stage2Config = None
) -> pd.Series:
    """
    计算多尺度统计差异评分
    
    公式：S(feature) = w₁ · effect_size + w₂ · |Δmean| + w₃ · quantile_shift + w₄ · significance
    
    Args:
        numeric_diff_df: 阶段1的连续特征差异结果（包含effect_size, mean_diff, delta_p95等）
        full_numeric_df: 全量统计DataFrame（可选，用于获取p90/p10）
        cohort_numeric_df: 圈定统计DataFrame（可选，用于获取p90/p10）
        config: 阶段2配置对象
    
    Returns:
        差异评分 Series，index为column_id
    """
    if config is None:
        from stage2_config import Stage2Config
        config = Stage2Config()
    
    scores = []
    indices = []
    
    # 计算全局最大值用于归一化
    max_effect_size = numeric_diff_df['effect_size'].abs().max() if len(numeric_diff_df) > 0 and 'effect_size' in numeric_diff_df.columns else 1.0
    max_delta_mean = numeric_diff_df['mean_diff'].abs().max() if len(numeric_diff_df) > 0 and 'mean_diff' in numeric_diff_df.columns else 1.0
    
    for idx, row in numeric_diff_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        indices.append(column_id)
        
        # 提取各项指标
        effect_size = abs(row.get('effect_size', 0)) if pd.notna(row.get('effect_size')) else 0
        delta_mean = abs(row.get('mean_diff', 0)) if pd.notna(row.get('mean_diff')) else 0
        is_significant = 1.0 if row.get('is_significant', False) else 0.0
        
        # 计算 quantile_shift = |p90_base - p90_full| + |p10_base - p10_full|
        # 如果阶段1输出中有delta_p95，可以使用它来近似
        quantile_shift = 0.0
        
        # 方法1：尝试从full_numeric_df和cohort_numeric_df获取p90/p10
        if full_numeric_df is not None and cohort_numeric_df is not None:
            try:
                # 获取stat_date（假设所有记录有相同的stat_date）
                stat_date = row.get('stat_date', None)
                if stat_date is None and len(numeric_diff_df) > 0:
                    stat_date = numeric_diff_df.iloc[0].get('stat_date', None)
                
                if stat_date is not None:
                    # 从全量数据获取p90, p10
                    full_row = full_numeric_df.loc[(stat_date, column_id)] if (stat_date, column_id) in full_numeric_df.index else None
                    cohort_row = cohort_numeric_df.loc[(stat_date, column_id)] if (stat_date, column_id) in cohort_numeric_df.index else None
                    
                    if full_row is not None and cohort_row is not None:
                        p90_full = full_row.get('p90', None)
                        p90_base = cohort_row.get('p90', None)
                        p10_full = full_row.get('p10', None)  # 可能不存在
                        p10_base = cohort_row.get('p10', None)  # 可能不存在
                        
                        if pd.notna(p90_full) and pd.notna(p90_base):
                            quantile_shift += abs(p90_base - p90_full)
                        if pd.notna(p10_full) and pd.notna(p10_base):
                            quantile_shift += abs(p10_base - p10_full)
            except (KeyError, IndexError):
                pass
        
        # 方法2：如果方法1失败，使用delta_p95和delta_median来近似
        if quantile_shift == 0:
            delta_p95 = abs(row.get('delta_p95', 0)) if pd.notna(row.get('delta_p95')) else 0
            delta_median = abs(row.get('delta_median', 0)) if pd.notna(row.get('delta_median')) else 0
            # 使用delta_p95和delta_median的加权平均作为quantile_shift的近似
            quantile_shift = delta_p95 * 0.6 + delta_median * 0.4
        
        # 如果还是没有，使用mean_diff的绝对值作为近似
        if quantile_shift == 0:
            quantile_shift = abs(delta_mean)
        
        # 归一化各项指标
        normalized_effect_size = effect_size / max_effect_size if max_effect_size > 0 else 0.0
        normalized_delta_mean = delta_mean / max_delta_mean if max_delta_mean > 0 else 0.0
        
        # quantile_shift归一化（使用delta_mean的最大值作为参考）
        normalized_quantile_shift = quantile_shift / max_delta_mean if max_delta_mean > 0 else 0.0
        # 限制在合理范围内
        normalized_quantile_shift = min(normalized_quantile_shift, 1.0)
        
        # 计算综合差异评分
        divergence_score = (
            config.w_effect_size * normalized_effect_size +
            config.w_delta_mean * normalized_delta_mean +
            config.w_quantile_shift * normalized_quantile_shift +
            config.w_significance * is_significant
        )
        
        scores.append(divergence_score)
    
    return pd.Series(scores, index=indices, name='divergence_score')


def calculate_stability_score(
    numeric_diff_df: pd.DataFrame,
    categorical_diff_df: pd.DataFrame,
    full_numeric_df: Optional[pd.DataFrame] = None,
    cohort_numeric_df: Optional[pd.DataFrame] = None,
    config: Stage2Config = None
) -> pd.Series:
    """
    计算规则稳定性评分
    
    公式：stability = a * (1 / (1 + σ)) + b * (1 / (1 + kurtosis)) + c * sample_weight
    
    Args:
        numeric_diff_df: 阶段1连续特征差异结果
        categorical_diff_df: 阶段1离散特征差异结果
        full_numeric_df: 全量统计DataFrame（可选，用于获取std等）
        cohort_numeric_df: 圈定统计DataFrame（可选，用于获取std等）
        config: 阶段2配置对象
    
    Returns:
        稳定性评分 Series，index为column_id
    """
    if config is None:
        from stage2_config import Stage2Config
        config = Stage2Config()
    
    scores = []
    indices = []
    
    # 处理连续特征
    for idx, row in numeric_diff_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        indices.append(column_id)
        
        # 提取统计指标
        # 优先从cohort_numeric_df获取std（更准确）
        std = None
        if cohort_numeric_df is not None:
            try:
                stat_date = row.get('stat_date', None)
                if stat_date is None and len(numeric_diff_df) > 0:
                    stat_date = numeric_diff_df.iloc[0].get('stat_date', None)
                if stat_date is not None:
                    cohort_row = cohort_numeric_df.loc[(stat_date, column_id)] if (stat_date, column_id) in cohort_numeric_df.index else None
                    if cohort_row is not None:
                        std = cohort_row.get('std', None)
            except (KeyError, IndexError):
                pass
        
        # 如果获取不到，使用默认值或从row中获取
        if std is None or pd.isna(std) or std <= 0:
            std = 1.0  # 默认值
        
        # 使用Stage1的新字段：dist_type, skew_proxy, tail_proxy
        dist_type = row.get('dist_type', row.get('distribution_type', 'heavy_tail'))
        if pd.isna(dist_type) or dist_type == '':
            dist_type = 'heavy_tail'
        
        # 使用tail_proxy（p99/p95或p95/p50）作为kurtosis的近似（优先使用tail_proxy）
        tail_proxy = row.get('tail_proxy', None)
        if pd.notna(tail_proxy) and tail_proxy > 0:
            kurtosis_approx = tail_proxy
        else:
            # 回退到tail_ratio
            tail_ratio = row.get('tail_ratio', 1.0) if pd.notna(row.get('tail_ratio')) else 1.0
            kurtosis_approx = tail_ratio  # tail_ratio越大，kurtosis越高
        
        # 样本量（优先使用sample_size，如果没有则使用total_count）
        sample_size = row.get('sample_size', None)
        if pd.notna(sample_size) and sample_size > 0:
            total_count = sample_size
        else:
            total_count = row.get('total_count', 1000) if pd.notna(row.get('total_count')) else 1000
        # 如果cohort_numeric_df中有total_count，优先使用
        if cohort_numeric_df is not None:
            try:
                stat_date = row.get('stat_date', None)
                if stat_date is None and len(numeric_diff_df) > 0:
                    stat_date = numeric_diff_df.iloc[0].get('stat_date', None)
                if stat_date is not None:
                    cohort_row = cohort_numeric_df.loc[(stat_date, column_id)] if (stat_date, column_id) in cohort_numeric_df.index else None
                    if cohort_row is not None:
                        cohort_total = cohort_row.get('total_count', None)
                        if pd.notna(cohort_total):
                            total_count = cohort_total
            except (KeyError, IndexError):
                pass
        
        # 缺失率（missing_ratio）：缺失率越高，稳定性越低
        missing_ratio = row.get('missing_ratio', 0.0) if pd.notna(row.get('missing_ratio')) else 0.0
        missing_component = 1.0 - min(missing_ratio, 0.5)  # 缺失率超过50%时，稳定性大幅下降
        
        # 归一化样本量（假设最大样本量为100000）
        max_sample = 100000.0
        sample_weight = min(total_count / max_sample, 1.0)
        
        # 计算稳定性各项
        std_component = 1.0 / (1.0 + std)  # std越大，稳定性越低
        kurtosis_component = 1.0 / (1.0 + kurtosis_approx)  # kurtosis越大，稳定性越低
        
        # 综合稳定性评分（加入missing_ratio权重）
        stability_score = (
            config.stability_w_std * std_component +
            config.stability_w_kurtosis * kurtosis_component +
            config.stability_w_sample * sample_weight * missing_component  # 样本量权重受缺失率影响
        )
        
        scores.append(stability_score)
    
    # 处理离散特征
    for idx, row in categorical_diff_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        if column_id not in indices:
            indices.append(column_id)
            # 离散特征的稳定性基于样本量、缺失率、top_value_ratio
            sample_size = row.get('sample_size', None)
            if pd.notna(sample_size) and sample_size > 0:
                total_count = sample_size
            else:
                total_count = row.get('total_count', 1000) if pd.notna(row.get('total_count')) else 1000
            
            missing_ratio = row.get('missing_ratio', 0.0) if pd.notna(row.get('missing_ratio')) else 0.0
            top_value_ratio = row.get('top_value_ratio', 0.5) if pd.notna(row.get('top_value_ratio')) else 0.5  # 最高频类别占比
            
            max_sample = 100000.0
            sample_weight = min(total_count / max_sample, 1.0)
            missing_component = 1.0 - min(missing_ratio, 0.5)
            # top_value_ratio越高，说明分布越集中，稳定性越高
            concentration_component = min(top_value_ratio, 0.8) / 0.8  # 归一化到0-1
            
            # 离散特征稳定性 = 样本量权重 * 缺失率权重 * 集中度权重
            stability_score = 0.3 + 0.7 * (sample_weight * missing_component * concentration_component)
            scores.append(stability_score)
    
    return pd.Series(scores, index=indices, name='stability_score')


def generate_numeric_atomic_rules(
    numeric_diff_df: pd.DataFrame,
    config: Stage2Config,
    full_numeric_df: Optional[pd.DataFrame] = None,
    cohort_numeric_df: Optional[pd.DataFrame] = None,
    divergence_scores: Optional[pd.Series] = None,
    stability_scores: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    生成连续特征原子规则（使用差异评分和稳定性评分）
    
    Args:
        numeric_diff_df: 阶段1的连续特征差异结果
        config: 阶段2配置对象
        full_numeric_df: 全量统计DataFrame（可选）
        cohort_numeric_df: 圈定统计DataFrame（可选）
        divergence_scores: 预计算的差异评分Series（可选）
        stability_scores: 预计算的稳定性评分Series（可选）
    
    Returns:
        DataFrame，包含 column_id, column_name, rule_type, rule_low, rule_high, 
        direction, divergence_score, stability_score等
    """
    # 计算差异评分和稳定性评分（如果未提供）
    if divergence_scores is None:
        divergence_scores = calculate_divergence_score(
            numeric_diff_df, full_numeric_df, cohort_numeric_df, config
        )
    if stability_scores is None:
        stability_scores = calculate_stability_score(
            numeric_diff_df, pd.DataFrame(), full_numeric_df, cohort_numeric_df, config
        )
    
    # 合并评分到numeric_diff_df
    numeric_diff_df = numeric_diff_df.copy()
    numeric_diff_df['divergence_score'] = divergence_scores
    numeric_diff_df['stability_score'] = stability_scores
    
    # 筛选有效规则：使用差异评分和稳定性评分
    valid_mask = (
        (numeric_diff_df['is_significant'] == 1) &
        (numeric_diff_df['divergence_score'] >= config.min_divergence_score) &
        (numeric_diff_df['stability_score'] >= config.min_stability_score)
    )
    # 如果有has_recommendation字段，进一步筛选
    if 'has_recommendation' in numeric_diff_df.columns:
        valid_mask = valid_mask & (numeric_diff_df['has_recommendation'] == True)
    
    valid_df = numeric_diff_df[valid_mask].copy()
    
    # 按差异评分排序，选择Top-K
    valid_df = valid_df.sort_values('divergence_score', ascending=False)
    valid_df = valid_df.head(config.top_k_features)
    
    logger.info(f"生成连续特征原子规则，有效特征数: {len(valid_df)}（差异评分筛选后）")
    
    rules = []
    use_family_first = getattr(config, 'use_candidate_family_first', True)
    max_per_col = getattr(config, 'max_atomic_per_column', 5)
    fallback_legacy = getattr(config, 'fallback_to_legacy_rec', True)

    for idx, row in valid_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        column_name = row.get('column_name', str(column_id))
        rule_reason_code = row.get('rule_reason_code', 'unknown')  # 推荐理由代码

        # 优先使用 Stage1 candidate_threshold_family（每候选一条原子规则，带 base_cov_est/sub_cov_est/lift_est/precision_est）
        family_raw = row.get('candidate_threshold_family', None)
        if use_family_first and family_raw is not None and pd.notna(family_raw):
            try:
                if isinstance(family_raw, str):
                    family_list = json.loads(family_raw) if family_raw.strip() else []
                elif isinstance(family_raw, list):
                    family_list = family_raw
                else:
                    family_list = []
            except (json.JSONDecodeError, TypeError):
                family_list = []
            if family_list:
                for c in family_list[:max_per_col]:
                    rules.append({
                        'column_id': column_id,
                        'column_name': column_name,
                        'rule_type': c.get('quantile_tag', 'family'),
                        'rule_low': c.get('low', np.nan),
                        'rule_high': c.get('high', np.nan),
                        'direction': c.get('direction', 'high'),
                        'divergence_score': row.get('divergence_score', 0.0),
                        'stability_score': row.get('stability_score', 0.0),
                        'distribution_type': row.get('dist_type', row.get('distribution_type', 'heavy_tail')),
                        'rule_reason_code': rule_reason_code,
                        'rule_meta': f"thr_source=candidate_family,quantile={c.get('quantile_tag','')},direction={c.get('direction','high')}",
                        'base_cov_est': c.get('base_cov_est'),
                        'sub_cov_est': c.get('sub_cov_est'),
                        'lift_est': c.get('lift_est'),
                        'precision_est': c.get('precision_est'),
                        'support_est': c.get('support_est'),
                    })
                continue  # 已从候选族生成，跳过下方 rec_rule_family / rec_low+tail 逻辑
            if not fallback_legacy:
                continue
        elif use_family_first and not fallback_legacy:
            continue

        # 处理rec_rule_family：主推荐区间 + 2~3个可调备选区间
        rec_rule_family = row.get('rec_rule_family', None)

        if rec_rule_family is not None and pd.notna(rec_rule_family):
            try:
                if isinstance(rec_rule_family, str):
                    rule_family = json.loads(rec_rule_family)
                else:
                    rule_family = rec_rule_family
                
                # 解析规则族：包含主推荐区间和备选区间
                main_rule = rule_family.get('main', {})
                alternatives = rule_family.get('alternatives', [])
                
                # 主规则
                direction = row.get('direction', 'high')
                if main_rule and pd.notna(main_rule.get('low')) and pd.notna(main_rule.get('high')):
                    rules.append({
                        'column_id': column_id,
                        'column_name': column_name,
                        'rule_type': 'main',
                        'rule_low': main_rule['low'],
                        'rule_high': main_rule['high'],
                        'direction': direction,
                        'divergence_score': row.get('divergence_score', 0.0),
                        'stability_score': row.get('stability_score', 0.0),
                        'distribution_type': row.get('dist_type', row.get('distribution_type', 'heavy_tail')),
                        'rule_reason_code': rule_reason_code,
                        'rule_meta': f"thr_source=rec_rule_family,thr_level=main,direction={direction}",
                    })
                
                # 备选区间（最多3个）
                for alt_idx, alt_rule in enumerate(alternatives[:3]):
                    if pd.notna(alt_rule.get('low')) and pd.notna(alt_rule.get('high')):
                        alt_type = alt_rule.get('type', f'alternative_{alt_idx+1}')  # narrower, wider, tail等
                        rules.append({
                            'column_id': column_id,
                            'column_name': column_name,
                            'rule_type': alt_type,
                            'rule_low': alt_rule['low'],
                            'rule_high': alt_rule['high'],
                            'direction': direction,
                            'divergence_score': row.get('divergence_score', 0.0) * 0.8,  # 备选区间评分略低
                            'stability_score': row.get('stability_score', 0.0),
                            'distribution_type': row.get('dist_type', row.get('distribution_type', 'heavy_tail')),
                            'rule_reason_code': rule_reason_code,
                            'rule_meta': f"thr_source=rec_rule_family,thr_level={alt_type},direction={direction}",
                        })
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"解析rec_rule_family失败 ({column_id}): {e}，使用默认规则")
                # 回退到旧逻辑
                rec_rule_family = None
        
        # 如果没有rec_rule_family，使用旧的逻辑（向后兼容）
        if rec_rule_family is None or not pd.notna(rec_rule_family):
            # 主规则：使用推荐的区间
            if pd.notna(row.get('rec_low')) and pd.notna(row.get('rec_high')):
                direction = row.get('direction', 'high')
                rules.append({
                    'column_id': column_id,
                    'column_name': column_name,
                    'rule_type': 'main',
                    'rule_low': row['rec_low'],
                    'rule_high': row['rec_high'],
                    'direction': direction,
                    'divergence_score': row.get('divergence_score', 0.0),
                    'stability_score': row.get('stability_score', 0.0),
                    'distribution_type': row.get('dist_type', row.get('distribution_type', 'heavy_tail')),
                    'rule_reason_code': rule_reason_code,
                    'rule_meta': f"thr_source=rec,thr_level=main,direction={direction}",
                })
            # Stage1 分位列存在时，为该字段补多条尾部分位规则（p60/p70/p80/p75/p90/p95 等，扩大原子池、增加窄规则）
            direction = row.get('direction', 'high')
            if direction == 'high':
                for qname in ('p60_full', 'p70_full', 'p80_full', 'p75_full', 'p90_full', 'p95_full'):
                    if qname in row.index and pd.notna(row.get(qname)) and np.isfinite(row.get(qname)):
                        thr = float(row[qname])
                        rules.append({
                            'column_id': column_id,
                            'column_name': column_name,
                            'rule_type': 'tail',
                            'rule_low': thr,
                            'rule_high': np.inf,
                            'direction': 'high',
                            'divergence_score': row.get('divergence_score', 0.0) * 0.85,
                            'stability_score': row.get('stability_score', 0.0),
                            'distribution_type': row.get('dist_type', row.get('distribution_type', 'symmetric')),
                            'rule_reason_code': rule_reason_code,
                            'rule_meta': f"thr_source={qname},thr_level={thr},direction=high",
                        })
            else:
                for qname in ('p10_full', 'p25_full'):
                    if qname in row.index and pd.notna(row.get(qname)) and np.isfinite(row.get(qname)):
                        thr = float(row[qname])
                        rules.append({
                            'column_id': column_id,
                            'column_name': column_name,
                            'rule_type': 'tail',
                            'rule_low': -np.inf,
                            'rule_high': thr,
                            'direction': 'low',
                            'divergence_score': row.get('divergence_score', 0.0) * 0.85,
                            'stability_score': row.get('stability_score', 0.0),
                            'distribution_type': row.get('dist_type', row.get('distribution_type', 'symmetric')),
                            'rule_reason_code': rule_reason_code,
                            'rule_meta': f"thr_source={qname},thr_level={thr},direction=low",
                        })
            
            # 对于长尾型分布，生成 >= p95 的尾部规则（如果稳定性足够）
            distribution_type = row.get('dist_type', row.get('distribution_type', 'heavy_tail'))
            stability_score = row.get('stability_score', 0.0)
            
            # 长尾分布的尾部规则需要更高的稳定性
            if distribution_type in ('heavy_tail', 'powerlaw', 'pareto') and stability_score >= config.min_stability_score * 0.8:
                # 尝试从cohort_numeric_df获取p95
                p95 = None
                if cohort_numeric_df is not None:
                    try:
                        stat_date = row.get('stat_date', None)
                        if stat_date is None and len(valid_df) > 0:
                            stat_date = valid_df.iloc[0].get('stat_date', None)
                        if stat_date is not None:
                            cohort_row = cohort_numeric_df.loc[(stat_date, column_id)] if (stat_date, column_id) in cohort_numeric_df.index else None
                            if cohort_row is not None:
                                p95 = cohort_row.get('p95', None)
                    except (KeyError, IndexError):
                        pass
                
                # 如果获取不到，跳过尾部规则
                if pd.notna(p95):
                    rules.append({
                        'column_id': column_id,
                        'column_name': column_name,
                        'rule_type': 'tail',
                        'rule_low': p95,
                        'rule_high': np.inf,
                        'direction': 'high',
                        'divergence_score': row.get('divergence_score', 0.0) * 0.7,  # 尾部规则差异评分略低
                        'stability_score': stability_score,
                        'distribution_type': distribution_type,
                        'rule_reason_code': rule_reason_code,
                        'rule_meta': f"thr_source=p95_cohort,thr_level={p95},direction=high",
                    })
    
    # 原子规则不足时扩大池：每个已生成 main 且 direction=high 的连续特征增加一条低尾规则（-inf, main_rule_low）
    main_rule_by_col: Dict[str, dict] = {}
    for r in rules:
        if r.get('rule_type') == 'main' and r.get('direction') == 'high' and r['column_id'] not in main_rule_by_col:
            main_rule_by_col[r['column_id']] = r
    for cid, main_r in main_rule_by_col.items():
        rule_high_low = main_r['rule_low']
        rules.append({
            'column_id': cid,
            'column_name': main_r['column_name'],
            'rule_type': 'low_tail',
            'rule_low': -np.inf,
            'rule_high': rule_high_low,
            'direction': 'low',
            'divergence_score': main_r['divergence_score'] * 0.7,
            'stability_score': main_r['stability_score'],
            'distribution_type': main_r.get('distribution_type', 'heavy_tail'),
            'rule_reason_code': main_r.get('rule_reason_code', 'unknown'),
            'rule_meta': f"thr_source=low_tail,thr_level={rule_high_low},direction=low",
        })
        logger.debug("为 column_id=%s 增加低尾原子规则（扩大原子池）", cid)
    
    if not rules:
        logger.warning("未生成任何连续特征原子规则")
        return pd.DataFrame(columns=[
            'column_id', 'column_name', 'rule_type', 'rule_low', 'rule_high',
            'direction', 'divergence_score', 'stability_score', 'distribution_type'
        ])
    
    # 同 column_id 同 (rule_low, rule_high) 去重，避免 p75_full 与 p75 等重复
    n_before_dedup = len(rules)
    seen_key: Dict[tuple, bool] = {}
    deduped = []
    for r in rules:
        key = (r['column_id'], r.get('rule_low'), r.get('rule_high'))
        if key not in seen_key:
            seen_key[key] = True
            deduped.append(r)
    rules = deduped
    if n_before_dedup > len(rules):
        logger.debug("连续特征 tail 分位去重：去重前 %d 条，去重后 %d 条", n_before_dedup, len(rules))
    
    # v2.0: cap 6 rules per column_id（主+多尾+低尾）；height 前置限用：最多 0 或 1 条
    from collections import defaultdict
    by_col = defaultdict(list)
    for r in rules:
        by_col[r['column_id']].append(r)
    height_regex_list = getattr(config, 'height_prune_regex_list', None) or []
    max_height_rules = getattr(config, 'max_atomic_rules_per_height_column', 0)
    capped = []
    cap_per_col = getattr(config, 'max_atomic_rules_per_numeric_column', 6)
    height_rule_count = 0
    for cid, rlist in by_col.items():
        is_height = False
        if height_regex_list and cid:
            cid_str = str(cid).strip().lower()
            for pat in height_regex_list:
                try:
                    if re.search(pat, cid_str, re.IGNORECASE):
                        is_height = True
                        break
                except re.error:
                    if pat.lower() in cid_str:
                        is_height = True
                        break
        if is_height:
            keep = rlist[:max_height_rules]
            height_rule_count += len(keep)
            if len(rlist) > max_height_rules:
                logger.info("height 前置限用: 字段 %s 原子规则数 %d，仅保留 %d 条（max_atomic_rules_per_height_column=%d）", cid, len(rlist), len(keep), max_height_rules)
        else:
            keep = rlist[:cap_per_col]
            if len(rlist) > cap_per_col:
                logger.info(f"字段 {cid} 原子规则数 {len(rlist)} 超过 {cap_per_col}，截断保留前 {cap_per_col} 条")
        thrs = [f"{r.get('rule_low', '')}~{r.get('rule_high', '')}" for r in keep]
        logger.debug("column_id=%s 生成 %d 条连续规则: %s", cid, len(keep), thrs)
        capped.extend(keep)
    if height_rule_count > 0 or height_regex_list:
        logger.info("height 相关原子规则数（前置限用后）: %d", height_rule_count)
    rules = capped
    
    result_df = pd.DataFrame(rules)
    per_col_sample = result_df.groupby('column_id').size().head(5).to_dict() if len(result_df) > 0 else {}
    logger.info("[stage2] atomic_rules (numeric) total=%d (target 40~60) per_column_sample=%s", len(result_df), per_col_sample)
    return result_df


def generate_categorical_atomic_rules(
    categorical_diff_df: pd.DataFrame,
    config: Stage2Config,
    divergence_scores: Optional[pd.Series] = None,
    stability_scores: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    生成离散特征原子规则（使用差异评分和稳定性评分）
    
    Args:
        categorical_diff_df: 阶段1的离散特征差异结果
        config: 阶段2配置对象
        divergence_scores: 预计算的差异评分Series（可选）
        stability_scores: 预计算的稳定性评分Series（可选）
    
    Returns:
        DataFrame，包含 column_id, column_name, rule_categories, divergence_score, stability_score等
    """
    # 计算差异评分（如果未提供）
    if divergence_scores is None:
        divergence_scores = calculate_categorical_divergence_score(categorical_diff_df, config)
    
    # 合并评分
    categorical_diff_df = categorical_diff_df.copy()
    categorical_diff_df['divergence_score'] = divergence_scores
    if stability_scores is not None:
        categorical_diff_df['stability_score'] = stability_scores
    
    # 筛选有效规则：使用差异评分和稳定性评分
    # 允许 rec_categories 为空/NaN：业务上“缺失”作为一类时，Stage1 可能输出 rec_categories=nan、rule_desc 为 "xxx in {nan}"
    has_rec = categorical_diff_df['rec_categories'].notna()
    if 'rule_desc' in categorical_diff_df.columns:
        rule_desc_str = categorical_diff_df['rule_desc'].astype(str)
        has_rule_desc = rule_desc_str.str.contains(r'in\s*\{', na=False)
        has_rec = has_rec | has_rule_desc
    valid_mask = (
        (categorical_diff_df['delta_ratio'] >= config.min_delta) &
        has_rec &
        (categorical_diff_df['divergence_score'] >= config.min_divergence_score)
    )
    
    # 如果有稳定性评分，进一步筛选（离散可单独放宽阈值以扩大原子池）
    stability_thr = getattr(config, 'min_stability_score_categorical', None)
    if stability_thr is None:
        stability_thr = config.min_stability_score
    else:
        logger.info("使用放宽的离散稳定性阈值 min_stability_score_categorical=%.4f，有效特征数将按此筛选", stability_thr)
    if 'stability_score' in categorical_diff_df.columns:
        valid_mask = valid_mask & (categorical_diff_df['stability_score'] >= stability_thr)
    
    valid_df = categorical_diff_df[valid_mask].copy()
    
    # 按差异评分排序，选择Top-K
    valid_df = valid_df.sort_values('divergence_score', ascending=False)
    valid_df = valid_df.head(config.top_k_features)
    
    logger.info(f"生成离散特征原子规则，有效特征数: {len(valid_df)}（差异评分和稳定性评分筛选后）")

    rules = []
    use_family_first = getattr(config, 'use_candidate_family_first', True)
    max_per_col = getattr(config, 'max_atomic_per_column', 5)
    fallback_legacy = getattr(config, 'fallback_to_legacy_rec', True)
    max_cats = getattr(config, 'max_categories', 3)

    for idx, row in valid_df.iterrows():
        column_id = idx if isinstance(idx, str) else row.get('column_id', idx)
        column_name = row.get('column_name', str(column_id))

        # 优先使用 Stage1 candidate_category_family（每候选一条原子规则，带 base_cov/sub_cov/lift/precision）
        family_raw = row.get('candidate_category_family', None)
        if use_family_first and family_raw is not None and pd.notna(family_raw):
            try:
                if isinstance(family_raw, str):
                    family_list = json.loads(family_raw) if family_raw.strip() else []
                elif isinstance(family_raw, list):
                    family_list = family_raw
                else:
                    family_list = []
            except (json.JSONDecodeError, TypeError):
                family_list = []
            if family_list:
                for c in family_list[:max_per_col]:
                    cats = c.get('categories', [])
                    if not cats:
                        continue
                    rules.append({
                        'column_id': column_id,
                        'column_name': column_name,
                        'rule_categories': ','.join(str(x) for x in cats),
                        'divergence_score': row.get('divergence_score', 0.0),
                        'stability_score': row.get('stability_score', 0.5),
                        'delta_ratio': row.get('delta_ratio', 0.0),
                        'rec_category_count': len(cats),
                        'base_cov_est': c.get('base_cov'),
                        'sub_cov_est': c.get('sub_cov'),
                        'lift_est': c.get('lift'),
                        'precision_est': c.get('precision'),
                    })
                continue
            if not fallback_legacy:
                continue
        elif use_family_first and not fallback_legacy:
            continue

        # 收集 1~3 条原子规则的类别集合：top-1、top-2（差异为正）、recommended
        category_sets_to_emit: List[List[str]] = []
        seen_signatures: set = set()

        def _add_rule_cats(cats: List[str], source: str) -> None:
            if not cats or len(cats) > max_cats:
                return
            sig = tuple(sorted(cats))
            if sig in seen_signatures:
                return
            seen_signatures.add(sig)
            category_sets_to_emit.append((cats, source))
        
        # 1) top_diff_categories：解析 "类别1(+0.25); 类别2(+0.12); 类别3(-0.08)"，取差异为正的 top-1 和 top-2
        top_diff_str = row.get('top_diff_categories', '') or row.get('差异最大类别及差值', '')
        if isinstance(top_diff_str, str) and top_diff_str.strip():
            # 格式: "val1(+0.25); val2(+0.12); val3(-0.08)"
            positive_cats = []
            for part in top_diff_str.split(';'):
                part = part.strip()
                m = re.match(r'^([^(]+)\(\s*([+-]?[\d.]+)\s*\)\s*$', part.strip())
                if m:
                    cat_name, diff_str = m.group(1).strip(), m.group(2)
                    try:
                        diff_val = float(diff_str)
                        if diff_val > 0:
                            positive_cats.append(cat_name)
                    except ValueError:
                        pass
            if positive_cats:
                _add_rule_cats(positive_cats[:1], 'top1')
            if len(positive_cats) >= 2:
                _add_rule_cats(positive_cats[:2], 'top2')
        
        # 2) rec_categories / rule_desc 作为 recommended（若与已有不同则作为第三条）
        rec_categories = row.get('rec_categories', '')
        if pd.isna(rec_categories) or rec_categories == '':
            rule_desc = row.get('rule_desc', '')
            if isinstance(rule_desc, str) and 'in {' in rule_desc:
                m = re.search(r'in\s*\{([^}]*)\}', rule_desc)
                if m:
                    rec_list = [c.strip() for c in m.group(1).split(',') if c.strip()]
                else:
                    rec_list = ['nan']
            else:
                rec_list = ['nan']
        else:
            if isinstance(rec_categories, str):
                rec_list = [c.strip() for c in rec_categories.split(',') if c.strip()]
            elif isinstance(rec_categories, list):
                rec_list = [str(c) for c in rec_categories]
            else:
                rec_list = [str(rec_categories)]
        if rec_list and len(rec_list) <= max_cats:
            _add_rule_cats(rec_list[:max_cats], 'recommended')
        
        if not category_sets_to_emit:
            continue
        
        for cats, _src in category_sets_to_emit:
            rules.append({
                'column_id': column_id,
                'column_name': column_name,
                'rule_categories': ','.join(cats),
                'divergence_score': row.get('divergence_score', 0.0),
                'stability_score': row.get('stability_score', 0.5),
                'delta_ratio': row.get('delta_ratio', 0.0),
                'rec_category_count': len(cats)
            })
        logger.debug("离散字段 column_id=%s 产出 %d 条规则（top_diff/rec）", column_id, len(category_sets_to_emit))
    
    if not rules:
        logger.warning("未生成任何离散特征原子规则")
        return pd.DataFrame(columns=[
            'column_id', 'column_name', 'rule_categories', 
            'divergence_score', 'stability_score', 'delta_ratio', 'rec_category_count'
        ])
    
    result_df = pd.DataFrame(rules)
    logger.info("[stage2] atomic_rules (categorical) total=%d", len(result_df))
    return result_df


def merge_atomic_rules(
    numeric_rules: pd.DataFrame,
    categorical_rules: pd.DataFrame
) -> pd.DataFrame:
    """
    合并原子规则库
    
    Args:
        numeric_rules: 连续特征原子规则DataFrame
        categorical_rules: 离散特征原子规则DataFrame
    
    Returns:
        统一格式的原子规则库DataFrame
    """
    # 为连续特征规则添加类型标识
    if len(numeric_rules) > 0:
        numeric_rules = numeric_rules.copy()
        numeric_rules['rule_type_feature'] = 'numeric'
    
    # 为离散特征规则添加类型标识
    if len(categorical_rules) > 0:
        categorical_rules = categorical_rules.copy()
        categorical_rules['rule_type_feature'] = 'categorical'
    
    # 合并（使用外连接，保留所有列）
    if len(numeric_rules) == 0 and len(categorical_rules) == 0:
        return pd.DataFrame()
    
    if len(numeric_rules) == 0:
        return categorical_rules
    
    if len(categorical_rules) == 0:
        return numeric_rules
    
    # 合并两个DataFrame
    # 由于列结构不同，我们需要创建一个统一的格式
    all_rules = []
    
    # 处理连续特征规则
    for _, row in numeric_rules.iterrows():
        rule_dict = {
            'column_id': row['column_id'],
            'column_name': row['column_name'],
            'rule_type_feature': 'numeric',
            'rule_type': row.get('rule_type', 'main'),
            'rule_low': row.get('rule_low', np.nan),
            'rule_high': row.get('rule_high', np.nan),
            'rule_categories': np.nan,
            'direction': row.get('direction', 'high'),
            'divergence_score': row.get('divergence_score', 0.0),
            'stability_score': row.get('stability_score', 0.0),
            'distribution_type': row.get('distribution_type', 'heavy_tail')
        }
        if 'rule_meta' in row.index and pd.notna(row.get('rule_meta')):
            rule_dict['rule_meta'] = row['rule_meta']
        for k in ('base_cov_est', 'sub_cov_est', 'lift_est', 'precision_est', 'support_est'):
            if k in row.index and pd.notna(row.get(k)):
                rule_dict[k] = row[k]
        all_rules.append(rule_dict)
    
    # 处理离散特征规则
    for _, row in categorical_rules.iterrows():
        rule_dict = {
            'column_id': row['column_id'],
            'column_name': row['column_name'],
            'rule_type_feature': 'categorical',
            'rule_type': 'categorical',
            'rule_low': np.nan,
            'rule_high': np.nan,
            'rule_categories': row.get('rule_categories', ''),
            'direction': None,
            'divergence_score': row.get('divergence_score', 0.0),
            'stability_score': row.get('stability_score', 0.5),
            'delta_ratio': row.get('delta_ratio', 0.0),
            'rec_category_count': row.get('rec_category_count', 0),
            'distribution_type': row.get('distribution_type', np.nan)
        }
        for k in ('base_cov_est', 'sub_cov_est', 'lift_est', 'precision_est'):
            if k in row.index and pd.notna(row.get(k)):
                rule_dict[k] = row[k]
        all_rules.append(rule_dict)

    result_df = pd.DataFrame(all_rules)
    per_col = result_df.groupby('column_id').size()
    sample = per_col.head(5).to_dict() if len(per_col) > 0 else {}
    logger.info("[stage2] atomic_rules total=%d (target 40~60) per_column_sample=%s", len(result_df), sample)
    return result_df