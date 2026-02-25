"""
分布类型检测模块

基于统计指标判断连续特征的分布类型，支持 normal, skewed, heavy_tail, powerlaw, multimodal 等类型。
"""
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple
from scipy import stats
from scipy.stats import skew, kurtosis

# 配置日志
logger = logging.getLogger(__name__)


def detect_distribution_type(
    mean: float,
    median: float,
    std: float,
    p05: Optional[float] = None,
    p95: Optional[float] = None,
    p25: Optional[float] = None,
    p75: Optional[float] = None,
    p90: Optional[float] = None,
    p99: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    skewness: Optional[float] = None,
    kurt: Optional[float] = None,
    total_count: Optional[int] = None
) -> Tuple[str, Dict[str, float]]:
    """
    基于统计指标判断分布类型（使用 approx_skew 和 tail_ratio）
    
    判断逻辑（基于新的规则）：
    1. std≈0 → 直接当 normal/constant
    2. 计算 approx_skew = (mean - median) / std
    3. 计算 tail_ratio = (p99 - p95) / (p95 - p90)
    4. 按新规则判断分布类型
    
    Args:
        mean: 均值
        median: 中位数
        std: 标准差
        p05: 5分位数（可选）
        p95: 95分位数（可选）
        p25: 25分位数（可选）
        p75: 75分位数（可选）
        p90: 90分位数（可选）
        p99: 99分位数（可选）
        min_val: 最小值（可选）
        max_val: 最大值（可选）
        skewness: 偏度（可选，如果不提供则基于 mean/median/std 估算）
        kurt: 峰度（可选）
        total_count: 总样本数（可选，用于某些检验）
    
    Returns:
        Tuple[分布类型字符串, 统计指标字典]:
        - 分布类型: "normal", "skewed", "heavy_tail", "powerlaw", "multimodal"
        - 统计指标字典: 包含 approx_skew, tail_ratio 等
    """
    # 初始化统计指标字典
    stats_dict = {
        'approx_skew': np.nan,
        'tail_ratio': np.nan
    }
    
    # 1. std≈0 → 直接当 normal/constant
    if std is None or std <= 1e-6 or np.isnan(std):
        stats_dict['approx_skew'] = 0.0
        stats_dict['tail_ratio'] = 1.0
        return "normal", stats_dict
    
    # 处理无效值
    if np.isnan(mean) or np.isnan(median):
        stats_dict['approx_skew'] = 0.0
        stats_dict['tail_ratio'] = 1.0
        return "normal", stats_dict
    
    # 计算近似偏度 approx_skew = (mean - median) / std
    if std > 0:
        approx_skew = (mean - median) / std
    else:
        approx_skew = 0.0
    stats_dict['approx_skew'] = approx_skew
    
    # 计算尾部比率 tail_ratio = (p99 - p95) / (p95 - p90)
    if p90 is not None and p95 is not None and p99 is not None:
        if not (np.isnan(p90) or np.isnan(p95) or np.isnan(p99)):
            if p95 > p90 and p99 > p95:
                tail_ratio = (p99 - p95) / (p95 - p90)
            else:
                tail_ratio = np.nan
        else:
            tail_ratio = np.nan
    else:
        tail_ratio = np.nan
    
    stats_dict['tail_ratio'] = tail_ratio if not np.isnan(tail_ratio) else 1.0
    
    # 基于新规则判断分布类型
    abs_skew = abs(approx_skew)
    tail_ratio_val = tail_ratio if not np.isnan(tail_ratio) else 1.0
    
    # 判断 powerlaw: abs(approx_skew) >= 2 AND tail_ratio >= 3 AND p99 >> p95
    if abs_skew >= 2.0 and tail_ratio_val >= 3.0:
        if p95 is not None and p99 is not None and not (np.isnan(p95) or np.isnan(p99)):
            if p99 > 0 and p95 > 0 and p99 / p95 > 1.5:  # p99 >> p95
                return "powerlaw", stats_dict
    
    # 判断 heavy_tail: abs(approx_skew) >= 1 OR tail_ratio >= 2
    if abs_skew >= 1.0 or tail_ratio_val >= 2.0:
        return "heavy_tail", stats_dict
    
    # 判断 multimodal: (p75 - median) 与 (median - p25) 差异特别大 AND tail_ratio < 1.5
    if p25 is not None and p75 is not None and not (np.isnan(p25) or np.isnan(p75)):
        upper_half = p75 - median if p75 > median else 0
        lower_half = median - p25 if median > p25 else 0
        if upper_half > 0 and lower_half > 0:
            half_ratio = max(upper_half, lower_half) / min(upper_half, lower_half)
            if half_ratio > 2.0 and tail_ratio_val < 1.5:
                return "multimodal", stats_dict
    
    # 判断 skewed: 0.3 <= abs(approx_skew) < 1 OR 1.5 <= tail_ratio < 2
    if (0.3 <= abs_skew < 1.0) or (1.5 <= tail_ratio_val < 2.0):
        return "skewed", stats_dict
    
    # 判断 normal: abs(approx_skew) < 0.3 AND tail_ratio < 1.5
    if abs_skew < 0.3 and tail_ratio_val < 1.5:
        return "normal", stats_dict
    
    # 默认 normal
    return "normal", stats_dict


def detect_distribution_type_from_stats(
    stats_row: pd.Series,
    use_skewness: bool = True,
    use_kurtosis: bool = True
) -> Tuple[str, Dict[str, float]]:
    """
    从统计DataFrame的一行中检测分布类型
    
    这是一个便捷函数，从 numeric_stats DataFrame 的一行中提取所需统计指标，
    然后调用 detect_distribution_type()。
    
    Args:
        stats_row: 统计DataFrame的一行，应包含 mean, median, std 等列
        use_skewness: 是否使用已有的 skewness 列（如果存在）
        use_kurtosis: 是否使用已有的 kurtosis 列（如果存在）
    
    Returns:
        Tuple[分布类型字符串, 统计指标字典]:
        - 分布类型: "normal", "skewed", "heavy_tail", "powerlaw", "multimodal"
        - 统计指标字典: 包含 approx_skew, tail_ratio 等
    """
    # 提取基本统计指标
    mean = float(stats_row.get('mean', np.nan))
    
    # 提取中位数（支持多种列名）
    median = stats_row.get('median', None)
    if median is None or pd.isna(median):
        median = stats_row.get('q2_cont', None)
    if median is None or pd.isna(median):
        median = np.nan
    else:
        median = float(median)
    
    std = float(stats_row.get('std', np.nan))
    
    # 提取分位数（新格式）
    p05 = stats_row.get('p05', None)
    if p05 is None or pd.isna(p05):
        p05 = stats_row.get('q05_cont', None)
    
    p90 = stats_row.get('p90', None)
    if p90 is None or pd.isna(p90):
        p90 = stats_row.get('q9_cont', None)  # q9_cont 对应 90%
    
    p95 = stats_row.get('p95', None)
    if p95 is None or pd.isna(p95):
        p95 = stats_row.get('q95_cont', None)
    
    p99 = stats_row.get('p99', None)
    if p99 is None or pd.isna(p99):
        p99 = stats_row.get('q99_cont', None)
    
    p25 = stats_row.get('p25', None)
    if p25 is None or pd.isna(p25):
        p25 = stats_row.get('q1_cont', None)
        if p25 is None or pd.isna(p25):
            p25 = stats_row.get('q1', None)
    
    p75 = stats_row.get('p75', None)
    if p75 is None or pd.isna(p75):
        p75 = stats_row.get('q3_cont', None)
        if p75 is None or pd.isna(p75):
            p75 = stats_row.get('q3', None)
    
    # 提取最值
    min_val = stats_row.get('min_val', None)
    if min_val is None or pd.isna(min_val):
        min_val = None
    
    max_val = stats_row.get('max_val', None)
    if max_val is None or pd.isna(max_val):
        max_val = None
    
    # 提取偏度和峰度（如果存在）
    skewness = None
    if use_skewness and 'skewness' in stats_row:
        skewness = stats_row.get('skewness', None)
        if pd.isna(skewness):
            skewness = None
    
    kurt = None
    if use_kurtosis and 'kurtosis' in stats_row:
        kurt = stats_row.get('kurtosis', None)
        if pd.isna(kurt):
            kurt = None
    
    # 提取样本数
    total_count = stats_row.get('total_count', None)
    
    # 转换分位数为 float（如果存在且有效）
    if p05 is not None:
        try:
            p05 = float(p05) if not pd.isna(p05) else None
        except (ValueError, TypeError):
            p05 = None
    if p95 is not None:
        try:
            p95 = float(p95) if not pd.isna(p95) else None
        except (ValueError, TypeError):
            p95 = None
    if p25 is not None:
        try:
            p25 = float(p25) if not pd.isna(p25) else None
        except (ValueError, TypeError):
            p25 = None
    if p75 is not None:
        try:
            p75 = float(p75) if not pd.isna(p75) else None
        except (ValueError, TypeError):
            p75 = None
    if min_val is not None:
        try:
            min_val = float(min_val) if not pd.isna(min_val) else None
        except (ValueError, TypeError):
            min_val = None
    if max_val is not None:
        try:
            max_val = float(max_val) if not pd.isna(max_val) else None
        except (ValueError, TypeError):
            max_val = None
    
    return detect_distribution_type(
        mean=mean,
        median=median,
        std=std,
        p05=p05,
        p95=p95,
        p25=p25,
        p75=p75,
        p90=p90,
        p99=p99,
        min_val=min_val,
        max_val=max_val,
        skewness=skewness,
        kurt=kurt,
        total_count=total_count
    )


# 四类分布仅允许：zero_inflated, heavy_tail, skewed, symmetric
DISTRIBUTION_TYPES = frozenset({'zero_inflated', 'heavy_tail', 'skewed', 'symmetric'})
SKEW_EPS = 0.2
TAIL_THR = 0.5
EPS = 1e-9


def detect_distribution_types_from_quantiles(
    numeric_stats_df: pd.DataFrame,
    skew_eps: float = SKEW_EPS,
    tail_thr: float = TAIL_THR,
    eps: float = EPS,
) -> pd.DataFrame:
    """
    基于 full 行分位数检测分布类型（仅输出四类）。
    优先级：zero_inflated > heavy_tail > skewed > symmetric。
    仅使用 source_type==1 且 group_id=='ALL' 的全量行。
    分位口径：P25=q3_cont, P50=q6_cont, P75=q9_cont；p90_cont, p95_cont, p99_cont 为真实分位。
    """
    if numeric_stats_df is None or len(numeric_stats_df) == 0:
        return pd.DataFrame(columns=['distribution_type', 'approx_skew', 'tail_ratio'])
    df = numeric_stats_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if 'column_id' not in df.columns and df.shape[1] >= 2:
            df['column_id'] = df.iloc[:, 1]
    if 'column_id' not in df.columns:
        df['column_id'] = df.index
    if 'source_type' in df.columns and 'group_id' in df.columns:
        base = df[(df['source_type'].astype(str) == '1') & (df['group_id'].astype(str).str.upper() == 'ALL')]
        if len(base) > 0:
            df = base
    def _get(row, *keys):
        for k in keys:
            if k in row.index:
                v = row.get(k)
                if pd.notna(v) and v != '':
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
        return np.nan
    results = []
    for col_id, grp in df.groupby('column_id', sort=False):
        row = grp.iloc[0]
        p25 = _get(row, 'q3_cont', 'q1')
        p50 = _get(row, 'q6_cont', 'median')
        p75 = _get(row, 'q9_cont', 'q3')
        p90 = _get(row, 'p90_cont', 'p90')
        p95 = _get(row, 'p95_cont', 'p95')
        p99 = _get(row, 'p99_cont', 'p99')
        iqr_raw = _get(row, 'IQR')
        if pd.isna(iqr_raw) or iqr_raw <= 0:
            iqr = max((p75 - p25) if all(pd.notna(x) for x in [p25, p75]) else 0.0, eps)
        else:
            iqr = max(iqr_raw, eps)
        numer = (p75 + p25 - 2.0 * p50) if all(pd.notna(x) for x in [p25, p50, p75]) else 0.0
        approx_skew = numer / iqr if iqr > 0 else 0.0
        denom = (p95 - p50) if pd.notna(p95) and pd.notna(p50) else np.nan
        if pd.isna(denom) or denom <= 0:
            denom = eps
        tail_ratio = (p99 - p95) / denom if pd.notna(p99) and pd.notna(p95) else 0.0
        missing_quantiles = int(pd.isna(p25) or pd.isna(p50) or pd.isna(p75))
        if missing_quantiles:
            dist_type = 'heavy_tail'
            rec = {'column_id': col_id, 'distribution_type': dist_type, 'approx_skew': approx_skew, 'tail_ratio': tail_ratio, 'debug_tag': 'missing_quantiles=1'}
        else:
            is_zero_inflated = (p50 == 0 or abs(float(p50)) < eps) and pd.notna(p90) and float(p90) > 0
            if is_zero_inflated:
                dist_type = 'zero_inflated'
            elif pd.notna(tail_ratio) and tail_ratio >= tail_thr and pd.notna(p95) and p95 > p50:
                dist_type = 'heavy_tail'
            elif pd.notna(approx_skew) and abs(approx_skew) > skew_eps:
                dist_type = 'skewed'
            else:
                dist_type = 'symmetric'
            rec = {'column_id': col_id, 'distribution_type': dist_type, 'approx_skew': approx_skew, 'tail_ratio': tail_ratio}
        results.append(rec)
    out = pd.DataFrame(results)
    out = out.set_index('column_id')
    return out


def batch_detect_distribution_types(
    numeric_stats_df: pd.DataFrame,
    use_skewness: bool = True,
    use_kurtosis: bool = True
) -> pd.DataFrame:
    """
    批量检测多个字段的分布类型（仅返回四类：zero_inflated, heavy_tail, skewed, symmetric）。
    若 DataFrame 含分位列（q3_cont/q6_cont/q9_cont 或 p90_cont 等），优先走 detect_distribution_types_from_quantiles。
    """
    if numeric_stats_df is not None and len(numeric_stats_df) > 0:
        cols = numeric_stats_df.columns if hasattr(numeric_stats_df, 'columns') else []
        has_quantiles = any(c in cols for c in ('q3_cont', 'q6_cont', 'q9_cont', 'q1', 'median', 'q3', 'p90_cont', 'p95_cont'))
        if has_quantiles:
            out = detect_distribution_types_from_quantiles(numeric_stats_df)
            if 'CV' in numeric_stats_df.columns:
                try:
                    cv_ser = numeric_stats_df.reset_index().groupby('column_id')['CV'].first()
                    out['cv'] = out.index.map(lambda x: cv_ser.get(x, np.nan))
                except Exception:
                    out['cv'] = np.nan
            return out
    results = []
    for idx, row in numeric_stats_df.iterrows():
        try:
            dist_type, stats_dict = detect_distribution_type_from_stats(
                row, use_skewness=use_skewness, use_kurtosis=use_kurtosis
            )
            column_id = idx[1] if isinstance(idx, tuple) else idx
            cv_val = row.get('CV', np.nan)
            try:
                cv_val = float(cv_val) if cv_val is not None and not pd.isna(cv_val) else np.nan
            except (TypeError, ValueError):
                cv_val = np.nan
            # 映射到四类（旧函数可能返回 normal/powerlaw/multimodal）
            if dist_type == 'normal':
                dist_type = 'symmetric'
            elif dist_type in ('powerlaw', 'multimodal'):
                dist_type = 'heavy_tail'
            elif dist_type not in DISTRIBUTION_TYPES:
                dist_type = 'symmetric'
            results.append({
                'column_id': column_id,
                'distribution_type': dist_type,
                'approx_skew': stats_dict.get('approx_skew', np.nan),
                'tail_ratio': stats_dict.get('tail_ratio', np.nan),
                'cv': cv_val
            })
        except Exception as e:
            logger.warning(f"检测分布类型失败 (index={idx}): {e}")
            column_id = idx[1] if isinstance(idx, tuple) else idx
            results.append({
                'column_id': column_id,
                'distribution_type': 'heavy_tail',
                'approx_skew': 0.0,
                'tail_ratio': 1.0,
                'cv': np.nan
            })
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index('column_id')
    return result_df

