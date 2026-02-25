"""
连续特征推荐区间模块

基于分布类型检测，为连续特征推荐最优阈值区间。
支持 normal, skewed, heavy_tail, powerlaw, multimodal 等分布类型。
"""
import pandas as pd
import numpy as np
import logging
from scipy.stats import norm
from typing import Optional, Tuple, Dict, Any

try:
    from .analyze_distributions import detect_distribution_type_from_stats
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    from analyze_distributions import detect_distribution_type_from_stats

# 配置日志
logger = logging.getLogger(__name__)


def estimate_coverage_from_quantiles(
    quantile_map: Dict[float, float],
    low: float,
    high: float
) -> float:
    """
    基于分位数映射估算覆盖率（线性插值）
    
    Args:
        quantile_map: 分位数映射，格式 {quantile_value: quantile_prob}
                      例如 {100: 0.05, 200: 0.25, 300: 0.5, 400: 0.75, 500: 0.95}
        low: 区间下界
        high: 区间上界
    
    Returns:
        覆盖率（0-1之间）
    """
    if not quantile_map or len(quantile_map) < 2:
        return 0.0
    
    # 将分位数映射转换为排序列表
    sorted_quantiles = sorted(quantile_map.items())
    quantile_values = [q[0] for q in sorted_quantiles]
    quantile_probs = [q[1] for q in sorted_quantiles]
    
    def interpolate_F(x: float) -> float:
        """线性插值得到 F(x)"""
        if x <= quantile_values[0]:
            return quantile_probs[0]
        if x >= quantile_values[-1]:
            return quantile_probs[-1]
        
        # 找到 x 所在区间
        for i in range(len(quantile_values) - 1):
            if quantile_values[i] <= x <= quantile_values[i + 1]:
                # 线性插值
                x1, p1 = quantile_values[i], quantile_probs[i]
                x2, p2 = quantile_values[i + 1], quantile_probs[i + 1]
                if x2 == x1:
                    return p1
                ratio = (x - x1) / (x2 - x1)
                return p1 + ratio * (p2 - p1)
        
        return 0.0
    
    # 计算覆盖率
    F_low = interpolate_F(low)
    F_high = interpolate_F(high)
    coverage = F_high - F_low
    
    return max(0.0, min(1.0, coverage))


def estimate_coverage(
    mean: float,
    std: float,
    low: float,
    high: float,
    distribution_type: str = "normal",
    median: Optional[float] = None,
    p05: Optional[float] = None,
    p95: Optional[float] = None
) -> float:
    """
    根据分布类型估算覆盖率
    
    Args:
        mean: 均值
        std: 标准差
        low: 区间下界
        high: 区间上界
        distribution_type: 分布类型 ("normal", "skewed", "heavy_tail", "powerlaw", "multimodal")
        median: 中位数（用于某些分布类型）
        p05: 5分位数（可选）
        p95: 95分位数（可选）
    
    Returns:
        覆盖率（0-1之间）
    """
    # 处理 std 很小的情况（近似固定值）
    if std is None or std <= 1e-6 or np.isnan(std):
        # 近似"固定值"，只在区间覆盖 mean 时返回 1
        if low <= mean <= high:
            return 1.0
        else:
            return 0.0
    
    try:
        if distribution_type == "normal":
            # 正态分布：使用标准方法
            z_low = (low - mean) / std
            z_high = (high - mean) / std
            coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
            
        elif distribution_type == "skewed":
            # 偏态分布：使用对数正态分布近似（如果数据为正）
            if mean > 0 and low > 0:
                # 估算对数正态参数
                log_mean = np.log(mean) - 0.5 * np.log(1 + (std/mean)**2)
                log_std = np.sqrt(np.log(1 + (std/mean)**2))
                coverage = float(
                    norm.cdf((np.log(high) - log_mean) / log_std) -
                    norm.cdf((np.log(low) - log_mean) / log_std)
                )
            else:
                # 回退到正态分布
                z_low = (low - mean) / std
                z_high = (high - mean) / std
                coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
                
        elif distribution_type == "heavy_tail":
            # 重尾分布：使用 t 分布（自由度=5）近似
            from scipy.stats import t
            df = 5  # 自由度
            t_low = (low - mean) / std
            t_high = (high - mean) / std
            coverage = float(t.cdf(t_high, df) - t.cdf(t_low, df))
            
        elif distribution_type == "powerlaw":
            # 幂律分布：使用分位数方法
            if p05 is not None and p95 is not None:
                # 基于分位数的线性插值
                if high <= p05:
                    coverage = 0.05 * (high - low) / max(p05 - (p05 - 2*std), 1e-6)
                elif low >= p95:
                    coverage = 0.05 * (high - low) / max((p95 + 2*std) - p95, 1e-6)
                else:
                    # 中间部分使用正态近似
                    z_low = (low - mean) / std
                    z_high = (high - mean) / std
                    coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
            else:
                # 回退到正态分布
                z_low = (low - mean) / std
                z_high = (high - mean) / std
                coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
                
        elif distribution_type == "multimodal":
            # 多峰分布：使用分位数方法
            if p05 is not None and p95 is not None:
                # 简单线性插值
                total_range = p95 - p05
                if total_range > 0:
                    # 计算区间在分位数范围内的比例
                    low_clipped = max(low, p05)
                    high_clipped = min(high, p95)
                    if high_clipped > low_clipped:
                        coverage = (high_clipped - low_clipped) / total_range * 0.9
                    else:
                        coverage = 0.0
                else:
                    coverage = 0.0
            else:
                # 回退到正态分布
                z_low = (low - mean) / std
                z_high = (high - mean) / std
                coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
        else:
            # 未知类型：回退到正态分布
            z_low = (low - mean) / std
            z_high = (high - mean) / std
            coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
        
        return max(0.0, min(1.0, coverage))
        
    except Exception as e:
        logger.warning(f"估算覆盖率失败: distribution_type={distribution_type}, error={e}")
        # 回退到正态分布
        try:
            z_low = (low - mean) / std
            z_high = (high - mean) / std
            coverage = float(norm.cdf(z_high) - norm.cdf(z_low))
            return max(0.0, min(1.0, coverage))
        except:
            return 0.0


def _recommend_normal_threshold(
    full_row: pd.Series,
    cohort_row: pd.Series,
    diff_row: pd.Series,
    config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    正态分布阈值推荐算法
    
    使用 mean/std，枚举基于正态 CDF 的候选区间
    
    Args:
        full_row: 全量统计行
        cohort_row: 圈定统计行
        diff_row: 差异结果行
        config: 配置字典，包含 min_cohort_coverage, min_lift, target_ratio（可选）等
    
    Returns:
        推荐结果字典，包含 rec_low, rec_high, direction, coverage, lift 等
    """
    min_cohort_coverage = config.get('min_cohort_coverage', 0.3)
    min_lift = config.get('min_lift', 1.5)
    target_ratio = config.get('target_ratio')
    
    # 转换为标量的辅助函数
    def to_scalar(val, default):
        if val is None:
            return default
        if isinstance(val, pd.Series):
            if len(val) == 0:
                return default
            return float(val.iloc[0])
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                return float(val[0])
            except:
                return default
        try:
            return float(val)
        except:
            return default
    
    # 获取统计指标（安全提取标量值）
    mean_full = to_scalar(full_row.get('mean'), 0.0)
    std_full = to_scalar(full_row.get('std'), 1.0)
    mean_cohort = to_scalar(cohort_row.get('mean'), 0.0)
    std_cohort = to_scalar(cohort_row.get('std'), 1.0)
    
    # 获取分位数作为anchor点（新格式，已通过stats_loader映射）
    p05_val = to_scalar(cohort_row.get('p05'), None)
    if p05_val is None:
        p05_val = mean_cohort - 2 * std_cohort
    
    p90_val = to_scalar(cohort_row.get('p90'), None)
    if p90_val is None:
        p90_val = mean_cohort + 1.28 * std_cohort
    
    p95_val = to_scalar(cohort_row.get('p95'), None)
    if p95_val is None:
        p95_val = mean_cohort + 1.65 * std_cohort
    
    p99_val = to_scalar(cohort_row.get('p99'), None)
    if p99_val is None:
        p99_val = mean_cohort + 2.33 * std_cohort
    
    anchors_cohort = {
        'p05': p05_val,
        'mean': mean_cohort,
        'p90': p90_val,
        'p95': p95_val,
        'p99': p99_val
    }
    
    # 根据 effect_size 判断方向（安全提取标量值）
    effect_size = to_scalar(diff_row.get('effect_size'), 0.0)
    if effect_size >= 0:
        direction = "high"
        candidate_lows = [anchors_cohort['mean'], anchors_cohort['p90']]
        candidate_highs = [anchors_cohort['p95'], anchors_cohort['p99']]
    else:
        direction = "low"
        candidate_lows = [anchors_cohort['p05']]
        candidate_highs = [anchors_cohort['mean'], anchors_cohort['p90']]
    
    # 若提供 target_ratio，用 k = Φ^{-1}(1−target_ratio) 反推阈值区间（方案对齐）
    if target_ratio is not None and isinstance(target_ratio, (int, float)) and 0 < float(target_ratio) < 1:
        tr = float(target_ratio)
        column_name = diff_row.get('column_name', '')
        if direction == "high":
            k = norm.ppf(1 - tr)
            rec_low = mean_cohort + k * std_cohort
            rec_high = p99_val if p99_val is not None and p99_val > rec_low else rec_low + 2 * std_cohort
            if rec_high <= rec_low:
                rec_high = rec_low + 2 * std_cohort
        else:
            # 低值侧：P(X <= rec_high) = target_ratio => rec_high = μ + Φ^{-1}(target_ratio)*σ
            k = norm.ppf(tr)
            rec_high = mean_cohort + k * std_cohort
            rec_low = p05_val if p05_val is not None and p05_val < rec_high else rec_high - 2 * std_cohort
            if rec_low >= rec_high:
                rec_low = rec_high - 2 * std_cohort
        coverage_cohort = estimate_coverage(mean_cohort, std_cohort, rec_low, rec_high, "normal")
        coverage_full = estimate_coverage(mean_full, std_full, rec_low, rec_high, "normal")
        lift_est = coverage_cohort / coverage_full if coverage_full and coverage_full > 0 else np.nan
        return {
            'rec_low': rec_low,
            'rec_high': rec_high,
            'direction': direction,
            'cohort_coverage_est': coverage_cohort,
            'full_coverage_est': coverage_full,
            'lift_est': lift_est,
            'rule_desc': f"推荐区间：{column_name} ∈ [{rec_low:.2f}, {rec_high:.2f}]（k=Φ^{{-1}}(1−{tr})）",
            'rule_reason': f"normal: target_ratio={tr}, k=Φ^{{-1}}(1−target_ratio)",
            'has_recommendation': True
        }
    
    # 枚举候选区间：先收集满足严格条件的；若无则放宽为任意 lift>1 的候选
    candidate_intervals = []
    all_candidates = []  # 用于放宽时选最佳
    for low in candidate_lows:
        for high in candidate_highs:
            if low >= high:
                continue
            
            # 使用正态分布 CDF 估算覆盖率
            coverage_cohort = estimate_coverage(mean_cohort, std_cohort, low, high, "normal")
            coverage_full = estimate_coverage(mean_full, std_full, low, high, "normal")
            
            if coverage_full <= 0:
                continue
            
            lift = coverage_cohort / coverage_full
            score = lift * coverage_cohort
            rec = {
                'low': low,
                'high': high,
                'coverage_cohort': coverage_cohort,
                'coverage_full': coverage_full,
                'lift': lift,
                'score': score
            }
            all_candidates.append(rec)
            if coverage_cohort >= min_cohort_coverage and lift >= min_lift:
                candidate_intervals.append(rec)
    
    # 优先选满足严格条件的；若无则放宽：取 lift>1 且得分最高的区间
    chosen = candidate_intervals if candidate_intervals else [
        c for c in all_candidates if c['lift'] > 1.0 and c['coverage_cohort'] > 0
    ]
    if chosen:
        best = max(chosen, key=lambda x: x['score'])
        column_name = diff_row.get('column_name', '')
        relaxed = not candidate_intervals
        return {
            'rec_low': best['low'],
            'rec_high': best['high'],
            'direction': direction,
            'cohort_coverage_est': best['coverage_cohort'],
            'full_coverage_est': best['coverage_full'],
            'lift_est': best['lift'],
            'rule_desc': f"推荐区间：{column_name} ∈ [{best['low']:.2f}, {best['high']:.2f}]",
            'rule_reason': "normal: relaxed (below min_coverage/min_lift)" if relaxed else "normal: using normal CDF strategy",
            'has_recommendation': True
        }
    
    return None


def _recommend_tail_quantile_threshold(
    full_row: pd.Series,
    cohort_row: pd.Series,
    diff_row: pd.Series,
    config: Dict[str, Any],
    dist_type: str = "heavy_tail"
) -> Optional[Dict[str, Any]]:
    """
    分位数阈值推荐算法（用于 heavy_tail, powerlaw, skewed, multimodal）
    
    使用分位数估算覆盖率，无需原始数据
    
    对于 heavy_tail: 使用 p90 / p95 / p99 作为右端 anchor 组合区间
    对于 powerlaw: 高值基于 p95–p99–max，低值基于 min–p01–p05
    
    Args:
        full_row: 全量统计行
        cohort_row: 圈定统计行
        diff_row: 差异结果行
        config: 配置字典
        dist_type: 分布类型（"heavy_tail", "powerlaw", "skewed", "multimodal"）
    
    Returns:
        推荐结果字典，包含 rule_reason
    """
    min_cohort_coverage = config.get('min_cohort_coverage', 0.3)
    min_lift = config.get('min_lift', 1.5)
    
    # 转换为标量的辅助函数
    def to_scalar(val, default):
        if val is None:
            return default
        if isinstance(val, pd.Series):
            if len(val) == 0:
                return default
            return float(val.iloc[0])
        if hasattr(val, '__iter__') and not isinstance(val, str):
            try:
                return float(val[0])
            except:
                return default
        try:
            return float(val)
        except:
            return default
    
    # 构建分位数映射
    def build_quantile_map(row, prefix=''):
        """构建分位数映射 {value: probability}"""
        q_map = {}
        
        # 提取分位数（新格式，已通过stats_loader映射为p05, p90, p95, p99等）
        p05 = to_scalar(row.get('p05'), None)
        q1 = to_scalar(row.get('q1'), None)
        median = to_scalar(row.get('median'), None)
        q3 = to_scalar(row.get('q3'), None)
        p90 = to_scalar(row.get('p90'), None)
        p95 = to_scalar(row.get('p95'), None)
        p99 = to_scalar(row.get('p99'), None)
        
        if p05 is not None:
            q_map[p05] = 0.05
        if q1 is not None:
            q_map[q1] = 0.25
        if median is not None:
            q_map[median] = 0.5
        if q3 is not None:
            q_map[q3] = 0.75
        if p90 is not None:
            q_map[p90] = 0.90
        if p95 is not None:
            q_map[p95] = 0.95
        if p99 is not None:
            q_map[p99] = 0.99
        
        return q_map
    
    quantile_map_full = build_quantile_map(full_row)
    quantile_map_cohort = build_quantile_map(cohort_row)
    
    if len(quantile_map_full) < 2 or len(quantile_map_cohort) < 2:
        return None
    
    # 根据 effect_size 判断方向（安全提取标量值）
    effect_size = to_scalar(diff_row.get('effect_size'), 0.0)
    
    # 获取分位数和极值（新格式，已通过stats_loader映射）
    p01 = to_scalar(cohort_row.get('p01'), None)  # p01可能不存在，使用None
    p05 = to_scalar(cohort_row.get('p05'), None)
    p90 = to_scalar(cohort_row.get('p90'), None)
    p95 = to_scalar(cohort_row.get('p95'), None)
    p99 = to_scalar(cohort_row.get('p99'), None)
    min_val = to_scalar(cohort_row.get('min_val'), None)
    max_val = to_scalar(cohort_row.get('max_val'), None)
    
    # 根据分布类型选择候选区间策略
    if dist_type == "powerlaw":
        # powerlaw: 高值基于 p95–p99–max，低值基于 min–p01–p05
        if effect_size >= 0:
            direction = "high"
            # 高值区间：p95–p99–max
            candidate_lows = [v for v in [p95] if v is not None]
            candidate_highs = [v for v in [p99, max_val] if v is not None and v > (candidate_lows[0] if candidate_lows else 0)]
            if not candidate_lows or not candidate_highs:
                # 回退到普通策略
                median_cohort = to_scalar(cohort_row.get('median') or cohort_row.get('q2_cont'), None)
                all_quantile_values = sorted(quantile_map_cohort.keys())
                candidate_values = [v for v in all_quantile_values if v >= (median_cohort or 0)]
                candidate_lows = candidate_values[:1] if candidate_values else []
                candidate_highs = candidate_values[-1:] if candidate_values else []
        else:
            direction = "low"
            # 低值区间：min–p01–p05
            candidate_lows = [v for v in [min_val, p01] if v is not None]
            candidate_highs = [v for v in [p05] if v is not None and v > (candidate_lows[0] if candidate_lows else 0)]
            if not candidate_lows or not candidate_highs:
                # 回退到普通策略
                median_cohort = to_scalar(cohort_row.get('median') or cohort_row.get('q2_cont'), None)
                all_quantile_values = sorted(quantile_map_cohort.keys())
                candidate_values = [v for v in all_quantile_values if v <= (median_cohort or float('inf'))]
                candidate_lows = candidate_values[:1] if candidate_values else []
                candidate_highs = candidate_values[-1:] if candidate_values else []
    else:
        # heavy_tail, skewed, multimodal: 使用 p90 / p95 / p99 作为右端 anchor
        if effect_size >= 0:
            direction = "high"
            # 高值区间：使用 p90 / p95 / p99 作为 anchor
            candidate_lows = [v for v in [p90, p95] if v is not None]
            candidate_highs = [v for v in [p95, p99] if v is not None]
            if not candidate_lows or not candidate_highs:
                # 回退到普通策略
                median_cohort = to_scalar(cohort_row.get('median') or cohort_row.get('q2_cont'), None)
                all_quantile_values = sorted(quantile_map_cohort.keys())
                candidate_values = [v for v in all_quantile_values if v >= (median_cohort or 0)]
                candidate_lows = candidate_values[:1] if candidate_values else []
                candidate_highs = candidate_values[-1:] if candidate_values else []
        else:
            direction = "low"
            # 低值区间：使用较低的分位数
            p25 = to_scalar(cohort_row.get('q1') or cohort_row.get('q1_cont'), None)
            candidate_lows = [v for v in [p05] if v is not None]
            candidate_highs = [v for v in [p25, p90] if v is not None]
            if not candidate_lows or not candidate_highs:
                # 回退到普通策略
                median_cohort = to_scalar(cohort_row.get('median') or cohort_row.get('q2_cont'), None)
                all_quantile_values = sorted(quantile_map_cohort.keys())
                candidate_values = [v for v in all_quantile_values if v <= (median_cohort or float('inf'))]
                candidate_lows = candidate_values[:1] if candidate_values else []
                candidate_highs = candidate_values[-1:] if candidate_values else []
    
    # 枚举候选区间：先收集满足严格条件的；若无则放宽为任意 lift>1 的候选
    candidate_intervals = []
    all_candidates = []
    for low in candidate_lows:
        for high in candidate_highs:
            if low >= high:
                continue
            
            coverage_cohort = estimate_coverage_from_quantiles(quantile_map_cohort, low, high)
            coverage_full = estimate_coverage_from_quantiles(quantile_map_full, low, high)
            
            if coverage_full <= 0:
                continue
            
            lift = coverage_cohort / coverage_full
            score = lift * coverage_cohort
            rec = {
                'low': low,
                'high': high,
                'coverage_cohort': coverage_cohort,
                'coverage_full': coverage_full,
                'lift': lift,
                'score': score
            }
            all_candidates.append(rec)
            if coverage_cohort >= min_cohort_coverage and lift >= min_lift:
                candidate_intervals.append(rec)
    
    chosen = candidate_intervals if candidate_intervals else [
        c for c in all_candidates if c['lift'] > 1.0 and c['coverage_cohort'] > 0
    ]
    if chosen:
        best = max(chosen, key=lambda x: x['score'])
        column_name = diff_row.get('column_name', '')
        relaxed = not candidate_intervals
        if dist_type == "powerlaw":
            base_reason = f"powerlaw: using {'p95–p99–max' if direction == 'high' else 'min–p01–p05'} quantile range"
        elif dist_type == "heavy_tail":
            base_reason = "heavy_tail: using p90–p99 quantile range"
        else:
            base_reason = f"{dist_type}: using quantile range strategy"
        rule_reason = f"{base_reason}; relaxed" if relaxed else base_reason
        return {
            'rec_low': best['low'],
            'rec_high': best['high'],
            'direction': direction,
            'cohort_coverage_est': best['coverage_cohort'],
            'full_coverage_est': best['coverage_full'],
            'lift_est': best['lift'],
            'rule_desc': f"推荐区间：{column_name} ∈ [{best['low']:.2f}, {best['high']:.2f}]",
            'rule_reason': rule_reason,
            'has_recommendation': True
        }
    
    return None


def _to_scalar(val, default):
    """从 Series 或标量安全取 float。"""
    if val is None:
        return default
    if isinstance(val, pd.Series):
        if len(val) == 0:
            return default
        return float(val.iloc[0])
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        try:
            return float(val[0])
        except (IndexError, TypeError, ValueError):
            return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _sanitize_no_inf(low: Any, high: Any, rule_desc: Optional[str]) -> Tuple[Any, Any, str]:
    """禁止输出 Infinity；无界用 NaN；rule_desc 不得含 Infinity 文本。"""
    rec_low = low
    rec_high = high
    if rec_low is not None and (isinstance(rec_low, float) and (np.isinf(rec_low) or rec_low <= -1e30)):
        rec_low = np.nan
    if rec_high is not None and (isinstance(rec_high, float) and (np.isinf(rec_high) or rec_high >= 1e30)):
        rec_high = np.nan
    if rule_desc is None:
        rule_desc = ""
    rule_desc = str(rule_desc).replace("Infinity", "").replace("inf", "").replace("∞", "").strip()
    return rec_low, rec_high, rule_desc


def _get_full_quantile(row: pd.Series, p25_keys, p75_keys, p90_keys, p95_keys, p99_keys, p05_keys, p10_keys, p01_keys):
    """从 full 行取 v2 分位：P25=q3_cont, P75=q9_cont；p90_cont, p95_cont, p99_cont, p05_cont, p10_cont, p01_cont。"""
    def _get(row, *keys):
        for k in keys:
            if k in row.index:
                v = row.get(k)
                if pd.notna(v):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
        return np.nan
    p25 = _get(row, *p25_keys)
    p75 = _get(row, *p75_keys)
    p90 = _get(row, *p90_keys)
    p95 = _get(row, *p95_keys)
    p99 = _get(row, *p99_keys)
    p05 = _get(row, *p05_keys)
    p10 = _get(row, *p10_keys)
    p01 = _get(row, *p01_keys)
    return p25, p75, p90, p95, p99, p05, p10, p01


def _value_at_quantile(quantile_map: Dict[float, float], p: float) -> Optional[float]:
    """给定概率 p，从 value->prob 的分位映射反推分位值（线性插值）。"""
    if not quantile_map or len(quantile_map) < 2 or pd.isna(p):
        return None
    sorted_by_val = sorted(quantile_map.items())
    vals = [x[0] for x in sorted_by_val]
    probs = [x[1] for x in sorted_by_val]
    if p <= probs[0]:
        return float(vals[0])
    if p >= probs[-1]:
        return float(vals[-1])
    for i in range(len(probs) - 1):
        if probs[i] <= p <= probs[i + 1]:
            if probs[i + 1] == probs[i]:
                return float(vals[i])
            r = (p - probs[i]) / (probs[i + 1] - probs[i])
            return float(vals[i] + r * (vals[i + 1] - vals[i]))
    return None


def _recommend_symmetric_tail(
    full_row: pd.Series,
    cohort_row: pd.Series,
    diff_row: pd.Series,
    max_base_cov: float,
) -> Optional[Dict[str, Any]]:
    """
    对称分布使用方向性单侧尾部阈值（不再用 P25~P75 常态区间）。
    上尾 q in [0.90, 0.85, 0.80, 0.75]，下尾 q in [0.10, 0.15, 0.20, 0.25]；
    选满足 full_coverage_est <= max_base_cov 且 lift 最大的；若无则取最稀有（0.90 或 0.10）。
    """
    column_name = diff_row.get('column_name', '')
    p25_keys = ('q3_cont', 'q1')
    p75_keys = ('q9_cont', 'q3')
    p90_keys = ('p90_cont', 'p90')
    p95_keys = ('p95_cont', 'p95')
    p99_keys = ('p99_cont', 'p99')
    p05_keys = ('p05_cont', 'p05')
    p10_keys = ('p10_cont', 'p10')
    p01_keys = ('p01_cont', 'p01')
    P25_full, P75_full, p90_full, p95_full, p99_full, p05_full, p10_full, p01_full = _get_full_quantile(
        full_row, p25_keys, p75_keys, p90_keys, p95_keys, p99_keys, p05_keys, p10_keys, p01_keys
    )
    med_full = _to_scalar(full_row.get('q6_cont'), _to_scalar(full_row.get('median'), np.nan))
    med_cohort = _to_scalar(cohort_row.get('q6_cont'), _to_scalar(cohort_row.get('median'), np.nan))
    mean_full = _to_scalar(full_row.get('mean_val'), np.nan)
    mean_cohort = _to_scalar(cohort_row.get('mean_val'), np.nan)
    delta_median = _to_scalar(diff_row.get('delta_median'), None)
    if delta_median is not None and not np.isnan(delta_median):
        direction = "high" if delta_median >= 0 else "low"
    elif pd.notna(med_cohort) and pd.notna(med_full):
        direction = "high" if med_cohort >= med_full else "low"
    elif pd.notna(mean_cohort) and pd.notna(mean_full):
        direction = "high" if mean_cohort >= mean_full else "low"
    else:
        direction = "high" if _to_scalar(diff_row.get('effect_size'), 0) >= 0 else "low"

    def build_quantile_map(row):
        q_map = {}
        for key, prob in [
            ('p05_cont', 0.05), ('p05', 0.05), ('q3_cont', 0.25), ('q1', 0.25),
            ('q6_cont', 0.5), ('median', 0.5), ('q9_cont', 0.75), ('q3', 0.75),
            ('p90_cont', 0.9), ('p90', 0.9), ('p95_cont', 0.95), ('p95', 0.95), ('p99_cont', 0.99), ('p99', 0.99),
        ]:
            if key not in row.index:
                continue
            v = _to_scalar(row.get(key), np.nan)
            if pd.notna(v):
                q_map[float(v)] = prob
        return q_map

    qm_full = build_quantile_map(full_row)
    qm_cohort = build_quantile_map(cohort_row)
    if len(qm_full) < 2 or len(qm_cohort) < 2:
        return None

    big = 1e30
    candidates = []
    if direction == "high":
        q_list = [0.90, 0.85, 0.80, 0.75]
        for q in q_list:
            Pq = _value_at_quantile(qm_full, q)
            if Pq is None:
                continue
            full_cov = estimate_coverage_from_quantiles(qm_full, Pq, big)
            cohort_cov = estimate_coverage_from_quantiles(qm_cohort, Pq, big)
            if full_cov <= 0:
                continue
            lift = cohort_cov / full_cov if full_cov else np.nan
            if pd.notna(lift):
                candidates.append((q, Pq, full_cov, cohort_cov, lift))
        if not candidates:
            return None
        allowed = [(q, Pq, fc, cc, lift) for (q, Pq, fc, cc, lift) in candidates if fc <= max_base_cov]
        if allowed:
            best = max(allowed, key=lambda x: (x[4], -x[2]))
        else:
            best = max(candidates, key=lambda x: x[0])
        q_best, Pq_best, full_cov, cohort_cov, lift_est = best
        rec_low, rec_high = float(Pq_best), np.nan
        rule_reason = f"symmetric: tail high P{int(q_best*100)}_full"
    else:
        q_list = [0.10, 0.15, 0.20, 0.25]
        for q in q_list:
            Pq = _value_at_quantile(qm_full, q)
            if Pq is None:
                continue
            full_cov = estimate_coverage_from_quantiles(qm_full, -big, Pq)
            cohort_cov = estimate_coverage_from_quantiles(qm_cohort, -big, Pq)
            if full_cov <= 0:
                continue
            lift = cohort_cov / full_cov if full_cov else np.nan
            if pd.notna(lift):
                candidates.append((q, Pq, full_cov, cohort_cov, lift))
        if not candidates:
            return None
        allowed = [(q, Pq, fc, cc, lift) for (q, Pq, fc, cc, lift) in candidates if fc <= max_base_cov]
        if allowed:
            best = max(allowed, key=lambda x: (x[4], -x[2]))
        else:
            best = min(candidates, key=lambda x: x[0])
        q_best, Pq_best, full_cov, cohort_cov, lift_est = best
        rec_low, rec_high = np.nan, float(Pq_best)
        rule_reason = f"symmetric: tail low P{int(q_best*100)}_full"
    rec_low, rec_high, _ = _sanitize_no_inf(rec_low, rec_high, None)
    low_str = f"{rec_low:.2f}" if pd.notna(rec_low) and np.isfinite(rec_low) else "无下界"
    high_str = f"{rec_high:.2f}" if pd.notna(rec_high) and np.isfinite(rec_high) else "无上界"
    rule_desc = f"推荐区间：{column_name} ∈ [{low_str}, {high_str}]"
    rec_low, rec_high, rule_desc = _sanitize_no_inf(rec_low, rec_high, rule_desc)
    return {
        'rec_low': rec_low,
        'rec_high': rec_high,
        'direction': direction,
        'cohort_coverage_est': cohort_cov,
        'full_coverage_est': full_cov,
        'lift_est': lift_est,
        'rule_desc': rule_desc,
        'rule_reason': rule_reason,
        'has_recommendation': True,
    }


def _recommend_fixed_strategy(
    full_row: pd.Series,
    cohort_row: pd.Series,
    diff_row: pd.Series,
    distribution_type: str,
    max_base_cov_for_numeric: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    按分布类型固定策略给出区间，不依赖 coverage/lift 决策。
    全部使用 full 行 v2 分位：q3_cont=P25, q9_cont=P75；p95_cont, p90_cont, p99_cont, p05_cont, p10_cont, p01_cont。
    无上界/无下界一律用 NaN，禁止 inf/Infinity。
    """
    column_name = diff_row.get('column_name', '')
    effect_size = _to_scalar(diff_row.get('effect_size'), 0.0)
    direction = "high" if effect_size >= 0 else "low"
    p25_keys = ('q3_cont', 'q1')
    p75_keys = ('q9_cont', 'q3')
    p90_keys = ('p90_cont', 'p90')
    p95_keys = ('p95_cont', 'p95')
    p99_keys = ('p99_cont', 'p99')
    p05_keys = ('p05_cont', 'p05')
    p10_keys = ('p10_cont', 'p10')
    p01_keys = ('p01_cont', 'p01')
    P25_full, P75_full, p90_full, p95_full, p99_full, p05_full, p10_full, p01_full = _get_full_quantile(
        full_row, p25_keys, p75_keys, p90_keys, p95_keys, p99_keys, p05_keys, p10_keys, p01_keys
    )
    P25_cohort = _to_scalar(cohort_row.get('q3_cont'), _to_scalar(cohort_row.get('q1'), np.nan))
    P75_cohort = _to_scalar(cohort_row.get('q9_cont'), _to_scalar(cohort_row.get('q3'), np.nan))
    rec_low, rec_high = np.nan, np.nan
    rule_reason = ""
    if distribution_type == "symmetric":
        max_cov = max_base_cov_for_numeric if max_base_cov_for_numeric is not None else 0.15
        rec = _recommend_symmetric_tail(full_row, cohort_row, diff_row, max_cov)
        if rec is not None:
            return rec
        return None
    elif distribution_type == "skewed":
        if direction == "high":
            if pd.notna(P75_full):
                rec_low, rec_high = float(P75_full), np.nan
                rule_reason = "skewed: P75_full (high side)"
            else:
                return None
        else:
            if pd.notna(P25_full):
                rec_low, rec_high = np.nan, float(P25_full)
                rule_reason = "skewed: P25_full (low side)"
            else:
                return None
    elif distribution_type == "heavy_tail":
        if direction == "high":
            thresh = p95_full
            if not (pd.notna(thresh) and np.isfinite(thresh)):
                thresh = p90_full
            if not (pd.notna(thresh) and np.isfinite(thresh)):
                thresh = p99_full
            if pd.notna(thresh) and np.isfinite(thresh):
                rec_low, rec_high = float(thresh), np.nan
                rule_reason = "heavy_tail: P95_full (fallback P90_full, P99_full)"
            else:
                return None
        else:
            thresh = p05_full
            if not (pd.notna(thresh) and np.isfinite(thresh)):
                thresh = p10_full
            if not (pd.notna(thresh) and np.isfinite(thresh)):
                thresh = p01_full
            if pd.notna(thresh) and np.isfinite(thresh):
                rec_low, rec_high = np.nan, float(thresh)
                rule_reason = "heavy_tail: P05_full (fallback P10_full, P01_full)"
            else:
                return None
    elif distribution_type == "zero_inflated":
        if direction == "high":
            rec_low, rec_high = 0.0, np.nan
            rule_reason = "zero_inflated: >0"
        else:
            rec_low, rec_high = np.nan, 0.0
            rule_reason = "zero_inflated: =0"
    else:
        return None
    rec_low, rec_high, _ = _sanitize_no_inf(rec_low, rec_high, None)
    def build_quantile_map(row):
        q_map = {}
        for key, prob in [
            ('p05_cont', 0.05), ('p05', 0.05), ('q3_cont', 0.25), ('q1', 0.25),
            ('q6_cont', 0.5), ('median', 0.5), ('q9_cont', 0.75), ('q3', 0.75),
            ('p90_cont', 0.9), ('p90', 0.9), ('p95_cont', 0.95), ('p95', 0.95), ('p99_cont', 0.99), ('p99', 0.99),
        ]:
            if key not in row.index:
                continue
            v = _to_scalar(row.get(key), np.nan)
            if pd.notna(v):
                q_map[float(v)] = prob
        return q_map
    qm_full = build_quantile_map(full_row)
    qm_cohort = build_quantile_map(cohort_row)
    cohort_coverage_est = estimate_coverage_from_quantiles(qm_cohort, rec_low, rec_high) if len(qm_cohort) >= 2 else np.nan
    full_coverage_est = estimate_coverage_from_quantiles(qm_full, rec_low, rec_high) if len(qm_full) >= 2 else np.nan
    lift_est = cohort_coverage_est / full_coverage_est if full_coverage_est and full_coverage_est > 0 else np.nan
    low_str = f"{rec_low:.2f}" if pd.notna(rec_low) and np.isfinite(rec_low) else "无下界"
    high_str = f"{rec_high:.2f}" if pd.notna(rec_high) and np.isfinite(rec_high) else "无上界"
    rule_desc = f"推荐区间：{column_name} ∈ [{low_str}, {high_str}]"
    if distribution_type == "zero_inflated":
        rule_desc = f"推荐区间：{column_name} > 0" if direction == "high" else f"推荐区间：{column_name} = 0"
    rec_low, rec_high, rule_desc = _sanitize_no_inf(rec_low, rec_high, rule_desc)
    return {
        'rec_low': rec_low,
        'rec_high': rec_high,
        'direction': direction,
        'cohort_coverage_est': cohort_coverage_est,
        'full_coverage_est': full_coverage_est,
        'lift_est': lift_est,
        'rule_desc': rule_desc,
        'rule_reason': rule_reason,
        'has_recommendation': True,
    }


def recommend_numeric_thresholds(
    numeric_diff_df: pd.DataFrame,
    full_numeric_df: pd.DataFrame,
    cohort_numeric_df: pd.DataFrame,
    min_cohort_coverage: float = 0.1,
    min_lift: float = 1.2,
    target_ratio: Optional[float] = None,
    max_base_cov_for_numeric: Optional[float] = None,
) -> pd.DataFrame:
    """
    为连续特征推荐最优阈值区间（按分布类型固定策略，不依赖 coverage/lift 决策）。
    
    分布类型与策略：symmetric→[P25_full, P75_full]；skewed→单侧 P75/P25；
    heavy_tail→P95_full（fallback P90/P99）；zero_inflated→>0。
    cohort_coverage_est / full_coverage_est / lift_est 仅作为 backward-compatible debug 列输出。
    
    Args:
        numeric_diff_df: 差异计算结果DataFrame，index为column_id，必须包含 distribution_type 列
        full_numeric_df: 全量统计DataFrame，index=['stat_date', 'column_id']
        cohort_numeric_df: 圈定统计DataFrame，index=['stat_date', 'column_id']
        min_cohort_coverage: 保留参数，不再参与决策
        min_lift: 保留参数，不再参与决策
        target_ratio: 保留参数，不再参与决策
    
    Returns:
        DataFrame，index为column_id，包含 rec_low, rec_high, direction,
        cohort_coverage_est, full_coverage_est, lift_est（debug）, rule_desc, rule_reason, has_recommendation
    """
    logger.info("开始推荐连续特征阈值区间...")
    
    # 配置字典
    config = {
        'min_cohort_coverage': min_cohort_coverage,
        'min_lift': min_lift,
        'target_ratio': target_ratio
    }
    
    # 存储推荐结果
    recommendations = []
    
    # 遍历每个字段
    for column_id in numeric_diff_df.index:
        try:
            # 获取差异结果中的统计信息（若 index 重复则取首行得标量）
            diff_row = numeric_diff_df.loc[column_id]
            if isinstance(diff_row, pd.DataFrame):
                diff_row = diff_row.iloc[0]
            stat_date = diff_row['stat_date']
            if isinstance(stat_date, pd.Series):
                stat_date = stat_date.iloc[0]
            
            # 获取分布类型（从差异结果中读取；与 detect_distribution_types_from_quantiles 输出一致）
            distribution_type = diff_row.get('distribution_type', 'symmetric')
            if isinstance(distribution_type, pd.Series):
                distribution_type = distribution_type.iloc[0]
            if pd.isna(distribution_type) or distribution_type == '':
                distribution_type = 'symmetric'
            # 兼容旧输出：normal→symmetric，powerlaw/multimodal→heavy_tail
            if distribution_type == "normal":
                distribution_type = "symmetric"
            elif distribution_type in ("powerlaw", "multimodal"):
                distribution_type = "heavy_tail"
            
            # 从全量和圈定数据中获取详细统计（若 xs 返回多行则取首行）
            try:
                full_row = full_numeric_df.xs((stat_date, column_id), level=['stat_date', 'column_id'])
            except KeyError:
                logger.warning(f"全量数据中未找到 (stat_date={stat_date}, column_id={column_id})")
                continue
            if isinstance(full_row, pd.DataFrame):
                full_row = full_row.iloc[0]
            
            try:
                cohort_row = cohort_numeric_df.xs((stat_date, column_id), level=['stat_date', 'column_id'])
            except KeyError:
                logger.warning(f"圈定数据中未找到 (stat_date={stat_date}, column_id={column_id})")
                continue
            if isinstance(cohort_row, pd.DataFrame):
                cohort_row = cohort_row.iloc[0]
            
            # 全量/圈定分位列（供 Stage2 多阈值）
            p25_keys = ('q3_cont', 'q1')
            p75_keys = ('q9_cont', 'q3')
            p90_keys = ('p90_cont', 'p90')
            p95_keys = ('p95_cont', 'p95')
            p99_keys = ('p99_cont', 'p99')
            p05_keys = ('p05_cont', 'p05')
            p10_keys = ('p10_cont', 'p10')
            p01_keys = ('p01_cont', 'p01')
            P25_f, P75_f, p90_f, p95_f, p99_f, p05_f, p10_f, p01_f = _get_full_quantile(
                full_row, p25_keys, p75_keys, p90_keys, p95_keys, p99_keys, p05_keys, p10_keys, p01_keys
            )
            p50_f = _to_scalar(full_row.get('q6_cont'), _to_scalar(full_row.get('median'), np.nan))
            P25_c, P75_c, p90_c, p95_c, p99_c, p05_c, p10_c, p01_c = _get_full_quantile(
                cohort_row, p25_keys, p75_keys, p90_keys, p95_keys, p99_keys, p05_keys, p10_keys, p01_keys
            )
            p50_c = _to_scalar(cohort_row.get('q6_cont'), _to_scalar(cohort_row.get('median'), np.nan))
            quantile_cols = {
                'p10_full': p10_f, 'p25_full': P25_f, 'p50_full': p50_f, 'p75_full': P75_f, 'p90_full': p90_f, 'p95_full': p95_f,
                'p10_cohort': p10_c, 'p25_cohort': P25_c, 'p50_cohort': p50_c, 'p75_cohort': P75_c, 'p90_cohort': p90_c, 'p95_cohort': p95_c,
            }
            max_cov = max_base_cov_for_numeric if max_base_cov_for_numeric is not None else 0.15
            # 按分布类型固定策略推荐；coverage/lift 仅作 debug 输出，不参与决策
            rec = _recommend_fixed_strategy(full_row, cohort_row, diff_row, distribution_type, max_base_cov_for_numeric=max_cov)
            
            # 生成推荐结果（再次 sanitize 禁止 Infinity 输出）
            if rec is not None:
                rl, rh, rd = _sanitize_no_inf(rec['rec_low'], rec['rec_high'], rec.get('rule_desc'))
                recommendations.append({
                    'column_id': column_id,
                    'rec_low': rl,
                    'rec_high': rh,
                    'direction': rec['direction'],
                    'cohort_coverage_est': rec['cohort_coverage_est'],
                    'full_coverage_est': rec['full_coverage_est'],
                    'lift_est': rec['lift_est'],
                    'rule_desc': rd or rec.get('rule_desc', ''),
                    'rule_reason': rec.get('rule_reason', ''),
                    'has_recommendation': rec['has_recommendation'],
                    **quantile_cols,
                })
            else:
                # 没有推荐
                column_name = diff_row.get('column_name', column_id)
                # 根据 effect_size 判断方向（即使没有推荐也记录方向）
                effect_size = float(diff_row.get('effect_size', 0))
                direction = "high" if effect_size >= 0 else "low"
                
                recommendations.append({
                    'column_id': column_id,
                    'rec_low': np.nan,
                    'rec_high': np.nan,
                    'direction': direction,
                    'cohort_coverage_est': np.nan,
                    'full_coverage_est': np.nan,
                    'lift_est': np.nan,
                    'rule_desc': None,
                    'rule_reason': f"{distribution_type}: no suitable threshold found",
                    'has_recommendation': False,
                    **quantile_cols,
                })
                
        except Exception as e:
            logger.warning(f"处理字段 {column_id} 时出错: {e}")
            continue
    
    if not recommendations:
        logger.warning("未生成任何推荐区间")
        return pd.DataFrame()
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(recommendations)
    result_df = result_df.set_index('column_id')
    # PATCH-4: 禁止 Infinity 出现在输出
    for col in ('rec_low', 'rec_high'):
        if col in result_df.columns:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
    if 'rule_desc' in result_df.columns:
        result_df['rule_desc'] = result_df['rule_desc'].astype(str).str.replace('Infinity', '', regex=False).str.replace('inf', '', regex=False).str.strip()
    logger.info(f"阈值推荐完成，共 {len(result_df)} 个字段")
    return result_df

