"""
阶段2主入口模块

串联所有模块，实现多特征组合规则生成。

Stage2 为 coverage-free：所有评分与筛选仅使用差异评分、稳定性评分、多样性评分，
不使用任何覆盖率或 lift 参与计算。Stage1 输出中的 cohort_coverage_est、full_coverage_est、
lift_est（连续）以及 cohort_coverage、full_coverage、lift（离散）等列若存在则接受并保留，
仅用于 backward-compatible 与 debug，不参与原子规则生成、Beam Search、客群评分或 portfolio 选择；
上述列不存在时不得报错（optional debug columns）。
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .stage2_config import Stage2Config
    from .atomic_rules import (
        generate_numeric_atomic_rules,
        generate_categorical_atomic_rules,
        merge_atomic_rules,
        calculate_divergence_score,
        calculate_categorical_divergence_score,
        calculate_stability_score
    )
    from .rule_combination import combine_rules_beam_search
    from .segment_scoring import filter_candidate_segments
    from .segment_portfolio import build_segment_portfolio, dedup_same_structure_candidates
    from .pair_assoc import PairAssocIndex
    from .rule_output import (
        export_atomic_rules,
        export_candidate_segments,
        export_segment_portfolio,
        segment_canonical_key,
    )
except ImportError:
    from stage2_config import Stage2Config
    from atomic_rules import (
        generate_numeric_atomic_rules,
        generate_categorical_atomic_rules,
        merge_atomic_rules,
        calculate_divergence_score,
        calculate_categorical_divergence_score,
        calculate_stability_score
    )
    from rule_combination import combine_rules_beam_search
    from segment_scoring import filter_candidate_segments
    from segment_portfolio import build_segment_portfolio, dedup_same_structure_candidates
    try:
        from pair_assoc import PairAssocIndex
    except ImportError:
        PairAssocIndex = None
    from rule_output import (
        export_atomic_rules,
        export_candidate_segments,
        export_segment_portfolio,
        segment_canonical_key,
    )

try:
    from stage1.threshold_numeric import estimate_coverage_from_quantiles
except ImportError:
    try:
        import sys
        from pathlib import Path
        _root = Path(__file__).resolve().parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from stage1.threshold_numeric import estimate_coverage_from_quantiles
    except ImportError:
        estimate_coverage_from_quantiles = None  # optional for per-rule coverage

# Stage1 输出若为业务可读表头时，反向映射为技术列名以便后续计算。
# 其中 cohort_coverage_est / full_coverage_est / lift_est 等为 optional debug 列，存在则映射，不存在不报错。
STAGE1_NUMERIC_HEADER_REVERSE = {
    '字段ID': 'column_id', '字段名称': 'column_name', '统计月份': 'stat_date',
    '全量均值': 'mean_full', '对比客群均值': 'mean_base', '均值差异（对比−全量）': 'mean_diff',
    '相对差异': 'mean_diff_ratio', '效应量': 'effect_size', '中位数差异': 'delta_median',
    'P95分位数差异': 'delta_p95', '四分位距差异': 'delta_IQR', '变异系数差异': 'delta_CV',
    '综合差异分数': 'diff_score', '是否显著差异': 'is_significant', '分布类型': 'distribution_type',
    '近似偏度': 'approx_skew', '尾部厚度': 'tail_ratio', '变异系数': 'cv',
    '推荐区间下界': 'rec_low', '推荐区间上界': 'rec_high', '推荐方向': 'direction',
    '对比客群估算覆盖率': 'cohort_coverage_est', '全量估算覆盖率': 'full_coverage_est', '估算Lift': 'lift_est',
    '规则描述': 'rule_desc', '推荐理由': 'rule_reason', '是否给出阈值推荐': 'has_recommendation',
}
STAGE1_CATEGORICAL_HEADER_REVERSE = {
    '字段ID': 'column_id', '字段名称': 'column_name', '统计月份': 'stat_date',
    '占比差异绝对值之和': 'sum_abs_diff', '最大单类占比差异': 'max_abs_diff', '差异最大类别及差值': 'top_diff_categories',
    '推荐类别（圈定>全量）': 'recommended_categories', '熵差异': 'entropy_diff', '基尼系数差异': 'gini_diff',
    '综合差异分数': 'diff_score', '推荐类别集合': 'rec_categories',
    '对比客群命中占比': 'cohort_coverage', '全量命中占比': 'full_coverage', '覆盖增幅': 'lift',
    '全量命中人数': 'full_hit_count', '对比客群命中人数': 'cohort_hit_count', '推荐类别个数': 'rec_category_count',
    '规则描述': 'rule_desc',
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数：解析命令行参数并执行阶段2流程"""
    parser = argparse.ArgumentParser(
        description='阶段2：多特征组合规则生成',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--stage1-output-dir',
        type=str,
        default='../data/stage1_output',
        help='阶段1输出目录（默认: ../data/stage1_output）'
    )
    
    parser.add_argument(
        '--stat-date',
        type=str,
        required=True,
        help='统计日期，格式为YYYYMM，如20251204（必填）'
    )
    
    parser.add_argument(
        '--cohort-name',
        type=str,
        required=True,
        help='客群名称/ID，如PRODA（必填）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/stage2_output',
        help='阶段2输出目录（默认: ./data/stage2_output）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='统一配置文件路径（JSON，与 Stage1 共用 config.json 时使用其中的 stage2 段；不指定则尝试项目根目录 config.json）'
    )
    
    parser.add_argument(
        '--full-stats-dir',
        type=str,
        default='./data/full_stats',
        help='全量统计目录，用于自动查找字段对关联表 ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx（默认: ./data/full_stats）'
    )
    parser.add_argument(
        '--pair-assoc',
        type=str,
        default=None,
        help='字段对关联统计表路径（可选）；不指定时自动在 full-stats-dir 下查找 ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx'
    )
    # 输出目标约束（平衡准确率与覆盖率）
    parser.add_argument('--target-precision-min', type=float, default=None,
                        help='期望准确率下限（默认 0.8，即高准确率 80%% 以上）；估计准确率低于此的 segment 不进入输出')
    parser.add_argument('--target-coverage-min', type=float, default=None,
                        help='期望覆盖率下限，即目标召回率（默认 0.7）')
    parser.add_argument('--target-segments', type=int, default=None,
                        help='预期输出客群数；不指定时使用 config 的 max_segments')
    parser.add_argument('--target-user-10k', type=float, default=None,
                        help='预期圈定用户数（万人），可选；需全量人数时参与换算')
    parser.add_argument('--use-f1-balance', type=lambda x: x.lower() in ('1', 'true', 'yes'), default=None,
                        help='是否用 F1 平衡准确率与覆盖率（默认 True）')
    parser.add_argument('--target-priority', type=str, default=None, choices=['f1', 'precision_first', 'coverage_first'],
                        help='无法同时满足时优先级（默认 f1）')

    args = parser.parse_args()

    try:
        run_stage2_analysis(
            stage1_output_dir=args.stage1_output_dir,
            stat_date=args.stat_date,
            cohort_name=args.cohort_name,
            output_dir=args.output_dir,
            config_path=args.config,
            pair_assoc_path=args.pair_assoc,
            full_stats_dir=args.full_stats_dir,
            target_precision_min=args.target_precision_min,
            target_coverage_min=args.target_coverage_min,
            target_segment_count=args.target_segments,
            target_user_count_10k=args.target_user_10k,
            use_f1_balance=args.use_f1_balance,
            target_priority=args.target_priority,
        )
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


# 字段对关联表默认文件名（全量目录下）
PAIR_ASSOC_FILENAME_TEMPLATE = "ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx"


def _build_quantile_map_from_row(row, full_prefix: str = 'full'):
    """Build {value: prob} from numeric_diff row: p10_full->0.1, p25_full->0.25, ..."""
    q_specs = [
        ('p10_' + full_prefix, 0.10), ('p25_' + full_prefix, 0.25), ('p50_' + full_prefix, 0.50),
        ('p75_' + full_prefix, 0.75), ('p90_' + full_prefix, 0.90), ('p95_' + full_prefix, 0.95),
    ]
    q_map = {}
    for col_name, prob in q_specs:
        if col_name not in row.index:
            continue
        v = row.get(col_name)
        if pd.notna(v) and np.isfinite(v):
            try:
                q_map[float(v)] = prob
            except (TypeError, ValueError):
                pass
    return q_map


def _compute_atomic_cov_lift(
    atomic_rules_df: pd.DataFrame,
    numeric_diff_df: pd.DataFrame,
    categorical_diff_df: pd.DataFrame,
) -> pd.DataFrame:
    """为原子规则表补齐 base_cov、sub_cov、lift、cov_unknown（v2.0 每条连续规则按区间算覆盖率）。"""
    eps = 1e-9
    base_covs = []
    sub_covs = []
    lifts = []
    cov_unknowns = []
    for _, row in atomic_rules_df.iterrows():
        base_cov = None
        sub_cov = None
        cov_unknown = False
        col_id = row.get('column_id')
        if row.get('rule_type_feature') == 'numeric':
            rule_low = row.get('rule_low')
            rule_high = row.get('rule_high')
            low_ok = rule_low is not None and pd.notna(rule_low) and (isinstance(rule_low, (int, float)) and np.isfinite(rule_low) or rule_low == -np.inf)
            high_ok = rule_high is not None and pd.notna(rule_high) and (isinstance(rule_high, (int, float)) and np.isfinite(rule_high) or rule_high == np.inf)
            if low_ok and high_ok and col_id in numeric_diff_df.index and estimate_coverage_from_quantiles is not None:
                r = numeric_diff_df.loc[col_id]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                qm_full = _build_quantile_map_from_row(r, 'full')
                qm_cohort = _build_quantile_map_from_row(r, 'cohort')
                try:
                    low = float(rule_low) if rule_low != -np.inf else -1e30
                    high = float(rule_high) if rule_high != np.inf else 1e30
                    if len(qm_full) >= 2:
                        base_cov = float(estimate_coverage_from_quantiles(qm_full, low, high))
                    if len(qm_cohort) >= 2:
                        sub_cov = float(estimate_coverage_from_quantiles(qm_cohort, low, high))
                    if base_cov is None and sub_cov is None:
                        cov_unknown = True
                except Exception:
                    cov_unknown = True
            if base_cov is None or sub_cov is None:
                if row.get('rule_type') == 'main' and col_id in numeric_diff_df.index:
                    r = numeric_diff_df.loc[col_id]
                    if isinstance(r, pd.DataFrame):
                        r = r.iloc[0]
                    base_cov = base_cov or _safe_float(r.get('full_coverage_est'))
                    sub_cov = sub_cov or _safe_float(r.get('cohort_coverage_est'))
                if base_cov is None or sub_cov is None:
                    cov_unknown = True
        else:
            if col_id in categorical_diff_df.index:
                r = categorical_diff_df.loc[col_id]
                base_cov = _safe_float(r.get('full_coverage'))
                sub_cov = _safe_float(r.get('cohort_coverage'))
            if base_cov is None and sub_cov is None:
                cov_unknown = True
        base_covs.append(base_cov)
        sub_covs.append(sub_cov)
        cov_unknowns.append(cov_unknown)
        if base_cov is not None and sub_cov is not None and base_cov >= eps:
            lifts.append(sub_cov / base_cov)
        else:
            lifts.append(None)
    df = atomic_rules_df.copy()
    df['base_cov'] = base_covs
    df['sub_cov'] = sub_covs
    df['_lift'] = lifts
    df['cov_unknown'] = cov_unknowns
    return df


def _effective_max_base_cov(row: pd.Series, max_base_cov: float, config: 'Stage2Config') -> float:
    """按规则类型返回用于过滤的 base_cov 上限，便于 main/离散规则放宽进入 beam。"""
    if row.get('base_cov') is None and row.get('sub_cov') is None:
        return max_base_cov
    if getattr(config, 'allow_main_for_beam', False) and row.get('rule_type') == 'main':
        return getattr(config, 'main_max_base_cov', 0.30)
    if row.get('rule_type_feature') == 'categorical':
        return getattr(config, 'categorical_max_base_cov', 0.40)
    return max_base_cov


def _apply_one_round_filter(
    df: pd.DataFrame,
    max_base_cov: float,
    min_lift: float,
    min_sub_cov: float,
    use_precision: bool,
    prior_pi: float,
    min_precision_mult: float,
) -> Tuple[pd.DataFrame, dict]:
    """应用一轮精度条件，返回 (保留的 DataFrame, 本轮被过滤原因计数)。"""
    eps = 1e-9
    reason_counts = {'base_cov_high': 0, 'sub_cov_low': 0, 'lift_low': 0, 'precision_low': 0, 'no_cov': 0}
    keep_mask = []
    for _, row in df.iterrows():
        base_cov = row.get('base_cov')
        sub_cov = row.get('sub_cov')
        lift = row.get('_lift')
        if base_cov is None or sub_cov is None:
            reason_counts['no_cov'] += 1
            keep_mask.append(True)
            continue
        if base_cov > max_base_cov:
            reason_counts['base_cov_high'] += 1
            keep_mask.append(False)
        elif sub_cov < min_sub_cov:
            reason_counts['sub_cov_low'] += 1
            keep_mask.append(False)
        elif lift is None or lift < min_lift:
            reason_counts['lift_low'] += 1
            keep_mask.append(False)
        elif use_precision and prior_pi > 0:
            precision_proxy = (sub_cov * prior_pi) / max(base_cov, eps)
            if precision_proxy < min_precision_mult * prior_pi:
                reason_counts['precision_low'] += 1
                keep_mask.append(False)
            else:
                keep_mask.append(True)
        else:
            keep_mask.append(True)
    filtered = df.loc[np.array(keep_mask)].reset_index(drop=True)
    return filtered, reason_counts


def _apply_one_round_filter_typed(
    df: pd.DataFrame,
    max_base_cov: float,
    min_lift: float,
    min_sub_cov: float,
    use_precision: bool,
    prior_pi: float,
    min_precision_mult: float,
    config: Stage2Config,
) -> Tuple[pd.DataFrame, dict]:
    """按规则类型使用不同 base_cov 上限的一轮精度过滤，使 main/离散规则有机会进入 beam。"""
    eps = 1e-9
    reason_counts = {'base_cov_high': 0, 'sub_cov_low': 0, 'lift_low': 0, 'precision_low': 0, 'no_cov': 0}
    keep_mask = []
    for _, row in df.iterrows():
        base_cov = row.get('base_cov')
        sub_cov = row.get('sub_cov')
        lift = row.get('_lift')
        if base_cov is None or sub_cov is None:
            reason_counts['no_cov'] += 1
            keep_mask.append(True)
            continue
        effective_max = _effective_max_base_cov(row, max_base_cov, config)
        if base_cov > effective_max:
            reason_counts['base_cov_high'] += 1
            keep_mask.append(False)
        elif sub_cov < min_sub_cov:
            reason_counts['sub_cov_low'] += 1
            keep_mask.append(False)
        elif lift is None or lift < min_lift:
            reason_counts['lift_low'] += 1
            keep_mask.append(False)
        elif use_precision and prior_pi and prior_pi > 0:
            precision_proxy = (sub_cov * prior_pi) / max(base_cov, eps)
            if precision_proxy < min_precision_mult * prior_pi:
                reason_counts['precision_low'] += 1
                keep_mask.append(False)
            else:
                keep_mask.append(True)
        else:
            keep_mask.append(True)
    filtered = df.loc[np.array(keep_mask)].reset_index(drop=True)
    return filtered, reason_counts


def _filter_atomic_rules_by_precision(
    atomic_rules_df: pd.DataFrame,
    numeric_diff_df: pd.DataFrame,
    categorical_diff_df: pd.DataFrame,
    config: Stage2Config,
) -> Tuple[pd.DataFrame, float, dict]:
    """
    原子规则精度过滤（永不清零 + 自动降级）：Round1 严格 -> Round2 放宽 -> 仅按 lift 取 TopN -> 仍为 0 则跳过过滤。
    返回 (过滤后带 base_cov 列的 DataFrame, prior_pi, 最终轮次的原因计数或空)。
    """
    if len(atomic_rules_df) == 0:
        return atomic_rules_df, None, {}
    enable = getattr(config, 'enable_precision_filter', True)
    min_for_search = getattr(config, 'min_atomic_rules_for_search', 30)
    min_sub_cov = getattr(config, 'min_sub_cov', 0.02)
    min_precision_mult = getattr(config, 'min_precision_mult', 3.0)
    prior_pi_raw = getattr(config, 'expected_cohort_ratio', None)
    if prior_pi_raw is None or (isinstance(prior_pi_raw, float) and (prior_pi_raw <= 0 or prior_pi_raw >= 1)):
        prior_pi = None
        use_precision_proxy = False
        logger.warning(
            "expected_cohort_ratio 未配置或无效，已跳过 precision 硬过滤；不输出 precision_est/fp_rate_est 有效值。"
            " 建议在 config.json stage2 中配置 expected_cohort_ratio=cohort_size/full_size。"
        )
    else:
        prior_pi = float(prior_pi_raw)
        use_precision_proxy = True

    df = _compute_atomic_cov_lift(atomic_rules_df, numeric_diff_df, categorical_diff_df)
    n_before = len(df)

    def _add_precision_fp(out_df: pd.DataFrame, pi: Optional[float]) -> pd.DataFrame:
        out_df = out_df.copy()
        out_df['lift'] = out_df.get('_lift')
        if pi is not None and pi > 0 and pi < 1:
            def _prec(lift_val):
                if lift_val is None or (isinstance(lift_val, float) and (np.isnan(lift_val) or lift_val <= 0)):
                    return np.nan
                denom = pi * float(lift_val) + (1 - pi)
                return (pi * float(lift_val)) / denom if denom else np.nan
            def _fp(bc):
                if bc is None or (isinstance(bc, float) and np.isnan(bc)):
                    return np.nan
                return float(bc) * (1 - pi)
            out_df['precision_est'] = out_df['_lift'].apply(_prec)
            out_df['fp_rate_est'] = out_df['base_cov'].apply(_fp)
        else:
            out_df['precision_est'] = np.nan
            out_df['fp_rate_est'] = np.nan
        return out_df.drop(columns=['_lift'], errors='ignore')

    if not enable:
        logger.info("  原子规则精度过滤已关闭（enable_precision_filter=False），保留全部 %d 条", n_before)
        return _add_precision_fp(df, prior_pi), prior_pi, {}

    max_base_cov = getattr(config, 'max_base_cov', 0.20)
    min_lift = getattr(config, 'min_lift_atomic', 2.0)
    round1, reason1 = _apply_one_round_filter_typed(
        df, max_base_cov, min_lift, min_sub_cov, use_precision_proxy, prior_pi, min_precision_mult, config
    )
    n_r1 = len(round1)
    logger.info("  原子规则精度 Round1（max_base_cov=%.2f, min_lift=%.2f）保留: %d 条；原因计数: base_cov过高=%d, sub_cov过低=%d, lift不足=%d, precision不足=%d, 缺cov=%d",
                max_base_cov, min_lift, n_r1,
                reason1.get('base_cov_high', 0), reason1.get('sub_cov_low', 0), reason1.get('lift_low', 0),
                reason1.get('precision_low', 0), reason1.get('no_cov', 0))

    if n_r1 >= min_for_search:
        return _add_precision_fp(round1, prior_pi), prior_pi, reason1

    fallback_max_base_cov = getattr(config, 'fallback_max_base_cov', 0.50)
    fallback_min_lift = getattr(config, 'fallback_min_lift_atomic', 1.2)
    logger.warning("原子规则过少（%d < %d），触发降级：放宽 max_base_cov->%.2f, min_lift->%.2f", n_r1, min_for_search, fallback_max_base_cov, fallback_min_lift)
    round2, reason2 = _apply_one_round_filter_typed(
        df, fallback_max_base_cov, fallback_min_lift, min_sub_cov, use_precision_proxy, prior_pi, min_precision_mult, config
    )
    n_r2 = len(round2)
    logger.info("  原子规则精度 Round2（fallback）保留: %d 条；原因计数: base_cov过高=%d, sub_cov过低=%d, lift不足=%d, precision不足=%d, 缺cov=%d",
                n_r2,
                reason2.get('base_cov_high', 0), reason2.get('sub_cov_low', 0), reason2.get('lift_low', 0),
                reason2.get('precision_low', 0), reason2.get('no_cov', 0))

    if n_r2 >= min_for_search:
        return _add_precision_fp(round2, prior_pi), prior_pi, reason2

    logger.warning(
        "原子规则仍过少（%d < %d），保留 Round2 结果继续 beam；规则不足可能导致无法凑齐 k=3 的候选",
        n_r2, min_for_search
    )
    return _add_precision_fp(round2, prior_pi), prior_pi, reason2


def _safe_float(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _compute_atomic_rule_scores(atomic_df: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    """
    为原子规则表计算统一 rule_score。
    公式：rule_score = w1*precision_est + w2*lift_est + w3*coverage_est + w4*divergence + w5*stability，
    权重归一化后缺失项置 0 参与。
    """
    w1 = getattr(config, 'rule_score_w_precision', 0.25)
    w2 = getattr(config, 'rule_score_w_lift', 0.25)
    w3 = getattr(config, 'rule_score_w_coverage', 0.1)
    w4 = getattr(config, 'rule_score_w_divergence', 0.2)
    w5 = getattr(config, 'rule_score_w_stability', 0.2)
    total = w1 + w2 + w3 + w4 + w5
    if total <= 0:
        total = 1.0
    w1, w2, w3, w4, w5 = w1 / total, w2 / total, w3 / total, w4 / total, w5 / total

    def _norm_prec(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or v < 0)):
            return 0.0
        return min(1.0, float(v))

    def _norm_lift(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0)):
            return 0.0
        return min(1.0, float(v) / 10.0)  # cap lift at 10

    def _norm_cov(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return min(1.0, max(0.0, float(v)))

    def _norm_01(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return max(0.0, min(1.0, float(v)))

    scores = []
    for _, row in atomic_df.iterrows():
        prec = _norm_prec(row.get('precision_est'))
        lift = _norm_lift(row.get('lift'))
        cov = _norm_cov(row.get('base_cov'))
        div = _norm_01(row.get('divergence_score', 0.0))
        stab = _norm_01(row.get('stability_score', 0.0))
        s = w1 * prec + w2 * lift + w3 * cov + w4 * div + w5 * stab
        scores.append(s)
    out = atomic_df.copy()
    out['rule_score'] = scores
    return out


def _atomic_overlap_ratio(row_a: pd.Series, row_b: pd.Series, col_id: str) -> float:
    """同 column 两条原子规则的重叠度：连续用 base_cov 比或区间重叠，离散用 rec_categories Jaccard。"""
    feat = row_a.get('rule_type_feature', 'numeric')
    if feat == 'categorical':
        ra = row_a.get('rec_categories') or ''
        rb = row_b.get('rec_categories') or ''
        if isinstance(ra, str):
            set_a = set(c.strip() for c in ra.split(',') if c.strip())
        elif hasattr(ra, '__iter__') and not isinstance(ra, str):
            set_a = set(ra)
        else:
            set_a = set()
        if isinstance(rb, str):
            set_b = set(c.strip() for c in rb.split(',') if c.strip())
        elif hasattr(rb, '__iter__') and not isinstance(rb, str):
            set_b = set(rb)
        else:
            set_b = set()
        if not set_a and not set_b:
            return 1.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return (inter / union) if union else 0.0
    # numeric: base_cov ratio
    ba = _safe_float(row_a.get('base_cov'))
    bb = _safe_float(row_b.get('base_cov'))
    if ba is not None and bb is not None and max(ba, bb) > 0:
        return min(ba, bb) / max(ba, bb)
    # fallback: interval overlap
    low_a = row_a.get('rule_low')
    high_a = row_a.get('rule_high')
    low_b = row_b.get('rule_low')
    high_b = row_b.get('rule_high')
    try:
        la = float(low_a) if low_a is not None and np.isfinite(float(low_a)) else -np.inf
        ha = float(high_a) if high_a is not None and np.isfinite(float(high_a)) else np.inf
        lb = float(low_b) if low_b is not None and np.isfinite(float(low_b)) else -np.inf
        hb = float(high_b) if high_b is not None and np.isfinite(float(high_b)) else np.inf
        inter_len = max(0.0, min(ha, hb) - max(la, lb))
        union_len = max(ha, hb) - min(la, lb)
        if union_len <= 0:
            return 0.0
        return inter_len / union_len
    except (TypeError, ValueError):
        return 0.0


def _dedup_atomic_rules_after_precision(
    atomic_df: pd.DataFrame, config: Stage2Config
) -> Tuple[pd.DataFrame, dict]:
    """
    精度过滤后的原子规则去重：① 同 column 覆盖重叠>=阈值只保留 rule_score 最优；② 同特征多阈值只保留 topK。
    返回 (去重后 DataFrame, 统计 dict)。
    """
    if len(atomic_df) == 0:
        return atomic_df, {'n_before': 0, 'n_after_overlap': 0, 'n_after_same_feature': 0, 'overlap_removed': 0, 'same_feature_removed': 0}
    thresh = getattr(config, 'atomic_overlap_dedup_threshold', 0.8)
    max_per_col = getattr(config, 'max_atomic_rules_per_column_after_dedup', 1)

    df = atomic_df.reset_index(drop=True)
    n_before = len(df)
    if 'column_id' not in df.columns:
        return df, {'n_before': n_before, 'n_after_overlap': n_before, 'n_after_same_feature': n_before, 'overlap_removed': 0, 'same_feature_removed': 0}

    # ① 按 column_id 分组，每组内用“重叠图”连通分量，每分量保留 rule_score 最大的一条
    keep_idx = []
    for col_id, grp in df.groupby('column_id', sort=False):
        indices = grp.index.tolist()
        if len(indices) <= 1:
            keep_idx.extend(indices)
            continue
        # 两两重叠度 -> 并查集合并 >= thresh 的对
        parent = {i: i for i in indices}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

        for ii, i in enumerate(indices):
            for j in indices[ii + 1 :]:
                if _atomic_overlap_ratio(df.loc[i], df.loc[j], col_id) >= thresh:
                    union(i, j)
        components = {}
        for i in indices:
            r = find(i)
            components.setdefault(r, []).append(i)
        for comp in components.values():
            best = max(comp, key=lambda idx: df.loc[idx].get('rule_score', 0.0))
            keep_idx.append(best)
    df1 = df.loc[sorted(keep_idx)].copy()
    n_after_overlap = len(df1)
    overlap_removed = n_before - n_after_overlap

    # ② 同特征多阈值：每 column_id 只保留 rule_score top max_per_col
    keep_idx2 = []
    for col_id, grp in df1.groupby('column_id', sort=False):
        top = grp.nlargest(max_per_col, 'rule_score').index.tolist()
        keep_idx2.extend(top)
    df2 = df1.loc[sorted(keep_idx2)].copy()
    n_after_same_feature = len(df2)
    same_feature_removed = n_after_overlap - n_after_same_feature

    return df2, {
        'n_before': n_before,
        'n_after_overlap': n_after_overlap,
        'n_after_same_feature': n_after_same_feature,
        'overlap_removed': overlap_removed,
        'same_feature_removed': same_feature_removed,
    }


def dedup_candidates_by_signature(candidate_rules: list) -> list:
    """按 rule_signature 去重，保留每组中 score 最大的一条。"""
    if not candidate_rules:
        return []
    from collections import defaultdict
    by_sig = defaultdict(list)
    for r in candidate_rules:
        sig = getattr(r, 'rule_signature', None)
        if sig is None or sig == '':
            sig = segment_canonical_key(r)
            setattr(r, 'rule_signature', sig)
        by_sig[sig].append(r)
    out = []
    for sig, group in by_sig.items():
        best = max(group, key=lambda x: getattr(x, 'score', 0.0) or 0.0)
        out.append(best)
    return out


def run_stage2_analysis(
    stage1_output_dir: str,
    stat_date: str,
    cohort_name: str,
    output_dir: str = './data/stage2_output',
    config_path: Optional[str] = None,
    pair_assoc_path: Optional[str] = None,
    full_stats_dir: Optional[str] = './data/full_stats',
    *,
    target_precision_min: Optional[float] = None,
    target_coverage_min: Optional[float] = None,
    target_segment_count: Optional[int] = None,
    target_user_count_10k: Optional[float] = None,
    use_f1_balance: Optional[bool] = None,
    target_priority: Optional[str] = None,
) -> None:
    """
    执行阶段2分析流程（coverage-free：仅使用差异/稳定性/多样性评分，不使用 coverage/lift）。
    Stage1 中的 cohort_coverage_est、full_coverage_est、lift_est、cohort_coverage、full_coverage、lift 为 optional debug 列，存在则保留，不参与任何评分。
    字段对关联表：若未传 pair_assoc_path，则在 full_stats_dir 下自动查找 ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx；存在则启用相关性剪枝，否则仅用结构约束。
    """
    # 未显式指定时，在全量统计目录下自动查找字段对关联表 ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx
    if pair_assoc_path is None and full_stats_dir:
        auto_pair = Path(full_stats_dir) / PAIR_ASSOC_FILENAME_TEMPLATE.format(stat_date=stat_date)
        if auto_pair.exists():
            pair_assoc_path = str(auto_pair)
        else:
            logger.info(f"  未发现字段对关联表 {auto_pair.name}，将使用结构约束（即插即用：有表则启用相关性剪枝）")
    
    logger.info("=" * 60)
    logger.info("阶段2：多特征组合规则生成")
    logger.info("=" * 60)
    
    # 1. 加载配置（支持统一 config.json：使用 stage2 段；若文件仅含 stage2 键则整份作为 stage2）
    logger.info("\n[1/6] 加载配置...")
    _config_path = config_path or str(Path(__file__).resolve().parent.parent / "config.json")
    if Path(_config_path).exists():
        try:
            import sys
            _root = Path(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from config import load_config
            app_config = load_config(_config_path)
            stage2_dict = app_config.get_stage2_dict()
            config = Stage2Config.from_dict(stage2_dict) if stage2_dict else Stage2Config()
            logger.info(f"  从配置文件加载: {_config_path}")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
            config = Stage2Config()
    else:
        config = Stage2Config()
        logger.info("  使用默认配置")

    # 命令行覆盖输出目标约束（平衡准确率与覆盖率）
    if target_precision_min is not None:
        config.target_precision_min = target_precision_min
    if target_coverage_min is not None:
        config.target_coverage_min = target_coverage_min
    if target_segment_count is not None:
        config.target_segment_count = target_segment_count
    if target_user_count_10k is not None:
        config.target_user_count_10k = target_user_count_10k
    if use_f1_balance is not None:
        config.use_f1_balance = use_f1_balance
    if target_priority is not None:
        config.target_priority = target_priority

    # 2. 读取阶段1的输出文件
    logger.info("\n[2/6] 读取阶段1输出文件...")
    stage1_dir = Path(stage1_output_dir)
    
    numeric_diff_file = stage1_dir / f"numeric_diff_{cohort_name}_{stat_date}.csv"
    categorical_diff_file = stage1_dir / f"categorical_diff_{cohort_name}_{stat_date}.csv"
    
    if not numeric_diff_file.exists():
        raise FileNotFoundError(f"阶段1输出文件不存在: {numeric_diff_file}")
    if not categorical_diff_file.exists():
        raise FileNotFoundError(f"阶段1输出文件不存在: {categorical_diff_file}")
    
    numeric_diff_df = pd.read_csv(numeric_diff_file, encoding='utf-8-sig')
    categorical_diff_df = pd.read_csv(categorical_diff_file, encoding='utf-8-sig')
    # 若 Stage1 导出为业务可读表头，映射回技术列名；coverage/lift 列为 optional，存在则保留，不参与评分
    if '字段ID' in numeric_diff_df.columns:
        numeric_diff_df.rename(columns=STAGE1_NUMERIC_HEADER_REVERSE, inplace=True)
    numeric_diff_df.set_index('column_id', inplace=True)
    if '字段ID' in categorical_diff_df.columns:
        categorical_diff_df.rename(columns=STAGE1_CATEGORICAL_HEADER_REVERSE, inplace=True)
    categorical_diff_df.set_index('column_id', inplace=True)
    
    # 去重：避免重复 column_id 导致后续 reindex 报错
    if numeric_diff_df.index.duplicated().any():
        dup_count = numeric_diff_df.index.duplicated().sum()
        logger.warning(f"连续特征差异结果存在重复 column_id，已去重保留首条: {dup_count} 条")
        numeric_diff_df = numeric_diff_df[~numeric_diff_df.index.duplicated(keep='first')].copy()
    if categorical_diff_df.index.duplicated().any():
        dup_count = categorical_diff_df.index.duplicated().sum()
        logger.warning(f"离散特征差异结果存在重复 column_id，已去重保留首条: {dup_count} 条")
        categorical_diff_df = categorical_diff_df[~categorical_diff_df.index.duplicated(keep='first')].copy()

    # 兼容 Stage1 输出：若缺少 delta_ratio，则用 sum_abs_diff 作为替代
    if 'delta_ratio' not in categorical_diff_df.columns:
        if 'sum_abs_diff' in categorical_diff_df.columns:
            categorical_diff_df['delta_ratio'] = categorical_diff_df['sum_abs_diff']
            logger.warning("离散特征差异结果缺少 delta_ratio，已使用 sum_abs_diff 作为替代")
        else:
            categorical_diff_df['delta_ratio'] = 0.0
            logger.warning("离散特征差异结果缺少 delta_ratio/sum_abs_diff，已填充为 0.0")

    logger.info(f"  连续特征差异结果: {len(numeric_diff_df)} 个特征")
    logger.info(f"  离散特征差异结果: {len(categorical_diff_df)} 个特征")

    # 从 Stage1 元数据读取全量/圈定人数，用于 total_population 与 pi（严禁静默默认）
    # 策略：先自动算 pi_from_meta = N_cohort/N_full；若计算结果 > 配置值则用计算结果，否则用配置值
    pi_input = getattr(config, 'expected_cohort_ratio', None)
    meta_file = stage1_dir / f"stage1_meta_{cohort_name}_{stat_date}.json"
    pi_from_meta = None
    n_full = getattr(config, 'total_population', None)
    n_cohort = getattr(config, 'cohort_size', None)
    if meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            n_full = meta.get('total_population_full')
            if n_full is not None and getattr(config, 'total_population', None) is None:
                config.total_population = int(n_full)
                logger.info(f"  已从 Stage1 元数据读取全量用户数: {config.total_population}")
            n_cohort = meta.get('total_population_cohort')
            if n_cohort is not None:
                config.cohort_size = int(n_cohort)
            if n_full is not None and n_cohort is not None and int(n_full) > 0:
                pi_from_meta = int(n_cohort) / int(n_full)
                logger.info(f"  已从 Stage1 元数据计算 pi（候选）: {pi_from_meta:.6f} = {n_cohort}/{n_full}")
        except Exception as e:
            logger.warning(f"读取 Stage1 元数据失败: {e}")
    else:
        n_full = getattr(config, 'total_population', None)
        n_cohort = getattr(config, 'cohort_size', None)
    pi_candidate = pi_from_meta  # N_sub / N_total from meta when available
    pi_config = None
    if pi_input is not None and isinstance(pi_input, (int, float)) and 0 < float(pi_input) < 1:
        pi_config = float(pi_input)
    if pi_from_meta is not None and (pi_config is None or pi_from_meta >= pi_config):
        pi_used = pi_from_meta
        pi_source = "computed"
        config.expected_cohort_ratio = pi_used
        if pi_config is not None and pi_from_meta > pi_config:
            logger.info(f"  pi 使用元数据计算结果（%.6f > 配置 %.6f）", pi_from_meta, pi_config)
    elif pi_config is not None:
        pi_used = pi_config
        pi_source = "config"
    else:
        pi_used = None
        pi_source = "missing"
        if pi_from_meta is None and pi_config is None:
            logger.warning(
                "expected_cohort_ratio (pi) 未配置且无法从 Stage1 元数据计算（缺 total_population_cohort 或 total_population_full）。"
                " precision_est/combo_precision_est 将为 None，排序时降权。"
            )
    logger.info("pi_input=%s, pi_candidate=%s, pi_used=%s, pi_source=%s", pi_input, pi_candidate, pi_used, pi_source)
    logger.info(
        "N_sub 对应 config.cohort_size / Stage1 元数据键 total_population_cohort；"
        "N_all 对应 config.total_population / Stage1 元数据键 total_population_full"
    )
    pi_reference = getattr(config, 'pi_reference', None)
    pi_tolerance = getattr(config, 'pi_tolerance', 1e-9)
    if pi_reference is not None and pi_candidate is not None and abs(pi_candidate - pi_reference) > pi_tolerance:
        logger.warning(
            "pi 校验未通过: abs(pi_candidate - pi_reference)=%.2e > pi_tolerance=%.2e",
            abs(pi_candidate - pi_reference), pi_tolerance
        )
    if pi_used is None:
        logger.warning(
            "expected_cohort_ratio (pi) 未配置或无法从元数据计算，precision 相关字段将置空或跳过过滤；菜单仍照常输出。"
        )
    logger.info(
        "覆盖率口径: non_sub/sub/all；用户数口径: N_all_est=N_sub_est+N_non_sub_est，估计用户数区间=[N_all_lb, N_all_ub]"
    )

    # 3. 计算差异评分和稳定性评分
    logger.info("\n[3/7] 计算差异评分和稳定性评分...")
    numeric_divergence_scores = calculate_divergence_score(numeric_diff_df, None, None, config)
    categorical_divergence_scores = calculate_categorical_divergence_score(categorical_diff_df, config)
    stability_scores = calculate_stability_score(numeric_diff_df, categorical_diff_df, None, None, config)
    
    # 合并连续和离散特征的差异评分
    all_divergence_scores = pd.concat([numeric_divergence_scores, categorical_divergence_scores])
    
    logger.info(f"  连续特征差异评分: {len(numeric_divergence_scores)} 个特征")
    logger.info(f"  离散特征差异评分: {len(categorical_divergence_scores)} 个特征")
    logger.info(f"  稳定性评分: {len(stability_scores)} 个特征")
    
    # 4. 生成原子规则库（使用新评分）
    logger.info("\n[4/7] 生成原子规则库...")
    numeric_atomic_rules = generate_numeric_atomic_rules(
        numeric_diff_df, config, None, None, numeric_divergence_scores, stability_scores
    )
    categorical_atomic_rules = generate_categorical_atomic_rules(
        categorical_diff_df, config, categorical_divergence_scores, stability_scores
    )
    atomic_rules_df = merge_atomic_rules(numeric_atomic_rules, categorical_atomic_rules)
    
    logger.info(f"  原子规则总数（过滤前）: {len(atomic_rules_df)}")
    
    # 4.2 原子规则精度过滤（永不清零 + 自动降级）：Round1 -> Round2 -> 仅 lift TopN -> 仍为 0 则跳过
    n_before_atomic = len(atomic_rules_df)
    atomic_rules_df, prior_pi, filter_reasons = _filter_atomic_rules_by_precision(
        atomic_rules_df, numeric_diff_df, categorical_diff_df, config
    )
    n_after_atomic = len(atomic_rules_df)
    min_for_search = getattr(config, 'min_atomic_rules_for_search', 30)
    if n_after_atomic < min_for_search:
        # Round3：放宽 Stage1 diff/stability 阈值（不放宽 demographic/height 硬剪枝），重新生成原子规则
        old_div = getattr(config, 'min_divergence_score', 0.1)
        old_stab = getattr(config, 'min_stability_score', 0.3)
        config_relaxed = Stage2Config.from_dict(config.to_dict())
        config_relaxed.min_divergence_score = max(0.05, old_div * 0.5)
        config_relaxed.min_stability_score = max(0.15, old_stab * 0.5)
        config_relaxed.min_stability_score_categorical = 0.15  # fallback 时放宽离散稳定性，保证原子数尽量 ≥30
        numeric_atomic_relaxed = generate_numeric_atomic_rules(
            numeric_diff_df, config_relaxed, None, None, numeric_divergence_scores, stability_scores
        )
        categorical_atomic_relaxed = generate_categorical_atomic_rules(
            categorical_diff_df, config_relaxed, categorical_divergence_scores, stability_scores
        )
        atomic_rules_df_r3 = merge_atomic_rules(numeric_atomic_relaxed, categorical_atomic_relaxed)
        atomic_rules_df_r3, _, _ = _filter_atomic_rules_by_precision(
            atomic_rules_df_r3, numeric_diff_df, categorical_diff_df, config
        )
        n_r3 = len(atomic_rules_df_r3)
        if n_r3 >= min_for_search:
            n_old = n_after_atomic
            atomic_rules_df = atomic_rules_df_r3
            n_after_atomic = n_r3
            logger.info(
                "放宽了 min_divergence_score 从 %.3f 到 %.3f、min_stability_score 从 %.3f 到 %.3f，原子规则从 %d 增至 %d",
                old_div, config_relaxed.min_divergence_score, old_stab, config_relaxed.min_stability_score,
                n_old, n_after_atomic,
            )
        else:
            logger.warning(
                "Round3 放宽 diff/stability 后原子规则仍不足 %d（%d 条），保留 Round2 结果",
                min_for_search, n_r3,
            )
    logger.info(f"  最终用于 beam search 的原子规则数: {n_after_atomic}")
    # 统一 rule_score + 原子覆盖重叠去重 + 同特征多阈值去重
    atomic_rules_df = _compute_atomic_rule_scores(atomic_rules_df, config)
    atomic_rules_df, dedup_stats = _dedup_atomic_rules_after_precision(atomic_rules_df, config)
    n_after_atomic = len(atomic_rules_df)
    logger.info(
        "  原子规则 rule_score 与去重: 去重前 %d, 重叠去重后 %d, 同特征多阈值后 %d (overlap_removed=%d, same_feature_removed=%d)",
        dedup_stats['n_before'], dedup_stats['n_after_overlap'], dedup_stats['n_after_same_feature'],
        dedup_stats['overlap_removed'], dedup_stats['same_feature_removed'],
    )
    if 'column_id' in atomic_rules_df.columns:
        n_cols = atomic_rules_df['column_id'].nunique()
        logger.info("  原子规则去重后涉及特征数（按 column_id）: %d", n_cols)
    if 'cov_unknown' in atomic_rules_df.columns:
        n_cov_unknown = atomic_rules_df['cov_unknown'].sum() if hasattr(atomic_rules_df['cov_unknown'], 'sum') else sum(1 for x in atomic_rules_df['cov_unknown'] if x)
        n_cov_known = n_after_atomic - n_cov_unknown
        logger.info(f"  原子规则 cov_known: {n_cov_known}, cov_unknown: {n_cov_unknown}")
    if 'column_id' in atomic_rules_df.columns:
        atomic_rules_per_feature = atomic_rules_df['column_id'].value_counts()
        parts = [f"{cid}={cnt}" for cid, cnt in atomic_rules_per_feature.items()]
        logger.info("  atomic_rules_per_feature: %s", ", ".join(parts))
    
    # 4.5 可选：加载字段对关联统计表（缺失则仅用结构约束）
    pair_assoc_df = None
    pair_assoc_index = None
    if pair_assoc_path and Path(pair_assoc_path).exists():
        try:
            p = Path(pair_assoc_path)
            if p.suffix.lower() in ('.xlsx', '.xls'):
                pair_assoc_df = pd.read_excel(pair_assoc_path)
            else:
                pair_assoc_df = pd.read_csv(pair_assoc_path, encoding='utf-8-sig')
            if pair_assoc_df is not None and len(pair_assoc_df) > 0 and getattr(config, 'enable_pair_assoc_pruning', True) and PairAssocIndex is not None:
                min_sup = getattr(config, 'pair_assoc_min_support', 200)
                pair_assoc_index = PairAssocIndex(pair_assoc_df, stat_date=stat_date, min_support=min_sup)
                logger.info(
                    "  已加载字段对关联表: %s，行数 %d，去重后对数 %d，支持度过滤后对数 %d，已构建 PairAssocIndex",
                    pair_assoc_path, pair_assoc_index.n_raw, pair_assoc_index.n_dedup, pair_assoc_index.n_after_support
                )
            else:
                if pair_assoc_df is not None and len(pair_assoc_df) > 0:
                    logger.info(f"  已加载字段对关联表: {pair_assoc_path}（未启用 PairAssocIndex 或未安装）")
                else:
                    logger.info(f"  已加载字段对关联表: {pair_assoc_path}（表为空，使用结构约束）")
        except Exception as e:
            logger.warning(f"加载字段对关联表失败，退化为结构约束: {e}")
    else:
        if pair_assoc_path:
            logger.info("  未发现字段对关联表，将使用结构约束")
    
    # 5. 规则组合搜索（Coverage-free Beam Search）
    logger.info("\n[5/7] 规则组合搜索（Coverage-free Beam Search）...")
    
    # 构建业务分组字典（从元数据或字段名推断）
    business_groups = {}
    
    candidate_rules, beam_stats = combine_rules_beam_search(
        atomic_rules_df,
        numeric_diff_df,
        categorical_diff_df,
        config,
        all_divergence_scores,
        stability_scores,
        business_groups=business_groups if business_groups else None,
        max_fields_per_business_group=2,
        pair_assoc_df=pair_assoc_df,
        pair_assoc_index=pair_assoc_index,
        prior_pi=prior_pi,
    )
    n_before_dedup = len(candidate_rules)
    candidate_rules = dedup_candidates_by_signature(candidate_rules)
    n_after_dedup = len(candidate_rules)
    logger.info(
        "candidates_before_dedup=%d, after_dedup=%d, dedup_removed=%d",
        n_before_dedup, n_after_dedup, n_before_dedup - n_after_dedup,
    )
    if getattr(config, 'same_structure_at_candidate', True):
        n_before_same = len(candidate_rules)
        candidate_rules = dedup_same_structure_candidates(candidate_rules, config)
        n_after_same = len(candidate_rules)
        logger.info(
            "[stage2] candidate_dedup before=%d after=%d removed=%d",
            n_before_same, n_after_same, n_before_same - n_after_same,
        )
    feat_count: Counter = Counter()
    for r in candidate_rules:
        for fr in r.feature_rules:
            cid = fr.get('column_id', '')
            if cid:
                feat_count[cid] += 1
    if feat_count:
        parts = [f"{cid}={cnt}" for cid, cnt in feat_count.most_common()]
        logger.info("  candidate_rules_per_feature: %s", ", ".join(parts))
    if getattr(config, 'require_exact_k', False):
        k = getattr(config, 'max_features_per_segment', 3)
        n_k = sum(1 for r in candidate_rules if len(r.feature_rules) == k)
        logger.info(f"  其中规则数严格为 k=%d 的候选: %d 条", k, n_k)
        if n_k == 0:
            logger.warning("无满足 k=3 且 combo 约束的候选，候选客群与 portfolio 可能为空")
    
    # 6. 客群评分与筛选（使用新评分）
    logger.info("\n[6/7] 客群评分与筛选...")
    filtered_rules = filter_candidate_segments(
        candidate_rules,
        config
    )
    
    logger.info(f"  筛选后候选规则数: {len(filtered_rules)}")
    
    # 导出与 portfolio 仅使用规则数 >= min_rules_per_segment 的候选（单规则不进入任何导出）
    min_k = getattr(config, 'min_rules_per_segment', 2)
    candidate_rules_for_export = [r for r in filtered_rules if len(r.feature_rules) >= min_k]
    if len(candidate_rules_for_export) == 0:
        logger.warning("筛选后无规则数>=%d的候选，候选客群与组合将为空", min_k)
    
    # 7. 多客群组合与去重（差异最大化原则）
    logger.info("\n[7/7] 多客群组合与去重（差异最大化原则）...")
    # 构建差异评分字典
    divergence_scores_dict = all_divergence_scores.to_dict()
    portfolio = build_segment_portfolio(candidate_rules_for_export, config, divergence_scores_dict)
    if getattr(portfolio, 'portfolio_metrics', None) is not None and isinstance(beam_stats, dict):
        portfolio.portfolio_metrics.update(beam_stats)
    
    logger.info(f"  portfolio 预选客群数（filter_similar_rules）: {len(portfolio.segments)}")
    
    # 8. 导出结果
    logger.info("\n[8/8] 导出结果...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建字段名映射
    column_name_map = {}
    for idx, row in numeric_diff_df.iterrows():
        column_name_map[idx] = row.get('column_name', str(idx))
    for idx, row in categorical_diff_df.iterrows():
        column_name_map[idx] = row.get('column_name', str(idx))
    
    # 导出原子规则库（保留 base_cov/sub_cov/lift/precision_est/fp_rate_est 便于业务解释）
    atomic_rules_export = atomic_rules_df.copy()
    if 'lift' not in atomic_rules_export.columns and '_lift' in atomic_rules_export.columns:
        atomic_rules_export['lift'] = atomic_rules_export['_lift']
    atomic_rules_export = atomic_rules_export.drop(columns=['_lift'], errors='ignore')
    atomic_rules_output = output_path / f"atomic_rules_{cohort_name}_{stat_date}"
    export_atomic_rules(atomic_rules_export, atomic_rules_output, format='csv')
    export_atomic_rules(atomic_rules_export, atomic_rules_output, format='json')
    
    # 导出候选客群规则（仅含 k>=min_rules_per_segment，不含单规则；导出为 dedup 后列表）
    candidate_segments_output = output_path / f"candidate_segments_{cohort_name}_{stat_date}"
    export_candidate_segments(
        candidate_rules_for_export, candidate_segments_output, column_name_map,
        format='csv', total_population=getattr(config, 'total_population', None),
        cohort_size=getattr(config, 'cohort_size', None), pi_used=pi_used,
        dedup_before=n_before_dedup, dedup_after=n_after_dedup, dedup_removed=n_before_dedup - n_after_dedup,
    )
    export_candidate_segments(
        candidate_rules_for_export, candidate_segments_output, column_name_map,
        format='json', total_population=getattr(config, 'total_population', None),
        cohort_size=getattr(config, 'cohort_size', None), pi_used=pi_used,
        dedup_before=n_before_dedup, dedup_after=n_after_dedup, dedup_removed=n_before_dedup - n_after_dedup,
    )
    
    # 导出推荐客群组合方案（v2.0: 传入 config 则按档位输出 tiers；metadata 含 pi_used/pi_source）
    # 两阶段选择始终用全部候选，以保证 recommended 数量与测试 ROC（若改从预选选则 recommended 变少、ROC 下降）
    portfolio_output = output_path / f"segment_portfolio_{cohort_name}_{stat_date}.json"
    export_segment_portfolio(
        portfolio, portfolio_output, column_name_map, config=config,
        menu_candidates=candidate_rules_for_export if candidate_rules_for_export else None,
        pi_used=pi_used, pi_source=pi_source
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("阶段2执行完成！")
    logger.info("=" * 60)
    logger.info(f"\n输出目录: {output_path}")
    logger.info(f"  - 原子规则库: atomic_rules_{cohort_name}_{stat_date}.csv/json")
    logger.info(f"  - 候选客群规则: candidate_segments_{cohort_name}_{stat_date}.csv/json")
    logger.info(f"  - 推荐客群组合: segment_portfolio_{cohort_name}_{stat_date}.json")


if __name__ == '__main__':
    main()

