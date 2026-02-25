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
import logging
import sys
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
    from .segment_portfolio import build_segment_portfolio
    from .rule_output import (
        export_atomic_rules,
        export_candidate_segments,
        export_segment_portfolio
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
    from segment_portfolio import build_segment_portfolio
    from rule_output import (
        export_atomic_rules,
        export_candidate_segments,
        export_segment_portfolio
    )

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
    
    args = parser.parse_args()
    
    try:
        # 执行阶段2流程
        run_stage2_analysis(
            stage1_output_dir=args.stage1_output_dir,
            stat_date=args.stat_date,
            cohort_name=args.cohort_name,
            output_dir=args.output_dir,
            config_path=args.config,
            pair_assoc_path=args.pair_assoc,
            full_stats_dir=args.full_stats_dir
        )
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


# 字段对关联表默认文件名（全量目录下）
PAIR_ASSOC_FILENAME_TEMPLATE = "ST_ANA_FEAT_PAIR_ASSOC_ALL_{stat_date}.xlsx"


def _compute_atomic_cov_lift(
    atomic_rules_df: pd.DataFrame,
    numeric_diff_df: pd.DataFrame,
    categorical_diff_df: pd.DataFrame,
) -> pd.DataFrame:
    """为原子规则表补齐 base_cov、sub_cov、lift、precision_proxy（prior_pi 由调用方传入时再算）。"""
    eps = 1e-9
    base_covs = []
    sub_covs = []
    lifts = []
    for _, row in atomic_rules_df.iterrows():
        base_cov = None
        sub_cov = None
        col_id = row.get('column_id')
        if row.get('rule_type_feature') == 'numeric':
            # 仅 main 从 numeric_diff 取全量/圈定覆盖率；tail 不填，避免用列级代表值误杀
            if row.get('rule_type') == 'main' and col_id in numeric_diff_df.index:
                r = numeric_diff_df.loc[col_id]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                base_cov = _safe_float(r.get('full_coverage_est'))
                sub_cov = _safe_float(r.get('cohort_coverage_est'))
        else:
            if col_id in categorical_diff_df.index:
                r = categorical_diff_df.loc[col_id]
                base_cov = _safe_float(r.get('full_coverage'))
                sub_cov = _safe_float(r.get('cohort_coverage'))
        base_covs.append(base_cov)
        sub_covs.append(sub_cov)
        if base_cov is not None and sub_cov is not None and base_cov >= eps:
            lifts.append(sub_cov / base_cov)
        else:
            lifts.append(None)
    df = atomic_rules_df.copy()
    df['base_cov'] = base_covs
    df['sub_cov'] = sub_covs
    df['_lift'] = lifts
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
        return atomic_rules_df, 0.05, {}
    enable = getattr(config, 'enable_precision_filter', True)
    min_for_search = getattr(config, 'min_atomic_rules_for_search', 30)
    min_sub_cov = getattr(config, 'min_sub_cov', 0.02)
    min_precision_mult = getattr(config, 'min_precision_mult', 3.0)
    prior_pi_raw = getattr(config, 'expected_cohort_ratio', None)
    allow_missing_pi = getattr(config, 'allow_missing_pi', False)
    if prior_pi_raw is None or (isinstance(prior_pi_raw, float) and (prior_pi_raw <= 0 or prior_pi_raw >= 1)):
        if not allow_missing_pi:
            raise ValueError(
                "expected_cohort_ratio 未配置或不在 (0,1)，Stage2 必须填写真实先验（cohort_size/full_size）。"
                " 若确需跳过 precision 约束，请在 config.json stage2 中显式设置 allow_missing_pi=true（不推荐）。"
            )
        prior_pi = None
        use_precision_proxy = False
        logger.warning("expected_cohort_ratio 未配置或无效，已跳过 precision 硬过滤；不输出 precision_est/fp_rate_est 有效值")
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


def run_stage2_analysis(
    stage1_output_dir: str,
    stat_date: str,
    cohort_name: str,
    output_dir: str = './data/stage2_output',
    config_path: Optional[str] = None,
    pair_assoc_path: Optional[str] = None,
    full_stats_dir: Optional[str] = './data/full_stats'
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
    logger.info(f"  最终用于 beam search 的原子规则数: {n_after_atomic}")
    
    # 4.5 可选：加载字段对关联统计表（缺失则仅用结构约束）
    pair_assoc_df = None
    if pair_assoc_path and Path(pair_assoc_path).exists():
        try:
            p = Path(pair_assoc_path)
            if p.suffix.lower() in ('.xlsx', '.xls'):
                pair_assoc_df = pd.read_excel(pair_assoc_path)
            else:
                pair_assoc_df = pd.read_csv(pair_assoc_path, encoding='utf-8-sig')
            logger.info(f"  已加载字段对关联表: {pair_assoc_path}（启用相关性剪枝）")
        except Exception as e:
            logger.warning(f"加载字段对关联表失败，退化为结构约束: {e}")
    else:
        if pair_assoc_path:
            logger.info("  未找到字段对关联表，使用结构约束")
    
    # 5. 规则组合搜索（Coverage-free Beam Search）
    logger.info("\n[5/7] 规则组合搜索（Coverage-free Beam Search）...")
    
    # 构建业务分组字典（从元数据或字段名推断）
    business_groups = {}
    
    candidate_rules = combine_rules_beam_search(
        atomic_rules_df,
        numeric_diff_df,
        categorical_diff_df,
        config,
        all_divergence_scores,
        stability_scores,
        business_groups=business_groups if business_groups else None,
        max_fields_per_business_group=2,
        pair_assoc_df=pair_assoc_df
    )
    
    logger.info(f"  生成候选规则数: {len(candidate_rules)}")
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
    
    logger.info(f"  最终客群组合数: {len(portfolio.segments)}")
    
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
    
    # 导出候选客群规则（仅含 k>=min_rules_per_segment，不含单规则）
    candidate_segments_output = output_path / f"candidate_segments_{cohort_name}_{stat_date}"
    export_candidate_segments(candidate_rules_for_export, candidate_segments_output, column_name_map, format='csv')
    export_candidate_segments(candidate_rules_for_export, candidate_segments_output, column_name_map, format='json')
    
    # 导出推荐客群组合方案
    portfolio_output = output_path / f"segment_portfolio_{cohort_name}_{stat_date}.json"
    export_segment_portfolio(portfolio, portfolio_output, column_name_map)
    
    logger.info("\n" + "=" * 60)
    logger.info("阶段2执行完成！")
    logger.info("=" * 60)
    logger.info(f"\n输出目录: {output_path}")
    logger.info(f"  - 原子规则库: atomic_rules_{cohort_name}_{stat_date}.csv/json")
    logger.info(f"  - 候选客群规则: candidate_segments_{cohort_name}_{stat_date}.csv/json")
    logger.info(f"  - 推荐客群组合: segment_portfolio_{cohort_name}_{stat_date}.json")


if __name__ == '__main__':
    main()

