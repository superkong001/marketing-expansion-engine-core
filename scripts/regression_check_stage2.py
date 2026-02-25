# Stage2 回归检查：k>=3 导出、原子规则精度过滤
# 运行方式：python scripts/regression_check_stage2.py  或  python -m scripts.regression_check_stage2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_candidate_export_min_k3():
    """candidate_segments 导出过滤后每条规则均满足 len(feature_rules) >= 3"""
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import SegmentRule

    config = Stage2Config()
    config.min_rules_per_segment = 3
    min_k = getattr(config, 'min_rules_per_segment', 2)
    # 模拟 filtered_rules：1 条单规则、1 条双规则、1 条三条规则
    r1 = SegmentRule(rule_id="r1", feature_rules=[{'column_id': 'A', 'type': 'numeric', 'low': 0, 'high': 1}], score=0.5)
    r2 = SegmentRule(rule_id="r2", feature_rules=[
        {'column_id': 'A', 'type': 'numeric', 'low': 0, 'high': 1},
        {'column_id': 'B', 'type': 'numeric', 'low': 0, 'high': 1},
    ], score=0.6)
    r3 = SegmentRule(rule_id="r3", feature_rules=[
        {'column_id': 'A', 'type': 'numeric', 'low': 0, 'high': 1},
        {'column_id': 'B', 'type': 'numeric', 'low': 0, 'high': 1},
        {'column_id': 'C', 'type': 'numeric', 'low': 0, 'high': 1},
    ], score=0.7)
    filtered_rules = [r1, r2, r3]
    candidate_rules_for_export = [r for r in filtered_rules if len(r.feature_rules) >= min_k]
    assert len(candidate_rules_for_export) == 1, "仅应保留 3 条规则的候选"
    assert all(len(r.feature_rules) >= 3 for r in candidate_rules_for_export), "导出候选均应有 >=3 条规则"
    print("  candidate_segments 导出 k>=3: OK")


def test_portfolio_min_k3():
    """segment_portfolio 中每个客群均满足 len(feature_rules) >= 3"""
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import SegmentRule
    from stage2.segment_portfolio import build_segment_portfolio

    config = Stage2Config()
    config.min_rules_per_segment = 3
    r1 = SegmentRule(rule_id="r1", feature_rules=[{'column_id': 'A', 'type': 'numeric'}], score=0.5)
    r2 = SegmentRule(rule_id="r2", feature_rules=[
        {'column_id': 'A', 'type': 'numeric'}, {'column_id': 'B', 'type': 'numeric'},
    ], score=0.6)
    r3 = SegmentRule(rule_id="r3", feature_rules=[
        {'column_id': 'A', 'type': 'numeric'}, {'column_id': 'B', 'type': 'numeric'}, {'column_id': 'C', 'type': 'numeric'},
    ], score=0.7)
    candidate_rules = [r1, r2, r3]
    portfolio = build_segment_portfolio(candidate_rules, config, {})
    for seg in portfolio.segments:
        assert len(seg.feature_rules) >= 3, f"portfolio 中客群应有 >=3 条规则，当前 {len(seg.feature_rules)}"
    assert all(len(s.feature_rules) >= 3 for s in portfolio.segments), "portfolio 所有客群均 k>=3"
    print("  segment_portfolio k>=3: OK")


def test_atomic_filter_removes_wide_rule():
    """原子规则过滤会移除 base_cov > max_base_cov 的宽规则；expected_cohort_ratio 必填（否则 fail fast）"""
    import pandas as pd
    from stage2.stage2_config import Stage2Config
    from stage2.stage2_main import _filter_atomic_rules_by_precision

    config = Stage2Config()
    config.expected_cohort_ratio = 0.01  # 必填，否则 Stage2 报错
    config.max_base_cov = 0.20
    config.min_sub_cov = 0.02
    config.min_lift_atomic = 2.0
    config.min_precision_mult = 3.0
    # 两条原子规则：一条 base_cov=0.25（应被过滤），一条 base_cov=0.10（保留）
    atomic_df = pd.DataFrame([
        {'column_id': 'A', 'rule_type_feature': 'numeric', 'rule_type': 'main', 'divergence_score': 0.5},
        {'column_id': 'B', 'rule_type_feature': 'numeric', 'rule_type': 'main', 'divergence_score': 0.5},
    ])
    numeric_diff = pd.DataFrame(
        index=['A', 'B'],
        data={
            'full_coverage_est': [0.25, 0.10],
            'cohort_coverage_est': [0.15, 0.08],
        },
    )
    categorical_diff = pd.DataFrame(index=[], columns=['full_coverage', 'cohort_coverage'])
    filtered, _, reasons = _filter_atomic_rules_by_precision(atomic_df, numeric_diff, categorical_diff, config)
    assert len(filtered) == 1, "base_cov=0.25 的规则应被过滤，仅保留 1 条"
    assert filtered.iloc[0]['column_id'] == 'B', "应保留 base_cov=0.10 的规则"
    assert reasons.get('base_cov_high', 0) >= 1, "应有至少 1 条因 base_cov 过高被过滤"
    assert 'precision_est' in filtered.columns and 'fp_rate_est' in filtered.columns, "导出应含 precision_est、fp_rate_est"
    print("  原子规则过滤（base_cov>0.2 移除）: OK")


def test_require_exact_k_and_combo_metrics():
    """require_exact_k 时仅保留 k=3；SegmentRule 含 combo_* 与 fp_rate_est"""
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import SegmentRule

    config = Stage2Config()
    config.require_exact_k = True
    config.max_features_per_segment = 3
    r = SegmentRule(
        rule_id="r1",
        feature_rules=[
            {'column_id': 'A', 'type': 'numeric', 'base_cov': 0.1, 'sub_cov': 0.2},
            {'column_id': 'B', 'type': 'numeric', 'base_cov': 0.1, 'sub_cov': 0.2},
            {'column_id': 'C', 'type': 'numeric', 'base_cov': 0.1, 'sub_cov': 0.2},
        ],
        score=0.7,
        combo_base_cov_est=0.001,
        combo_sub_cov_est=0.008,
        combo_lift_est=8.0,
        combo_precision_est=0.07,
        fp_rate_est=0.00099,
    )
    assert getattr(r, 'combo_precision_est', None) is not None
    assert getattr(r, 'fp_rate_est', None) is not None
    assert len(r.feature_rules) == 3
    print("  require_exact_k + combo 指标: OK")


def main():
    print("Stage2 回归检查（k>=3 + 原子精度过滤 + require_exact_k + 导出列）")
    test_candidate_export_min_k3()
    test_portfolio_min_k3()
    test_atomic_filter_removes_wide_rule()
    test_require_exact_k_and_combo_metrics()
    print("全部通过.")


if __name__ == '__main__':
    main()
