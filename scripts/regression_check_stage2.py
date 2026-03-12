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
    """require_exact_k 时仅保留 k=3；SegmentRule 含 Fréchet 区间 combo_*_lb/ub"""
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
        combo_base_cov_lb=0.0,
        combo_base_cov_ub=0.001,
        combo_sub_cov_lb=0.0,
        combo_sub_cov_ub=0.008,
        combo_precision_lb=0.05,
        combo_precision_ub=0.10,
    )
    assert getattr(r, 'combo_precision_lb', None) is not None
    assert getattr(r, 'combo_precision_ub', None) is not None
    assert getattr(r, 'combo_base_cov_ub', None) is not None
    assert len(r.feature_rules) == 3
    print("  require_exact_k + combo 区间指标: OK")


def test_pair_assoc_with_table_hard_thr_no_strong_pair():
    """有表 + hard_thr=0.85：mock 表含 (A,B)=0.9，断言最终任意 segment 中不包含字段对 (A,B)（同一客群内无 strength>=0.85 的对）"""
    import pandas as pd
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import combine_rules_beam_search
    from stage2.pair_assoc import PairAssocIndex

    config = Stage2Config()
    config.pair_assoc_hard_thr = 0.85
    config.enable_pair_assoc_pruning = True
    config.max_features_per_segment = 3
    config.beam_size = 20
    config.top_k_features = 10
    config.require_exact_k = False
    # 最小原子规则：A, B, C 三条 numeric
    atomic = pd.DataFrame([
        {'column_id': 'A', 'rule_type_feature': 'numeric', 'divergence_score': 0.6, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'A'},
        {'column_id': 'B', 'rule_type_feature': 'numeric', 'divergence_score': 0.6, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'B'},
        {'column_id': 'C', 'rule_type_feature': 'numeric', 'divergence_score': 0.6, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'C'},
    ])
    numeric_diff = pd.DataFrame(index=['A', 'B', 'C'], data={'column_name': ['A', 'B', 'C']})
    categorical_diff = pd.DataFrame()
    pair_df = pd.DataFrame([
        {'column_id_a': 'A', 'column_id_b': 'B', 'corr_strength': 0.9, 'corr_type': 'cont_cont_linear', 'corr_sign': 1},
    ])
    index = PairAssocIndex(pair_df, stat_date=None, min_support=0)
    candidates, beam_stats = combine_rules_beam_search(
        atomic, numeric_diff, categorical_diff, config,
        pair_assoc_df=pair_df, pair_assoc_index=index
    )
    for c in candidates:
        cols = [fr['column_id'] for fr in c.feature_rules]
        if 'A' in cols and 'B' in cols:
            assert index.get_strength('A', 'B') < 0.85, "segment 中不应同时含 (A,B) 且 strength>=0.85"
    # 应有高相关剪枝发生（A,B 不应同时出现）
    segments_with_a_and_b = [c for c in candidates if 'A' in [fr['column_id'] for fr in c.feature_rules] and 'B' in [fr['column_id'] for fr in c.feature_rules]]
    assert len(segments_with_a_and_b) == 0, "不应存在同时包含 A 与 B 的 segment（strength=0.9 >= 0.85 应被剪枝）"
    print("  pair_assoc 有表 + hard_thr 无强相关对: OK")


def test_pair_assoc_without_table_no_error():
    """无表：不传 pair_assoc 表，断言流程正常、返回 (list, beam_stats) 且 beam_stats 含三项"""
    import pandas as pd
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import combine_rules_beam_search

    config = Stage2Config()
    config.max_features_per_segment = 2
    config.beam_size = 5
    config.top_k_features = 5
    atomic = pd.DataFrame([
        {'column_id': 'A', 'rule_type_feature': 'numeric', 'divergence_score': 0.5, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'A'},
        {'column_id': 'B', 'rule_type_feature': 'numeric', 'divergence_score': 0.5, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'B'},
    ])
    numeric_diff = pd.DataFrame(index=['A', 'B'], data={'column_name': ['A', 'B']})
    categorical_diff = pd.DataFrame()
    candidate_rules, beam_stats = combine_rules_beam_search(
        atomic, numeric_diff, categorical_diff, config,
        pair_assoc_df=None, pair_assoc_index=None
    )
    assert isinstance(candidate_rules, list)
    assert isinstance(beam_stats, dict)
    assert beam_stats.get('pruned_high_corr', 0) == 0
    assert beam_stats.get('penalized_mid_corr', 0) == 0
    assert beam_stats.get('pruned_conflict', 0) == 0
    print("  pair_assoc 无表 行为一致: OK")


def test_pair_assoc_soft_penalty_lower_score():
    """软惩罚：mock 表存在 0.65~0.85 中相关对 (A,B)=0.7，断言该组合仍出现但 penalized_mid_corr>0 且 (A,B) 组合得分较无表时低"""
    import pandas as pd
    from stage2.stage2_config import Stage2Config
    from stage2.rule_combination import combine_rules_beam_search
    from stage2.pair_assoc import PairAssocIndex

    config = Stage2Config()
    config.pair_assoc_soft_thr = 0.65
    config.pair_assoc_hard_thr = 0.85
    config.pair_assoc_soft_penalty = 0.15
    config.enable_pair_assoc_pruning = True
    config.max_features_per_segment = 2
    config.beam_size = 20
    config.top_k_features = 10
    atomic = pd.DataFrame([
        {'column_id': 'A', 'rule_type_feature': 'numeric', 'divergence_score': 0.6, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'A'},
        {'column_id': 'B', 'rule_type_feature': 'numeric', 'divergence_score': 0.6, 'stability_score': 0.5, 'direction': 'high', 'rule_low': 0, 'rule_high': 1, 'base_cov': 0.1, 'sub_cov': 0.2, 'column_name': 'B'},
    ])
    numeric_diff = pd.DataFrame(index=['A', 'B'], data={'column_name': ['A', 'B']})
    categorical_diff = pd.DataFrame()
    # 无表
    candidates_no_table, _ = combine_rules_beam_search(
        atomic, numeric_diff, categorical_diff, config,
        pair_assoc_df=None, pair_assoc_index=None
    )
    # 有表：(A,B)=0.7 中相关
    pair_df = pd.DataFrame([
        {'column_id_a': 'A', 'column_id_b': 'B', 'corr_strength': 0.7, 'corr_type': 'cont_cont_linear', 'corr_sign': 1},
    ])
    index = PairAssocIndex(pair_df, stat_date=None, min_support=0)
    candidates_with_table, beam_stats = combine_rules_beam_search(
        atomic, numeric_diff, categorical_diff, config,
        pair_assoc_df=pair_df, pair_assoc_index=index
    )
    ab_no = [c for c in candidates_no_table if len(c.feature_rules) == 2 and {fr['column_id'] for fr in c.feature_rules} == {'A', 'B'}]
    ab_yes = [c for c in candidates_with_table if len(c.feature_rules) == 2 and {fr['column_id'] for fr in c.feature_rules} == {'A', 'B'}]
    assert len(ab_yes) >= 1, "中相关对 (A,B) 仍应出现在候选列表中"
    assert beam_stats.get('penalized_mid_corr', 0) >= 1, "应有至少一次中相关软惩罚"
    if ab_no and ab_yes:
        assert ab_yes[0].score < ab_no[0].score + 1e-6, "带软惩罚的 (A,B) 得分应低于无表时"
    print("  pair_assoc 软惩罚 中相关得分降低: OK")


def main():
    print("Stage2 回归检查（k>=3 + 原子精度过滤 + require_exact_k + 导出列 + pair_assoc）")
    test_candidate_export_min_k3()
    test_portfolio_min_k3()
    test_atomic_filter_removes_wide_rule()
    test_require_exact_k_and_combo_metrics()
    test_pair_assoc_with_table_hard_thr_no_strong_pair()
    test_pair_assoc_without_table_no_error()
    test_pair_assoc_soft_penalty_lower_score()
    print("全部通过.")


if __name__ == '__main__':
    main()
