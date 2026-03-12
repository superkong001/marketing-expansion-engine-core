"""
规则结构化输出模块

将客群和规则转换为SQL和JSON格式
"""
import math
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any


def _json_safe_number(v: Any) -> Any:
    """导出 JSON 时：inf/nan 转为 None，避免写出 Infinity/NaN。"""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return v

try:
    from .rule_combination import SegmentRule
    from .segment_portfolio import SegmentPortfolio, assign_tiers, select_final_segments
    from .segment_diversity import calculate_rule_structure_distance
except ImportError:
    from rule_combination import SegmentRule
    from segment_portfolio import SegmentPortfolio, assign_tiers, select_final_segments
    from segment_diversity import calculate_rule_structure_distance

logger = logging.getLogger(__name__)


def _write_then_replace(file_path: Path, write_fn) -> None:
    """先写入临时文件再替换目标，避免目标被占用时丢失结果。write_fn 接受一个 Path 参数并写入该路径。"""
    file_path = Path(file_path)
    tmp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    write_fn(tmp_path)
    try:
        tmp_path.replace(file_path)
    except PermissionError:
        logger.warning(
            f"目标文件被占用，无法替换: {file_path}\n"
            f"结果已写入: {tmp_path}\n"
            f"请关闭 Excel 等程序后，将 .tmp 文件重命名或重新运行。"
        )


# 导出用：技术列名 -> 业务可读列名（无需数据字典即可理解）
ATOMIC_RULES_HEADER_CN = {
    'column_id': '字段ID',
    'column_name': '字段名称',
    'rule_type_feature': '特征类型',
    'rule_type': '规则子类型',
    'rule_low': '区间下界',
    'rule_high': '区间上界',
    'direction': '推荐方向',
    'rule_categories': '推荐类别集合',
    'divergence_score': '差异评分',
    'stability_score': '稳定性评分',
    'distribution_type': '分布类型',
    'rule_reason_code': '推荐理由代码',
    'delta_ratio': '差异比',
    'rec_category_count': '推荐类别个数',
    'base_cov': '全量覆盖率',
    'sub_cov': '圈定覆盖率',
    'lift': 'Lift',
    'precision_est': '精度估计',
    'fp_rate_est': '误判率估计',
    'cov_unknown': '覆盖率未知',
    'rule_score': '统一规则得分',
}
CANDIDATE_SEGMENTS_HEADER_CN = {
    'rule_id': '规则编号',
    'rule_expression': '规则表达式',
    'sql_where_clause': 'SQL筛选条件',
    'divergence_score': '差异评分',
    'stability_score': '稳定性评分',
    'diversity_score': '多样性评分',
    'score': '综合得分',
    'rule_score': '统一规则得分',
    'anchor_feature': '锚点特征',
    'confidence_level': '可信度档位',
    'feature_count': '包含字段数',
    'combo_non_sub_cov_est': '组合非圈定覆盖率估计',
    'combo_sub_cov_est': '组合圈定覆盖率估计',
    'combo_all_cov_est': '组合全量覆盖率估计',
    'combo_lift_est': '组合Lift估计',
    'combo_precision_est': '组合精度估计',
    'fp_rate_est': '误判率估计',
    'combo_cov_unknown': '组合覆盖率未知',
    'tier': '档位',
    'k': '规则数',
    'kappa': '条件数',
    'missing_pair_count': '缺失字段对数',
    'rule_signature': '去重键（字段+阈值摘要）',
}
PORTFOLIO_METRICS_HEADER_CN = {
    'total_segments': '客群数量',
    'avg_divergence_score': '平均差异评分',
    'avg_stability_score': '平均稳定性评分',
    'avg_diversity_score': '平均多样性评分',
    'avg_score': '平均综合得分',
    'min_structure_distance': '最小结构距离',
    'pruned_high_corr': '高相关剪枝数',
    'penalized_mid_corr': '中相关惩罚数',
    'pruned_conflict': '冲突剪枝数',
    'pruned_demographic': 'demographic 剪枝数',
    'pruned_height': 'height 强剪枝数',
    'pruned_kappa': '条件数剪枝数',
}
SEGMENT_KEYS_CN = {
    'segment_id': '客群编号',
    'rule_id': '规则编号',
    'rules': '规则明细',
    'sql_where_clause': 'SQL筛选条件',
    'metrics': '评分指标',
}
RULE_KEYS_CN = {
    'column_id': '字段ID',
    'column_name': '字段名称',
    'type': '类型',
    'low': '下界',
    'high': '上界',
    'direction': '方向',
    'categories': '类别集合',
}
METRICS_KEYS_CN = {
    'divergence_score': '差异评分',
    'stability_score': '稳定性评分',
    'diversity_score': '多样性评分',
    'score': '综合得分',
}


def generate_sql_where_clause(
    segment_rule: SegmentRule,
    column_name_map: Optional[Dict[str, str]] = None
) -> str:
    """
    生成SQL WHERE子句
    
    Args:
        segment_rule: SegmentRule对象
        column_name_map: 字段名映射（column_id -> column_name），如果为None则使用rule中的column_name
    
    Returns:
        SQL WHERE子句字符串
    """
    conditions = []
    
    for feature_rule in segment_rule.feature_rules:
        col_id = feature_rule['column_id']
        col_name = column_name_map.get(col_id, feature_rule.get('column_name', col_id)) if column_name_map else feature_rule.get('column_name', col_id)
        
        if feature_rule['type'] == 'numeric':
            low = feature_rule.get('low')
            high = feature_rule.get('high')
            
            if pd.notna(low) and pd.notna(high):
                if high == float('inf'):
                    # 只有下界
                    conditions.append(f"{col_name} >= {low}")
                elif low == float('-inf'):
                    # 只有上界
                    conditions.append(f"{col_name} < {high}")
                else:
                    # 区间
                    conditions.append(f"{col_name} >= {low} AND {col_name} < {high}")
            elif pd.notna(low):
                conditions.append(f"{col_name} >= {low}")
            elif pd.notna(high):
                conditions.append(f"{col_name} < {high}")
        
        elif feature_rule['type'] == 'categorical':
            categories = feature_rule.get('categories', [])
            if categories:
                # 转义单引号
                escaped_cats = [("'" + str(cat).replace("'", "''") + "'") for cat in categories]
                conditions.append(f"{col_name} IN ({', '.join(escaped_cats)})")
    
    return " AND ".join(conditions) if conditions else "1=1"


def export_atomic_rules(
    atomic_rules_df: pd.DataFrame,
    output_path: Path,
    format: str = 'csv'
) -> None:
    """
    导出原子规则库
    
    Args:
        atomic_rules_df: 原子规则库DataFrame
        output_path: 输出文件路径（不含扩展名）
        format: 输出格式（'csv' 或 'json'）
    """
    output_path = Path(output_path)
    
    if format == 'csv':
        file_path = output_path.with_suffix('.csv')
        export_df = atomic_rules_df.copy()
        export_df.columns = [ATOMIC_RULES_HEADER_CN.get(c, c) for c in export_df.columns]
        _write_then_replace(file_path, lambda p: export_df.to_csv(p, index=False, encoding='utf-8-sig'))
        logger.info(f"原子规则库已导出到: {file_path}")
    
    elif format == 'json':
        file_path = output_path.with_suffix('.json')
        records = atomic_rules_df.to_dict('records')
        def _write_json(p):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        _write_then_replace(file_path, _write_json)
        logger.info(f"原子规则库已导出到: {file_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")


def export_candidate_segments(
    candidate_rules: List[SegmentRule],
    output_path: Path,
    column_name_map: Optional[Dict[str, str]] = None,
    format: str = 'csv',
    total_population: Optional[int] = None,
    cohort_size: Optional[int] = None,
    pi_used: Optional[float] = None,
    dedup_before: Optional[int] = None,
    dedup_after: Optional[int] = None,
    dedup_removed: Optional[int] = None,
) -> None:
    """
    导出候选客群规则

    Args:
        candidate_rules: 候选规则列表（应为 dedup 后列表）
        output_path: 输出文件路径（不含扩展名）
        column_name_map: 字段名映射
        format: 输出格式（'csv' 或 'json'）
        total_population: 可选，全量人数 N_total，用于估计用户数区间
        cohort_size: 可选，圈定人数 N_sub，用于 N_sub_est/N_all_est
        pi_used: 可选，实际使用的 pi，写入 JSON metadata
        dedup_before: 可选，去重前候选数，写入 JSON metadata
        dedup_after: 可选，去重后候选数，写入 JSON metadata
        dedup_removed: 可选，去重删除数，写入 JSON metadata
    """
    output_path = Path(output_path)
    
    # 构建导出数据
    records = []
    for rule in candidate_rules:
        # 构建规则表达式
        rule_expr_parts = []
        for fr in rule.feature_rules:
            col_name = column_name_map.get(fr['column_id'], fr.get('column_name', fr['column_id'])) if column_name_map else fr.get('column_name', fr['column_id'])
            
            if fr['type'] == 'numeric':
                low = fr.get('low')
                high = fr.get('high')
                if pd.notna(low) and pd.notna(high):
                    if high == float('inf'):
                        rule_expr_parts.append(f"{col_name} >= {low}")
                    else:
                        rule_expr_parts.append(f"{col_name} ∈ [{low}, {high})")
            elif fr['type'] == 'categorical':
                cats = fr.get('categories', [])
                if cats:
                    rule_expr_parts.append(f"{col_name} ∈ {{{', '.join(cats)}}}")
        
        rule_expr = " AND ".join(rule_expr_parts)
        
        # 生成SQL
        sql_clause = generate_sql_where_clause(rule, column_name_map)
        combo_non_sub_cov_lb = getattr(rule, 'combo_non_sub_cov_lb', None)
        combo_non_sub_cov_ub = getattr(rule, 'combo_non_sub_cov_ub', None)
        combo_sub_cov_lb = getattr(rule, 'combo_sub_cov_lb', None)
        combo_sub_cov_ub = getattr(rule, 'combo_sub_cov_ub', None)
        combo_all_cov_lb = getattr(rule, 'combo_all_cov_lb', None)
        combo_all_cov_ub = getattr(rule, 'combo_all_cov_ub', None)
        combo_precision_lb = getattr(rule, 'combo_precision_lb', None)
        combo_precision_ub = getattr(rule, 'combo_precision_ub', None)
        missing_cnt = getattr(rule, 'missing_pair_count', None)
        missing_cnt = int(missing_cnt) if missing_cnt is not None else 0
        prec_lb = getattr(rule, 'combo_precision_lb', None)
        kappa_val = getattr(rule, 'kappa', None)
        kappa_ok = kappa_val is None or (isinstance(kappa_val, (int, float)) and float(kappa_val) < 50)
        if prec_lb is not None and prec_lb >= 0.35 and missing_cnt <= 1 and kappa_ok:
            confidence_level = 'high'
        elif (prec_lb is not None and prec_lb >= 0.2) or missing_cnt <= 1:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        rec = {
            'rule_id': rule.rule_id,
            'rule_expression': rule_expr,
            'sql_where_clause': sql_clause,
            'divergence_score': rule.divergence_score,
            'stability_score': rule.stability_score,
            'diversity_score': rule.diversity_score,
            'score': rule.score,
            'rule_score': getattr(rule, 'rule_score', None),
            'anchor_feature': getattr(rule, 'anchor_feature', None),
            'confidence_level': confidence_level,
            'feature_count': len(rule.feature_rules),
            'k': len(rule.feature_rules),
            'combo_cov_unknown': getattr(rule, 'combo_cov_unknown', None),
            'combo_non_sub_cov_lb': combo_non_sub_cov_lb,
            'combo_non_sub_cov_ub': combo_non_sub_cov_ub,
            'combo_sub_cov_lb': combo_sub_cov_lb,
            'combo_sub_cov_ub': combo_sub_cov_ub,
            'combo_all_cov_lb': combo_all_cov_lb,
            'combo_all_cov_ub': combo_all_cov_ub,
            'combo_precision_lb': combo_precision_lb,
            'combo_precision_ub': combo_precision_ub,
        }
        for key in ('combo_non_sub_cov_est', 'combo_sub_cov_est', 'combo_all_cov_est', 'combo_lift_est', 'combo_precision_est', 'fp_rate_est'):
            rec[key] = getattr(rule, key, None)
        rec['kappa'] = getattr(rule, 'kappa', None)
        rec['missing_pair_count'] = getattr(rule, 'missing_pair_count', None)
        rec['combo_cov_est_source'] = getattr(rule, 'combo_cov_est_source', None)
        rec['precision_ub_shrunk_by_corr'] = getattr(rule, 'precision_ub_shrunk_by_corr', None)
        rec['rule_signature'] = getattr(rule, 'rule_signature', None) or segment_canonical_key(rule)
        # 估计用户数：N_sub_est = sub_cov_est*N_sub, N_non_sub_est = non_sub_cov_est*(N_total-N_sub), N_all_est = N_sub_est+N_non_sub_est；区间 = [N_all_lb, N_all_ub]
        n_total = total_population
        n_sub = cohort_size
        sub_est = getattr(rule, 'combo_sub_cov_est', None)
        non_sub_est = getattr(rule, 'combo_non_sub_cov_est', None)
        if n_total is not None and n_sub is not None and sub_est is not None and non_sub_est is not None:
            rec['N_sub_est'] = round(sub_est * n_sub)
            rec['N_non_sub_est'] = round(non_sub_est * (n_total - n_sub))
            rec['N_all_est'] = rec['N_sub_est'] + rec['N_non_sub_est']
        else:
            rec['N_sub_est'] = rec['N_non_sub_est'] = rec['N_all_est'] = None
        if combo_all_cov_lb is not None and combo_all_cov_ub is not None and n_total is not None:
            rec['估计用户数区间'] = [round(combo_all_cov_lb * n_total), round(combo_all_cov_ub * n_total)]
        else:
            rec['估计用户数区间'] = None
        rec['估计准确率区间'] = [combo_precision_lb, combo_precision_ub] if (combo_precision_lb is not None or combo_precision_ub is not None) else None
        # 估计准确率说明：adj/ind，是否被 pair 上界收紧
        cov_src = getattr(rule, 'combo_cov_est_source', None)
        shrunk = getattr(rule, 'precision_ub_shrunk_by_corr', False)
        if getattr(rule, 'combo_pi_missing', True):
            rec['估计准确率说明'] = '不可计算（缺少pi）'
        else:
            parts = []
            if cov_src:
                parts.append(f'{cov_src}_est')
            if shrunk:
                parts.append('上界由 pair 共现收紧')
            rec['估计准确率说明'] = '，'.join(parts) if parts else None
        records.append(rec)
    
    if format == 'csv':
        file_path = output_path.with_suffix('.csv')
        df = pd.DataFrame(records)
        # 区间字段序列化为 JSON 字符串；所有字段不得出现 NaN
        def _cell_for_csv(x):
            if isinstance(x, list):
                return json.dumps(x, ensure_ascii=False)
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ''
            return x
        for col in df.columns:
            df[col] = df[col].apply(_cell_for_csv)
        df.columns = [CANDIDATE_SEGMENTS_HEADER_CN.get(c, c) for c in df.columns]
        _write_then_replace(file_path, lambda p: df.to_csv(p, index=False, encoding='utf-8-sig'))
        logger.info(f"候选客群规则已导出到: {file_path}")
    
    elif format == 'json':
        file_path = output_path.with_suffix('.json')
        def _write_json(p):
            out = {'candidate_segments': records}
            meta = {}
            if pi_used is not None:
                meta['pi_used'] = pi_used
            if dedup_before is not None:
                meta['candidates_before_dedup'] = dedup_before
            if dedup_after is not None:
                meta['candidates_after_dedup'] = dedup_after
            if dedup_removed is not None:
                meta['dedup_removed'] = dedup_removed
            out['metadata'] = meta
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        _write_then_replace(file_path, _write_json)
        logger.info(f"候选客群规则已导出到: {file_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")


def segment_canonical_key(segment: SegmentRule) -> str:
    """
    生成组合唯一键：按 column_id 排序后，每条规则追加阈值/类别摘要，
    保证同字段同条件唯一、同字段不同条件不同（避免重复 segment）。
    """
    parts = []
    rules = sorted(segment.feature_rules, key=lambda fr: fr.get('column_id', ''))
    for fr in rules:
        cid = fr.get('column_id', '')
        if fr.get('type') == 'numeric':
            low, high = fr.get('low'), fr.get('high')
            direction = fr.get('direction')
            if low is not None and high is not None:
                summary = f"{low}_{high}"
            else:
                summary = str(direction) if direction else "numeric"
            parts.append(f"{cid}:{summary}")
        elif fr.get('type') == 'categorical':
            cats = fr.get('categories') or []
            summary = ",".join(sorted(str(c) for c in cats))
            parts.append(f"{cid}:{summary}")
        else:
            parts.append(cid)
    return "|".join(parts)


def _segment_to_export_dict(
    segment: SegmentRule,
    idx: int,
    tier_name: str,
    column_name_map: Optional[Dict[str, str]] = None,
    total_population: Optional[int] = None,
    config: Optional[object] = None,
) -> dict:
    """v2.0: 单条 segment 转为导出用 dict，含 tier/k/combo_*/估计准确率/估计用户数；config 用于 precision_range_delta。"""
    rules_list = []
    for fr in segment.feature_rules:
        col_name = column_name_map.get(fr['column_id'], fr.get('column_name', fr['column_id'])) if column_name_map else fr.get('column_name', fr['column_id'])
        rule_dict = {
            RULE_KEYS_CN.get('column_id', 'column_id'): fr['column_id'],
            RULE_KEYS_CN.get('column_name', 'column_name'): col_name,
            RULE_KEYS_CN.get('type', 'type'): fr['type'],
            'base_cov': fr.get('base_cov'),
            'sub_cov': fr.get('sub_cov'),
            'feature_family': fr.get('feature_family'),
            'direction': fr.get('direction'),
        }
        if fr['type'] == 'numeric':
            rule_dict[RULE_KEYS_CN.get('low', 'low')] = _json_safe_number(fr.get('low'))
            rule_dict[RULE_KEYS_CN.get('high', 'high')] = _json_safe_number(fr.get('high'))
            rule_dict[RULE_KEYS_CN.get('direction', 'direction')] = fr.get('direction')
        elif fr['type'] == 'categorical':
            rule_dict[RULE_KEYS_CN.get('categories', 'categories')] = fr.get('categories', [])
        rules_list.append(rule_dict)
    sql_clause = generate_sql_where_clause(segment, column_name_map)
    metrics_cn = {
        METRICS_KEYS_CN.get(k, k): v
        for k, v in {
            'divergence_score': segment.divergence_score,
            'stability_score': segment.stability_score,
            'diversity_score': segment.diversity_score,
            'score': segment.score,
            'rule_score': getattr(segment, 'rule_score', None),
        }.items()
    }
    combo_non_sub_cov_lb = getattr(segment, 'combo_non_sub_cov_lb', None)
    combo_non_sub_cov_ub = getattr(segment, 'combo_non_sub_cov_ub', None)
    combo_sub_cov_lb = getattr(segment, 'combo_sub_cov_lb', None)
    combo_sub_cov_ub = getattr(segment, 'combo_sub_cov_ub', None)
    combo_all_cov_lb = getattr(segment, 'combo_all_cov_lb', None)
    combo_all_cov_ub = getattr(segment, 'combo_all_cov_ub', None)
    combo_precision_lb = getattr(segment, 'combo_precision_lb', None)
    combo_precision_ub = getattr(segment, 'combo_precision_ub', None)
    out = {
        SEGMENT_KEYS_CN.get('segment_id', 'segment_id'): f"segment_{idx}",
        SEGMENT_KEYS_CN.get('rule_id', 'rule_id'): segment.rule_id,
        'tier': tier_name,
        'k': len(segment.feature_rules),
        'rule_desc': sql_clause,
        SEGMENT_KEYS_CN.get('sql_where_clause', 'sql_where_clause'): sql_clause,
        SEGMENT_KEYS_CN.get('rules', 'rules'): rules_list,
        SEGMENT_KEYS_CN.get('metrics', 'metrics'): metrics_cn,
        'combo_non_sub_cov_lb': combo_non_sub_cov_lb,
        'combo_non_sub_cov_ub': combo_non_sub_cov_ub,
        'combo_sub_cov_lb': combo_sub_cov_lb,
        'combo_sub_cov_ub': combo_sub_cov_ub,
        'combo_all_cov_lb': combo_all_cov_lb,
        'combo_all_cov_ub': combo_all_cov_ub,
        'combo_all_cov_ub_shrunk': getattr(segment, 'combo_all_cov_ub_shrunk', None),
        'combo_sub_cov_ub_shrunk': getattr(segment, 'combo_sub_cov_ub_shrunk', None),
        'combo_precision_lb': combo_precision_lb,
        'combo_precision_ub': combo_precision_ub,
        'combo_non_sub_cov_est': getattr(segment, 'combo_non_sub_cov_est', None),
        'combo_sub_cov_est': getattr(segment, 'combo_sub_cov_est', None),
        'combo_all_cov_est': getattr(segment, 'combo_all_cov_est', None),
        'combo_lift_est': getattr(segment, 'combo_lift_est', None),
        'combo_precision_est': getattr(segment, 'combo_precision_est', None),
        'fp_rate_est': getattr(segment, 'fp_rate_est', None),
        'combo_cov_unknown': getattr(segment, 'combo_cov_unknown', None),
        'combo_non_sub_cov_ind_est': getattr(segment, 'combo_non_sub_cov_ind_est', None),
        'combo_sub_cov_ind_est': getattr(segment, 'combo_sub_cov_ind_est', None),
        'combo_non_sub_cov_adj_est': getattr(segment, 'combo_non_sub_cov_adj_est', None),
        'combo_sub_cov_adj_est': getattr(segment, 'combo_sub_cov_adj_est', None),
        'combo_pi_missing': getattr(segment, 'combo_pi_missing', None),
        'precision_ub_shrunk_by_corr': getattr(segment, 'precision_ub_shrunk_by_corr', None),
        'r_max': getattr(segment, 'r_max', None),
        'combo_cov_est_source': getattr(segment, 'combo_cov_est_source', None),
        'precision_est_source': getattr(segment, 'combo_cov_est_source', None),
        'kappa': getattr(segment, 'kappa', None),
        'missing_pair_count': getattr(segment, 'missing_pair_count', None),
        'rule_score': getattr(segment, 'rule_score', None),
        'anchor_feature': getattr(segment, 'anchor_feature', None),
    }
    if config and getattr(config, 'include_score_breakdown', True):
        out['score_breakdown'] = getattr(segment, 'score_breakdown', None)
    if config and getattr(config, 'include_redundancy_group_id', True):
        out['redundancy_group_id'] = getattr(segment, 'redundancy_group_id', None)
    if config and getattr(config, 'include_selection_reason', True):
        out['selection_reason'] = getattr(segment, 'selection_reason', None)
    if config and getattr(config, 'include_feature_clusters', True):
        out['feature_clusters'] = getattr(segment, 'feature_clusters', None) or (getattr(segment, 'rule_feature_ids', None) and list(segment.rule_feature_ids)) or []
    missing_cnt = getattr(segment, 'missing_pair_count', None)
    missing_cnt = int(missing_cnt) if missing_cnt is not None else 0
    thr = getattr(config, 'recommended_max_missing_pair_count', 1) if config else 1
    out['low_confidence'] = missing_cnt > thr
    # confidence_level: high/medium/low 由 combo_precision_lb、missing_pair_count、kappa 推导
    prec_lb = getattr(segment, 'combo_precision_lb', None)
    kappa_val = getattr(segment, 'kappa', None)
    kappa_ok = kappa_val is None or (isinstance(kappa_val, (int, float)) and float(kappa_val) < 50)
    if prec_lb is not None and prec_lb >= 0.35 and missing_cnt <= thr and kappa_ok:
        out['confidence_level'] = 'high'
    elif (prec_lb is not None and prec_lb >= 0.2) or missing_cnt <= thr:
        out['confidence_level'] = 'medium'
    else:
        out['confidence_level'] = 'low'
    combo_pi_missing = getattr(segment, 'combo_pi_missing', True)
    n_total = total_population
    n_sub = getattr(config, 'cohort_size', None) if config is not None else None
    sub_est = getattr(segment, 'combo_sub_cov_est', None)
    non_sub_est = getattr(segment, 'combo_non_sub_cov_est', None)
    if n_total is not None and n_sub is not None and sub_est is not None and non_sub_est is not None:
        out['N_sub_est'] = round(sub_est * n_sub)
        out['N_non_sub_est'] = round(non_sub_est * (n_total - n_sub))
        out['N_all_est'] = out['N_sub_est'] + out['N_non_sub_est']
    else:
        out['N_sub_est'] = out['N_non_sub_est'] = out['N_all_est'] = None
    if combo_all_cov_lb is not None and combo_all_cov_ub is not None and n_total is not None:
        out['估计用户数区间'] = [round(combo_all_cov_lb * n_total), round(combo_all_cov_ub * n_total)]
    else:
        out['估计用户数区间'] = None
    if combo_pi_missing:
        out['估计准确率区间'] = None
        out['估计准确率说明'] = '不可计算（缺少pi）'
    else:
        plb = _json_safe_number(combo_precision_lb) if combo_precision_lb is not None else None
        pub = _json_safe_number(combo_precision_ub) if combo_precision_ub is not None else None
        out['估计准确率区间'] = [plb, pub] if (plb is not None or pub is not None) else None
        cov_src = getattr(segment, 'combo_cov_est_source', None)
        shrunk = getattr(segment, 'precision_ub_shrunk_by_corr', False)
        parts = [f'{cov_src}_est'] if cov_src else []
        if shrunk:
            parts.append('上界由 pair 共现收紧')
        if getattr(segment, 'combo_all_cov_ub_shrunk', None) is not None:
            parts.append('cov_ub 已用 pair 联合上界收紧')
        if (getattr(segment, 'missing_pair_count', 0) or 0) > 0:
            parts.append('pair缺失导致区间可信度下降')
        out['估计准确率说明'] = '，'.join(parts) if parts else None
    return out


def _recommended_rule_detail(
    segment: SegmentRule,
    column_name_map: Optional[Dict[str, str]] = None,
    total_population: Optional[int] = None,
    config: Optional[object] = None,
) -> dict:
    """从 SegmentRule 构建一条 recommended_rule_details 条目，便于审计。"""
    full = _segment_to_export_dict(segment, 0, 'recommended', column_name_map, total_population, config)
    rule_id_key = SEGMENT_KEYS_CN.get('rule_id', 'rule_id')
    return {
        'rule_id': full.get(rule_id_key, segment.rule_id),
        'rule_desc': full.get('rule_desc'),
        'rules': full.get(SEGMENT_KEYS_CN.get('rules', 'rules')),
        'combo_precision_lb': full.get('combo_precision_lb'),
        'combo_precision_ub': full.get('combo_precision_ub'),
        'combo_precision_est': full.get('combo_precision_est'),
        'combo_lift_est': full.get('combo_lift_est'),
        'kappa': full.get('kappa'),
        'combo_all_cov_lb': full.get('combo_all_cov_lb'),
        'combo_all_cov_ub': full.get('combo_all_cov_ub'),
        'combo_all_cov_est': full.get('combo_all_cov_est'),
        'combo_sub_cov_est': full.get('combo_sub_cov_est'),
        'missing_pair_count': full.get('missing_pair_count'),
        'rule_score': full.get('rule_score'),
        'anchor_feature': full.get('anchor_feature'),
        'confidence_level': full.get('confidence_level'),
        'score_breakdown': full.get('score_breakdown'),
        'redundancy_group_id': full.get('redundancy_group_id'),
        'selection_reason': full.get('selection_reason'),
        'feature_clusters': full.get('feature_clusters'),
    }


def _menu_sort_key(segment: SegmentRule):
    """按估计准确率（区间/点估计）排序：prec_lb 或 precision_est 降序，缺 pi 时用 lift_est 降权排后。"""
    prec_lb = getattr(segment, 'combo_precision_lb', None)
    prec_est = getattr(segment, 'combo_precision_est', None)
    lift_est = getattr(segment, 'combo_lift_est', None)
    pi_missing = getattr(segment, 'combo_pi_missing', True)
    cov_est = getattr(segment, 'combo_non_sub_cov_est', None)
    has_prec = (prec_est is not None or prec_lb is not None) and not pi_missing
    sort_prec = -(prec_est if prec_est is not None else prec_lb or 0.0)
    if pi_missing and sort_prec == 0 and lift_est is not None:
        sort_prec = -lift_est
    return (
        not has_prec,
        sort_prec,
        cov_est is None,
        cov_est if cov_est is not None else 1.0,
    )


def export_segment_portfolio(
    portfolio: SegmentPortfolio,
    output_path: Path,
    column_name_map: Optional[Dict[str, str]] = None,
    config: Optional[object] = None,
    menu_candidates: Optional[list] = None,
    pi_used: Optional[float] = None,
    pi_source: Optional[str] = None,
) -> None:
    """
    导出推荐客群组合方案。v2.0: 若提供 config 则按准确率排序、条数截断、档位分层，并输出估计准确率/估计用户数。
    当提供 menu_candidates 时，菜单式输出使用该候选列表（按准确率排序取 top N），否则使用 portfolio.segments。
    顶层 metadata 含 pi_used（实际用于 precision 的值）、pi_source（config|computed|missing）。
    """
    output_path = Path(output_path)
    portfolio_metrics_cn = {
        PORTFOLIO_METRICS_HEADER_CN.get(k, k): v
        for k, v in portfolio.portfolio_metrics.items()
    }

    if config is not None:
        total_pop = getattr(config, 'total_population', None)
        max_est = getattr(config, 'max_estimated_users', None)
        max_cov_menu = getattr(config, 'max_combo_cov_est_menu', None)
        min_est = getattr(config, 'min_estimated_users', None)
        out_n = getattr(config, 'output_max_segments', None) or getattr(config, 'max_segments', 5)

        # 菜单模式：有 menu_candidates 时用完整候选列表按准确率排序输出多条
        if menu_candidates:
            segments = list(menu_candidates)
        else:
            segments = list(portfolio.segments)
        # 按 canonical_key（含阈值/类别摘要）去重：同一条件只保留一条，按 precision_lb / precision_est / score 取最优
        if segments:
            by_key: Dict[str, list] = {}
            for s in segments:
                key = segment_canonical_key(s)
                by_key.setdefault(key, []).append(s)
            segments = []
            for key, group in by_key.items():
                group.sort(key=lambda x: (
                    getattr(x, 'combo_precision_lb', None) is None,
                    -(getattr(x, 'combo_precision_lb') or 0.0),
                    -(getattr(x, 'combo_precision_est') or 0.0),
                    -(getattr(x, 'score', 0.0)),
                ))
                segments.append(group[0])
        # 先分层，再保底过滤，保证 elite 不被过滤/截断
        tiers = assign_tiers(segments, config)
        elite_set = set(id(s) for s in tiers['elite'])
        # 保底过滤：elite 一律保留；其余按 max_est/min_est/max_cov_menu 过滤
        if segments:
            kept = []
            for s in segments:
                if id(s) in elite_set:
                    kept.append(s)
                    continue
                all_ub = getattr(s, 'combo_all_cov_ub', None)
                all_lb = getattr(s, 'combo_all_cov_lb', None)
                if total_pop is not None and all_ub is not None:
                    est_users_ub = round(all_ub * total_pop)
                    if max_est is not None and est_users_ub > max_est:
                        continue
                if total_pop is not None and all_lb is not None and min_est is not None:
                    est_users_lb = round(all_lb * total_pop)
                    if est_users_lb < min_est:
                        continue
                if max_cov_menu is not None and all_ub is not None and all_ub > max_cov_menu:
                    continue
                kept.append(s)
            segments = kept
        # 截断：先保留所有 elite，再按 _menu_sort_key 填满至 out_n
        if segments and all(getattr(s, 'combo_precision_lb', None) is None for s in segments):
            logger.warning("pi 缺失，precision 区间无效；菜单按 _menu_sort_key 排序")
        if segments:
            result = list(tiers['elite'])
            remaining = [s for s in segments if s not in result]
            remaining.sort(key=_menu_sort_key)
            for s in remaining:
                if len(result) >= out_n:
                    break
                result.append(s)
            segments = result[:out_n]
        tiers = assign_tiers(segments, config)
        # 日志：cov 模式（ind/adj）与 pi 缺失数量
        n_adj = sum(1 for s in segments if getattr(s, 'combo_non_sub_cov_adj_est', None) is not None)
        n_pi_missing = sum(1 for s in segments if getattr(s, 'combo_pi_missing', True))
        logger.info("菜单导出 %d 条客群：cov 模式 ind=%d、adj=%d；pi 缺失 %d 条", len(segments), len(segments) - n_adj, n_adj, n_pi_missing)
        total_pop_for_export = total_pop
        rank = 0
        elite_cn = []
        for s in tiers['elite']:
            rank += 1
            elite_cn.append(_segment_to_export_dict(s, rank, 'elite', column_name_map, total_pop_for_export, config))
        standard_cn = []
        for s in tiers['standard']:
            rank += 1
            standard_cn.append(_segment_to_export_dict(s, rank, 'standard', column_name_map, total_pop_for_export, config))
        expand_cn = []
        for s in tiers['expand']:
            rank += 1
            expand_cn.append(_segment_to_export_dict(s, rank, 'expand', column_name_map, total_pop_for_export, config))
        # 两阶段选择：elite 非空时强制选 1 个，其余按结构距离+综合得分
        selected_list, selection_steps, elite_forced = select_final_segments(segments, tiers, config)
        logger.info("推荐客群数=%d（两阶段选择结果）", len(selected_list))
        if selected_list:
            mp_parts = [f"{s.rule_id}={getattr(s, 'missing_pair_count', None) or 0}" for s in selected_list]
            logger.info("recommended 各条 missing_pair_count: %s", ", ".join(mp_parts))
        # 组合指标基于最终推荐 selected_list 重算（与 JSON recommended 一致）
        n_sel = len(selected_list)
        if n_sel > 0:
            recommended_metrics = {
                'total_segments': n_sel,
                'avg_divergence_score': sum(s.divergence_score for s in selected_list) / n_sel,
                'avg_stability_score': sum(s.stability_score for s in selected_list) / n_sel,
                'avg_diversity_score': sum(s.diversity_score for s in selected_list) / n_sel,
                'avg_score': sum(s.score for s in selected_list) / n_sel,
                'min_structure_distance': min(
                    calculate_rule_structure_distance(selected_list[i], selected_list[j])
                    for i in range(n_sel) for j in range(i + 1, n_sel)
                ) if n_sel > 1 else 1.0,
            }
        else:
            recommended_metrics = {
                'total_segments': 0,
                'avg_divergence_score': 0.0,
                'avg_stability_score': 0.0,
                'avg_diversity_score': 0.0,
                'avg_score': 0.0,
                'min_structure_distance': 1.0,
            }
        portfolio_metrics_cn = {PORTFOLIO_METRICS_HEADER_CN.get(k, k): v for k, v in recommended_metrics.items()}
        # 为 selection_steps 补充 rule_desc（便于直接读出推荐组合），并打日志
        seg_by_id = {s.rule_id: s for s in segments}
        for step in selection_steps:
            rid = step.get('rule_id')
            if rid and rid in seg_by_id:
                rule_desc = generate_sql_where_clause(seg_by_id[rid], column_name_map)
                step['rule_desc'] = rule_desc
                if step.get('reason') == 'force_elite':
                    logger.info("组合选择阶段1：强制选入 elite 规则 rule_id=%s, rule_desc=%s", rid, rule_desc[:200] + "..." if len(rule_desc) > 200 else rule_desc)
        selected_segments = [
            {
                'rule_id': s.rule_id,
                'segment_id': f"segment_{i+1}",
                'k': len(s.feature_rules),
                'rule_feature_ids': getattr(s, 'rule_feature_ids', None) or [fr['column_id'] for fr in s.feature_rules],
            }
            for i, s in enumerate(selected_list, 1)
        ]
        target_n = getattr(config, 'target_segment_count', None) or getattr(config, 'max_segments', 5)
        selected_count = len(selected_segments)
        metadata = {
            'pi_used': pi_used,
            'pi_source': pi_source if pi_source is not None else 'missing',
            'target_segment_count': target_n,
            'selected_count': selected_count,
            'recommended_meaning': '两阶段选择后的最终推荐客群',
        }
        if selected_count != target_n:
            reason = '候选不足' if selected_count < target_n else '超过目标条数'
            metadata['reason'] = reason
            logger.info(
                "selected_count=%d, target_segment_count=%d, reason=%s",
                selected_count, target_n, reason,
            )
        reason_parts = [f"elite_forced={str(elite_forced).lower()}", f"target_segment_count={target_n}"]
        for st in selection_steps:
            reason_parts.append(f"step{st.get('step', '')}: {st.get('reason', '')}")
        selection_reason = "; ".join(reason_parts)
        menu_list = elite_cn + standard_cn + expand_cn
        rule_id_key = SEGMENT_KEYS_CN.get('rule_id', 'rule_id')
        menu_rule_ids = {d.get(rule_id_key) or d.get('rule_id') for d in menu_list if d.get(rule_id_key) or d.get('rule_id')}
        appended_for_recommended = []
        for s in selected_list:
            if s.rule_id not in menu_rule_ids:
                rank = len(menu_list) + 1
                menu_list.append(_segment_to_export_dict(s, rank, 'recommended', column_name_map, total_pop_for_export, config))
                menu_rule_ids.add(s.rule_id)
                appended_for_recommended.append(s.rule_id)
        if appended_for_recommended:
            logger.info("menu 已强制包含 recommended 的 %d 条 rule_id: %s", len(appended_for_recommended), ", ".join(appended_for_recommended))
        else:
            logger.info("menu 已包含全部 recommended 的 %d 条 rule_id", len(selected_list))
        recommended_rule_details = [
            _recommended_rule_detail(s, column_name_map, total_pop_for_export, config) for s in selected_list
        ]
        logger.info("recommended_rule_details 已写入，共 %d 条", len(recommended_rule_details))
        target_count = getattr(config, 'target_segment_count', None) or getattr(config, 'max_segments', 5)
        has_breakdown = any(getattr(s, 'score_breakdown', None) for s in selected_list) if selected_list else False
        logger.info(
            "[stage2] recommended count=%d (target %s) has_score_breakdown=%s risk_score_api=ok",
            len(selected_list), target_count, has_breakdown,
        )
        portfolio_data = {
            'metadata': metadata,
            'selected_segments': selected_segments,
            'recommended': selected_segments,
            'recommended_rule_details': recommended_rule_details,
            'menu': menu_list,
            'selection_trace': selection_steps,
            'selection_reason': selection_reason,
            'elite_forced': elite_forced,
            'selection_steps': selection_steps,
            'tiers': {
                'elite': elite_cn,
                'standard': standard_cn,
                'expand': expand_cn,
            },
            '组合指标': portfolio_metrics_cn,
        }
    else:
        total_pop = None
        segments_cn = [
            _segment_to_export_dict(s, idx, '', column_name_map, total_pop)
            for idx, s in enumerate(portfolio.segments, 1)
        ]
        for d in segments_cn:
            d.pop('tier', None)
        selected_segments = [
            {
                'rule_id': s.rule_id,
                'segment_id': f"segment_{i+1}",
                'k': len(s.feature_rules),
                'rule_feature_ids': getattr(s, 'rule_feature_ids', None) or [fr['column_id'] for fr in s.feature_rules],
            }
            for i, s in enumerate(portfolio.segments, 1)
        ]
        selection_reason = "基于规则结构距离的差异最大化"
        portfolio_data = {
            'metadata': {
                'pi_used': pi_used,
                'pi_source': pi_source if pi_source is not None else 'missing',
            },
            'selected_segments': selected_segments,
            'selection_reason': selection_reason,
            '组合指标': portfolio_metrics_cn,
            '客群列表': segments_cn
        }

    def _write_json(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
    _write_then_replace(output_path, _write_json)
    logger.info(f"推荐客群组合方案已导出到: {output_path}")

