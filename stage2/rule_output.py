"""
规则结构化输出模块

将客群和规则转换为SQL和JSON格式
"""
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .rule_combination import SegmentRule
    from .segment_portfolio import SegmentPortfolio
except ImportError:
    from rule_combination import SegmentRule
    from segment_portfolio import SegmentPortfolio

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
}
CANDIDATE_SEGMENTS_HEADER_CN = {
    'rule_id': '规则编号',
    'rule_expression': '规则表达式',
    'sql_where_clause': 'SQL筛选条件',
    'divergence_score': '差异评分',
    'stability_score': '稳定性评分',
    'diversity_score': '多样性评分',
    'score': '综合得分',
    'feature_count': '包含字段数',
    'combo_base_cov_est': '组合全量覆盖率估计',
    'combo_sub_cov_est': '组合圈定覆盖率估计',
    'combo_lift_est': '组合Lift估计',
    'combo_precision_est': '组合精度估计',
    'fp_rate_est': '误判率估计',
}
PORTFOLIO_METRICS_HEADER_CN = {
    'total_segments': '客群数量',
    'avg_divergence_score': '平均差异评分',
    'avg_stability_score': '平均稳定性评分',
    'avg_diversity_score': '平均多样性评分',
    'avg_score': '平均综合得分',
    'min_structure_distance': '最小结构距离',
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
    format: str = 'csv'
) -> None:
    """
    导出候选客群规则
    
    Args:
        candidate_rules: 候选规则列表
        output_path: 输出文件路径（不含扩展名）
        column_name_map: 字段名映射
        format: 输出格式（'csv' 或 'json'）
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
        
        rec = {
            'rule_id': rule.rule_id,
            'rule_expression': rule_expr,
            'sql_where_clause': sql_clause,
            'divergence_score': rule.divergence_score,
            'stability_score': rule.stability_score,
            'diversity_score': rule.diversity_score,
            'score': rule.score,
            'feature_count': len(rule.feature_rules),
        }
        for key in ('combo_base_cov_est', 'combo_sub_cov_est', 'combo_lift_est', 'combo_precision_est', 'fp_rate_est'):
            rec[key] = getattr(rule, key, None)
        records.append(rec)
    
    if format == 'csv':
        file_path = output_path.with_suffix('.csv')
        df = pd.DataFrame(records)
        df.columns = [CANDIDATE_SEGMENTS_HEADER_CN.get(c, c) for c in df.columns]
        _write_then_replace(file_path, lambda p: df.to_csv(p, index=False, encoding='utf-8-sig'))
        logger.info(f"候选客群规则已导出到: {file_path}")
    
    elif format == 'json':
        file_path = output_path.with_suffix('.json')
        def _write_json(p):
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        _write_then_replace(file_path, _write_json)
        logger.info(f"候选客群规则已导出到: {file_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")


def export_segment_portfolio(
    portfolio: SegmentPortfolio,
    output_path: Path,
    column_name_map: Optional[Dict[str, str]] = None
) -> None:
    """
    导出推荐客群组合方案
    
    Args:
        portfolio: SegmentPortfolio对象
        output_path: 输出文件路径（JSON格式）
        column_name_map: 字段名映射
    """
    output_path = Path(output_path)
    
    # 构建导出数据（使用业务可读键名）
    portfolio_metrics_cn = {
        PORTFOLIO_METRICS_HEADER_CN.get(k, k): v
        for k, v in portfolio.portfolio_metrics.items()
    }
    segments_cn = []
    for idx, segment in enumerate(portfolio.segments, 1):
        rules_list = []
        for fr in segment.feature_rules:
            col_name = column_name_map.get(fr['column_id'], fr.get('column_name', fr['column_id'])) if column_name_map else fr.get('column_name', fr['column_id'])
            rule_dict = {
                RULE_KEYS_CN.get('column_id', 'column_id'): fr['column_id'],
                RULE_KEYS_CN.get('column_name', 'column_name'): col_name,
                RULE_KEYS_CN.get('type', 'type'): fr['type']
            }
            if fr['type'] == 'numeric':
                rule_dict[RULE_KEYS_CN.get('low', 'low')] = fr.get('low')
                rule_dict[RULE_KEYS_CN.get('high', 'high')] = fr.get('high')
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
                'score': segment.score
            }.items()
        }
        segment_data = {
            SEGMENT_KEYS_CN.get('segment_id', 'segment_id'): f"segment_{idx}",
            SEGMENT_KEYS_CN.get('rule_id', 'rule_id'): segment.rule_id,
            SEGMENT_KEYS_CN.get('rules', 'rules'): rules_list,
            SEGMENT_KEYS_CN.get('sql_where_clause', 'sql_where_clause'): sql_clause,
            SEGMENT_KEYS_CN.get('metrics', 'metrics'): metrics_cn
        }
        segments_cn.append(segment_data)
    portfolio_data = {
        '组合指标': portfolio_metrics_cn,
        '客群列表': segments_cn
    }
    
    def _write_json(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
    _write_then_replace(output_path, _write_json)
    logger.info(f"推荐客群组合方案已导出到: {output_path}")

