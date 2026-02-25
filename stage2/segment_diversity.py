"""
规则多样性评估模块

计算规则之间的多样性、相似度等指标
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


def calculate_feature_diversity(rule1: Any, rule2: Any) -> float:
    """
    计算两个规则的特征多样性
    
    基于特征集合的差异（Jaccard距离）
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        多样性得分（0-1），1表示完全不同，0表示完全相同
    """
    # 获取两个规则使用的字段集合
    columns1 = {fr['column_id'] for fr in rule1.feature_rules}
    columns2 = {fr['column_id'] for fr in rule2.feature_rules}
    
    if not columns1 and not columns2:
        return 0.0
    
    if not columns1 or not columns2:
        return 1.0
    
    # 计算Jaccard距离 = 1 - Jaccard相似度
    intersection = len(columns1 & columns2)
    union = len(columns1 | columns2)
    
    if union == 0:
        return 0.0
    
    jaccard_similarity = intersection / union
    jaccard_distance = 1.0 - jaccard_similarity
    
    return jaccard_distance


def calculate_interval_overlap(rule1: Any, rule2: Any) -> float:
    """
    计算区间重叠度
    
    对于相同字段，计算数值区间重叠比例
    对于离散字段，计算类别集合重叠比例
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        重叠度（0-1），1表示完全重叠，0表示无重叠
    """
    # 构建字段到规则的映射
    rule1_map = {fr['column_id']: fr for fr in rule1.feature_rules}
    rule2_map = {fr['column_id']: fr for fr in rule2.feature_rules}
    
    # 找出共同字段
    common_columns = set(rule1_map.keys()) & set(rule2_map.keys())
    
    if not common_columns:
        return 0.0
    
    overlap_scores = []
    
    for col_id in common_columns:
        fr1 = rule1_map[col_id]
        fr2 = rule2_map[col_id]
        
        # 连续特征：计算区间重叠比例
        if fr1.get('type') == 'numeric' and fr2.get('type') == 'numeric':
            low1 = fr1.get('low', float('-inf'))
            high1 = fr1.get('high', float('inf'))
            low2 = fr2.get('low', float('-inf'))
            high2 = fr2.get('high', float('inf'))
            
            # 计算重叠区间
            overlap_low = max(low1, low2)
            overlap_high = min(high1, high2)
            
            if overlap_low >= overlap_high:
                # 无重叠
                overlap_ratio = 0.0
            else:
                # 计算重叠比例（使用并集区间长度）
                range1 = high1 - low1 if high1 != float('inf') and low1 != float('-inf') else 1.0
                range2 = high2 - low2 if high2 != float('inf') and low2 != float('-inf') else 1.0
                
                # 并集区间
                union_low = min(low1, low2)
                union_high = max(high1, high2)
                union_range = union_high - union_low if union_high != float('inf') and union_low != float('-inf') else 1.0
                
                overlap_range = overlap_high - overlap_low
                
                if union_range > 0:
                    overlap_ratio = overlap_range / union_range
                else:
                    # 如果两个区间都是无限，认为重叠度为0.5（不确定）
                    overlap_ratio = 0.5
            
            overlap_scores.append(overlap_ratio)
        
        # 离散特征：计算类别集合重叠比例
        elif fr1.get('type') == 'categorical' and fr2.get('type') == 'categorical':
            cats1 = set(fr1.get('categories', []))
            cats2 = set(fr2.get('categories', []))
            
            if not cats1 or not cats2:
                overlap_ratio = 0.0
            else:
                intersection = len(cats1 & cats2)
                union = len(cats1 | cats2)
                overlap_ratio = intersection / union if union > 0 else 0.0
            
            overlap_scores.append(overlap_ratio)
    
    # 返回平均重叠度
    if overlap_scores:
        return np.mean(overlap_scores)
    else:
        return 0.0


def calculate_rule_structure_distance(rule1: Any, rule2: Any) -> float:
    """
    计算规则结构距离（用于多客群去重）
    
    基于字段集合差异和区间重叠度，计算规则之间的结构距离
    距离越大，规则差异越大
    
    公式：distance = 1 - similarity
    similarity = w_field · field_similarity + w_interval · interval_overlap
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        结构距离（0-1之间），0表示完全相同，1表示完全不同
    """
    # 获取字段集合
    columns1 = {fr['column_id'] for fr in rule1.feature_rules}
    columns2 = {fr['column_id'] for fr in rule2.feature_rules}
    
    # 字段集合相似度（Jaccard相似度）
    if not columns1 and not columns2:
        field_similarity = 1.0
    elif not columns1 or not columns2:
        field_similarity = 0.0
    else:
        intersection = len(columns1 & columns2)
        union = len(columns1 | columns2)
        field_similarity = intersection / union if union > 0 else 0.0
    
    # 区间重叠度
    interval_overlap = calculate_interval_overlap(rule1, rule2)
    
    # 综合相似度（字段集合权重0.4，区间重叠权重0.6）
    similarity = 0.4 * field_similarity + 0.6 * interval_overlap
    
    # 结构距离 = 1 - 相似度
    distance = 1.0 - similarity
    
    return max(0.0, min(1.0, distance))


def calculate_rule_similarity(rule1: Any, rule2: Any) -> float:
    """
    计算规则相似度（兼容旧接口）
    
    公式：similarity = 共享字段数量 + 区间重叠度
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        相似度（0-1之间）
    """
    # 获取字段集合
    columns1 = {fr['column_id'] for fr in rule1.feature_rules}
    columns2 = {fr['column_id'] for fr in rule2.feature_rules}
    
    # 共享字段数量（归一化）
    shared_count = len(columns1 & columns2)
    total_count = len(columns1 | columns2)
    shared_ratio = shared_count / total_count if total_count > 0 else 0.0
    
    # 区间重叠度
    interval_overlap = calculate_interval_overlap(rule1, rule2)
    
    # 综合相似度（加权平均）
    similarity = 0.4 * shared_ratio + 0.6 * interval_overlap
    
    return min(1.0, similarity)


def calculate_rule_diversity_score(rule: Any, existing_rules: List[Any]) -> float:
    """
    计算规则相对于现有规则集的多样性得分
    
    基于规则结构距离（字段集合 + 区间重叠）计算多样性
    
    Args:
        rule: 待评估规则
        existing_rules: 现有规则列表
    
    Returns:
        多样性得分（0-1），越高表示与现有规则差异越大
    """
    if not existing_rules:
        return 1.0  # 第一个规则，多样性最高
    
    # 计算与所有现有规则的平均结构距离
    distances = [calculate_rule_structure_distance(rule, existing_rule) for existing_rule in existing_rules]
    avg_distance = np.mean(distances) if distances else 0.0
    
    # 多样性得分 = 平均结构距离
    diversity = avg_distance
    
    return max(0.0, min(1.0, diversity))


def filter_similar_rules(
    rules: List[Any],
    similarity_threshold: float = 0.5,
    max_rules: Optional[int] = None
) -> List[Any]:
    """
    基于规则结构距离进行多客群去重
    
    使用贪心算法：按得分从高到低，保留与已选规则结构距离足够大的规则
    
    Args:
        rules: 候选规则列表（应已按得分排序）
        similarity_threshold: 相似度阈值，超过此值的规则将被过滤（默认0.5）
        max_rules: 最大保留规则数（可选）
    
    Returns:
        去重后的规则列表
    """
    if not rules:
        return []
    
    # 按得分排序（如果规则有score属性）
    sorted_rules = sorted(rules, key=lambda r: getattr(r, 'score', 0.0), reverse=True)
    
    selected_rules = []
    
    for rule in sorted_rules:
        if max_rules and len(selected_rules) >= max_rules:
            break
        
        if not selected_rules:
            # 第一个规则直接加入
            selected_rules.append(rule)
        else:
            # 计算与所有已选规则的平均相似度
            similarities = [
                calculate_rule_similarity(rule, selected_rule)
                for selected_rule in selected_rules
            ]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # 如果平均相似度低于阈值，则保留该规则
            if avg_similarity < similarity_threshold:
                selected_rules.append(rule)
    
    return selected_rules

