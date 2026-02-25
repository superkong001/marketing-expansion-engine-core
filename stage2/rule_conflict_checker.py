"""
规则逻辑冲突检测模块

检测规则之间的逻辑冲突
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def check_numeric_conflict(rule1: Any, rule2: Any) -> Tuple[bool, str]:
    """
    检测数值规则冲突
    
    检查相同字段的区间是否矛盾（如 [10, 20] 和 [30, 40] 无重叠但都要求满足）
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        (是否有冲突, 冲突描述)
    """
    # 构建字段到规则的映射
    rule1_map = {fr['column_id']: fr for fr in rule1.feature_rules if fr['type'] == 'numeric'}
    rule2_map = {fr['column_id']: fr for fr in rule2.feature_rules if fr['type'] == 'numeric'}
    
    # 找出共同字段
    common_columns = set(rule1_map.keys()) & set(rule2_map.keys())
    
    for col_id in common_columns:
        fr1 = rule1_map[col_id]
        fr2 = rule2_map[col_id]
        
        low1 = fr1.get('low', float('-inf'))
        high1 = fr1.get('high', float('inf'))
        low2 = fr2.get('low', float('-inf'))
        high2 = fr2.get('high', float('inf'))
        
        # 检查区间是否完全分离（无重叠）
        if high1 <= low2 or high2 <= low1:
            return True, f"字段 {col_id} 的区间完全分离: [{low1}, {high1}) 和 [{low2}, {high2})"
        
        # 检查是否有逻辑矛盾（例如一个要求>=x，另一个要求<x）
        # 这里简化处理，主要检查完全分离的情况
    
    return False, ""


def check_categorical_conflict(rule1: Any, rule2: Any) -> Tuple[bool, str]:
    """
    检测离散规则冲突
    
    检查相同字段的类别集合是否矛盾（交集为空）
    
    Args:
        rule1: 规则1
        rule2: 规则2
    
    Returns:
        (是否有冲突, 冲突描述)
    """
    # 构建字段到规则的映射
    rule1_map = {fr['column_id']: fr for fr in rule1.feature_rules if fr['type'] == 'categorical'}
    rule2_map = {fr['column_id']: fr for fr in rule2.feature_rules if fr['type'] == 'categorical'}
    
    # 找出共同字段
    common_columns = set(rule1_map.keys()) & set(rule2_map.keys())
    
    for col_id in common_columns:
        fr1 = rule1_map[col_id]
        fr2 = rule2_map[col_id]
        
        cats1 = set(fr1.get('categories', []))
        cats2 = set(fr2.get('categories', []))
        
        # 检查类别集合是否完全不相交
        if cats1 and cats2 and len(cats1 & cats2) == 0:
            return True, f"字段 {col_id} 的类别集合完全不相交: {cats1} 和 {cats2}"
    
    return False, ""


def check_rule_conflicts(rule: Any, existing_rules: List[Any]) -> List[Dict[str, str]]:
    """
    检查规则与现有规则集的冲突
    
    Args:
        rule: 待检查规则
        existing_rules: 现有规则列表
    
    Returns:
        冲突列表，每个冲突包含 'type' 和 'description'
    """
    conflicts = []
    
    for existing_rule in existing_rules:
        # 检查数值规则冲突
        has_numeric_conflict, numeric_desc = check_numeric_conflict(rule, existing_rule)
        if has_numeric_conflict:
            conflicts.append({
                'type': 'numeric',
                'description': numeric_desc,
                'conflict_with': existing_rule.rule_id
            })
        
        # 检查离散规则冲突
        has_categorical_conflict, categorical_desc = check_categorical_conflict(rule, existing_rule)
        if has_categorical_conflict:
            conflicts.append({
                'type': 'categorical',
                'description': categorical_desc,
                'conflict_with': existing_rule.rule_id
            })
    
    return conflicts

