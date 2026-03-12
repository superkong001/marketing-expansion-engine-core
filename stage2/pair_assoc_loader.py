"""
字段对关联表加载：从 CSV / Excel / 路径加载 ST_ANA_FEAT_PAIR_ASSOC_ALL，返回 DataFrame 供 PairAssocIndex 使用。
并提供冗余剪枝判断：corr_strength >= thr 且 corr_type in {cont_cont, disc_disc, disc_cont} 时视为冗余对。
"""
from pathlib import Path
from typing import Optional, Tuple, Any
import pandas as pd

try:
    from .pair_assoc import PairAssocIndex
except ImportError:
    try:
        from pair_assoc import PairAssocIndex
    except ImportError:
        PairAssocIndex = None

# 冗余剪枝允许的 corr_type 前缀（归一化后匹配）
CORR_PRUNE_TYPE_PREFIXES = ("cont_cont", "disc_disc", "disc_cont", "cont_disc")


def _normalize_corr_type(ct: Optional[str]) -> str:
    """归一化 corr_type 为小写、连续下划线，便于匹配 cont_cont / disc_disc / disc_cont。"""
    if not ct or not isinstance(ct, str):
        return ""
    return ct.strip().lower().replace("-", "_").replace(" ", "_")


def load_pair_assoc(
    path: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    stat_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    从文件路径或已有 DataFrame 加载字段对关联表。
    支持 .csv / .xlsx / .xls；返回标准化列名的 DataFrame，若无有效数据则返回 None。
    """
    if df is not None and len(df) > 0:
        return _norm_columns(df.copy())
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            out = pd.read_excel(path)
        else:
            out = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None
    if out is None or len(out) == 0:
        return None
    return _norm_columns(out)


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名标准化为小写+下划线，兼容 column_id_a/b, corr_strength, pair_lift_full/cohort, pair_cov_full/cohort。"""
    mapping = {}
    for c in df.columns:
        nc = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        if nc in (
            "column_id_a",
            "column_id_b",
            "corr_strength",
            "corr_sign",
            "corr_type",
            "assoc_method",
            "support_count",
            "stat_date",
            "pair_lift_full",
            "pair_lift_cohort",
            "pair_cov_full",
            "pair_cov_cohort",
        ):
            mapping[c] = nc
    return df.rename(columns=mapping) if mapping else df


def should_prune_pair(
    pair_index: Optional[Any],
    col_a: str,
    col_b: str,
    corr_prune_th: float = 0.85,
    corr_type_prefixes: Tuple[str, ...] = CORR_PRUNE_TYPE_PREFIXES,
) -> bool:
    """
    若 corr_strength >= corr_prune_th 且 corr_type 属于 cont_cont / disc_disc / disc_cont（或 cont_disc），
    认为两规则冗余，应剪枝（不允许同时出现在同一组合）。
    无 pair_index 或查不到对则返回 False。
    """
    if pair_index is None:
        return False
    st = pair_index.get_strength(col_a, col_b)
    if st is None or st < corr_prune_th:
        return False
    ct = pair_index.get_corr_type(col_a, col_b)
    norm = _normalize_corr_type(ct)
    if not norm:
        return True  # 无类型信息时按高相关剪枝
    return any(norm.startswith(prefix) or prefix in norm for prefix in corr_type_prefixes)
