"""
字段对关联统计表索引（PairAssocIndex）

对 ST_ANA_FEAT_PAIR_ASSOC_ALL_YYYYMM 做预处理，提供 O(1) 成对查询，
支持 corr_strength / corr_sign / corr_type / assoc_method / support_count。
"""
from typing import Optional, Dict, Tuple, Any
import pandas as pd
import numpy as np

# 列名与 rule_combination 一致，并兼容小写/下划线
COL_A = "column_id_a"
COL_B = "column_id_b"
CORR_STRENGTH = "corr_strength"
CORR_TYPE = "corr_type"
CORR_SIGN = "corr_sign"
SUPPORT_COUNT = "support_count"
STAT_DATE = "stat_date"
ASSOC_METHOD = "assoc_method"
# 可选：pair-level lift/cov 用于组合覆盖率几何校正
PAIR_LIFT_FULL = "pair_lift_full"
PAIR_LIFT_COHORT = "pair_lift_cohort"
PAIR_COV_FULL = "pair_cov_full"
PAIR_COV_COHORT = "pair_cov_cohort"


def _norm_key(a: str, b: str) -> Tuple[str, str]:
    """规范化为无序对 (min, max) 便于去重与对称查询。"""
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


_CANONICAL = {
    "column_id_a": COL_A,
    "column_id_b": COL_B,
    "corr_strength": CORR_STRENGTH,
    "corr_type": CORR_TYPE,
    "corr_sign": CORR_SIGN,
    "support_count": SUPPORT_COUNT,
    "stat_date": STAT_DATE,
    "assoc_method": ASSOC_METHOD,
    "pair_lift_full": PAIR_LIFT_FULL,
    "pair_lift_cohort": PAIR_LIFT_COHORT,
    "pair_cov_full": PAIR_COV_FULL,
    "pair_cov_cohort": PAIR_COV_COHORT,
}


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """列名标准化为小写+下划线，映射到已知键。"""
    mapping = {}
    for c in df.columns:
        nc = str(c).strip().lower().replace("-", "_").replace(" ", "_")
        if nc in _CANONICAL:
            mapping[c] = _CANONICAL[nc]
    return df.rename(columns=mapping) if mapping else df


class PairAssocIndex:
    """
    字段对关联表索引：预处理后 O(1) 查询 strength/sign/method/corr_type，
    支持支持度过滤与对称查询。
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stat_date: Optional[str] = None,
        min_support: int = 0,
    ) -> None:
        """
        Args:
            df: 原始 DataFrame，需含 column_id_a, column_id_b, corr_strength；可选 corr_type, corr_sign, support_count, stat_date.
            stat_date: 若表有 stat_date 列，只保留该月份行（与 str/int 一致即可）。
            min_support: support_count 列存在时，低于此值的行不进入索引。
        """
        self.n_raw = len(df)
        if df is None or len(df) == 0:
            self._index: Dict[Tuple[str, str], Dict[str, Any]] = {}
            self.n_dedup = 0
            self.n_after_support = 0
            return

        df = _norm_cols(df.copy())
        need = {COL_A, COL_B, CORR_STRENGTH}
        if not need.issubset(df.columns):
            self._index = {}
            self.n_dedup = 0
            self.n_after_support = 0
            return

        # 可选：按 stat_date 过滤
        if stat_date is not None and STAT_DATE in df.columns:
            stat_str = str(stat_date).strip()
            df = df[df[STAT_DATE].astype(str).str.strip() == stat_str]
        if len(df) == 0:
            self._index = {}
            self.n_dedup = 0
            self.n_after_support = 0
            return

        # 去重：同一无序对只保留一行（首行）；过滤无效 strength 与 support
        seen = set()
        rows_ok = []
        for _, row in df.iterrows():
            try:
                a, b = str(row[COL_A]).strip(), str(row[COL_B]).strip()
            except Exception:
                continue
            key = _norm_key(a, b)
            if key in seen:
                continue
            try:
                strength = row.get(CORR_STRENGTH)
                if strength is None or (isinstance(strength, float) and np.isnan(strength)):
                    continue
                strength = float(strength)
                if not (0 <= strength <= 1):
                    continue
            except (TypeError, ValueError):
                continue
            if SUPPORT_COUNT in row.index and row.get(SUPPORT_COUNT) is not None:
                try:
                    sc = int(row[SUPPORT_COUNT])
                    if sc < min_support:
                        continue
                except (TypeError, ValueError):
                    continue
            seen.add(key)
            rec = {
                "corr_strength": strength,
                "corr_type": str(row[CORR_TYPE]).strip() if CORR_TYPE in row.index and pd.notna(row.get(CORR_TYPE)) and row.get(CORR_TYPE) not in (None, "") else None,
                "corr_sign": None,
                "assoc_method": str(row[ASSOC_METHOD]).strip() if ASSOC_METHOD in row.index and pd.notna(row.get(ASSOC_METHOD)) else None,
                "support_count": int(row[SUPPORT_COUNT]) if SUPPORT_COUNT in row.index and pd.notna(row.get(SUPPORT_COUNT)) else None,
            }
            try:
                if CORR_SIGN in row.index and row.get(CORR_SIGN) is not None and not (isinstance(row[CORR_SIGN], float) and np.isnan(row[CORR_SIGN])):
                    rec["corr_sign"] = int(row[CORR_SIGN])
            except (TypeError, ValueError):
                pass
            # 可选：pair-level lift（用于组合覆盖率几何校正）
            for lift_key, col_name in [(PAIR_LIFT_FULL, PAIR_LIFT_FULL), (PAIR_LIFT_COHORT, PAIR_LIFT_COHORT)]:
                if col_name in row.index and row.get(col_name) is not None and not (isinstance(row.get(col_name), float) and np.isnan(row.get(col_name))):
                    try:
                        rec[lift_key] = float(row[col_name])
                    except (TypeError, ValueError):
                        rec[lift_key] = None
                else:
                    rec[lift_key] = None
            rows_ok.append((key, rec))

        self.n_dedup = len(rows_ok)
        self.n_after_support = self.n_dedup  # 已在上面按 min_support 过滤
        self._index = dict(rows_ok)

    def _get(self, col1: str, col2: str) -> Optional[Dict[str, Any]]:
        key = _norm_key(str(col1).strip(), str(col2).strip())
        return self._index.get(key)

    def get_strength(self, col1: str, col2: str) -> Optional[float]:
        rec = self._get(col1, col2)
        return rec["corr_strength"] if rec else None

    def get_sign(self, col1: str, col2: str) -> Optional[int]:
        rec = self._get(col1, col2)
        return rec.get("corr_sign") if rec else None

    def get_method(self, col1: str, col2: str) -> Optional[str]:
        rec = self._get(col1, col2)
        return rec.get("assoc_method") if rec else None

    def get_corr_type(self, col1: str, col2: str) -> Optional[str]:
        rec = self._get(col1, col2)
        return rec.get("corr_type") if rec else None

    def get_pair_lift(self, col1: str, col2: str, scope: str = "full") -> Optional[float]:
        """
        返回字段对的 pair-level lift。scope="full" 用 pair_lift_full，scope="cohort" 用 pair_lift_cohort。
        表无该列或缺失时返回 None。
        """
        rec = self._get(col1, col2)
        if not rec:
            return None
        key = PAIR_LIFT_COHORT if (scope and str(scope).strip().lower() == "cohort") else PAIR_LIFT_FULL
        val = rec.get(key)
        if val is None or (isinstance(val, float) and (np.isnan(val) or val <= 0)):
            return None
        return float(val)

    def is_highly_correlated(
        self,
        col1: str,
        col2: str,
        thr: float,
        min_support: int = 0,
    ) -> bool:
        rec = self._get(col1, col2)
        if not rec:
            return False
        if rec["corr_strength"] is None or rec["corr_strength"] < thr:
            return False
        if min_support > 0 and rec.get("support_count") is not None and rec["support_count"] < min_support:
            return False
        return True
