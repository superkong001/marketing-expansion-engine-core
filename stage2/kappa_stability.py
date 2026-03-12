"""
病态度量/条件数稳定性评估：仅用字段对关联表 ST_ANA_FEAT_PAIR_ASSOC_ALL，不依赖明细。

对每个候选组合的字段集合构造相关矩阵 R，计算条件数 kappa = σ_max/σ_min，
用于 beam search 剪枝或惩罚。与现有 corr_strength 两两剪枝并存，不替代。
"""
from typing import List, Optional, Tuple, Any
import numpy as np
import math

_R_MAX = 0.99  # 截断 r <= 0.99


def _r_from_pair(
    strength: float,
    corr_type: Optional[str],
) -> float:
    """
    按 corr_type 将强度映射为相关矩阵元素 r_ij：
    cont_cont_linear -> r = abs(corr_strength)
    cont_cat_effect（η²）-> r = sqrt(corr_strength)
    cat_cat_assoc（Cramer's V）-> r = corr_strength
    缺失类型时按 abs(strength)，再截断 min(r, 0.99)。
    """
    s = float(strength)
    if s <= 0 or (corr_type is None or not isinstance(corr_type, str)):
        r = min(abs(s), _R_MAX)
    else:
        ct = str(corr_type).strip().lower().replace("-", "_").replace(" ", "_")
        if "cont_cont" in ct or "linear" in ct or "rank" in ct:
            r = abs(s)
        elif "cont_cat" in ct or "cat_cont" in ct or "disc_cont" in ct or "cont_disc" in ct or "effect" in ct or "eta" in ct:
            r = math.sqrt(s)
        else:
            r = s  # Cramer's V / cat_cat_assoc / disc_disc
        r = min(r, _R_MAX)
    return r


def build_correlation_matrix(
    column_ids: List[str],
    pair_index: Any,
) -> Tuple[Optional[np.ndarray], int]:
    """
    由字段集合与字段对表构造相关矩阵 R：对角线=1，非对角从 pair_index 取强度并按 corr_type 映射，缺失对 r=0。
    返回 (R, missing_pair_count)。若 column_ids 长度 < 2，返回 (None, 0)。
    """
    k = len(column_ids)
    if k < 2:
        return None, 0
    R = np.eye(k, dtype=float)
    missing = 0
    for i in range(k):
        for j in range(i + 1, k):
            strength = pair_index.get_strength(column_ids[i], column_ids[j]) if pair_index else None
            corr_type = pair_index.get_corr_type(column_ids[i], column_ids[j]) if pair_index else None
            if strength is None or (isinstance(strength, float) and np.isnan(strength)):
                r = 0.0
                missing += 1
            else:
                r = _r_from_pair(float(strength), corr_type)
            R[i, j] = R[j, i] = r
    return R, missing


def smooth_matrix(R: np.ndarray, lambda_diag: float) -> np.ndarray:
    """R' = (1-λ)R + λI，数值稳健。"""
    lam = max(0.0, min(1.0, float(lambda_diag)))
    return (1.0 - lam) * R + lam * np.eye(R.shape[0], dtype=float)


def condition_number(R: np.ndarray) -> float:
    """条件数 kappa = σ_max / σ_min，用奇异值计算。σ_min 过小则返回大数。"""
    try:
        s = np.linalg.svd(R, compute_uv=False)
        if s.size == 0:
            return float("inf")
        s_min = float(s[-1])
        s_max = float(s[0])
        if s_min <= 0 or not np.isfinite(s_min):
            return 1e10
        return s_max / s_min
    except Exception:
        return 1e10


def compute_kappa_stability(
    column_ids: List[str],
    pair_index: Optional[Any],
    lambda_diag: float = 0.02,
) -> Tuple[Optional[float], int]:
    """
    对给定字段集合计算条件数 kappa 与缺失对数。
    返回 (kappa, missing_pair_count)。若 pair_index 为 None 或 len(column_ids)<2，返回 (None, 0)。
    """
    if pair_index is None or len(column_ids) < 2:
        return None, 0
    R, missing = build_correlation_matrix(column_ids, pair_index)
    if R is None:
        return None, missing
    R_smooth = smooth_matrix(R, lambda_diag)
    kappa = condition_number(R_smooth)
    return kappa, missing


def kappa_to_stability_score(kappa: float) -> float:
    """stability_score = 1 / (1 + log(kappa))，kappa>=1。"""
    if kappa < 1.0:
        kappa = 1.0
    return 1.0 / (1.0 + math.log(kappa))


def kappa_penalty(kappa: float, alpha: float, log_threshold: float = 10.0) -> float:
    """penalty = alpha * max(0, log(kappa) - log(threshold))。"""
    if kappa < 1.0:
        kappa = 1.0
    return alpha * max(0.0, math.log(kappa) - math.log(log_threshold))
