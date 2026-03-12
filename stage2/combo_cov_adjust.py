"""
组合覆盖率与精度：Fréchet-Hoeffding 边界 + 相关性校正估计 + 准确率区间

- 严格范围：combo_non_sub_cov_lb/ub (P(rule|non-sub)), combo_sub_cov_lb/ub (P(rule|sub)), combo_all_cov_lb/ub (P(rule|all))
- 独立性点估计：ind_est = Π cov_i
- 相关性校正：有 pair_lift 时用 log(p_adj)=log(p_ind)+α*avg(log(lift_ij)) 几何校正；无则用 corr 插值或 None
- 准确率区间：precision_lb/ub 由 lift 与 pi 的 Bayes 公式计算
- cont-cont 对上界收紧：高斯 copula 近似 P(X满足∧Y满足)，用 corr_strength/corr_sign 作 ρ。
"""
from typing import List, Optional, Tuple, Any
import numpy as np

_EPS = 1e-12
_COV_CLAMP = 1e-6  # 避免 ppf(0)/ppf(1) 为 inf
_RHO_CLAMP = 0.9999


def frechet_bounds(covs: List[float]) -> Tuple[float, float]:
    """Fréchet-Hoeffding: lb = max(0, sum(cov_i) - (k-1)), ub = min(cov_i)."""
    if not covs:
        return 0.0, 0.0
    k = len(covs)
    s = sum(covs)
    lb = max(0.0, s - (k - 1))
    ub = min(covs)
    return lb, ub


def _bivariate_normal_survival(a: float, b: float, rho: float) -> float:
    """
    P(Z1 >= a, Z2 >= b) 其中 (Z1,Z2) 为标准二维正态，相关系数 rho。
    公式：1 - Phi(a) - Phi(b) + Phi_2(a, b; rho)。依赖 scipy。
    """
    from scipy.stats import norm, multivariate_normal
    rho = max(-_RHO_CLAMP, min(_RHO_CLAMP, float(rho)))
    phi_a = norm.cdf(a)
    phi_b = norm.cdf(b)
    cov = [[1.0, rho], [rho, 1.0]]
    phi2_ab = multivariate_normal.cdf([a, b], mean=[0.0, 0.0], cov=cov)
    surv = 1.0 - phi_a - phi_b + phi2_ab
    return float(max(0.0, min(1.0, surv)))


def _copula_joint_ub_pair(p_a: float, p_b: float, rho: float) -> Optional[float]:
    """
    连续-连续规则对：上尾覆盖率 p_a, p_b（满足区间的边际概率），
    用高斯 copula 近似联合上界 P(X满足 ∧ Y满足)。
    u_A = 1 - p_A => a = Phi^{-1}(u_A)，joint = P(Z1>=a, Z2>=b)。
    返回 clamp(joint, 0, min(p_a, p_b))。无 scipy 时退回 min(p_a, p_b)。
    """
    try:
        from scipy.stats import norm
    except ImportError:
        return min(p_a, p_b)
    p_a = max(_COV_CLAMP, min(1.0 - _COV_CLAMP, float(p_a)))
    p_b = max(_COV_CLAMP, min(1.0 - _COV_CLAMP, float(p_b)))
    rho = max(-_RHO_CLAMP, min(_RHO_CLAMP, float(rho)))
    u_a = 1.0 - p_a
    u_b = 1.0 - p_b
    a = float(norm.ppf(u_a))
    b = float(norm.ppf(u_b))
    joint = _bivariate_normal_survival(a, b, rho)
    cap = min(p_a, p_b)
    return max(0.0, min(cap, joint))


def tighten_ub_with_copula(
    non_sub_vals: List[float],
    sub_vals: List[float],
    column_ids: List[str],
    pair_index: Any,
    non_sub_ub_frechet: float,
    sub_ub_frechet: float,
) -> Tuple[float, float]:
    """
    对连续-连续规则对（字段对关联表中 pearson/corr 存在）用高斯 copula 收紧上界。
    返回 (non_sub_ub_tight, sub_ub_tight)，不超过 Fréchet 上界。
    """
    non_sub_ub = non_sub_ub_frechet
    sub_ub = sub_ub_frechet
    if pair_index is None or len(column_ids) < 2:
        return non_sub_ub, sub_ub
    k = len(non_sub_vals)
    for i in range(k):
        for j in range(i + 1, k):
            strength = pair_index.get_strength(column_ids[i], column_ids[j]) if pair_index else None
            if strength is None:
                continue
            sign = pair_index.get_sign(column_ids[i], column_ids[j]) if pair_index else None
            rho = float(strength) * (int(sign) if sign is not None else 1)
            rho = max(-_RHO_CLAMP, min(_RHO_CLAMP, rho))
            joint_non_sub = _copula_joint_ub_pair(non_sub_vals[i], non_sub_vals[j], rho)
            joint_sub = _copula_joint_ub_pair(sub_vals[i], sub_vals[j], rho)
            if joint_non_sub is not None:
                non_sub_ub = min(non_sub_ub, joint_non_sub)
            if joint_sub is not None:
                sub_ub = min(sub_ub, joint_sub)
    return non_sub_ub, sub_ub


def precision_from_lift(lift: Optional[float], pi: Optional[float]) -> Optional[float]:
    """precision = (pi * lift) / (pi * lift + (1 - pi)); pi/lift 无效则 None。"""
    if lift is None or pi is None or not (0 < pi < 1):
        return None
    denom = pi * float(lift) + (1 - pi)
    return (pi * float(lift)) / denom if denom else None


def avg_corr_for_combo(
    column_ids: List[str],
    pair_index: Any,
) -> Tuple[float, float]:
    """
    对组合内所有规则对 (i,j) 取 corr_strength 与 corr_sign（无 sign 默认 +1），
    返回 (avg_pos_corr, avg_neg_corr)。
    """
    pos_strengths = []
    neg_strengths = []
    for i in range(len(column_ids)):
        for j in range(i + 1, len(column_ids)):
            st = pair_index.get_strength(column_ids[i], column_ids[j]) if pair_index else None
            if st is None:
                continue
            sign = pair_index.get_sign(column_ids[i], column_ids[j]) if pair_index else None
            if sign is not None and sign < 0:
                neg_strengths.append(float(st))
            else:
                pos_strengths.append(float(st))
    avg_pos = float(np.mean(pos_strengths)) if pos_strengths else 0.0
    avg_neg = float(np.mean(neg_strengths)) if neg_strengths else 0.0
    return avg_pos, avg_neg


def adj_est_from_corr(
    ind: float,
    ub: float,
    lb: float,
    avg_pos_corr: float,
    avg_neg_corr: float,
    gamma: float = 1.0,
) -> float:
    """
    alpha = clamp(avg_pos_corr - avg_neg_corr, 0, 1)
    combo_non_sub_cov_adj_est = ind + alpha * (min(ub, ind*(ub/ind)**gamma) - ind)，再 clamp 到 [lb, ub]。
    若 ind<=0 或 ub<=0 则返回 clamp(ind, lb, ub)。
    """
    if ind <= 0 or ub <= 0:
        return max(lb, min(ub, ind))
    alpha = max(0.0, min(1.0, avg_pos_corr - avg_neg_corr))
    ratio = ub / ind if ind else 1.0
    if ratio <= 0:
        mid = ind
    else:
        try:
            mid = ind * (ratio ** gamma)
        except Exception:
            mid = ind
        mid = min(ub, mid)
    adj = ind + alpha * (mid - ind)
    return max(lb, min(ub, adj))


def _adj_est_from_pairwise_lift(
    ind: float,
    lb: float,
    ub: float,
    alpha: float,
    log_lifts: List[float],
) -> Optional[float]:
    """
    log(p_adj) = log(p_ind) + α * avg(log(lift_ij))；夹逼到 [lb, ub]。
    若 log_lifts 为空返回 None。lift 须 >0，log_lifts 中已为 log 值。
    """
    if not log_lifts or ind <= 0:
        return None
    import math
    avg_log = float(np.mean(log_lifts))
    try:
        log_ind = math.log(max(ind, _EPS))
        log_adj = log_ind + alpha * avg_log
        adj = math.exp(log_adj)
    except (ValueError, OverflowError):
        return None
    return max(lb, min(ub, adj))


def _avg_log_pair_lifts(
    column_ids: List[str],
    pair_index: Any,
    scope: str,
) -> Tuple[List[float], int]:
    """对组合内所有 (i,j) 取 get_pair_lift(scope)；返回 (list of log(lift)), count。仅 lift>0 取 log。"""
    log_vals = []
    for i in range(len(column_ids)):
        for j in range(i + 1, len(column_ids)):
            lift = getattr(pair_index, "get_pair_lift", None)
            if lift is None:
                continue
            val = lift(column_ids[i], column_ids[j], scope)
            if val is not None and val > 0:
                try:
                    log_vals.append(float(np.log(val)))
                except (ValueError, FloatingPointError):
                    pass
    return log_vals, len(log_vals)


def compute_combo_metrics(
    base_vals: List[float],
    sub_vals: List[float],
    column_ids: List[str],
    pi: Optional[float],
    pair_index: Optional[Any] = None,
    gamma: float = 1.0,
    pairwise_alpha: float = 0.5,
    inflation_cap: float = 2.0,
    cov_shrunk_missing_penalty: float = 0.8,
    min_cov_lb_floor: float = 1e-9,
    precision_ub_cap: float = 0.99,
) -> dict:
    """
    base_vals: P(rule_i|non-sub) 边际覆盖率；sub_vals: P(rule_i|sub)。
    返回 dict：
      non_sub_lb, non_sub_ub, sub_lb, sub_ub
      all_lb, all_ub, all_est (all = pi*sub + (1-pi)*non_sub，仅当 pi 有效时)
      ind_non_sub, ind_sub, adj_non_sub, adj_sub
      cov_est_non_sub, cov_est_sub (优先 adj，否则 ind)
      lift_lb, lift_ub, lift_est
      precision_lb, precision_ub, precision_est
      pi_missing
    """
    k = len(base_vals)
    non_sub_lb_f, non_sub_ub = frechet_bounds(base_vals)
    sub_lb_f, sub_ub = frechet_bounds(sub_vals)
    if pair_index is not None and k >= 2:
        non_sub_ub, sub_ub = tighten_ub_with_copula(
            base_vals, sub_vals, column_ids, pair_index, non_sub_ub, sub_ub
        )
    ind_non_sub = float(np.prod(base_vals))
    ind_sub = float(np.prod(sub_vals))
    if ind_non_sub > 0 and non_sub_ub > 0:
        weak_non_sub_lb = min(ind_non_sub / inflation_cap, non_sub_ub)
        non_sub_lb = max(non_sub_lb_f, weak_non_sub_lb)
    else:
        non_sub_lb = non_sub_lb_f
    if ind_sub > 0 and sub_ub > 0:
        weak_sub_lb = min(ind_sub / inflation_cap, sub_ub)
        sub_lb = max(sub_lb_f, weak_sub_lb)
    else:
        sub_lb = sub_lb_f

    # 下界 floor：当 ind>0 时禁止 lb=0，避免 lift_ub=inf 导致 precision_ub 无定义或 JSON 写出 Infinity
    if ind_non_sub > 0 and (non_sub_lb is None or non_sub_lb <= 0 or non_sub_lb < min_cov_lb_floor):
        non_sub_lb = max(min_cov_lb_floor, min(ind_non_sub / inflation_cap, non_sub_ub) if non_sub_ub > 0 else min_cov_lb_floor)
    if ind_sub > 0 and (sub_lb is None or sub_lb <= 0 or sub_lb < min_cov_lb_floor):
        sub_lb = max(min_cov_lb_floor, min(ind_sub / inflation_cap, sub_ub) if sub_ub > 0 else min_cov_lb_floor)

    lift_lb = sub_lb / non_sub_ub if non_sub_ub and non_sub_ub > 0 else None
    lift_ub = sub_ub / non_sub_lb if non_sub_lb and non_sub_lb > 0 else None

    adj_non_sub = None
    adj_sub = None
    if pair_index is not None and k >= 2:
        log_full, n_full = _avg_log_pair_lifts(column_ids, pair_index, "full")
        log_cohort, n_cohort = _avg_log_pair_lifts(column_ids, pair_index, "cohort")
        if n_full > 0:
            adj_non_sub = _adj_est_from_pairwise_lift(ind_non_sub, non_sub_lb, non_sub_ub, pairwise_alpha, log_full)
        if n_cohort > 0:
            adj_sub = _adj_est_from_pairwise_lift(ind_sub, sub_lb, sub_ub, pairwise_alpha, log_cohort)
        if adj_non_sub is None or adj_sub is None:
            avg_pos, avg_neg = avg_corr_for_combo(column_ids, pair_index)
            if adj_non_sub is None:
                adj_non_sub = adj_est_from_corr(ind_non_sub, non_sub_ub, non_sub_lb, avg_pos, avg_neg, gamma)
            if adj_sub is None:
                adj_sub = adj_est_from_corr(ind_sub, sub_ub, sub_lb, avg_pos, avg_neg, gamma)

    cov_est_non_sub = adj_non_sub if adj_non_sub is not None else ind_non_sub
    cov_est_sub = adj_sub if adj_sub is not None else ind_sub
    lift_est = cov_est_sub / cov_est_non_sub if cov_est_non_sub and cov_est_non_sub > 0 else None

    pi_missing = pi is None or not (0 < pi < 1)
    precision_lb = precision_from_lift(lift_lb, pi)
    _precision_ub_raw = precision_from_lift(lift_ub, pi) if lift_ub is not None else None
    # 避免 lift_ub 过大导致 precision_ub 接近 1 或数值不稳定；业务展示用区间上界 cap
    precision_ub = min(_precision_ub_raw, precision_ub_cap) if _precision_ub_raw is not None else None
    precision_est = precision_from_lift(lift_est, pi) if not pi_missing and lift_est is not None else None

    # all_cov = pi*sub + (1-pi)*non_sub (P(rule|all))
    if not pi_missing and pi is not None:
        all_lb = pi * sub_lb + (1 - pi) * non_sub_lb
        all_ub = pi * sub_ub + (1 - pi) * non_sub_ub
        all_est = pi * cov_est_sub + (1 - pi) * cov_est_non_sub
    else:
        all_lb = all_ub = all_est = None

    # 收紧 cov_ub：all_ub_shrunk = min(all_ub, min over pairs joint_all_ub_ij)；缺失对按 fallback_corr=0 不参与收紧，仅统计 missing_pair_count 供日志与下游温和惩罚
    all_ub_shrunk = all_ub
    sub_ub_shrunk = sub_ub
    missing_pair_count = 0
    missing_pairs_log: List[Tuple[str, str]] = []
    if k >= 2 and pair_index is not None:
        joint_all_ubs = []
        for i in range(k):
            for j in range(i + 1, k):
                strength = pair_index.get_strength(column_ids[i], column_ids[j]) if pair_index else None
                if strength is None:
                    missing_pair_count += 1
                    missing_pairs_log.append((column_ids[i], column_ids[j]))
                    continue
                sign = pair_index.get_sign(column_ids[i], column_ids[j]) if pair_index else None
                rho = float(strength) * (int(sign) if sign is not None else 1)
                rho = max(-_RHO_CLAMP, min(_RHO_CLAMP, rho))
                joint_non_sub_ij = _copula_joint_ub_pair(base_vals[i], base_vals[j], rho)
                joint_sub_ij = _copula_joint_ub_pair(sub_vals[i], sub_vals[j], rho)
                if joint_non_sub_ij is not None and joint_sub_ij is not None and not pi_missing and pi is not None:
                    joint_all_ij = pi * joint_sub_ij + (1 - pi) * joint_non_sub_ij
                    joint_all_ubs.append(joint_all_ij)
        if joint_all_ubs and all_ub is not None:
            all_ub_shrunk = min(all_ub, min(joint_all_ubs))
        # 缺失对不参与收紧：不再对 all_ub_shrunk 乘 cov_shrunk_missing_penalty，仅保留 missing_pair_count 供下游温和惩罚
        if missing_pairs_log:
            import logging
            logging.getLogger(__name__).debug(
                "PairAssoc 查不到的字段对（column_id）: %s", missing_pairs_log[:20] if len(missing_pairs_log) > 20 else missing_pairs_log
            )

    return {
        "non_sub_lb": non_sub_lb,
        "non_sub_ub": non_sub_ub,
        "sub_lb": sub_lb,
        "sub_ub": sub_ub,
        "all_lb": all_lb,
        "all_ub": all_ub,
        "all_est": all_est,
        "all_ub_shrunk": all_ub_shrunk,
        "sub_ub_shrunk": sub_ub_shrunk,
        "missing_pair_count": missing_pair_count,
        "ind_non_sub": ind_non_sub,
        "ind_sub": ind_sub,
        "adj_non_sub": adj_non_sub,
        "adj_sub": adj_sub,
        "cov_est_non_sub": cov_est_non_sub,
        "cov_est_sub": cov_est_sub,
        "lift_lb": lift_lb,
        "lift_ub": lift_ub,
        "lift_est": lift_est,
        "precision_lb": precision_lb,
        "precision_ub": precision_ub,
        "precision_est": precision_est,
        "pi_missing": pi_missing,
    }
