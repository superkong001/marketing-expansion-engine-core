"""
阶段2配置模块

定义阶段2的配置参数和默认值
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class Stage2Config:
    """阶段2配置类
    
    包含所有可配置参数及其默认值
    """
    # 原子规则生成（保留旧参数以兼容）
    lift_min: float = 1.5
    min_delta: float = 0.02
    max_categories: int = 5
    # 候选族优先：优先用 Stage1 candidate_threshold_family / candidate_category_family
    use_candidate_family_first: bool = True
    max_atomic_per_column: int = 5  # 每字段最多原子条数（候选族时）
    fallback_to_legacy_rec: bool = True  # 无候选族时回退 rec_low/rec_high + 分位尾
    
    # 规则组合搜索
    max_features_per_segment: int = 3
    top_k_features: int = 30
    beam_size: int = 20
    coverage_discount_alpha: float = 0.8  # 保留但不再使用
    
    # 客群评分（保留旧参数以兼容）
    w1: float = 0.4  # lift权重（旧版）
    w2: float = 0.4  # coverage权重（旧版）
    w3: float = 0.2  # stability权重（旧版）
    full_cov_min: float = 0.01  # 保留但不再使用
    cohort_cov_min: float = 0.1  # 保留但不再使用
    lift_min_filter: float = 1.5  # 保留但不再使用
    top_n_candidates: int = 30
    
    # 多客群组合
    max_segments: int = 5
    overlap_max: float = 0.6
    different_field_overlap: float = 0.3
    
    # ========== 新增：多尺度统计差异评分权重 ==========
    w_effect_size: float = 0.4
    w_delta_mean: float = 0.3
    w_quantile_shift: float = 0.2
    w_significance: float = 0.1
    
    # ========== 新增：规则稳定性模型权重 ==========
    stability_w_std: float = 0.4
    stability_w_kurtosis: float = 0.3
    stability_w_sample: float = 0.3
    
    # ========== 新增：Coverage-free Beam Search权重 ==========
    beam_w_divergence: float = 0.5
    beam_w_stability: float = 0.3
    beam_w_diversity: float = 0.2
    
    # ========== 新增：多客群差异最大化 ==========
    similarity_threshold: float = 0.85  # 结构相似度阈值；仅“非常相似”才剔除，使预选 ≥3（0.72 易致预选仅 1 条）
    similarity_fallback_thresholds: Tuple[float, ...] = (0.85, 0.72, 0.5)  # 预选不足时依次放宽的阈值列表
    max_same_structure_in_recommended: int = 2  # recommended 中允许同一结构最多保留条数，>1 可增加客群数以提升 ROC
    
    # ========== 新增：稳定性评分阈值 ==========
    min_stability_score: float = 0.3  # 最小稳定性得分，低于此值的规则将被排除
    min_stability_score_categorical: Optional[float] = 0.15  # 离散规则单独阈值，None 则用 min_stability_score；0.15 放宽以扩大离散原子池
    
    # ========== 新增：差异评分阈值 ==========
    min_divergence_score: float = 0.1  # 最小差异评分，低于此值的特征将被排除
    
    # ========== 最终输出约束：每客群最少规则数 ==========
    min_rules_per_segment: int = 3  # 最终导出每个客群至少 3 条原子规则（k=3）

    # ========== 精度收紧（更平衡）：原子规则过滤 ==========
    enable_precision_filter: bool = True  # 是否启用原子规则精度过滤
    min_atomic_rules_for_search: int = 30  # 过滤后至少保留条数，不足则触发降级
    expected_cohort_ratio: Optional[float] = None  # 先验 sub 占比 prior_pi；缺失时不做了 precision_proxy 硬过滤，建议 config.json 由业务配置
    pi_reference: Optional[float] = None  # 可选：pi 校验参考值（如 5475/15000），与 pi_candidate 比较
    pi_tolerance: float = 1e-9  # 可选：abs(pi_candidate - pi_reference) <= 此值否则 WARN
    max_base_cov: float = 0.35  # Round1 通用 ALL 覆盖率上限（0.35 放宽以保留更多原子规则，目标 atomic_rules≥40）
    allow_main_for_beam: bool = True  # 为 True 时 main 型规则使用 main_max_base_cov 放宽进入 beam，便于产出有界区间+离散组合
    main_max_base_cov: float = 0.30  # main 型（有界区间）规则 ALL 覆盖率上限，放宽以保留 在网时长 40-203 等
    categorical_max_base_cov: float = 0.40  # 离散规则 ALL 覆盖率上限，放宽以保留 用户手机上网时间段偏好 IN(...) 等
    fallback_max_base_cov: float = 0.50  # 降级后 ALL 覆盖率上限
    min_sub_cov: float = 0.02  # sub 覆盖率下限（避免极小样本）
    min_lift_atomic: float = 2.0  # Round1 lift 下限
    fallback_min_lift_atomic: float = 1.2  # 降级后 lift 下限
    min_precision_mult: float = 3.0  # precision_proxy >= min_precision_mult * prior_pi（仅当 expected_cohort_ratio 存在时生效）

    # ========== 精度收紧（更平衡）：组合规则过滤 ==========
    max_combo_cov_est: float = 0.03  # 组合 ALL 覆盖率（独立近似）上限 3%
    min_combo_precision: float = 0.2  # 组合精度下限（precision_lb >= 此值才保留候选；Fréchet 区间下界）
    require_rare_anchor: bool = True  # 组合中至少一条规则 ALL 覆盖 <= rare_anchor_base_cov
    rare_anchor_base_cov: float = 0.10  # 稀有锚点 ALL 覆盖上限 10%
    require_exact_k: bool = True  # 仅输出规则数严格为 k（=max_features_per_segment）的客群
    allow_missing_pi: bool = False  # 为 True 时允许 expected_cohort_ratio 未配置继续跑（不推荐）

    # ========== v2.0 多样性约束（regex family） ==========
    enable_diversity_family_constraint: bool = True  # fee_family≤1, recency_family≤1, identity_family≤2
    demographic_family_regex_list: List[str] = field(default_factory=lambda: ["^age$", "height", "height_cm", "weight", "waist", "BMI", "sex", "gender"])  # height/height_cm 归入 demographic，防男性代理
    max_rules_per_demographic_family: int = 1  # 0=不允许；1=最多 1 条（软约束）
    # height：禁止作 anchor（第一层）；允许作第 3 特征时施加 height_as_third_penalty
    height_prune_regex_list: List[str] = field(default_factory=lambda: ["height", "height_cm"])
    height_as_third_penalty: float = 0.8  # height 作为第 3 特征加入时 score 乘以此系数
    max_atomic_rules_per_height_column: int = 0  # 原子规则阶段 height 最多生成条数，0=不生成（前置限用，避免大量生成再剪枝）

    # ========== v2.0 三档分层阈值 ==========
    elite_combo_cov_max: float = 0.02
    elite_combo_precision_min: float = 0.35
    standard_combo_cov_max: float = 0.05
    standard_combo_precision_min: float = 0.20
    expand_combo_cov_max: float = 0.10
    expand_require_discrete: bool = True
    expand_allow_cov_unknown: bool = True
    # elite 为空时降级：放宽阈值或按 score 取 top1
    elite_fallback_precision_min: float = 0.25
    elite_fallback_cov_max: float = 0.03
    elite_fallback_use_top1_by_score: bool = True

    # ========== 字段对关联表（ST_ANA_FEAT_PAIR_ASSOC_ALL_YYYYMM）即插即用 ==========
    pair_assoc_path: Optional[str] = None
    pair_redundant_thr: float = 0.7
    pair_conflict_thr: float = 0.8
    pair_conflict_sign: int = -1
    pair_redundant_types: tuple = ('cont_cont_linear', 'cont_cont_rank')
    # 相关性剪枝（PairAssocIndex）：v2.0 统计口径 corr_mid / corr_high
    enable_pair_assoc_pruning: bool = True
    pair_assoc_min_support: int = 200
    pair_assoc_hard_thr: float = 0.85
    corr_prune_th: float = 0.85  # 冗余剪枝（兼容）：corr_strength >= 此值且 corr_type in cont_cont/disc_disc 则不同时选
    corr_mid: float = 0.5   # abs(assoc) in [corr_mid, corr_high) 则降权
    corr_high: float = 0.8  # abs(assoc) >= corr_high 则直接剪枝
    corr_prune_threshold_high: float = 0.85  # >= 直接剪枝（优先）；无则用 pair_assoc_hard_thr/corr_high
    corr_penalty_threshold_mid: float = 0.60  # >= 进入惩罚（优先）；无则用 corr_mid
    corr_penalty_weight: float = 0.2  # 中相关扣分权重（优先）；无则用 pair_assoc_soft_penalty
    pair_assoc_soft_thr: float = 0.65
    pair_assoc_soft_penalty: float = 0.15
    pair_assoc_conflict_mode: str = "off"  # "off" | "light"
    # 相关性修正与估计区间（可用的估计范围）
    assoc_inflation_alpha: float = 0.3  # 组合覆盖率相关性修正：adj_est = min(ub, max(ind_est, alpha*ub))
    combo_pairwise_alpha: float = 0.5   # pairwise lift 几何校正：log(p_adj)=log(p_ind)+alpha*avg(log(lift_ij))
    precision_range_delta: float = 0.1  # 估计准确率区间上界 = min(1.0, precision_est + delta)
    precision_ub_shrink_beta: float = 0.8  # 相关性驱动 precision_ub 收紧：shrink = clamp(1 - beta*r_max, 0.5, 1.0)
    inflation_cap: float = 2.0  # 弱下界（旧逻辑，现用 Fréchet lb）
    combo_cov_gamma: float = 1.0  # 相关性校正插值：ind*(ub/ind)**gamma（无 pair_lift 时用）
    min_cov_lb_floor: float = 1e-9  # 组合覆盖率下界最小值，避免 0 导致 lift_ub=inf、precision_ub 无定义
    precision_ub_cap: float = 0.99  # 估计准确率区间上界上限，避免写出 Infinity

    # ========== 病态度量/条件数稳定性（仅用字段对表，不依赖明细） ==========
    enable_kappa_stability: bool = True  # 是否启用条件数剪枝/惩罚
    kappa_prune: float = 100.0  # kappa > 此值则剪枝（pruned_kappa）
    kappa_penalty_alpha: float = 0.1  # 惩罚项 alpha*max(0, log(kappa)-log(10))
    lambda_diag: float = 0.02  # R'=(1-λ)R+λI 的 λ，数值稳健

    # ========== 统一 rule_score 权重（原子+组合通用） ==========
    rule_score_w_precision: float = 0.25
    rule_score_w_lift: float = 0.25
    rule_score_w_coverage: float = 0.1
    rule_score_w_divergence: float = 0.2
    rule_score_w_stability: float = 0.2
    # ========== 统一 score_final（宽进严排：precision+lift - penalties + diversity） ==========
    use_unified_scoring: bool = True
    w1_precision: float = 0.35
    w2_lift: float = 0.25
    w3_all_cov_penalty: float = 0.15
    w4_instability_penalty: float = 0.10
    w5_redundancy_penalty: float = 0.05
    w6_proxy_penalty: float = 0.05
    w7_diversity_bonus: float = 0.05
    cov_penalty_threshold: float = 0.35  # base_cov 超过此值则施加 cov_penalty
    stability_floor: float = 0.2  # 低于此值施加 instability_penalty
    redundancy_similarity_threshold: float = 0.9
    # ========== 同特征集去冗余 ==========
    same_structure_at_candidate: bool = True  # 候选阶段同特征集只保留 score_final 最高一条
    same_structure_at_portfolio: bool = True  # portfolio 推荐确定后再做一次同特征集去冗余
    # ========== 多簇选择 ==========
    cluster_by: str = ""  # 非空时启用：anchor_feature | feature_family；空则用 filter_similar_rules
    max_per_cluster_first: int = 1  # 先每簇取 top1
    min_clusters_represented: int = 2  # 至少覆盖簇数
    # ========== 输出增强 ==========
    include_anchor_feature: bool = True
    include_feature_clusters: bool = True
    include_score_breakdown: bool = True
    include_redundancy_group_id: bool = True
    include_selection_reason: bool = True
    menu_contains_all_recommended: bool = True
    # ========== 连续风险分 ==========
    risk_scoring_enabled: bool = True
    risk_tier_weights: Optional[dict] = None  # {"strong": 3, "medium": 2, "weak": 1}
    risk_use_rule_score_directly: bool = False
    # 原子级覆盖重叠去重与同特征多阈值去重
    atomic_overlap_dedup_threshold: float = 0.8  # 同 column 覆盖重叠>=此值只保留 rule_score 最优一条
    max_atomic_rules_per_column_after_dedup: int = 2  # 同特征多阈值去重后每 column 最多保留条数（1 易致候选不足）
    # Greedy portfolio + 相似惩罚
    similarity_penalty_lambda: float = 0.2  # score_effective = rule_score - lambda * max_similarity_to_selected（略降以保留更多客群）
    portfolio_use_rule_score: bool = True  # 组合选择是否用 rule_score 排序/选人

    # ========== 最终组合两阶段选择（elite 强制 + 结构距离最大化） ==========
    selection_w_dist: float = 0.6  # 剩余名额选择：结构距离权重
    selection_w_quality: float = 0.4  # 剩余名额选择：综合得分权重
    max_segments_per_anchor: int = 2  # 同一锚点在推荐集中最多入选条数
    min_family_coverage: int = 2  # 推荐集至少覆盖的 feature_family 数量（软约束：新 family 候选加分）
    selection_min_score: Optional[float] = None  # 综合得分下限，None 则不做硬过滤
    selection_overlap_shared_fields_max: int = 1  # 共享字段数 > 此值视为高重叠（≥2 即高重叠）
    selection_jaccard_dist_min: float = 0.5  # Jaccard 距离低于此值视为高重叠，降权或剔除
    # ========== 区间上界收紧：pair 缺失惩罚 ==========
    cov_shrunk_missing_penalty: float = 0.8  # 已废弃：缺失对不再参与收紧，仅统计；保留配置兼容
    missing_pair_penalty_lambda: float = 0.1  # score 惩罚：score_penalty = min(lambda * missing_pair_ratio, cap)
    missing_pair_penalty_cap: float = 0.05  # 惩罚上限，温和惩罚
    recommended_max_missing_pair_count: int = 1  # recommended 仅允许 missing_pair_count 不超过此值；> 此值的仅进 menu 且标 low_confidence

    # ========== 输出目标约束（菜单模式：按准确率排序 + 条数限制，默认不做 target 硬过滤） ==========
    target_precision_min: float = 0.0  # 期望准确率下限；0 表示不过滤，菜单模式由业务自选
    target_coverage_min: float = 0.0  # 期望覆盖率下限；0 表示不过滤
    target_segment_count: Optional[int] = None  # 预期输出客群数，None 时使用 max_segments
    target_user_count_10k: Optional[float] = None  # 预期圈定用户数（万人），可选
    use_f1_balance: bool = True  # 是否用 F1 参与排序
    target_priority: str = "f1"  # 优先级: "precision_first" | "coverage_first" | "f1"
    # 菜单输出：全量人数与导出条数
    total_population: Optional[int] = None  # 全量用户数 N_total，用于计算每条 segment 的估计用户数
    cohort_size: Optional[int] = None  # 圈定人数 N_sub，用于 N_sub_est/N_all_est 等
    output_max_segments: Optional[int] = None  # 导出时最多保留条数（按准确率排序后 top N）；None 时用 max_segments
    # 名单规模保底上限（安全阀，避免营销误触过大）
    max_estimated_users: Optional[int] = None  # 单条组合估计用户数上限，如 50000；需 total_population 时生效
    max_combo_cov_est_menu: Optional[float] = None  # 未配置 total_population 时用比例上限兜底，如 0.10
    min_estimated_users: Optional[int] = None  # 单条组合估计用户数下限，过小样本不稳定时可设，如 200

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Stage2Config':
        """从字典创建配置对象；旧配置缺少新字段不报错；pair_redundant_types 若为 list 会转为 tuple；支持嵌套 atomic_rules。"""
        d = dict(config_dict)
        if isinstance(d.get('atomic_rules'), dict):
            for k, v in d['atomic_rules'].items():
                if k not in d:
                    d[k] = v
        if isinstance(d.get('unified_scoring'), dict):
            for k, v in d['unified_scoring'].items():
                if k not in d:
                    d[k] = v
        if isinstance(d.get('dedup'), dict):
            for k, v in d['dedup'].items():
                if k not in d:
                    d[k] = v
        if isinstance(d.get('clustering'), dict):
            for k, v in d['clustering'].items():
                if k not in d:
                    d[k] = v
        if isinstance(d.get('output'), dict):
            for k, v in d['output'].items():
                if k not in d:
                    d[k] = v
        if isinstance(d.get('risk_scoring'), dict):
            risk_map = {'enabled': 'risk_scoring_enabled', 'tier_weights': 'risk_tier_weights', 'use_rule_score_directly': 'risk_use_rule_score_directly'}
            for k, v in d['risk_scoring'].items():
                key = risk_map.get(k, k)
                if key not in d:
                    d[key] = v
        filtered = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if 'pair_redundant_types' in filtered and isinstance(filtered['pair_redundant_types'], list):
            filtered['pair_redundant_types'] = tuple(filtered['pair_redundant_types'])
        if 'similarity_fallback_thresholds' in filtered and isinstance(filtered['similarity_fallback_thresholds'], list):
            filtered['similarity_fallback_thresholds'] = tuple(filtered['similarity_fallback_thresholds'])
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

