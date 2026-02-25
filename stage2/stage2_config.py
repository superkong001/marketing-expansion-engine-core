"""
阶段2配置模块

定义阶段2的配置参数和默认值
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Stage2Config:
    """阶段2配置类
    
    包含所有可配置参数及其默认值
    """
    # 原子规则生成（保留旧参数以兼容）
    lift_min: float = 1.5
    min_delta: float = 0.02
    max_categories: int = 5
    
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
    similarity_threshold: float = 0.4  # 结构相似度阈值，越低则 portfolio 中允许更多结构相近的客群
    
    # ========== 新增：稳定性评分阈值 ==========
    min_stability_score: float = 0.3  # 最小稳定性得分，低于此值的规则将被排除
    
    # ========== 新增：差异评分阈值 ==========
    min_divergence_score: float = 0.1  # 最小差异评分，低于此值的特征将被排除
    
    # ========== 最终输出约束：每客群最少规则数 ==========
    min_rules_per_segment: int = 3  # 最终导出每个客群至少 3 条原子规则（k=3）

    # ========== 精度收紧（更平衡）：原子规则过滤 ==========
    enable_precision_filter: bool = True  # 是否启用原子规则精度过滤
    min_atomic_rules_for_search: int = 30  # 过滤后至少保留条数，不足则触发降级
    expected_cohort_ratio: Optional[float] = None  # 先验 sub 占比 prior_pi；缺失时不做了 precision_proxy 硬过滤，建议 config.json 由业务配置
    max_base_cov: float = 0.20  # Round1 通用 ALL 覆盖率上限（tail 无 cov 不参与此判）
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
    min_combo_precision: float = 0.2  # 组合精度下限（combo_precision_est >= 此值才保留）
    require_rare_anchor: bool = True  # 组合中至少一条规则 ALL 覆盖 <= rare_anchor_base_cov
    rare_anchor_base_cov: float = 0.10  # 稀有锚点 ALL 覆盖上限 10%
    require_exact_k: bool = True  # 仅输出规则数严格为 k（=max_features_per_segment）的客群
    allow_missing_pi: bool = False  # 为 True 时允许 expected_cohort_ratio 未配置继续跑（不推荐）

    # ========== 字段对关联表（ST_ANA_FEAT_PAIR_ASSOC_ALL_YYYYMM）即插即用 ==========
    pair_assoc_path: Optional[str] = None
    pair_redundant_thr: float = 0.7
    pair_conflict_thr: float = 0.8
    pair_conflict_sign: int = -1
    pair_redundant_types: tuple = ('cont_cont_linear', 'cont_cont_rank')

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Stage2Config':
        """从字典创建配置对象；旧配置缺少新字段不报错；pair_redundant_types 若为 list 会转为 tuple。"""
        filtered = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        if 'pair_redundant_types' in filtered and isinstance(filtered['pair_redundant_types'], list):
            filtered['pair_redundant_types'] = tuple(filtered['pair_redundant_types'])
        return cls(**filtered)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

