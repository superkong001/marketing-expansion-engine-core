"""
统一配置模块

从 config.json 加载 Stage1 / Stage2 的可配置参数与功能开关。
用户可通过修改项目根目录下的 config.json 调整参数，无需改代码。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

# 默认配置文件路径（相对项目根或当前工作目录）
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


@dataclass
class Stage1DistributionConfig:
    """分布类型检测参数（analyze_distributions）"""
    skew_eps: float = 0.2
    tail_thr: float = 0.5
    eps: float = 1e-9


@dataclass
class Stage1NumericThresholdConfig:
    """连续特征阈值推荐参数（threshold_numeric，仅作 debug 用，不参与决策）"""
    min_cohort_coverage: float = 0.1
    min_lift: float = 1.2
    max_base_cov_for_numeric: float = 0.15  # symmetric 尾部阈值满足的全量覆盖率上限，供 Stage2 多阈值


@dataclass
class Stage1CategoricalThresholdConfig:
    """离散特征类别推荐参数（threshold_categorical）"""
    min_delta: float = 0.01
    min_cov: float = 0.1
    min_increment: float = 0.01


@dataclass
class Stage1DiffNumericConfig:
    """连续特征差异计算参数（diff_numeric）"""
    effect_size_significant_threshold: float = 0.2
    w_effect_size: float = 0.5
    w_delta_median: float = 0.3
    w_delta_p95: float = 0.2


@dataclass
class Stage1DiffCategoricalConfig:
    """离散特征差异计算参数（diff_categorical）"""
    top_k: int = 3


@dataclass
class Stage1Config:
    """Stage1 可配置参数与功能开关"""
    # 阈值推荐 TopK
    top_k_numeric: int = 50
    top_k_categorical: int = 30
    # 正态分布时目标覆盖比例（可选，用于反推 k）
    target_ratio: Optional[float] = None
    # 分布检测
    distribution: Stage1DistributionConfig = field(default_factory=Stage1DistributionConfig)
    # 连续/离散阈值推荐
    numeric_threshold: Stage1NumericThresholdConfig = field(default_factory=Stage1NumericThresholdConfig)
    categorical_threshold: Stage1CategoricalThresholdConfig = field(default_factory=Stage1CategoricalThresholdConfig)
    # 差异计算
    diff_numeric: Stage1DiffNumericConfig = field(default_factory=Stage1DiffNumericConfig)
    diff_categorical: Stage1DiffCategoricalConfig = field(default_factory=Stage1DiffCategoricalConfig)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Stage1Config":
        if not d:
            return cls()
        top_k_numeric = d.get("top_k_numeric", 50)
        top_k_categorical = d.get("top_k_categorical", 30)
        target_ratio = d.get("target_ratio")
        dist = d.get("distribution", {})
        num_thr = d.get("numeric_threshold", {})
        cat_thr = d.get("categorical_threshold", {})
        diff_num = d.get("diff_numeric", {})
        diff_cat = d.get("diff_categorical", {})
        return cls(
            top_k_numeric=int(top_k_numeric),
            top_k_categorical=int(top_k_categorical),
            target_ratio=float(target_ratio) if target_ratio is not None else None,
            distribution=Stage1DistributionConfig(
                skew_eps=float(dist.get("skew_eps", 0.2)),
                tail_thr=float(dist.get("tail_thr", 0.5)),
                eps=float(dist.get("eps", 1e-9)),
            ),
            numeric_threshold=Stage1NumericThresholdConfig(
                min_cohort_coverage=float(num_thr.get("min_cohort_coverage", 0.1)),
                min_lift=float(num_thr.get("min_lift", 1.2)),
                max_base_cov_for_numeric=float(num_thr.get("max_base_cov_for_numeric", 0.15)),
            ),
            categorical_threshold=Stage1CategoricalThresholdConfig(
                min_delta=float(cat_thr.get("min_delta", 0.01)),
                min_cov=float(cat_thr.get("min_cov", 0.1)),
                min_increment=float(cat_thr.get("min_increment", 0.01)),
            ),
            diff_numeric=Stage1DiffNumericConfig(
                effect_size_significant_threshold=float(diff_num.get("effect_size_significant_threshold", 0.2)),
                w_effect_size=float(diff_num.get("w_effect_size", 0.5)),
                w_delta_median=float(diff_num.get("w_delta_median", 0.3)),
                w_delta_p95=float(diff_num.get("w_delta_p95", 0.2)),
            ),
            diff_categorical=Stage1DiffCategoricalConfig(
                top_k=int(diff_cat.get("top_k", 3)),
            ),
        )


@dataclass
class AppConfig:
    """统一应用配置：包含 Stage1 与 Stage2 配置"""
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Dict[str, Any] = field(default_factory=dict)

    def get_stage2_dict(self) -> Dict[str, Any]:
        """返回 Stage2 配置字典，供 Stage2Config.from_dict 使用"""
        return self.stage2


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    从 JSON 文件加载统一配置。

    配置文件结构示例：
    {
      "stage1": {
        "top_k_numeric": 50,
        "top_k_categorical": 30,
        "target_ratio": null,
        "distribution": { "skew_eps": 0.2, "tail_thr": 0.5, "eps": 1e-9 },
        "numeric_threshold": { "min_cohort_coverage": 0.1, "min_lift": 1.2 },
        "categorical_threshold": { "min_delta": 0.01, "min_cov": 0.1, "min_increment": 0.01 },
        "diff_numeric": { "effect_size_significant_threshold": 0.2, ... },
        "diff_categorical": { "top_k": 3 }
      },
      "stage2": {
        "beam_size": 10,
        "max_features_per_segment": 3,
        ...
      }
    }

    若文件仅包含 stage2 的键（无 "stage1"），则整份 JSON 作为 stage2 配置（兼容旧版）。

    Args:
        config_path: 配置文件路径；若为 None，则尝试 DEFAULT_CONFIG_PATH。

    Returns:
        AppConfig，包含 stage1 与 stage2 配置。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return AppConfig(stage1=Stage1Config(), stage2={})

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"加载配置文件失败 {path}: {e}，使用默认配置")
        return AppConfig(stage1=Stage1Config(), stage2={})

    if not isinstance(data, dict):
        return AppConfig(stage1=Stage1Config(), stage2={})

    # 兼容：若顶层没有 stage1/stage2，则整份作为 stage2
    if "stage1" in data:
        stage1 = Stage1Config.from_dict(data["stage1"])
    else:
        stage1 = Stage1Config()

    if "stage2" in data:
        stage2 = data["stage2"] if isinstance(data["stage2"], dict) else {}
    else:
        stage2 = data  # 整份作为 stage2

    return AppConfig(stage1=stage1, stage2=stage2)
