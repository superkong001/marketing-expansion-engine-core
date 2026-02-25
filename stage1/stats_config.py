"""
配置与文件路径管理模块

提供配置数据结构定义和工厂函数，用于管理全量客群和圈定客群的CSV文件路径。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class CohortFileConfig:
    """客群文件配置
    
    Attributes:
        stat_date: 统计日期，格式为YYYYMM，如"202510"
        cohort_name: 客群名称/ID，如"FULL_ALL"或"COHORT_产品A"
        numeric_csv_path: 连续型统计CSV文件路径
        categorical_csv_path: 离散型统计CSV文件路径
    """
    stat_date: str
    cohort_name: str
    numeric_csv_path: Union[str, Path]
    categorical_csv_path: Union[str, Path]
    
    def __post_init__(self):
        """后处理：确保路径为Path对象"""
        self.numeric_csv_path = Path(self.numeric_csv_path)
        self.categorical_csv_path = Path(self.categorical_csv_path)


@dataclass
class OverallConfig:
    """整体配置
    
    Attributes:
        full: 全量客群文件配置
        cohort: 当前圈定客群文件配置
    """
    full: CohortFileConfig
    cohort: CohortFileConfig


def load_config_from_args(
    full_base_dir: Union[str, Path],
    cohort_base_dir: Union[str, Path],
    stat_date: str,
    cohort_name: str
) -> OverallConfig:
    """
    根据命令行参数构建OverallConfig
    
    约定默认文件命名规则：
    - 全量连续：full_base_dir / f"numeric_stats_full_{stat_date}.csv"
    - 全量离散：full_base_dir / f"categorical_stats_full_{stat_date}.csv"
    - 圈定连续：cohort_base_dir / f"numeric_stats_{cohort_name}_{stat_date}.csv"
    - 圈定离散：cohort_base_dir / f"categorical_stats_{cohort_name}_{stat_date}.csv"
    
    Args:
        full_base_dir: 全量客群CSV文件所在目录
        cohort_base_dir: 圈定客群CSV文件所在目录
        stat_date: 统计日期，格式为YYYYMM，如"202510"
        cohort_name: 圈定客群名称/ID，如"PRODUCT_A"
    
    Returns:
        OverallConfig对象，包含全量和圈定客群的文件配置
    
    Example:
        >>> config = load_config_from_args(
        ...     full_base_dir="./full_stats",
        ...     cohort_base_dir="./cohort_stats",
        ...     stat_date="202510",
        ...     cohort_name="PRODUCT_A"
        ... )
    """
    # 转换为Path对象
    full_base_dir = Path(full_base_dir)
    cohort_base_dir = Path(cohort_base_dir)
    
    # 构建全量客群文件路径
    full_numeric_path = full_base_dir / f"numeric_stats_full_{stat_date}.csv"
    full_categorical_path = full_base_dir / f"categorical_stats_full_{stat_date}.csv"
    
    # 构建圈定客群文件路径
    cohort_numeric_path = cohort_base_dir / f"numeric_stats_{cohort_name}_{stat_date}.csv"
    cohort_categorical_path = cohort_base_dir / f"categorical_stats_{cohort_name}_{stat_date}.csv"
    
    # 创建配置对象
    full_config = CohortFileConfig(
        stat_date=stat_date,
        cohort_name="FULL_ALL",
        numeric_csv_path=full_numeric_path,
        categorical_csv_path=full_categorical_path
    )
    
    cohort_config = CohortFileConfig(
        stat_date=stat_date,
        cohort_name=cohort_name,
        numeric_csv_path=cohort_numeric_path,
        categorical_csv_path=cohort_categorical_path
    )
    
    return OverallConfig(
        full=full_config,
        cohort=cohort_config
    )


def load_config_from_excel_paths(
    full_numeric_path: Union[str, Path],
    full_categorical_path: Union[str, Path],
    cohort_numeric_path: Union[str, Path],
    cohort_categorical_path: Union[str, Path],
    stat_date: str,
    cohort_name: str,
) -> OverallConfig:
    """
    根据四份 Excel 文件路径构建 OverallConfig，用于全量/对比客群的连续与离散统计。
    约定：全量连续、全量离散、对比连续、对比离散；table_id 在加载时忽略；
    对比客群离散表中缺失的枚举值在差异计算中按比例 0 处理。
    """
    full_numeric_path = Path(full_numeric_path)
    full_categorical_path = Path(full_categorical_path)
    cohort_numeric_path = Path(cohort_numeric_path)
    cohort_categorical_path = Path(cohort_categorical_path)
    full_config = CohortFileConfig(
        stat_date=stat_date,
        cohort_name="FULL_ALL",
        numeric_csv_path=full_numeric_path,
        categorical_csv_path=full_categorical_path,
    )
    cohort_config = CohortFileConfig(
        stat_date=stat_date,
        cohort_name=cohort_name,
        numeric_csv_path=cohort_numeric_path,
        categorical_csv_path=cohort_categorical_path,
    )
    return OverallConfig(full=full_config, cohort=cohort_config)

