"""
CSV与Excel解析模块

提供函数用于解析连续型和离散型特征统计CSV/Excel文件，统一列名格式，计算衍生字段，设置合适的索引。
Excel 与 CSV 加载后均保证存在 table_id、group_id、source_type、ext_info（缺则补默认值）；离散型对比客群中缺失的枚举值在差异计算时按比例0处理。
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Optional, Dict

# 配置日志
logger = logging.getLogger(__name__)

# Excel 列名到内部列名。支持中文表头（按下列映射）或英文表头（202510 结构为英文，与内部名一致或仅大小写如 iqr→IQR、cv→CV）。
# v2 口径：q3_cont=P25, q6_cont=P50, q9_cont=P75；p90_cont/p95_cont/p99_cont/p05_cont 为真实分位列。
EXCEL_NUMERIC_COLUMN_MAP: Dict[str, str] = {
    "统计日期": "stat_date",
    "字段ID": "column_id",
    "字段名": "column_name",
    "总数": "total_count",
    "非空数": "nonull_count",
    "非空占比": "nonull_ratio",
    "均值": "avg_val",
    "方差": "var_val",
    "q1": "q1_cont",
    "中位数": "q6_cont",
    "q2": "q2_cont",
    "q3": "q3_cont",
    "q6": "q6_cont",
    "q9": "q9_cont",
    "p05": "p05_cont",
    "p05_cont": "p05_cont",
    "p90": "p90_cont",
    "p90_cont": "p90_cont",
    "p95": "p95_cont",
    "p95_cont": "p95_cont",
    "p99": "p99_cont",
    "p99_cont": "p99_cont",
    "p01": "p01_cont",
    "p01_cont": "p01_cont",
    "p10": "p10_cont",
    "p10_cont": "p10_cont",
    "iqr": "IQR",
    "cv": "CV",
}
# 保证输出的分位/key 列（缺失填 NaN）
QUANTILE_OUTPUT_COLS = [
    "q3_cont", "q6_cont", "q9_cont",
    "p05_cont", "p90_cont", "p95_cont", "p99_cont", "p01_cont", "p10_cont",
    "IQR", "CV",
]
# 可选分位列（202510 表结构含有，统一转数值供下游或扩展使用）
OPTIONAL_QUANTILE_COLS = ["q4_cont", "q5_cont", "q7_cont", "q8_cont", "q10_cont", "q11_cont"]
KEY_COLS = ["stat_date", "table_id", "group_id", "column_id", "column_name", "source_type", "ext_info"]
# 离散型：支持中文表头（按下列映射）或英文表头（202510 结构为英文）。
EXCEL_CATEGORICAL_COLUMN_MAP: Dict[str, str] = {
    "统计日期": "stat_date",
    "字段ID": "column_id",
    "字段名": "column_name",
    "总数": "total_count",
    "枚举值": "val_enum",
    "枚举计数": "val_count",
    "枚举占比": "val_ratio",
    "排名": "val_rank",
    "唯一数": "unique_count",
    "熵": "entropy",
    "基尼": "gini",
}


def load_numeric_stats(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    加载连续型特征统计CSV文件
    
    解析规则：
    - 使用pandas.read_csv读取CSV
    - 确保必需列存在
    - 统一列名为snake_case格式
    - 计算std = sqrt(var_val)，处理负值和NaN
    - 设置index=['stat_date', 'column_id']
    - 保留ext_info列用于标识客群
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        DataFrame，index为['stat_date', 'column_id']，包含所有统计指标
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    
    Example:
        >>> df = load_numeric_stats("numeric_stats_full_202510.csv")
        >>> print(df.index.names)
        ['stat_date', 'column_id']
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    logger.info(f"开始读取连续型统计文件: {csv_path}")
    
    # 读取CSV，尝试多种编码（处理中文编码问题）
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info(f"成功使用编码 {encoding} 读取文件")
            break
        except UnicodeDecodeError:
            # 继续尝试下一个编码
            continue
        except Exception as e:
            # 其他错误直接抛出
            logger.error(f"读取CSV文件失败 (编码 {encoding}): {e}")
            raise
    
    if df is None:
        raise ValueError(
            f"无法使用任何编码读取文件: {csv_path}\n"
            f"尝试的编码: {', '.join(encodings)}\n"
            f"文件可能已损坏或使用了不支持的编码格式"
        )
    
    logger.info(f"读取成功，共 {len(df)} 行")
    
    # 必需列（v2 最小集；分位列可缺失，后续补齐 NaN）
    required_columns = [
        'stat_date', 'column_id', 'column_name', 'total_count', 'nonull_count',
        'nonull_ratio', 'avg_val', 'var_val', 'source_type', 'ext_info'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV文件缺少必需列: {missing_columns}. "
            f"现有列: {list(df.columns)}"
        )
    logger.info(f"列名: {list(df.columns)}")
    # 仅做非分位的重命名
    df = df.rename(columns={
        'nonull_count': 'nonnull_count',
        'nonull_ratio': 'nonnull_ratio',
        'avg_val': 'mean',
    })
    # v2 分位：q3_cont=P25, q6_cont=P50, q9_cont=P75；不把 q9_cont 当作 p90
    if 'q6_cont' not in df.columns and 'q2_cont' in df.columns:
        df['q6_cont'] = df['q2_cont']
    for c in QUANTILE_OUTPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # IQR 缺失时用 q9_cont - q3_cont
    if 'q9_cont' in df.columns and 'q3_cont' in df.columns:
        iqr_val = df['q9_cont'].astype(float) - df['q3_cont'].astype(float)
        if 'IQR' not in df.columns:
            df['IQR'] = iqr_val
        else:
            df['IQR'] = df['IQR'].where(df['IQR'].notna(), iqr_val)
    # 兼容下游：median/p05/p90/p95/p99/q1/q3 供 diff_numeric / threshold_numeric 使用
    if 'median' not in df.columns:
        df['median'] = df['q6_cont'] if 'q6_cont' in df.columns else np.nan
    if 'p05' not in df.columns:
        df['p05'] = df['p05_cont'] if 'p05_cont' in df.columns else np.nan
    if 'p90' not in df.columns:
        df['p90'] = df['p90_cont'] if 'p90_cont' in df.columns else np.nan
    if 'p95' not in df.columns:
        df['p95'] = df['p95_cont'] if 'p95_cont' in df.columns else np.nan
    if 'p99' not in df.columns:
        df['p99'] = df['p99_cont'] if 'p99_cont' in df.columns else np.nan
    if 'q1' not in df.columns:
        df['q1'] = df['q3_cont'] if 'q3_cont' in df.columns else np.nan
    if 'q3' not in df.columns:
        df['q3'] = df['q9_cont'] if 'q9_cont' in df.columns else np.nan
    
    # 计算std = sqrt(var_val)，处理负值和NaN
    def safe_sqrt(var_val):
        """安全计算平方根，处理负值和NaN"""
        if pd.isna(var_val) or var_val < 0:
            return np.nan
        return np.sqrt(var_val)
    
    df['std'] = df['var_val'].apply(safe_sqrt)
    
    # 增加近似标准差计算（用于偏度）：std = (avg_str_2 - avg_str_1) / 6
    # 如果 std 缺失或无效，使用近似标准差
    if 'avg_str_1' in df.columns and 'avg_str_2' in df.columns:
        def calc_approx_std(row):
            """计算近似标准差"""
            try:
                avg_str_1 = row.get('avg_str_1', np.nan)
                avg_str_2 = row.get('avg_str_2', np.nan)
                if pd.isna(avg_str_1) or pd.isna(avg_str_2):
                    return np.nan
                avg_str_1 = float(avg_str_1)
                avg_str_2 = float(avg_str_2)
                if avg_str_2 > avg_str_1:
                    return (avg_str_2 - avg_str_1) / 6.0
                return np.nan
            except:
                return np.nan
        
        df['std_approx'] = df.apply(calc_approx_std, axis=1)
        # 如果 std 缺失或无效，使用近似标准差填充
        mask_invalid_std = df['std'].isna() | (df['std'] <= 0)
        df.loc[mask_invalid_std, 'std'] = df.loc[mask_invalid_std, 'std_approx']
        # 如果 std 缺失，使用 std_approx 填充
        df['std'] = df['std'].fillna(df['std_approx'])
    
    for col in ('table_id', 'group_id'):
        if col not in df.columns:
            df[col] = ''
    if 'ext_info' not in df.columns:
        df['ext_info'] = ''
    df = df.set_index(['stat_date', 'column_id'])
    logger.info(f"解析完成，索引: {df.index.names}, 列数: {len(df.columns)}")
    return df


def load_categorical_stats(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    加载离散型特征统计CSV文件
    
    解析规则：
    - 使用pandas.read_csv读取CSV
    - 确保必需列存在
    - 设置index=['stat_date', 'column_id', 'val_enum']
    - 保留ext_info列用于标识客群
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        DataFrame，index为['stat_date', 'column_id', 'val_enum']，包含所有统计指标
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    
    Example:
        >>> df = load_categorical_stats("categorical_stats_full_202510.csv")
        >>> print(df.index.names)
        ['stat_date', 'column_id', 'val_enum']
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    logger.info(f"开始读取离散型统计文件: {csv_path}")
    
    # 读取CSV，尝试多种编码（处理中文编码问题）
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig', 'latin1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info(f"成功使用编码 {encoding} 读取文件")
            break
        except UnicodeDecodeError:
            # 继续尝试下一个编码
            continue
        except Exception as e:
            # 其他错误直接抛出
            logger.error(f"读取CSV文件失败 (编码 {encoding}): {e}")
            raise
    
    if df is None:
        raise ValueError(
            f"无法使用任何编码读取文件: {csv_path}\n"
            f"尝试的编码: {', '.join(encodings)}\n"
            f"文件可能已损坏或使用了不支持的编码格式"
        )
    
    logger.info(f"读取成功，共 {len(df)} 行")
    
    # 定义必需列
    required_columns = [
        'stat_date', 'column_id', 'column_name', 'total_count', 'val_enum',
        'val_count', 'val_ratio', 'val_rank', 'unique_count', 'entropy',
        'gini', 'source_type', 'ext_info'
    ]
    
    # 检查必需列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV文件缺少必需列: {missing_columns}. "
            f"现有列: {list(df.columns)}"
        )
    
    logger.info(f"列名: {list(df.columns)}")
    
    # 设置索引
    df = df.set_index(['stat_date', 'column_id', 'val_enum'])
    for col in ('table_id', 'group_id'):
        if col not in df.columns:
            df[col] = ''
    if 'ext_info' not in df.columns:
        logger.warning("ext_info列不存在，将设置为空字符串")
        df['ext_info'] = ''
    logger.info(f"解析完成，索引: {df.index.names}, 列数: {len(df.columns)}")
    return df


def _apply_excel_column_map(
    df: pd.DataFrame, column_map: Optional[Dict[str, str]]
) -> pd.DataFrame:
    """应用列名映射（保留 table_id 等 key 列，不再删除）。"""
    if not column_map:
        return df
    rename = {}
    for excel_name, internal_name in column_map.items():
        if internal_name not in df.columns and excel_name in df.columns:
            rename[excel_name] = internal_name
    if rename:
        df = df.rename(columns=rename)
        logger.info(f"Excel 列名映射: {rename}")
    return df


def load_numeric_stats_from_excel(
    excel_path: Union[str, Path],
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    从 Excel 加载连续型特征统计，输出与 load_numeric_stats 一致的结构。
    列名可通过 column_map 或 EXCEL_NUMERIC_COLUMN_MAP 映射到内部名；table_id、group_id、source_type、ext_info 缺则补默认值。
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"文件不存在: {excel_path}")
    logger.info(f"开始读取连续型统计 Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"读取成功，共 {len(df)} 行")
    if column_map is None:
        column_map = EXCEL_NUMERIC_COLUMN_MAP
    df = _apply_excel_column_map(df, column_map)

    required_columns = [
        "stat_date", "column_id", "column_name", "total_count", "nonull_count",
        "nonull_ratio", "avg_val", "var_val", "source_type", "ext_info",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必需列: {missing}. 现有列: {list(df.columns)}")
    for col in ("min_val", "max_val"):
        if col not in df.columns:
            logger.warning(f"Excel 缺少 {col}，threshold_numeric 与分布分析将无法使用真实边界")
    for col in ("table_id", "group_id", "source_type", "ext_info"):
        if col not in df.columns:
            df[col] = ""
    # v2 分位列：确保存在（缺失填 NaN）；q6_cont 可由 q2_cont 补
    if "q6_cont" not in df.columns and "q2_cont" in df.columns:
        df["q6_cont"] = df["q2_cont"]
    for c in QUANTILE_OUTPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # 可选 p10_cont 插值：p10 ≈ q1_cont + 0.2*(q2_cont - q1_cont)
    if "p10_cont" in df.columns and df["p10_cont"].isna().all() and "q1_cont" in df.columns and "q2_cont" in df.columns:
        df["p10_cont"] = df["q1_cont"].astype(float) + 0.2 * (df["q2_cont"].astype(float) - df["q1_cont"].astype(float))
    # p01_cont 缺失时保守用 p05_cont（可选）
    if "p01_cont" in df.columns and df["p01_cont"].isna().all() and "p05_cont" in df.columns:
        df["p01_cont"] = df["p05_cont"]

    numeric_cols = [
        "total_count", "nonull_count", "nonull_ratio", "avg_val", "var_val",
        "min_val", "max_val",
        "q1_cont", "q2_cont", "q3_cont", "q6_cont", "q9_cont",
        "p05_cont", "p90_cont", "p95_cont", "p99_cont", "p01_cont", "p10_cont",
        "IQR", "CV",
    ]
    for col in numeric_cols + OPTIONAL_QUANTILE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"nonull_count": "nonnull_count", "nonull_ratio": "nonnull_ratio", "avg_val": "mean"})
    if "q9_cont" in df.columns and "q3_cont" in df.columns:
        iqr_val = df["q9_cont"].astype(float) - df["q3_cont"].astype(float)
        if "IQR" not in df.columns:
            df["IQR"] = iqr_val
        else:
            df["IQR"] = df["IQR"].where(df["IQR"].notna(), iqr_val)
    # 兼容下游 diff_numeric / threshold_numeric
    if "median" not in df.columns:
        df["median"] = df["q6_cont"] if "q6_cont" in df.columns else np.nan
    for alias, src in [("p05", "p05_cont"), ("p90", "p90_cont"), ("p95", "p95_cont"), ("p99", "p99_cont")]:
        if alias not in df.columns and src in df.columns:
            df[alias] = df[src]
    if "q1" not in df.columns and "q3_cont" in df.columns:
        df["q1"] = df["q3_cont"]
    if "q3" not in df.columns and "q9_cont" in df.columns:
        df["q3"] = df["q9_cont"]

    def safe_sqrt(var_val):
        if pd.isna(var_val) or var_val < 0:
            return np.nan
        return np.sqrt(var_val)
    df["std"] = df["var_val"].apply(safe_sqrt)
    for col in ("avg_str_1", "avg_str_2"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "avg_str_1" in df.columns and "avg_str_2" in df.columns:
        def calc_approx_std(row):
            try:
                a1, a2 = row.get("avg_str_1", np.nan), row.get("avg_str_2", np.nan)
                if pd.isna(a1) or pd.isna(a2) or a2 <= a1:
                    return np.nan
                return (float(a2) - float(a1)) / 6.0
            except Exception:
                return np.nan
        df["std_approx"] = df.apply(calc_approx_std, axis=1)
        mask = df["std"].isna() | (df["std"] <= 0)
        df.loc[mask, "std"] = df.loc[mask, "std_approx"]
        df["std"] = df["std"].fillna(df.get("std_approx", np.nan))

    df = df.set_index(["stat_date", "column_id"])
    logger.info(f"解析完成，索引: {df.index.names}, 列数: {len(df.columns)}")
    return df


def load_categorical_stats_from_excel(
    excel_path: Union[str, Path],
    column_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    从 Excel 加载离散型特征统计，输出与 load_categorical_stats 一致的结构。
    对比客群 Excel 中若某枚举值无记录（无该行），差异计算时该枚举比例按 0 处理；table_id、group_id、source_type、ext_info 缺则补默认值。
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"文件不存在: {excel_path}")
    logger.info(f"开始读取离散型统计 Excel: {excel_path}")
    df = pd.read_excel(excel_path)
    logger.info(f"读取成功，共 {len(df)} 行")
    if column_map is None:
        column_map = EXCEL_CATEGORICAL_COLUMN_MAP
    df = _apply_excel_column_map(df, column_map)

    required_columns = [
        "stat_date", "column_id", "column_name", "total_count", "val_enum",
        "val_count", "val_ratio", "val_rank", "unique_count", "entropy",
        "gini",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必需列: {missing}. 现有列: {list(df.columns)}")
    for col in ("table_id", "group_id", "source_type", "ext_info"):
        if col not in df.columns:
            df[col] = ""

    # 离散型数值列统一转数值，与连续型风格一致，避免 Excel 读成字符串导致下游异常
    categorical_numeric_cols = [
        "total_count", "nonull_count", "val_count", "val_ratio", "val_rank",
        "unique_count", "entropy", "gini",
    ]
    for col in categorical_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index(["stat_date", "column_id", "val_enum"])
    logger.info(f"解析完成，索引: {df.index.names}, 列数: {len(df.columns)}")
    return df
