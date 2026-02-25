"""
CLI入口模块

提供命令行接口，用于串联配置、加载和差异计算流程，输出结果到CSV文件。
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Any

import pandas as pd
import numpy as np

from stats_config import load_config_from_args, load_config_from_excel_paths, OverallConfig
from stats_loader import (
    load_numeric_stats,
    load_categorical_stats,
    load_numeric_stats_from_excel,
    load_categorical_stats_from_excel,
)
from diff_numeric import compute_numeric_diffs
from diff_categorical import compute_categorical_diffs
from threshold_numeric import recommend_numeric_thresholds
from threshold_categorical import recommend_categorical_thresholds

# 导出用：技术列名 -> 业务可读列名（无需数据字典即可理解）
NUMERIC_DIFF_HEADER_CN = {
    'column_id': '字段ID',
    'column_name': '字段名称',
    'stat_date': '统计月份',
    'mean_full': '全量均值',
    'mean_base': '对比客群均值',
    'mean_diff': '均值差异（对比−全量）',
    'mean_diff_ratio': '相对差异',
    'effect_size': '效应量',
    'delta_median': '中位数差异',
    'delta_p95': 'P95分位数差异',
    'delta_IQR': '四分位距差异',
    'delta_CV': '变异系数差异',
    'diff_score': '综合差异分数',
    'is_significant': '是否显著差异',
    'distribution_type': '分布类型',
    'approx_skew': '近似偏度',
    'tail_ratio': '尾部厚度',
    'cv': '变异系数',
    'rec_low': '推荐区间下界',
    'rec_high': '推荐区间上界',
    'direction': '推荐方向',
    'cohort_coverage_est': '对比客群估算覆盖率',
    'full_coverage_est': '全量估算覆盖率',
    'lift_est': '估算Lift',
    'rule_desc': '规则描述',
    'rule_reason': '推荐理由',
    'has_recommendation': '是否给出阈值推荐',
}
CATEGORICAL_DIFF_HEADER_CN = {
    'column_id': '字段ID',
    'column_name': '字段名称',
    'stat_date': '统计月份',
    'sum_abs_diff': '占比差异绝对值之和',
    'max_abs_diff': '最大单类占比差异',
    'top_diff_categories': '差异最大类别及差值',
    'recommended_categories': '推荐类别（圈定>全量）',
    'entropy_diff': '熵差异',
    'gini_diff': '基尼系数差异',
    'diff_score': '综合差异分数',
    'rec_categories': '推荐类别集合',
    'cohort_coverage': '对比客群命中占比',
    'full_coverage': '全量命中占比',
    'lift': '覆盖增幅',
    'full_hit_count': '全量命中人数',
    'cohort_hit_count': '对比客群命中人数',
    'rec_category_count': '推荐类别个数',
    'rule_desc': '规则描述',
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数：解析命令行参数并执行差异计算流程"""
    parser = argparse.ArgumentParser(
        description='计算全量客群vs圈定客群的特征差异度',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--full-dir',
        type=str,
        default='../data/full_stats',
        help='全量客群目录：CSV 时为 CSV 文件目录；Excel 时为含 st_ana_*_all_{stat_date}.xlsx 的目录（默认: ../data/full_stats）'
    )
    
    parser.add_argument(
        '--cohort-dir',
        type=str,
        default='../data/cohort_stats',
        help='对比客群目录：CSV 时为 CSV 文件目录；Excel 时为含 st_ana_*_sub_{stat_date}.xlsx 的目录（默认: ../data/cohort_stats）'
    )
    
    parser.add_argument(
        '--stat-date',
        type=str,
        default=None,
        help='统计日期，格式为YYYYMM，如202512（使用 Excel 输入时必填；CSV 时也必填）'
    )
    
    parser.add_argument(
        '--cohort-name',
        type=str,
        default=None,
        help='圈定客群名称/ID，如PRODUCT_A（CSV 模式必填；Excel 模式默认 sub）'
    )
    
    parser.add_argument(
        '--input-format',
        type=str,
        choices=['csv', 'excel'],
        default='excel',
        help='输入格式：csv 为 full-dir/cohort-dir 下 CSV 文件；excel 为 full-dir 下 all、cohort-dir 下 sub 的 xlsx（默认: excel）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/stage1_output',
        help='输出目录（默认: ../data/stage1_output）'
    )
    
    parser.add_argument(
        '--top-k-numeric',
        type=int,
        default=50,
        help='连续特征阈值推荐TopK（默认50）'
    )
    
    parser.add_argument(
        '--top-k-categorical',
        type=int,
        default=30,
        help='离散特征阈值推荐TopK（默认30）'
    )
    
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=None,
        help='正态分布时目标覆盖比例，用于反推 k=Φ^{-1}(1−target_ratio)（默认不设则用候选区间+覆盖率/Lift）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='统一配置文件路径（JSON，含 stage1/stage2）。不指定时尝试使用项目根目录 config.json'
    )
    
    args = parser.parse_args()
    
    stat_date = args.stat_date
    cohort_name = args.cohort_name
    if args.input_format == 'excel':
        cohort_name = cohort_name or 'sub'
        if not stat_date:
            parser.error('--stat-date 必填（Excel 模式下用于定位文件名，如 202512）')
    else:
        if not stat_date or not cohort_name:
            parser.error('CSV 模式下 --stat-date 与 --cohort-name 必填')
    
    # 加载统一配置（可选）：从配置文件读取 stage1 参数，命令行参数优先覆盖
    stage1_config = None
    config_path = args.config or str(Path(__file__).resolve().parent.parent / "config.json")
    if Path(config_path).exists():
        try:
            _root = Path(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from config import load_config
            app_config = load_config(config_path)
            stage1_config = app_config.stage1
            logger.info(f"已从配置文件加载 Stage1 参数: {config_path}")
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用命令行参数: {e}")
    
    try:
        run_diff_analysis(
            full_dir=args.full_dir,
            cohort_dir=args.cohort_dir,
            stat_date=stat_date,
            cohort_name=cohort_name,
            output_dir=args.output_dir,
            top_k_numeric=args.top_k_numeric,
            top_k_categorical=args.top_k_categorical,
            input_format=args.input_format,
            target_ratio=args.target_ratio,
            stage1_config=stage1_config,
        )
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


def run_diff_analysis(
    full_dir: str,
    cohort_dir: str,
    stat_date: str,
    cohort_name: str,
    output_dir: str = '../data/stage1_output',
    top_k_numeric: int = 50,
    top_k_categorical: int = 30,
    input_format: str = 'excel',
    target_ratio: Optional[float] = None,
    stage1_config: Optional[Any] = None,
) -> None:
    """
    执行差异分析流程
    
    Args:
        full_dir: 全量客群CSV文件所在目录
        cohort_dir: 圈定客群CSV文件所在目录
        stat_date: 统计日期，格式为YYYYMM
        cohort_name: 圈定客群名称/ID
        output_dir: 输出目录
        top_k_numeric: 连续特征阈值推荐TopK（默认50）
        top_k_categorical: 离散特征阈值推荐TopK（默认30）
        input_format: 输入格式 'csv' 或 'excel'；excel 时全量用 full_dir、对比用 cohort_dir 下的 xlsx
        target_ratio: 正态分布时目标覆盖比例，用于反推 k=Φ^{-1}(1−target_ratio)（可选）
        stage1_config: 从 config 加载的 Stage1Config，若提供则覆盖上述部分参数
    """
    if stage1_config is not None:
        top_k_numeric = stage1_config.top_k_numeric
        top_k_categorical = stage1_config.top_k_categorical
        if stage1_config.target_ratio is not None:
            target_ratio = stage1_config.target_ratio
    use_excel = (input_format == 'excel')
    
    logger.info("=" * 60)
    logger.info("开始特征差异度计算")
    logger.info("=" * 60)
    logger.info(f"输入格式: {input_format}")
    logger.info(f"全量目录: {full_dir}")
    logger.info(f"对比目录: {cohort_dir}")
    logger.info(f"统计日期: {stat_date}")
    logger.info(f"圈定客群: {cohort_name}")
    
    # 1. 构建配置
    logger.info("\n[1/5] 构建配置...")
    try:
        if use_excel:
            full_base = Path(full_dir)
            cohort_base = Path(cohort_dir)
            config = load_config_from_excel_paths(
                full_numeric_path=full_base / f"st_ana_continuous_feature_stats_all_{stat_date}.xlsx",
                full_categorical_path=full_base / f"st_ana_discrete_feature_stats_all_{stat_date}.xlsx",
                cohort_numeric_path=cohort_base / f"st_ana_continuous_feature_stats_sub_{stat_date}.xlsx",
                cohort_categorical_path=cohort_base / f"st_ana_discrete_feature_stats_sub_{stat_date}.xlsx",
                stat_date=stat_date,
                cohort_name=cohort_name,
            )
        else:
            config = load_config_from_args(
                full_base_dir=full_dir,
                cohort_base_dir=cohort_dir,
                stat_date=stat_date,
                cohort_name=cohort_name
            )
        logger.info("配置构建成功")
        logger.info(f"  全量连续: {config.full.numeric_csv_path}")
        logger.info(f"  全量离散: {config.full.categorical_csv_path}")
        logger.info(f"  圈定连续: {config.cohort.numeric_csv_path}")
        logger.info(f"  圈定离散: {config.cohort.categorical_csv_path}")
    except Exception as e:
        logger.error(f"配置构建失败: {e}")
        raise
    
    # 2. 加载统计文件（CSV 或 Excel；离散对比客群缺枚举按 0 处理）
    logger.info("\n[2/5] 加载统计文件...")
    try:
        if use_excel:
            full_numeric_df = load_numeric_stats_from_excel(config.full.numeric_csv_path)
            full_categorical_df = load_categorical_stats_from_excel(config.full.categorical_csv_path)
            cohort_numeric_df = load_numeric_stats_from_excel(config.cohort.numeric_csv_path)
            cohort_categorical_df = load_categorical_stats_from_excel(config.cohort.categorical_csv_path)
        else:
            full_numeric_df = load_numeric_stats(config.full.numeric_csv_path)
            full_categorical_df = load_categorical_stats(config.full.categorical_csv_path)
            cohort_numeric_df = load_numeric_stats(config.cohort.numeric_csv_path)
            cohort_categorical_df = load_categorical_stats(config.cohort.categorical_csv_path)

        # 补齐 key 列：确保 full/cohort 的 numeric 与 categorical 均包含 table_id, group_id, source_type, ext_info
        def _ensure_key_columns(df: pd.DataFrame) -> None:
            for col in ('table_id', 'group_id', 'source_type', 'ext_info'):
                if col not in df.columns:
                    df[col] = ''
        _ensure_key_columns(full_numeric_df)
        _ensure_key_columns(full_categorical_df)
        _ensure_key_columns(cohort_numeric_df)
        _ensure_key_columns(cohort_categorical_df)

        logger.info(f"  全量连续型: {len(full_numeric_df)} 条记录")
        logger.info(f"  全量离散型: {len(full_categorical_df)} 条记录")
        logger.info(f"  圈定连续型: {len(cohort_numeric_df)} 条记录")
        logger.info(f"  圈定离散型: {len(cohort_categorical_df)} 条记录")
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {e}")
        raise
    except ValueError as e:
        logger.error(f"格式错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载失败: {e}")
        raise
    
    # 3. 计算连续特征差异
    logger.info("\n[3/6] 计算连续特征差异...")
    try:
        numeric_diff_df = compute_numeric_diffs(full_numeric_df, cohort_numeric_df)
        logger.info(f"  计算完成，共 {len(numeric_diff_df)} 个字段")
        
        # 按diff_score降序排序（已在compute_numeric_diffs中完成）
        
        # 打印前20行
        logger.info("\n  连续特征差异Top 20:")
        print("\n" + "=" * 100)
        print("连续特征差异Top 20:")
        print("=" * 100)
        print(numeric_diff_df.head(20).to_string())
        print("=" * 100)
        
    except Exception as e:
        logger.error(f"连续特征差异计算失败: {e}")
        raise
    
    # 3.2. 基于分位数检测分布类型并合并到差异结果
    logger.info("\n[3.2/6] 基于分位数检测分布类型...")
    try:
        from analyze_distributions import detect_distribution_types_from_quantiles

        distribution_info = detect_distribution_types_from_quantiles(full_numeric_df)
        numeric_diff_df = numeric_diff_df.join(distribution_info, how='left')
        numeric_diff_df['distribution_type'] = numeric_diff_df['distribution_type'].fillna('symmetric')
        numeric_diff_df['approx_skew'] = numeric_diff_df['approx_skew'].fillna(0.0)
        numeric_diff_df['tail_ratio'] = numeric_diff_df['tail_ratio'].fillna(1.0)

        logger.info(f"  分布类型检测完成，共 {len(distribution_info)} 个字段")
        logger.info(f"  分布类型统计: {numeric_diff_df['distribution_type'].value_counts().to_dict()}")
    except Exception as e:
        logger.warning(f"分布类型检测失败: {e}，所有字段将使用默认 symmetric")
        numeric_diff_df['distribution_type'] = 'symmetric'
        numeric_diff_df['approx_skew'] = 0.0
        numeric_diff_df['tail_ratio'] = 1.0
    
    # 3.5. 推荐连续特征阈值区间
    logger.info("\n[3.5/6] 推荐连续特征阈值区间...")
    try:
        # 先过滤 TopK，只对差异度 TopN 字段做推荐
        numeric_diff_for_threshold = numeric_diff_df.head(top_k_numeric) if len(numeric_diff_df) > top_k_numeric else numeric_diff_df
        logger.info(f"  对差异度 Top {len(numeric_diff_for_threshold)} 个字段进行阈值推荐（共 {len(numeric_diff_df)} 个字段）")
        
        min_cov = stage1_config.numeric_threshold.min_cohort_coverage if stage1_config else 0.1
        min_lift = stage1_config.numeric_threshold.min_lift if stage1_config else 1.2
        max_base_cov = getattr(stage1_config.numeric_threshold, 'max_base_cov_for_numeric', 0.15) if stage1_config else 0.15
        numeric_thresholds_df = recommend_numeric_thresholds(
            numeric_diff_for_threshold, full_numeric_df, cohort_numeric_df,
            min_cohort_coverage=min_cov,
            min_lift=min_lift,
            target_ratio=target_ratio,
            max_base_cov_for_numeric=max_base_cov,
        )
        
        # 合并推荐结果到差异结果
        if len(numeric_thresholds_df) > 0:
            numeric_diff_df = numeric_diff_df.join(numeric_thresholds_df, how='left')
            logger.info(f"  阈值推荐完成，共 {len(numeric_thresholds_df)} 个字段有推荐")
            
            # 确保所有字段都有 direction（即使没有推荐）
            if 'direction' not in numeric_diff_df.columns:
                numeric_diff_df['direction'] = np.nan
            mask_no_direction = numeric_diff_df['direction'].isna()
            if mask_no_direction.any():
                # 根据 effect_size 设置默认 direction
                numeric_diff_df.loc[mask_no_direction, 'direction'] = np.where(
                    numeric_diff_df.loc[mask_no_direction, 'effect_size'] >= 0, 'high', 'low'
                )
            
            # 确保所有字段都有 rule_reason（即使没有推荐）
            if 'rule_reason' not in numeric_diff_df.columns:
                numeric_diff_df['rule_reason'] = np.nan
            mask_no_reason = numeric_diff_df['rule_reason'].isna()
            if mask_no_reason.any():
                # 根据 distribution_type 设置默认 rule_reason
                for idx in numeric_diff_df[mask_no_reason].index:
                    dist_type = numeric_diff_df.loc[idx, 'distribution_type']
                    dist_type = dist_type.iloc[0] if isinstance(dist_type, pd.Series) else dist_type
                    if pd.isna(dist_type) or dist_type == '':
                        dist_type = 'heavy_tail'
                    numeric_diff_df.loc[idx, 'rule_reason'] = f"{dist_type}: no suitable threshold found"
        else:
            logger.warning("  未生成任何阈值推荐")
            # 添加空列以保持结构一致
            numeric_diff_df['rec_low'] = np.nan
            numeric_diff_df['rec_high'] = np.nan
            # 根据 effect_size 设置默认 direction
            numeric_diff_df['direction'] = np.where(
                numeric_diff_df.get('effect_size', 0) >= 0, 'high', 'low'
            )
            numeric_diff_df['cohort_coverage_est'] = np.nan
            numeric_diff_df['full_coverage_est'] = np.nan
            numeric_diff_df['lift_est'] = np.nan
            numeric_diff_df['rule_desc'] = None
            # 根据 distribution_type 设置默认 rule_reason
            numeric_diff_df['rule_reason'] = None
            for idx in numeric_diff_df.index:
                dist_type = numeric_diff_df.loc[idx, 'distribution_type']
                dist_type = dist_type.iloc[0] if isinstance(dist_type, pd.Series) else dist_type
                if pd.isna(dist_type) or dist_type == '':
                    dist_type = 'heavy_tail'
                numeric_diff_df.loc[idx, 'rule_reason'] = f"{dist_type}: no suitable threshold found"
            numeric_diff_df['has_recommendation'] = False
        
    except Exception as e:
        logger.error(f"连续特征阈值推荐失败: {e}")
        # 即使推荐失败，也继续执行，只添加空列
        numeric_diff_df['rec_low'] = np.nan
        numeric_diff_df['rec_high'] = np.nan
        # 根据 effect_size 设置默认 direction
        numeric_diff_df['direction'] = np.where(
            numeric_diff_df.get('effect_size', 0) >= 0, 'high', 'low'
        )
        numeric_diff_df['cohort_coverage_est'] = np.nan
        numeric_diff_df['full_coverage_est'] = np.nan
        numeric_diff_df['lift_est'] = np.nan
        numeric_diff_df['rule_desc'] = None
        # 根据 distribution_type 设置默认 rule_reason
        numeric_diff_df['rule_reason'] = None
        for idx in numeric_diff_df.index:
            dist_type = numeric_diff_df.loc[idx, 'distribution_type']
            dist_type = dist_type.iloc[0] if isinstance(dist_type, pd.Series) else dist_type
            if pd.isna(dist_type) or dist_type == '':
                dist_type = 'heavy_tail'
            numeric_diff_df.loc[idx, 'rule_reason'] = f"{dist_type}: no suitable threshold found"
        numeric_diff_df['has_recommendation'] = False
    
    # 4. 计算离散特征差异
    logger.info("\n[4/6] 计算离散特征差异...")
    try:
        categorical_diff_df = compute_categorical_diffs(full_categorical_df, cohort_categorical_df)
        logger.info(f"  计算完成，共 {len(categorical_diff_df)} 个字段")
        
        # 检查是否有结果
        if len(categorical_diff_df) == 0:
            logger.warning("  未计算出任何离散特征差异，可能所有字段都处理失败")
            categorical_diff_df = pd.DataFrame()  # 创建空DataFrame
        else:
            # 按diff_score降序排序
            categorical_diff_df = categorical_diff_df.sort_values('diff_score', ascending=False)
        
        # 打印前20行
        if len(categorical_diff_df) > 0:
            logger.info("\n  离散特征差异Top 20:")
            print("\n" + "=" * 100)
            print("离散特征差异Top 20:")
            print("=" * 100)
            print(categorical_diff_df.head(20).to_string())
            print("=" * 100)
        else:
            logger.info("\n  无离散特征差异结果可显示")
            print("\n" + "=" * 100)
            print("离散特征差异: 无结果")
            print("=" * 100)
        
    except Exception as e:
        logger.error(f"离散特征差异计算失败: {e}")
        raise
    
    # 4.5. 推荐离散特征类别集合
    logger.info("\n[4.5/6] 推荐离散特征类别集合...")
    try:
        if len(categorical_diff_df) > 0:
            # 先过滤 TopK，只对差异度 TopN 字段做推荐
            categorical_diff_for_threshold = categorical_diff_df.head(top_k_categorical) if len(categorical_diff_df) > top_k_categorical else categorical_diff_df
            logger.info(f"  对差异度 Top {len(categorical_diff_for_threshold)} 个字段进行类别集合推荐（共 {len(categorical_diff_df)} 个字段）")
            
            min_delta = stage1_config.categorical_threshold.min_delta if stage1_config else 0.01
            min_cov_cat = stage1_config.categorical_threshold.min_cov if stage1_config else 0.1
            min_inc = stage1_config.categorical_threshold.min_increment if stage1_config else 0.01
            categorical_thresholds_df = recommend_categorical_thresholds(
                categorical_diff_for_threshold, full_categorical_df, cohort_categorical_df,
                min_delta=min_delta,
                min_cov=min_cov_cat,
                min_increment=min_inc,
            )
            
            # 合并推荐结果到差异结果
            if len(categorical_thresholds_df) > 0:
                categorical_diff_df = categorical_diff_df.join(categorical_thresholds_df, how='left')
                logger.info(f"  类别集合推荐完成，共 {len(categorical_thresholds_df)} 个字段有推荐")
            else:
                logger.warning("  未生成任何类别集合推荐")
                # 添加空列以保持结构一致
                categorical_diff_df['rec_categories'] = None
                categorical_diff_df['cohort_coverage'] = np.nan
                categorical_diff_df['full_coverage'] = np.nan
                categorical_diff_df['lift'] = np.nan
                categorical_diff_df['full_hit_count'] = np.nan
                categorical_diff_df['cohort_hit_count'] = np.nan
                categorical_diff_df['rec_category_count'] = 0
                categorical_diff_df['rule_desc'] = None
        else:
            logger.warning("  离散特征差异结果为空，跳过类别集合推荐")
            categorical_diff_df['rec_categories'] = None
            categorical_diff_df['cohort_coverage'] = np.nan
            categorical_diff_df['full_coverage'] = np.nan
            categorical_diff_df['lift'] = np.nan
            categorical_diff_df['full_hit_count'] = np.nan
            categorical_diff_df['cohort_hit_count'] = np.nan
            categorical_diff_df['rec_category_count'] = 0
            categorical_diff_df['rule_desc'] = None
        
    except Exception as e:
        logger.error(f"离散特征类别集合推荐失败: {e}")
        # 即使推荐失败，也继续执行，只添加空列
        if len(categorical_diff_df) > 0:
            categorical_diff_df['rec_categories'] = None
            categorical_diff_df['cohort_coverage'] = np.nan
            categorical_diff_df['full_coverage'] = np.nan
            categorical_diff_df['lift'] = np.nan
            categorical_diff_df['full_hit_count'] = np.nan
            categorical_diff_df['cohort_hit_count'] = np.nan
            categorical_diff_df['rec_category_count'] = 0
            categorical_diff_df['rule_desc'] = None
    
    # 5. 保存结果
    logger.info("\n[5/6] 保存结果...")
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存连续特征差异结果（先写临时文件，再替换，避免目标被占用时丢失结果）
        numeric_output_file = output_path / f"numeric_diff_{cohort_name}_{stat_date}.csv"
        numeric_tmp_file = output_path / f"numeric_diff_{cohort_name}_{stat_date}.csv.tmp"
        numeric_export = numeric_diff_df.copy()
        numeric_export.columns = [NUMERIC_DIFF_HEADER_CN.get(c, c) for c in numeric_export.columns]
        numeric_export.index.name = NUMERIC_DIFF_HEADER_CN.get(numeric_export.index.name, numeric_export.index.name or '字段ID')
        numeric_export.to_csv(numeric_tmp_file, encoding='utf-8-sig', index=True)
        try:
            numeric_tmp_file.replace(numeric_output_file)
            logger.info(f"  连续特征差异结果已保存: {numeric_output_file}")
        except PermissionError:
            logger.warning(
                f"目标文件被占用，无法替换: {numeric_output_file}\n"
                f"结果已写入: {numeric_tmp_file}\n"
                f"请关闭 Excel 等程序后，将 .tmp 文件重命名为 .csv 或重新运行。"
            )
        
        # 保存离散特征差异结果（先写临时文件，再替换）
        categorical_output_file = output_path / f"categorical_diff_{cohort_name}_{stat_date}.csv"
        categorical_tmp_file = output_path / f"categorical_diff_{cohort_name}_{stat_date}.csv.tmp"
        categorical_export = categorical_diff_df.copy()
        categorical_export.columns = [CATEGORICAL_DIFF_HEADER_CN.get(c, c) for c in categorical_export.columns]
        categorical_export.index.name = CATEGORICAL_DIFF_HEADER_CN.get(categorical_export.index.name, categorical_export.index.name or '字段ID')
        if len(categorical_diff_df) > 0:
            categorical_export.to_csv(categorical_tmp_file, encoding='utf-8-sig', index=True)
        else:
            categorical_export.to_csv(categorical_tmp_file, encoding='utf-8-sig', index=True)
            logger.warning("  离散特征差异结果为空，将创建空文件")
        try:
            categorical_tmp_file.replace(categorical_output_file)
            logger.info(f"  离散特征差异结果已保存: {categorical_output_file}")
        except PermissionError:
            logger.warning(
                f"目标文件被占用，无法替换: {categorical_output_file}\n"
                f"结果已写入: {categorical_tmp_file}\n"
                f"请关闭 Excel 等程序后，将 .tmp 文件重命名为 .csv 或重新运行。"
            )
        
    except PermissionError as e:
        logger.error(
            f"保存结果失败（权限错误）: {e}\n"
            f"可能的原因：\n"
            f"  1. 文件被其他程序打开（如Excel、文本编辑器）\n"
            f"  2. 没有写入权限\n"
            f"  3. 文件被设置为只读\n"
            f"请关闭相关程序或检查文件权限后重试。"
        )
        raise
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        raise
    
    logger.info("\n" + "=" * 60)
    logger.info("特征差异度计算完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

