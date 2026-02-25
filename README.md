# CSV特征统计解析与差异度计算模块

## 概述

本模块用于解析上游系统（GBase SQL）生成的统计CSV文件，并计算全量客群vs圈定客群的特征差异度。

## 模块说明

### 1. stats_config.py - 配置模块
- **功能**: 管理全量和圈定客群的CSV文件路径
- **主要类**:
  - `CohortFileConfig`: 单个客群的文件配置
  - `OverallConfig`: 整体配置（包含全量和圈定）
- **主要函数**:
  - `load_config_from_args()`: 根据目录和命名约定构建配置

### 2. stats_loader.py - CSV解析模块
- **功能**: 解析连续型和离散型统计CSV文件
- **主要函数**:
  - `load_numeric_stats()`: 加载连续型特征统计CSV
  - `load_categorical_stats()`: 加载离散型特征统计CSV
- **特性**:
  - 统一列名格式（snake_case）
  - 计算衍生字段（如std = sqrt(var_val)）
  - 设置合适的索引
  - 保留ext_info列用于标识客群

### 3. diff_numeric.py - 连续特征差异度计算
- **功能**: 计算全量vs圈定的连续特征差异
- **主要函数**:
  - `compute_numeric_diffs()`: 计算连续特征差异指标
- **计算指标**:
  - mean_diff: 均值差异
  - mean_diff_ratio: 均值差异比例
  - effect_size: 效应量（Cohen's d）
  - delta_median: 中位数差异
  - delta_p95: P95分位数差异
  - delta_IQR: 四分位距差异
  - diff_score: 综合差异分数

### 4. diff_categorical.py - 离散特征差异度计算
- **功能**: 计算全量vs圈定的离散特征差异
- **主要函数**:
  - `compute_categorical_diffs()`: 计算离散特征差异指标
- **计算指标**:
  - sum_abs_diff: 绝对频率差异之和
  - max_abs_diff: 最大绝对频率差异
  - top_diff_categories: 前3个差异最大的类别
  - entropy_diff: 熵差异
  - diff_score: 综合差异分数

### 5. diff_main.py - CLI入口
- **功能**: 提供命令行接口，串联整个流程
- **主要功能**:
  - 解析命令行参数
  - 加载CSV文件
  - 计算差异
  - 输出结果到CSV文件

## 使用方法

### 命令行使用

```bash
# 基本用法
python diff_main.py \
    --full-dir ./full_stats \
    --cohort-dir ./cohort_stats \
    --stat-date 202510 \
    --cohort-name PRODUCT_A

# 指定输出目录
python diff_main.py \
    --full-dir ./full_stats \
    --cohort-dir ./cohort_stats \
    --stat-date 202510 \
    --cohort-name PRODUCT_A \
    --output-dir ./output
```

### 参数说明

- `--full-dir`: 全量客群CSV文件所在目录（默认: ./full_stats）
- `--cohort-dir`: 圈定客群CSV文件所在目录（默认: ./cohort_stats）
- `--stat-date`: 统计日期，格式为YYYYMM，如202510（必填）
- `--cohort-name`: 圈定客群名称/ID，如PRODUCT_A（必填）
- `--output-dir`: 输出目录（默认: ./output）

### 文件命名约定

系统按照以下约定查找CSV文件：

- 全量连续型: `{full_dir}/numeric_stats_full_{stat_date}.csv`
- 全量离散型: `{full_dir}/categorical_stats_full_{stat_date}.csv`
- 圈定连续型: `{cohort_dir}/numeric_stats_{cohort_name}_{stat_date}.csv`
- 圈定离散型: `{cohort_dir}/categorical_stats_{cohort_name}_{stat_date}.csv`

### 输出文件

结果会保存到输出目录：

- `output/numeric_diff_{cohort_name}_{stat_date}.csv` - 连续特征差异结果
- `output/categorical_diff_{cohort_name}_{stat_date}.csv` - 离散特征差异结果

## CSV文件格式要求

### 连续型特征统计CSV必需列

- stat_date: 统计日期
- column_id: 字段ID
- column_name: 字段名称
- total_count: 总记录数
- nonnull_count: 非空记录数
- nonnull_ratio: 非空比例
- mean: 均值
- var_val: 方差
- q1_cont: 第一四分位数
- q2_cont: 中位数
- q3_cont: 第三四分位数
- p05_cont: 5%分位数
- p90_cont: 90%分位数
- p95_cont: 95%分位数
- p99_cont: 99%分位数
- IQR: 四分位距
- CV: 变异系数
- source_type: 数据源类型（1=全量，2=圈定）
- ext_info: 客群补充信息

### 离散型特征统计CSV必需列

- stat_date: 统计日期
- column_id: 字段ID
- column_name: 字段名称
- total_count: 总记录数
- val_enum: 枚举值
- val_count: 该枚举值的数量
- val_ratio: 该枚举值的比例
- val_rank: 该枚举值的排名
- unique_count: 唯一值数量
- entropy: 熵
- gini: 基尼系数
- source_type: 数据源类型（1=全量，2=圈定）
- ext_info: 客群补充信息

## 编程接口使用

### 示例代码

```python
from pathlib import Path
from stats_config import load_config_from_args
from stats_loader import load_numeric_stats, load_categorical_stats
from diff_numeric import compute_numeric_diffs
from diff_categorical import compute_categorical_diffs

# 1. 构建配置
config = load_config_from_args(
    full_base_dir="./full_stats",
    cohort_base_dir="./cohort_stats",
    stat_date="202510",
    cohort_name="PRODUCT_A"
)

# 2. 加载CSV文件
full_numeric_df = load_numeric_stats(config.full.numeric_csv_path)
full_categorical_df = load_categorical_stats(config.full.categorical_csv_path)
cohort_numeric_df = load_numeric_stats(config.cohort.numeric_csv_path)
cohort_categorical_df = load_categorical_stats(config.cohort.categorical_csv_path)

# 3. 计算差异
numeric_diff_df = compute_numeric_diffs(full_numeric_df, cohort_numeric_df)
categorical_diff_df = compute_categorical_diffs(full_categorical_df, cohort_categorical_df)

# 4. 查看结果
print(numeric_diff_df.head(20))
print(categorical_diff_df.head(20))
```

## 依赖要求

- Python 3.9+
- pandas
- numpy
- scipy（可选）

## 注意事项

1. **文件路径**: 确保CSV文件路径正确，系统会检查文件是否存在
2. **列名**: CSV文件必须包含所有必需列，否则会抛出ValueError
3. **索引**: 解析后的DataFrame使用多级索引，注意索引结构
4. **数据类型**: 系统会自动处理数据类型转换，但建议CSV中数值列为数字格式
5. **日志**: 使用logging模块记录关键步骤，可通过日志了解执行过程

## 错误处理

- **文件不存在**: 抛出FileNotFoundError，包含文件路径信息
- **必需列缺失**: 抛出ValueError，列出缺失的列名
- **数据类型错误**: 抛出TypeError或ValueError，包含详细信息
- **计算错误**: 记录warning日志，跳过有问题的字段，继续处理其他字段

## 扩展说明

本模块设计为可扩展的架构：

1. **计算逻辑与IO分离**: 可以轻松替换CSV读取为数据库查询
2. **模块化设计**: 各模块独立，可以单独使用或替换
3. **类型注解**: 所有函数都有完整的类型注解，便于IDE支持和类型检查
4. **日志系统**: 使用标准logging模块，便于集成到更大的系统中

