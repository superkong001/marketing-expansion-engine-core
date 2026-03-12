# Stage1 完整处理逻辑与输入输出说明

本文档说明 Stage1（特征差异分析）的**完整处理流程**、**输入约定**与**输出样式**。各列详细含义见 [Stage1 输出说明](Stage1_输出说明.md)。

---

## 一、概述

Stage1 在**不依赖明细数据**的前提下，基于全量客群与圈定客群的**预聚合统计表**，完成：

1. **连续型特征**：计算均值/分位数/效应量等差异，检测分布类型，并为差异显著的字段推荐阈值区间；
2. **离散型特征**：计算分布差异（占比差、熵、Gini 等），并为差异显著的字段推荐“圈定>全量”的类别集合。

输出为两类 CSV（连续型差异、离散型差异），表头为**业务可读列名（中文）**，供业务查看或作为 Stage2 的输入。

---

## 二、输入

### 2.1 输入格式与运行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--input-format` | 输入格式：`csv` 或 `excel` | 默认 `excel` |
| `--full-dir` | 全量客群统计文件所在目录 | `./data/full_stats` |
| `--cohort-dir` | 圈定（对比）客群统计文件所在目录 | `./data/cohort_stats` |
| `--stat-date` | 统计月份，YYYYMM | `202512` |
| `--cohort-name` | 圈定客群名称（Excel 模式可省略，默认 sub） | `sub`、`PRODA` |
| `--output-dir` | Stage1 输出目录 | `./data/stage1_output` |
| `--top-k-numeric` | 对差异度 Top K 的连续特征做阈值推荐 | 默认 50 |
| `--top-k-categorical` | 对差异度 Top K 的离散特征做类别推荐 | 默认 30 |
| `--target-ratio` | 正态分布时目标覆盖比例（可选） | 如 `0.1` |

### 2.2 文件命名与目录结构

**CSV 模式：**

- 全量连续：`{full-dir}/numeric_stats_full_{stat_date}.csv`
- 全量离散：`{full-dir}/categorical_stats_full_{stat_date}.csv`
- 圈定连续：`{cohort-dir}/numeric_stats_{cohort_name}_{stat_date}.csv`
- 圈定离散：`{cohort-dir}/categorical_stats_{cohort_name}_{stat_date}.csv`

**Excel 模式：**

- 全量连续：`{full-dir}/st_ana_continuous_feature_stats_all_{stat_date}.xlsx`
- 全量离散：`{full-dir}/st_ana_discrete_feature_stats_all_{stat_date}.xlsx`
- 圈定连续：`{cohort-dir}/st_ana_continuous_feature_stats_sub_{stat_date}.xlsx`
- 圈定离散：`{cohort-dir}/st_ana_discrete_feature_stats_sub_{stat_date}.xlsx`

### 2.3 连续型统计表（输入）结构

- **索引**：`(stat_date, column_id)`，即每行对应一个“统计月份 + 字段”。
- **必需列**（内部使用名）：  
  `stat_date`, `column_id`, `column_name`, `total_count`, `nonull_count`, `nonull_ratio`, `avg_val`, `var_val`,  
  `q1_cont`, `q2_cont`, `q3_cont`, `q6_cont`, `q9_cont`, `p05_cont`, `p90_cont`, `p95_cont`, `p99_cont`, `IQR`, `CV`, `source_type`, `ext_info`  
- **可选列**：`min_val`, `max_val`（供 threshold_numeric 与分布分析使用）；`q4_cont`～`q11_cont`（扩展分位）；`avg_str_1`, `avg_str_2`（用于 std 近似）。缺 `min_val`/`max_val` 时 loader 会打 warning。
- **Key 列**：加载后程序会补齐 `table_id`, `group_id`, `source_type`, `ext_info`（缺则补默认值），保证全量/圈定的 numeric 与 categorical 四个 DataFrame 均含上述四列，与统计表设计口径一致。
- **说明**：Excel 表头可为**中文**（如 统计日期、字段ID、均值、中位数、p90 等）或**英文**（当前 202510 结构为英文 snake_case），由 `stats_loader` 按映射转为上述内部列名；会由 `var_val` 计算 `std`，若缺则用 `avg_str_1`/`avg_str_2` 近似；并映射出 `mean`, `median`, `p05`, `p90`, `p95`, `p99` 等供差异与阈值推荐使用。**表中不包含 skewness、kurtosis**，偏度/峰度由程序内部基于分位数或 tail 近似计算。

### 2.4 离散型统计表（输入）结构

- **索引**：`(stat_date, column_id, val_enum)`，即每行对应一个“统计月份 + 字段 + 枚举值”。
- **必需列**：  
  `stat_date`, `column_id`, `column_name`, `total_count`, `val_enum`, `val_count`, `val_ratio`, `val_rank`, `unique_count`, `entropy`, `gini`, `source_type`, `ext_info`  
- **Key 列**：同上，加载后补齐 `table_id`, `group_id`, `source_type`, `ext_info`。
- **说明**：对比客群中若某枚举值无记录，差异计算时该枚举占比按 0 处理；Excel 表头可为中文或英文（202510 为英文），由 loader 映射；`total_count`, `val_count`, `val_ratio`, `entropy`, `gini` 等数值列在加载时会统一 `to_numeric`，避免类型异常。

---

## 三、处理逻辑（完整步骤）

整体顺序为：**配置 → 加载 → 连续差异 → 分布检测 → 连续阈值推荐 → 离散差异 → 离散类别推荐 → 保存**。以下按步骤写出**输入、判断逻辑、计算逻辑**及**如何得出结果**。

**流程概览：**

```
全量/圈定统计表(CSV或Excel) → 1.构建配置 → 2.加载并补齐 key 列 → 3.连续差异(common_index/effect_size/diff_score)
    → 3.2 基于分位数检测分布类型(zero_inflated/heavy_tail/skewed/symmetric) → 3.5 连续阈值(Top-K, 按分布类型固定策略, coverage/lift 仅 debug)
    → 4.离散差异(sum_abs_diff/diff_score) → 4.5 离散类别(Top-K, 贪心rec_categories)
    → 5.保存(中文表头CSV)
```

**步骤与对应小节：**

| 步骤 | 小节 | 要点 |
|------|------|------|
| 1. 构建配置 | 3.1 | 根据 input_format 拼 4 个文件路径 |
| 2. 加载统计文件 | 3.2 | 读 4 个 DF，列名映射、索引、std 计算；补齐 table_id/group_id/source_type/ext_info |
| 3. 连续特征差异 | 3.3 | common_index、effect_size、diff_score、is_significant |
| 3.2 分布类型 | 3.4 | 基于分位数：zero_inflated→heavy_tail→skewed→symmetric |
| 3.5 连续阈值推荐 | 3.5 | Top-K，按分布类型固定策略，coverage/lift 仅作 debug 输出 |
| 4. 离散特征差异 | 3.6 | 按 val_enum 外连接、sum_abs_diff、diff_score |
| 4.5 离散类别推荐 | 3.7 | Top-K，delta>min_delta，贪心 increment/min_cov |
| 5. 保存结果 | 3.8 | 列名映射中文后写 CSV |

---

### 3.1 步骤一：构建配置

- **输入**：命令行或调用参数 `input_format`、`full-dir`、`cohort-dir`、`stat_date`、`cohort-name`。
- **判断逻辑**：
  - 若 `input_format == 'excel'`：用 Excel 命名约定拼出 4 个 xlsx 路径（全量 all、圈定 sub）；`cohort-name` 缺省时为 `sub`。
  - 若 `input_format == 'csv'`：用 CSV 命名约定拼出 4 个 csv 路径；`stat_date` 与 `cohort-name` 必填。
- **计算逻辑**：无数值计算，仅路径拼接。
- **结果**：得到 `OverallConfig`，内含全量/圈定的 `numeric_csv_path`、`categorical_csv_path`（共 4 个文件路径）。

---

### 3.2 步骤二：加载统计文件

- **输入**：上述 4 个文件路径。
- **判断逻辑**：
  - 若为 Excel：调用 `load_numeric_stats_from_excel` / `load_categorical_stats_from_excel`；列名通过映射表转为内部名（如 统计日期→stat_date、均值→avg_val 等）；**保留** `table_id`，不再删除。
  - 若为 CSV：调用 `load_numeric_stats` / `load_categorical_stats`；检查必需列是否齐全，缺则抛错。
  - 加载完成后，对四个 DataFrame 分别**补齐 key 列**：若缺少 `table_id`、`group_id`、`source_type`、`ext_info` 中任意一列，则添加该列并赋默认值（如空字符串），保证下游拿到的四个 DF 均含上述四列。
- **计算逻辑**：
  - **连续型**：`std = sqrt(var_val)`（若缺或无效且存在 `avg_str_1`/`avg_str_2` 则用近似 `(avg_str_2 - avg_str_1)/6` 填充）；列重命名为 `mean`、`median`、`p05`、`p90`、`p95`、`p99` 等；索引设为 `(stat_date, column_id)`。
  - **离散型**：索引设为 `(stat_date, column_id, val_enum)`；不做额外计算。
- **结果**：4 个 DataFrame：`full_numeric_df`、`full_categorical_df`、`cohort_numeric_df`、`cohort_categorical_df`，均含 `table_id`, `group_id`, `source_type`, `ext_info`，供后续步骤使用。

---

### 3.3 步骤三：计算连续特征差异

- **输入**：`full_numeric_df`、`cohort_numeric_df`（索引均为 `(stat_date, column_id)`），且需包含列 `mean`、`std`、`median`、`p95`、`IQR`、`column_name`。
- **判断逻辑**：
  - 取**共同索引**：`common_index = full_index ∩ cohort_index`。仅对 `common_index` 内的字段计算；仅在 full 或仅在 cohort 的字段跳过并打 warning。
  - 若 `common_index` 为空，返回空 DataFrame 并报错。
- **计算逻辑**（对每个共同字段，向量化计算）：
  - **均值差异**：`mean_diff = mean_base - mean_full`（圈定 − 全量）；`mean_diff_ratio = mean_diff / max(|mean_full|, 1e-6)`。
  - **合并标准差**：`pooled_std = sqrt((std_full² + std_base²) / 2)`；若 ≤0 或 NaN 则置为 1e-6。
  - **效应量（Cohen's d）**：`effect_size = (mean_base - mean_full) / pooled_std`。
  - **分位数差异**：`delta_median = median_base - median_full`；`delta_p95 = p95_base - p95_full`；`delta_IQR = iqr_base - iqr_full`。若有 `CV` 列则 `delta_CV = cv_base - cv_full`，否则为 NaN。
  - **综合差异分数**：`std_full_safe = std_full`（若 ≤0 或 NaN 则 1e-6）；  
    `diff_score = |effect_size| + 0.5×|delta_median/std_full_safe| + 0.3×|delta_p95/std_full_safe|`。
  - **是否显著**：`is_significant = (|effect_size| >= 0.2)`。
- **结果**：DataFrame，索引为 `column_id`，列含 `column_id`、`column_name`、`stat_date`、`mean_full`、`mean_base`、`mean_diff`、`mean_diff_ratio`、`effect_size`、`delta_median`、`delta_p95`、`delta_IQR`、`delta_CV`、`diff_score`、`is_significant`；按 `diff_score` 降序排列。`column_name` 优先取圈定，缺失时回填全量。

---

### 3.4 步骤 3.2：基于分位数检测分布类型

- **输入**：`full_numeric_df`（全量连续统计，索引 `(stat_date, column_id)`），列需含 P25/P50/P75/P90/P95/P99/IQR（如 `q1`、`median`、`q3`、`p90`、`p95`、`p99`、`IQR`）。
- **判断逻辑**：调用 `detect_distribution_types_from_quantiles(full_numeric_df)`。若同一 `column_id` 有多行（如多 stat_date 或多 table_id），优先过滤 `source_type==1` 且 `group_id=='ALL'` 的全量基准行，再按 `column_id` 取代表行；然后按**优先级顺序**（每字段只命中第一条）：
  1. **zero_inflated**：`P50 == 0` 且 `P90 > 0`（最高优先级）。
  2. **heavy_tail**：`(P99−P95)/(P95−P50) ≥ 0.5`（分母≤0 时不计入）。
  3. **skewed**：`|(P75+P25−2×P50)/IQR| > 0.2`（IQR>0 时）。
  4. **symmetric**：其余情况。
- **计算逻辑**：
  - **approx_skew**：`(P75 + P25 − 2×P50) / IQR`，IQR≤0 时置 0。
  - **tail_ratio**：`(P99 − P95) / (P95 − P50)`，分母≤0 时置 0。
- **结果**：DataFrame 索引为 `column_id`，列 `distribution_type`、`approx_skew`、`tail_ratio`；与步骤三的连续差异表按 `column_id` **左连接**，缺失类型填 `symmetric`，缺失 `approx_skew`/`tail_ratio` 填 0 和 1。

---

### 3.5 步骤 3.5：推荐连续特征阈值

- **输入**：连续差异表（含 `distribution_type`，取值为 symmetric/skewed/heavy_tail/zero_inflated）、`full_numeric_df`、`cohort_numeric_df`。
- **判断逻辑**：
  - **参与推荐范围**：仅对**差异度 Top top_k_numeric**（默认 50）的连续字段做推荐；其余字段不调用推荐算法，后续统一补空或默认。
  - **决策仅依赖分布类型**：不再使用 coverage/lift 筛选或排序候选；按 `distribution_type` 使用**固定策略**直接给出区间，`cohort_coverage_est`、`full_coverage_est`、`lift_est` 仅在确定区间后计算并写入，作为 backward-compatible / debug 输出。
- **计算逻辑**（固定策略，与统计文档口径一致）：

**A. symmetric（对称）**  
- 以**全量行**分位数为阈值来源：`rec_low = P25_full`，`rec_high = P75_full`；若 full 缺失则回退 cohort。`rule_reason` 标注如 `symmetric: P25_full, P75_full`。

**B. skewed（偏态）**  
- 单侧：若 `effect_size ≥ 0` 则 `rec_low = P75`、`rec_high = +∞`；否则 `rec_low = −∞`、`rec_high = P25`（P25/P75 取自圈定行）。

**C. heavy_tail（重尾）**  
- 以**全量行**高位分位为下界：优先 `P95_full`，缺失则依次回退 `P90_full`、`P99_full`；`rec_high = +∞`。`rule_reason` 标注如 `heavy_tail: P95_full (fallback P90_full, P99_full)`。

**D. zero_inflated（零膨胀）**  
- 规则为 “>0”：`rec_low = 0`，`rec_high = +∞`，`rule_desc` 注明 “>0”。

- **coverage/lift**：上述区间确定后，用分位数线性插值或现有 `estimate_coverage_from_quantiles` 计算 `cohort_coverage_est`、`full_coverage_est`、`lift_est` 并写入输出列，仅作 debug，不参与是否推荐的决策。
- **结果**：每个参与字段得到一行推荐（或 None）；与连续差异表按 `column_id` **左连接**，无推荐的字段补 NaN 和 `has_recommendation=False`，并补全 `direction`、`rule_reason` 等默认值。

---

### 3.6 步骤四：计算离散特征差异

- **输入**：`full_categorical_df`、`cohort_categorical_df`（索引均为 `(stat_date, column_id, val_enum)`），且含列 `val_ratio`、`column_name`、`entropy`、`gini`、`total_count`。
- **判断逻辑**：
  - 取**共同 (stat_date, column_id)**：`common_keys = full_keys ∩ cohort_keys`。仅对 `common_keys` 内字段计算；仅在单侧出现的键跳过并打 warning。
  - 若 `common_keys` 为空，返回空 DataFrame 并报错。
  - 对每个 (stat_date, column_id)，取该字段下全量/圈定子表，按 `val_enum` **外连接**；缺失的 `val_ratio` 填 0（即某侧没有的枚举视为占比 0）。
- **计算逻辑**（对每个字段）：
  - **频率差**：`ratio_diff = val_ratio_base - val_ratio_full`（圈定 − 全量）；`abs_ratio_diff = |ratio_diff|`。
  - **聚合**：`sum_abs_diff = sum(abs_ratio_diff)`；`max_abs_diff = max(abs_ratio_diff)`。
  - **Top 差异类别**：取 `abs_ratio_diff` 最大的 top_k（默认 3）个枚举，格式化为 `"枚举1(+0.25); 枚举2(-0.08)"` 存入 `top_diff_categories`。
  - **推荐类别（圈定>全量）**：`recommended_categories = 所有 ratio_diff > 0 的 val_enum`，逗号拼接。
  - **熵差异**：`entropy_diff = entropy_base - entropy_full`（从该字段任一行取字段级 entropy）。
  - **Gini 差异**：`gini_diff = gini_base - gini_full`（若有列）。
  - **综合差异分数**：`weight = cohort_total_count / max(full_total_count, 1)`；`diff_score = (sum_abs_diff + 0.5×max_abs_diff) × weight`。
- **结果**：DataFrame，索引为 `column_id`，列含 `column_id`、`column_name`、`stat_date`、`sum_abs_diff`、`max_abs_diff`、`top_diff_categories`、`recommended_categories`、`entropy_diff`、`gini_diff`、`diff_score`；按 `diff_score` 降序。

---

### 3.7 步骤 4.5：推荐离散类别集合

- **输入**：离散差异表、`full_categorical_df`、`cohort_categorical_df`；参数 `min_delta`（默认 0.01）、`min_cov`（默认 0.1）、`min_increment`（默认 0.01）、`bad_tokens`（默认过滤 `__NULL__`、`__OTHER__`）。
- **判断逻辑**：
  - **参与推荐范围**：仅对**差异度 Top top_k_categorical**（默认 30）的离散字段做推荐。
  - 对每个字段，取该字段在全量/圈定中的子表，按 `val_enum` 外连接，缺失 `val_ratio` 填 0。
- **计算逻辑**（对每个参与字段）：
  - **delta**：`delta = val_ratio_base - val_ratio_full`。
  - **候选**：仅保留 `delta > min_delta` 的枚举，并按 `delta` 降序排列；再剔除 `bad_tokens` 中的枚举。
  - 若候选为空，返回 None，该字段无推荐。
  - **贪心构造**：初始 `rec_cats = []`，`cov_cohort = 0`，`cov_full = 0`。按 delta 从高到低遍历候选枚举：  
    - 若加入该类后圈定覆盖率增量 `increment < min_increment`，则跳过该类；  
    - 否则将该类加入 `rec_cats`，并更新 `cov_cohort += val_ratio_base`，`cov_full += val_ratio_full`；  
    - 若 `cov_cohort ≥ min_cov` 则停止。
  - **lift**：`lift = cov_cohort / max(cov_full, 1e-6)`。
  - **命中人数**：`full_hit_count = cov_full × full_total_count`，`cohort_hit_count = cov_cohort × cohort_total_count`。
- **结果**：每个字段得到 `rec_categories`（列表或逗号串）、`cohort_coverage`、`full_coverage`、`lift`、`full_hit_count`、`cohort_hit_count`、`rec_category_count`、`rule_desc`；与离散差异表左连接，无推荐字段补空。

---

### 3.8 步骤五：保存结果

- **输入**：连续差异表（含阈值推荐列）、离散差异表（含类别推荐列），以及业务可读列名映射表（如 字段ID、全量均值、综合差异分数、是否给出阈值推荐 等）。
- **判断逻辑**：若输出文件已存在，先尝试以追加模式打开以检测是否被占用；若权限错误则抛错并提示关闭 Excel 等。
- **计算逻辑**：复制 DataFrame，将列名与索引名按映射表替换为中文；不改变单元格数值。
- **结果**：写入 `numeric_diff_{cohort_name}_{stat_date}.csv`、`categorical_diff_{cohort_name}_{stat_date}.csv`，编码 UTF-8-sig，第一列为索引（中文列名为“字段ID”），其余列为业务可读中文表头。

---

## 四、输出

### 4.1 输出文件命名与位置

- **路径**：`{output-dir}/`，默认 `./data/stage1_output`。
- **文件名**：  
  - 连续型：`numeric_diff_{cohort_name}_{stat_date}.csv`  
  - 离散型：`categorical_diff_{cohort_name}_{stat_date}.csv`  
- **示例**：`numeric_diff_sub_202512.csv`、`categorical_diff_sub_202512.csv`。

### 4.2 输出格式与表头

- **格式**：CSV，UTF-8-sig，第一列为索引（字段ID），其余为数据列。
- **表头**：全部为**业务可读列名（中文）**，无需数据字典即可理解；与 [Stage1 输出说明](Stage1_输出说明.md) 中的“列说明”一一对应。

### 4.3 连续型输出样式（表头示例）

第一列为索引，列名为**中文**，例如：

| 字段ID | 字段名称 | 统计月份 | 全量均值 | 对比客群均值 | 均值差异（对比−全量） | 相对差异 | 效应量 | … | 推荐区间下界 | 推荐区间上界 | 推荐方向 | 规则描述 | 推荐理由 | 是否给出阈值推荐 |
|--------|----------|----------|----------|--------------|------------------------|----------|--------|---|--------------|--------------|----------|----------|----------|------------------|

每行一个连续字段；无推荐时推荐区间、规则描述等可为空；`是否给出阈值推荐` 为 True/False。

### 4.4 离散型输出样式（表头示例）

第一列为索引，列名为**中文**，例如：

| 字段ID | 字段名称 | 统计月份 | 占比差异绝对值之和 | 最大单类占比差异 | 差异最大类别及差值 | 推荐类别（圈定>全量） | 综合差异分数 | 推荐类别集合 | 对比客群命中占比 | 全量命中占比 | 覆盖增幅 | 规则描述 |
|--------|----------|----------|--------------------|------------------|--------------------|------------------------|--------------|--------------|------------------|--------------|----------|----------|

每行一个离散字段；推荐类别集合可为具体枚举或“缺失”（nan），规则描述中可能出现 `in {nan}` 表示缺失作为一类。

### 4.5 详细列含义

各列含义、取值说明及阅读建议见 **[Stage1 输出说明](Stage1_输出说明.md)**。输出文件可直接供 Stage2 读取（Stage2 会自动识别中文表头并映射回内部列名）。

---

## 五、与下游的衔接

- **Stage2**：读取 `numeric_diff_*.csv` 与 `categorical_diff_*.csv`，按 `stat_date`、`cohort_name` 匹配；若表头为中文会自动反映射为技术列名后再做原子规则生成与规则组合。
- **规则引擎 / 人工**：可直接根据中文表头阅读与使用，按“综合差异分数”排序优先关注高差异字段，结合“是否给出阈值推荐”“推荐类别集合”“规则描述”等做业务判断。
