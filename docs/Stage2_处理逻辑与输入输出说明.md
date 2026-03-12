# Stage2 完整处理逻辑与输入输出说明

本文档说明 Stage2（多特征组合规则生成）的**完整处理流程**、**输入约定**与**输出样式**。各列/键详细含义见 [Stage2 输出说明](Stage2_输出说明.md)。

**约定**：Stage2 **不依赖覆盖率预估**（无 coverage_estimator），所有评分与筛选仅基于差异评分、稳定性评分与多样性评分，不输出任何“估算覆盖率”字段。

---

## 一、概述

Stage2 在**仅依赖 Stage1 输出的差异与推荐结果**的前提下，完成：

1. **差异评分与稳定性评分**：对 Stage1 的连续/离散差异做多尺度差异评分与稳定性建模；
2. **原子规则库**：优先从 Stage1 候选族生成单字段原子规则，不足时可放宽离散稳定性或增加连续低尾规则（Round3）；原子规则经精度过滤（max_base_cov、min_lift_atomic）与降级；
3. **Coverage-free Beam Search**：用差异、稳定性、多样性评分组合多字段规则；每条候选带 **rule_feature_ids**、**rule_signature**；**min_rules_per_segment** 与 **require_exact_k** 控制是否允许 k=2（允许时不必严格 k=3）；
4. **候选按 rule_signature 去重**：保留 score 最大的一条，日志输出 before_dedup / after_dedup / dedup_removed；可选同特征集去冗余（same_structure_at_candidate）；
5. **客群评分与筛选**：使用 **unified_scoring**（精度、lift、覆盖率惩罚等）计算综合得分，含 missing_pair 惩罚，并取 Top N；
6. **多客群组合与两阶段选择**：filter_similar_rules → assign_tiers（elite/standard/expand）→ **select_final_segments**（elite 强制 + 结构距离最大化 + 锚点限流 + family 覆盖）；可选 **risk_scoring**（tier 权重或 use_rule_score_directly）供下游 AUC 对比；
7. **导出**：候选导出为**去重后**列表并写 metadata（dedup 统计）、rule_signature；Portfolio 输出 **recommended**（最终 1~K 条，K 可为 2 或 3）、**menu**（菜单全量）、**selection_trace**，**组合指标** 基于 recommended 重算。

输出为**原子规则库**（CSV/JSON）、**候选客群规则**（CSV/JSON，去重后）、**推荐客群组合**（JSON），表头/键名为**业务可读（中文）**。

---

## 二、输入

### 2.1 输入格式与运行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--stage1-output-dir` | Stage1 输出目录 | `../data/stage1_output` |
| `--stat-date` | 统计日期，YYYYMM | `202512`（必填） |
| `--cohort-name` | 客群名称 | `PRODA`、`sub`（必填） |
| `--output-dir` | Stage2 输出目录 | `./data/stage2_output` |
| `--config` | 配置文件路径（JSON，可选） | `stage2_config.json` |

### 2.2 输入文件命名与结构

- **路径**：`{stage1-output-dir}/`
- **文件名**：  
  - 连续型差异：`numeric_diff_{cohort_name}_{stat_date}.csv`  
  - 离散型差异：`categorical_diff_{cohort_name}_{stat_date}.csv`
- **格式**：CSV，UTF-8-sig；表头可为**业务可读列名（中文）**。若为中文，Stage2 会按 `STAGE1_NUMERIC_HEADER_REVERSE` / `STAGE1_CATEGORICAL_HEADER_REVERSE` 映射回技术列名后再处理。

### 2.3 连续型差异表（Stage1 输出）依赖列

- **索引列**：读取后设为 `column_id`（若表头为“字段ID”会先反映射）。
- **依赖列**（技术名）：`column_id`, `column_name`, `stat_date`, `mean_full`, `mean_base`, `mean_diff`, `effect_size`, `delta_median`, `delta_p95`, `diff_score`, `is_significant`, `distribution_type`, `approx_skew`, `tail_ratio`, `rec_low`, `rec_high`, `direction`, `rule_reason`, `has_recommendation` 等；可选 `rec_rule_family`（JSON）、`total_count`、`tail_proxy`、`missing_ratio`。
- **Optional debug 列**：`cohort_coverage_est`、`full_coverage_est`、`lift_est` 若存在则保留，**不参与任何评分**；不存在不得报错。

### 2.4 离散型差异表（Stage1 输出）依赖列

- **索引列**：`column_id`。
- **依赖列**：`column_id`, `column_name`, `sum_abs_diff`, `diff_score`, `rec_categories`, `rule_desc` 等；若缺少 `delta_ratio`，Stage2 会用 `sum_abs_diff` 填充并打 warning。
- **Optional debug 列**：`cohort_coverage`、`full_coverage`、`lift` 若存在则保留，**不参与任何评分**；不存在不得报错。

---

## 三、处理逻辑（完整步骤）

整体顺序为：**加载配置 → 读取 Stage1 输出 → 计算差异/稳定性评分 → 生成原子规则 → Beam Search 组合 → 客群评分与筛选 → 多客群组合与去重 → 导出**。以下按步骤写出**输入、判断逻辑、计算逻辑**及**如何得出结果**。

**流程概览：**

```
Stage1 两个 CSV → 1.加载配置 → 2.读取并反映射列名 → 3.差异评分+稳定性评分
    → 4.原子规则(连续+离散, Top-K；不足30可 Round3 放宽 diff/stability)
    → 5.Beam Search(div+stab+divs, rule_feature_ids/rule_signature)
    → 5b.候选按 rule_signature 去重(保留 score 最大)
    → 6.客群评分与筛选(含 missing_pair 惩罚, Top N)
    → 7.filter_similar_rules → assign_tiers → select_final_segments(两阶段: elite+结构距离/锚点/family)
    → 8.导出(候选=去重后+metadata+rule_signature; Portfolio=recommended/menu/selection_trace, 组合指标基于 recommended 重算)
```

**步骤与对应小节：**

| 步骤 | 小节 | 要点 |
|------|------|------|
| 1. 加载配置 | 3.1 | 配置文件或默认 Stage2Config |
| 2. 读取 Stage1 输出 | 3.2 | 两个 CSV、列名反映射、去重、delta_ratio 兼容 |
| 3. 差异评分与稳定性评分 | 3.3 | 连续/离散 divergence_score、stability_score |
| 4. 生成原子规则库 | 3.4 | 连续 rec_low/rec_high/rec_rule_family、离散 rec_categories；可选 min_stability_score_categorical、连续低尾规则；Round3 放宽 |
| 5. Coverage-free Beam Search | 3.5 | 种子→逐层扩展、rule_feature_ids/rule_signature、冲突与业务组约束 |
| 5b. 候选去重 | 3.5b | 按 rule_signature 去重，保留 score 最大，打 before/after/dedup_removed |
| 6. 客群评分与筛选 | 3.6 | 综合得分（含 missing_pair 惩罚）、Top N |
| 7. 多客群组合与两阶段选择 | 3.7 | filter_similar_rules → assign_tiers → select_final_segments（elite 强制、Jaccard 结构距离、锚点限流、family 覆盖） |
| 8. 导出结果 | 3.8 | 候选=去重后+metadata+rule_signature；Portfolio=recommended/menu/selection_trace，组合指标基于 recommended 重算 |

---

### 3.1 步骤一：加载配置

- **输入**：`config_path`（可选）、`Stage2Config` 默认值。
- **判断逻辑**：若 `config_path` 存在且可读，则 `json.load` 后 `Stage2Config.from_dict(config_dict)`；否则使用 `Stage2Config()` 默认值。
- **计算逻辑**：无数值计算；仅加载权重与阈值（如 `w_effect_size`、`beam_w_divergence`、`min_stability_score`、`similarity_threshold` 等）。
- **结果**：得到 `config: Stage2Config`，供后续所有步骤使用。

---

### 3.2 步骤二：读取 Stage1 输出文件

- **输入**：`stage1_output_dir`、`stat_date`、`cohort_name`；两个 CSV 路径。
- **判断逻辑**：
  - 若 `numeric_diff_*.csv` 或 `categorical_diff_*.csv` 不存在，抛出 `FileNotFoundError`。
  - 若列中存在「字段ID」，则对两表执行列名反映射（`STAGE1_NUMERIC_HEADER_REVERSE` / `STAGE1_CATEGORICAL_HEADER_REVERSE`），将中文表头转为技术列名。
  - 若 `numeric_diff_df.index` 或 `categorical_diff_df.index` 存在重复 `column_id`，去重保留首条并打 warning。
  - 若离散表无 `delta_ratio`，则用 `sum_abs_diff` 填充 `delta_ratio`；若两者都无则填 0.0 并打 warning。
  - Stage1 中的 coverage/lift 相关列（如 cohort_coverage_est、full_coverage_est、lift_est、cohort_coverage、full_coverage、lift）为 **optional debug 列**：存在则保留，不参与后续任何评分；不存在不得报错。
- **计算逻辑**：`pd.read_csv(..., encoding='utf-8-sig')`；`set_index('column_id')`；列重命名与去重、填充。
- **结果**：`numeric_diff_df`、`categorical_diff_df`，索引均为 `column_id`，列名为技术名，供步骤 3、4 使用。

---

### 3.3 步骤三：计算差异评分与稳定性评分

- **输入**：`numeric_diff_df`、`categorical_diff_df`、`config`；可选 `full_numeric_df`/`cohort_numeric_df`（当前主流程传 None）。
- **判断逻辑**：
  - **连续差异评分**：若 `full_numeric_df`/`cohort_numeric_df` 非空则尝试用其计算 `quantile_shift`（p90/p10 差）；否则用 `delta_p95`、`delta_median` 近似；再否则用 `mean_diff` 近似。
  - **离散差异评分**：若表无 `delta_ratio` 或全空，则 `max_delta_ratio` 取 1.0 避免除零。
  - **稳定性**：连续型若取不到 `std` 或无效则 std=1.0；若无 `tail_proxy` 则用 `tail_ratio` 作为 kurtosis 近似；离散型若无 `top_value_ratio` 则用 0.5。
- **计算逻辑**：

**A. 连续型多尺度差异评分（calculate_divergence_score）**

- 归一化：`max_effect_size = numeric_diff_df['effect_size'].abs().max()`（若空则 1.0），`max_delta_mean = numeric_diff_df['mean_diff'].abs().max()`（若空则 1.0）。
- 对每行：`effect_size = |effect_size|`，`delta_mean = |mean_diff|`，`is_significant = 1.0 if is_significant else 0.0`。
- `quantile_shift`：若有 full/cohort 表则 `quantile_shift = |p90_base−p90_full| + |p10_base−p10_full|`；否则 `quantile_shift = 0.6×|delta_p95| + 0.4×|delta_median|`；若仍为 0 则 `quantile_shift = |delta_mean|`。
- 归一化：`normalized_effect_size = effect_size / max_effect_size`，`normalized_delta_mean = delta_mean / max_delta_mean`，`normalized_quantile_shift = min(quantile_shift / max_delta_mean, 1.0)`。
- **公式**：`divergence_score = w_effect_size×normalized_effect_size + w_delta_mean×normalized_delta_mean + w_quantile_shift×normalized_quantile_shift + w_significance×is_significant`（默认权重 0.4、0.3、0.2、0.1）。

**B. 离散型差异评分（calculate_categorical_divergence_score）**

- `max_delta_ratio = categorical_diff_df['delta_ratio'].abs().max()`（若空或无列则 1.0）。
- 对每行：`delta_ratio = |delta_ratio|`，`is_significant = 1.0 if is_significant else 0.0`，`normalized_delta_ratio = delta_ratio / max_delta_ratio`。
- **公式**：`divergence_score = w_delta_mean×normalized_delta_ratio + w_significance×is_significant`。

**C. 稳定性评分（calculate_stability_score）**

- **连续型**：`std_component = 1/(1+std)`；`kurtosis_approx = tail_proxy` 或 `tail_ratio`；`kurtosis_component = 1/(1+kurtosis_approx)`；`sample_weight = min(total_count/100000, 1)`；`missing_component = 1 − min(missing_ratio, 0.5)`。
  - **公式**：`stability_score = stability_w_std×std_component + stability_w_kurtosis×kurtosis_component + stability_w_sample×sample_weight×missing_component`（默认 0.4、0.3、0.3）。
- **离散型**：`sample_weight = min(total_count/100000, 1)`，`missing_component = 1 − min(missing_ratio, 0.5)`，`concentration_component = min(top_value_ratio, 0.8)/0.8`；**公式**：`stability_score = 0.3 + 0.7×(sample_weight×missing_component×concentration_component)`。

- **结果**：`numeric_divergence_scores`、`categorical_divergence_scores`（Series，index=column_id）；合并为 `all_divergence_scores`；`stability_scores`（Series，index=column_id，含连续+离散）。传入步骤 4、5、6、7。

---

### 3.4 步骤四：生成原子规则库

- **输入**：`numeric_diff_df`、`categorical_diff_df`、`config`，以及 `numeric_divergence_scores`、`categorical_divergence_scores`、`stability_scores`。
- **判断逻辑**：

**连续型（generate_numeric_atomic_rules）**

- **有效规则**：`is_significant == 1` 且 `divergence_score >= min_divergence_score` 且 `stability_score >= min_stability_score`；若存在列 `has_recommendation` 则需 `has_recommendation == True`。
- 在有效规则中按 `divergence_score` 降序取 **Top top_k_features**（默认 20）。
- 对每行：若存在且有效则解析 `rec_rule_family`（JSON）：取 `main` 为主规则，`alternatives` 最多 3 条备选；若无或解析失败则用 `rec_low`/`rec_high` 生成主规则；若为 heavy_tail/powerlaw 且 `stability_score >= min_stability_score×0.8` 且能取到 p95，则追加一条 tail 规则（区间 [p95, +∞)）。

**离散型（generate_categorical_atomic_rules）**

- **有效规则**：`delta_ratio >= min_delta` 且（`rec_categories` 非空或 `rule_desc` 含 `in {`）且 `divergence_score >= min_divergence_score`；若有 `stability_score` 列则用 **min_stability_score_categorical**（若配置存在）或 `min_stability_score` 做阈值筛选，便于原子不足时放宽离散进入池子。
- 按 `divergence_score` 降序取 **Top top_k_features**。
- 对每行：若 `rec_categories` 为空/NaN，则从 `rule_desc` 用正则解析类别；类别数超过 `max_categories` 则截断。

**连续型低尾规则（原子不足时扩大池子）**

- 对每个已生成 main 且 direction=high 的连续特征，额外生成一条 **low_tail** 规则：`rule_low=-inf`、`rule_high=main_rule_low`、`direction=low`、divergence_score×0.7；每字段仅增加一条。

**Round3 放宽（原子规则 < min_atomic_rules_for_search 时）**

- 若精度过滤 Round2 后原子规则数仍 < 30，则复制 config 放宽 `min_divergence_score`、`min_stability_score`（不改 demographic/height 硬剪枝），用同一份 diff_df 与 scores 重新生成原子规则并再做一次精度过滤；若仍不足则保留 Round2 结果并打 WARNING。

- **计算逻辑**：连续型输出含 main/tail/low_tail；离散型输出同上。**合并**：`merge_atomic_rules` 统一格式并增加 `rule_type_feature`（numeric/categorical）。
- **结果**：`atomic_rules_df`，供步骤 5 Beam Search 使用。

---

### 3.5 步骤五：Coverage-free Beam Search 规则组合

- **输入**：`atomic_rules_df`、`numeric_diff_df`、`categorical_diff_df`、`config`、`all_divergence_scores`、`stability_scores`；可选 `business_groups`、`max_fields_per_business_group`（默认 2）。
- **判断逻辑**：
  - 若 `atomic_rules_df` 为空，返回空列表。
  - 按 `rule_type_feature` 拆分为 `numeric_rules` 与 `categorical_rules`。
  - **第一层**：每个原子规则生成一条单特征 `SegmentRule`，`diversity_score=1.0`，初始 `score=divergence_score`；按 score 降序取 **Top top_k_features** 作为当前层候选。
  - **第二层及以后**（depth=2 到 max_features_per_segment）：对当前层每条候选，与原子规则表中**未使用过的字段**做 AND 组合；对新组合计算 `combined_div_score`、`combined_stab_score`、`diversity_score`（相对已有 all_candidates）；再检查内部冲突、业务组约束、与已有候选的规则冲突；通过则加入 new_candidates；按 score 取 **Top beam_size** 作为下一层候选。
  - **冲突**：同字段多区间完全分离（high1≤low2 或 high2≤low1）判为数值冲突；同字段类别集合交集为空判为离散冲突；任一与已有候选冲突则丢弃该组合。
- **计算逻辑**：

**组合得分（每层权重可调）**

- depth=2：`w_div=0.5, w_stab=0.3, w_divs=0.2`；depth=3：`w_div=0.4, w_stab=0.4, w_divs=0.2`；depth≥4：`w_div=0.3, w_stab=0.5, w_divs=0.2`。
- **公式**：`combined_div_score = (candidate.divergence_score×len(feature_rules) + new_div_score) / (len(feature_rules)+1)`；`combined_stab_score` 同理。
- `diversity_score = calculate_rule_diversity_score(new_rule, all_candidates)`（见 3.7 结构距离与多样性）。
- **综合得分**：`score = w_div×combined_div_score + w_stab×combined_stab_score + w_divs×diversity_score`。

- **结果**：`candidate_rules: List[SegmentRule]`，每条带 **rule_feature_ids**（sorted column_id 列表）、**rule_signature**（与 rule_output.segment_canonical_key 一致，用于去重与导出）。

---

### 3.5b 步骤五 b：候选按 rule_signature 去重

- **输入**：Beam Search 输出的 `candidate_rules`（已过滤 require_exact_k 等）。
- **判断逻辑**：按 **rule_signature** 分组，同签名仅保留一条。
- **计算逻辑**：每组保留 **score 最大** 的一条；日志输出 `candidates_before_dedup`、`candidates_after_dedup`、`dedup_removed`。
- **结果**：`candidate_rules` 为去重后列表，供步骤 6、8（候选导出使用此列表）。

---

### 3.6 步骤六：客群评分与筛选

- **输入**：去重后的 `candidate_rules`、`config`。
- **判断逻辑**：对每条规则计算综合得分；若 `divergence_score` 或 `stability_score` 低于阈值则丢弃；否则保留。
- **计算逻辑**：`score_segment_rule(rule, config)`：先算 `score = w_div×divergence_score + w_stab×stability_score + w_divs×diversity_score`；若 **missing_pair_count > 0**，则 `pair_total = k*(k-1)//2`，`missing_pair_ratio = missing_pair_count / pair_total`，`score_penalty = missing_pair_penalty_lambda × missing_pair_ratio`，**score_final = score - score_penalty**（λ 默认 0.1）。将 `rule.score` 设为 score_final。按 score 降序取 **Top top_n_candidates**。
- **结果**：`filtered_rules: List[SegmentRule]`，供步骤 7 组合与两阶段选择。

---

### 3.7 步骤七：多客群组合与两阶段选择

- **输入**：`filtered_rules`、`config`。
- **判断逻辑**：
  1. **filter_similar_rules**：按结构相似度与 `similarity_threshold` 做预选，得到 `portfolio.segments`（预选客群），并计算 `portfolio_metrics`（基于该预选集合，仅用于日志/兼容）。
  2. **菜单路径**（导出时）：对候选做去重保底、按 max_est/min_est/max_cov_menu 过滤时 **elite 一律保留**；截断时先保证所有 elite（及可选 standard）在列表内，再按 _menu_sort_key 填满至 output_max_segments；**assign_tiers** 得到 elite / standard / expand。
  3. **select_final_segments**：**Phase 1**：若存在 elite，强制选入 1 条 elite；**Phase 2**：剩余名额按 **结构距离（Jaccard）+ 综合得分** 选人，且受 **锚点限流**（同一 anchor 最多 max_segments_per_anchor 条）、**family 覆盖**（未达 min_family_coverage 时对能带来新 family 的候选加分）约束。

**规则结构距离（用于两阶段选择）**

- 优先使用 **rule_feature_ids**：`jaccard_distance = 1 - |A∩B|/|A∪B|`（空集约定为 0）；若无则从 feature_rules 抽 column_id 再算 Jaccard。保证不同字段集合的规则 min_struct_dist > 0（除完全相同）。

**锚点与 family**

- 锚点：每条 segment 的 `rule_feature_ids[0]`（或 feature_rules 首字段）。Phase 2 中若某 anchor 已选条数达 `max_segments_per_anchor` 则跳过该候选。
- family：从 `feature_rules` 取 `feature_family`（无则 `'other'`）；若已选 family 数 < `min_family_coverage` 且候选能带来新 family，则对 score_selection 加固定 bonus（如 0.05）。

- **结果**：`selected_list`（最终 1~K 条）、`selection_steps`（选择过程）、`elite_forced`（是否强制选入过 elite）。导出时 **组合指标** 由 **selected_list** 重算（total_segments、avg_*、min_structure_distance），不沿用 portfolio.portfolio_metrics。

---

### 3.8 步骤八：导出结果

- **输入**：`atomic_rules_df`、**去重后的候选列表**（与日志 after_dedup 一致）、`portfolio`、两阶段选择得到的 `selected_list`/`selection_steps`/`tiers`、`output_dir`、`cohort_name`、`stat_date`、`column_name_map`；以及 dedup 计数（n_before_dedup、n_after_dedup）。
- **判断逻辑**：若输出目录不存在则创建；候选导出使用**去重后**列表；Portfolio 的 **组合指标** 基于 **selected_list**（即 recommended）重算；表头/键名使用业务可读中文（见 Stage2_输出说明.md）。
- **计算逻辑**：
  - **原子规则**：CSV/JSON，无变更。
  - **候选客群**：导出**去重后**的候选；每条记录含 **rule_signature**；JSON 的 metadata 写入 `candidates_before_dedup`、`candidates_after_dedup`、`dedup_removed`（若传入）。
  - **推荐客群组合**：JSON 含 `metadata`（target_segment_count、selected_count、recommended_meaning）、**recommended**（= selected_segments，最终 1~K 条，每项含 rule_feature_ids）、**menu**（elite_cn + standard_cn + expand_cn）、**selection_trace**（= selection_steps）；**组合指标** 由 selected_list 重算（total_segments、avg_*、min_structure_distance）。
- **结果**：  
  - `atomic_rules_{cohort_name}_{stat_date}.csv` / `.json`  
  - `candidate_segments_{cohort_name}_{stat_date}.csv` / `.json`（去重后，含 rule_signature 与 dedup metadata）  
  - `segment_portfolio_{cohort_name}_{stat_date}.json`（含 recommended、menu、selection_trace，组合指标基于 recommended）  

---

## 四、输出

### 4.1 输出文件命名与位置

- **路径**：`{output-dir}/`，默认 `./data/stage2_output`。
- **文件名**：  
  - 原子规则库：`atomic_rules_{cohort_name}_{stat_date}.csv`、`.json`  
  - 候选客群规则：`candidate_segments_{cohort_name}_{stat_date}.csv`、`.json`  
  - 推荐客群组合：`segment_portfolio_{cohort_name}_{stat_date}.json`  

### 4.2 输出格式与表头/键名

- **原子规则 / 候选客群**：CSV 为 UTF-8-sig，第一列为索引（字段ID或规则编号）；表头为**中文业务可读**。
- **推荐客群组合**：JSON，键名为**中文**（如 组合指标、客群列表、客群编号、规则明细、评分指标、最小结构距离 等）。

### 4.3 与 Stage1 文档的对应关系

各文件列含义、JSON 结构及阅读建议见 **[Stage2 输出说明](Stage2_输出说明.md)**。本文档仅描述**如何通过处理逻辑得到这些输出**。

---

## 五、与上下游的衔接

- **Stage1**：Stage2 读取 Stage1 的 `numeric_diff_*.csv` 与 `categorical_diff_*.csv`；若表头为中文会自动反映射为技术列名；**不修改** Stage1 的 CSV 结构约定。
- **下游**：原子规则库与候选客群可供规则引擎或人工查看；推荐客群组合中的 **SQL筛选条件** 可在受控环境执行以生成客群名单。  
- **不输出**：任何“估算覆盖率”“预估覆盖人数”等字段；Stage2 全程不使用覆盖率估计。
