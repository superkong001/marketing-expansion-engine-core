# Stage2 覆盖率与排序调参指南

当 segment **覆盖人数偏少**或希望**多出推荐客群、排序更贴合业务**时，可参考以下参数（以当前 `config.json` 为基准）。

---

## 1. 组合层：允许更大的组合客群

| 参数 | 含义 | 当前 config | 调参建议 |
|------|------|-------------|----------|
| `max_combo_cov_est` | 组合 ALL 覆盖率估计上限 | 0.15 | 提高到 0.18～0.20 可保留更宽组合 |
| `expand_combo_cov_max` | expand 档允许的组合覆盖率上限 | 0.15 | 同上 |
| `require_rare_anchor` | 是否要求组合中至少一条“稀有”规则 | true | 若日志 rare_anchor 跳过多，可改为 false 或提高 rare_anchor_base_cov |
| `rare_anchor_base_cov` | 稀有锚点规则 ALL 覆盖率上限 | 0.35 | 可调到 0.40；或 require_rare_anchor=false 取消约束 |

---

## 2. 原子规则层：多留原子

| 参数 | 含义 | 当前 config | 调参建议 |
|------|------|-------------|----------|
| `max_base_cov` | Round1 单条规则 ALL 覆盖率上限 | 0.42 | 提高到 0.45 可多留原子 |
| `min_lift_atomic` | Round1 原子规则 lift 下限 | 1.05 | 降到 1.0 可多留原子 |
| `fallback_max_base_cov` / `fallback_min_lift_atomic` | 降级后上限/下限 | 0.5 / 1.05 | 原子不足 30 时自动启用 |

---

## 3. 允许 k=2 与更多推荐条数

| 参数 | 含义 | 当前 config |
|------|------|-------------|
| `min_rules_per_segment` | 每个客群至少规则数 | 2（允许 k=2 或 k=3） |
| `require_exact_k` | 是否只保留严格 k=3 | false |
| `max_segments` / `target_segment_count` | 目标客群数 | 5 / null |

---

## 4. 统一评分与风险分（排序与 AUC 对齐）

| 参数 | 含义 | 调参建议 |
|------|------|----------|
| `unified_scoring.w1_precision` | 精度权重 | 想更偏「准」可略调高（如 0.40） |
| `unified_scoring.w2_lift` | 提升度权重 | 想更偏「稀」可略调高（如 0.30） |
| `unified_scoring.w3_all_cov_penalty` | 覆盖率惩罚 | 可略降（如 0.10）以放宽 |
| `risk_scoring.use_rule_score_directly` | 风险分是否直接用规则得分 | true 时与规则排序一致，便于和下游 AUC 对比；false 时用 tier_weights |

---

## 5. 输出目标约束（准确率 / 覆盖率 / F1）

| 参数 | 含义 | 默认 |
|------|------|------|
| `target_precision_min` | 期望准确率下限 | 0 |
| `target_coverage_min` | 期望覆盖率下限 | 0 |
| `use_f1_balance` | 是否用 F1 平衡排序 | true |
| `target_priority` | f1 / precision_first / coverage_first | f1 |

---

## 6. 其他

- **max_segments / top_n_candidates**：调大可增加候选与最终推荐条数。
- **expected_cohort_ratio**：影响精度过滤；多圈人场景可保持 0.01 或由 Stage1 元数据计算。
- **dedup.same_structure_at_candidate / same_structure_at_portfolio**：设为 false 会多保留“同特征、不同阈值”的规则，候选更多但可能重复度略高。
