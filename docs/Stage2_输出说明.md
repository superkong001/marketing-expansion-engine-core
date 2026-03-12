# Stage2 输出说明

Stage2 规则组合会生成三类输出：**原子规则库**（`atomic_rules_*.csv/json`）、**候选客群规则**（`candidate_segments_*.csv/json`）、**推荐客群组合**（`segment_portfolio_*.json`）。CSV 表头与 JSON 键名为**业务可读**（中文），无需数据字典即可理解。本文档说明各文件含义、列/字段结构及如何阅读结果。

---

## 一、atomic_rules_*.csv / .json（原子规则库）

**含义**：由 Stage1 差异与阈值推荐生成的**单字段规则**，作为后续 Beam Search 组合的“积木”。每条规则对应一个字段的一个约束（数值区间或离散类别集合）。

### 列说明（CSV 导出为业务可读表头）

| 列名 | 含义 |
|------|------|
| 字段ID | 字段 ID（如 FACT_FLOW、MAIN_GPRS_VALUE） |
| 字段名称 | 字段中文名（如 套内流量、主套餐包含总流量(M)） |
| 特征类型 | 规则特征类型：numeric（连续）或 categorical（离散） |
| 规则子类型 | 规则子类型（如 main、tail；仅连续型常见） |
| 区间下界 / 区间上界 | 数值区间下界/上界（仅连续型有；离散为空） |
| 推荐方向 | 推荐方向：high=高值侧规则，low=低值侧规则（仅连续型） |
| 推荐类别集合 | 离散类别集合，逗号分隔（仅离散型有；连续为空） |
| 差异评分 | 多尺度统计差异评分（越大表示该字段“全量 vs 对比”差异越明显） |
| 稳定性评分 | 规则稳定性评分（基于 std/峰度/样本量等，越大越稳） |
| 分布类型 | 分布类型（仅连续型：与 Stage1 一致，为 symmetric / skewed / heavy_tail / zero_inflated） |
| 推荐理由代码 | 推荐理由代码（如 unknown） |
| 差异比 / 推荐类别个数 | 离散型时存在：差异比、推荐类别个数 |

### 阅读建议

- 按 **差异评分** 从大到小看，越靠前的单字段规则差异越强，优先参与组合。
- **稳定性评分** 过低时，该规则在业务中可能对波动敏感，组合时可酌情控制。
- 连续型看 **区间下界/区间上界** 与 **推荐方向**；离散型看 **推荐类别集合**。

---

## 二、candidate_segments_*.csv / .json（候选客群规则）

**含义**：由 **Coverage-free Beam Search** 生成、并经 **rule_signature 去重** 后的**多字段组合规则**（候选客群）。导出内容为**去重后**的候选列表，与日志中的 `after_dedup` 一致。每条记录 = 一个“AND”组合规则，可映射为 SQL WHERE 条件。

### CSV 列说明（当前导出为业务可读表头）

| 列名 | 含义 |
|------|------|
| 规则编号 | 规则组合唯一 ID（如 rule_3feat_68、rule_3feat_94） |
| 规则表达式 | 规则表达式（中文），如「triglyceride >= 162.0 AND serum creatinine >= 1.1 AND Gtp >= 46.0」 |
| SQL筛选条件 | 可直接用于查询的 SQL WHERE 子句（字段名为中文，AND 连接） |
| 差异评分 | 组合内各字段差异评分的平均 |
| 稳定性评分 | 组合内各字段稳定性评分的平均 |
| 多样性评分 | 与已有候选规则的结构多样性（0–1，越高越不重复） |
| 综合得分 | 综合得分（含 missing_pair 惩罚后），用于排序与筛选 |
| 包含字段数 | 组合包含的字段数（k，如 3） |
| 去重键（字段+阈值摘要） | **rule_signature**：字段+阈值规范化后的签名，用于去重与追溯 |

### JSON 结构（若导出）

- **candidate_segments**：数组，每项与 CSV 行一一对应，键名含英文（如 rule_id、rule_signature、divergence_score、score 等）。
- **metadata**（仅 JSON 存在）：
  - `pi_used`：使用的先验圈定占比；
  - `candidates_before_dedup`：去重前候选数；
  - `candidates_after_dedup`：去重后候选数（= 本文件条数）；
  - `dedup_removed`：去重删除数量。

每条候选记录均包含 **rule_signature** 字段，便于与日志及去重逻辑对照。

### 阅读建议

- 按 **综合得分** 从大到小看，靠前的是“单条质量”更好的候选客群。
- **rule_signature** 相同即视为同一条规则（仅字段顺序或表述不同），导出中仅保留 score 最大的一条。
- **SQL筛选条件** 可直接在受控环境执行，生成客群名单。

---

## 三、segment_portfolio_*.json（推荐客群组合）

**含义**：在候选客群中经**两阶段选择**（elite 强制 + 结构距离/锚点限流/family 覆盖）得到的**最终推荐方案**。每条推荐客群可为 **k=2 或 k=3** 条规则组合（由 min_rules_per_segment、require_exact_k 控制）。**组合指标**仅基于最终推荐集合（recommended）重算，与菜单（menu）区分。JSON 使用**业务可读键名**（中文），并保留英文键以兼容下游。

### 顶层结构

| 键 | 含义 |
|------|------|
| metadata | 元数据：pi_used、target_segment_count、selected_count、recommended_meaning 等 |
| selected_segments | 最终推荐客群列表（与 recommended 内容一致，兼容旧版） |
| recommended | **最终推荐的 1~K 条客群**（与 selected_segments 相同，推荐使用此键） |
| menu | 菜单全量：elite + standard + expand 按档位顺序合并，供菜单展示 |
| selection_trace | 选择过程（与 selection_steps 相同），调试/审计用 |
| selection_steps | 选择步骤（兼容旧版） |
| elite_forced | 是否强制选入过 elite |
| tiers | 按档位分层的客群（elite / standard / expand） |
| 组合指标 | 基于 **recommended** 重算的整体指标（见下） |

### 组合指标说明

**组合指标** 由 **recommended** 中的客群重算得到（非预选 portfolio），保证与最终推荐一致。

| 键 | 含义 |
|------|------|
| 客群数量 | 推荐客群数量（= len(recommended)） |
| 平均差异评分 | 各推荐客群差异评分的平均 |
| 平均稳定性评分 | 各推荐客群稳定性评分的平均 |
| 平均多样性评分 | 各推荐客群多样性评分的平均 |
| 平均综合得分 | 各推荐客群综合得分的平均 |
| 最小结构距离 | 推荐客群两两之间的最小结构距离（基于 rule_feature_ids 的 Jaccard 距离，越大越不相似） |

### recommended / selected_segments 中每条说明

| 键 | 含义 |
|------|------|
| rule_id | 规则 ID（与 candidate_segments 一致） |
| segment_id | 客群编号（如 segment_1、segment_2） |
| k | 规则数（字段数） |
| rule_feature_ids | 字段 ID 列表（用于结构距离与去重） |

### menu / tiers 中每个客群说明（中文键）

| 键 | 含义 |
|------|------|
| 客群编号 | 客群编号 |
| 规则编号 | 对应的候选规则 ID |
| tier | 档位：elite / standard / expand |
| 规则明细 | 该客群的字段规则列表（见下） |
| SQL筛选条件 | 该客群的 SQL WHERE 子句 |
| 评分指标 | 差异/稳定性/多样性/综合得分 |

### 规则明细中每条规则说明

| 键 | 含义 |
|------|------|
| 字段ID | 字段 ID |
| 字段名称 | 字段中文名 |
| 类型 | numeric（连续）或 categorical（离散） |
| 下界 / 上界 | 数值区间（仅连续型） |
| 方向 | high / low（仅连续型） |
| 类别集合 | 类别列表（仅离散型） |

### 阅读建议

- **recommended**（或 selected_segments）即最终要交付的“多客群”列表；**menu** 为菜单全量，条数可能多于 recommended。
- **最小结构距离** 基于字段集合 Jaccard 距离，越大说明推荐客群之间越不相似。
- 将 **SQL筛选条件**（在 menu/tiers 的规则明细中）交给下游在受控环境执行，即可得到各客群用户清单。

---

## 四、文件在流程中的位置

- **atomic_rules_***：Stage2 内部使用，也可供规则引擎或人工查看单字段规则质量；原子规则不足时可放宽离散稳定性阈值或为连续字段增加低尾规则以扩大池子。
- **candidate_segments_***：Beam Search 后经 **rule_signature 去重** 的候选，与日志 `after_dedup` 一致；含 **rule_signature** 与 metadata 中的 dedup 统计；其 **sql_where_clause** 可单独执行。
- **segment_portfolio_***：**最终推荐客群组合**；**recommended** 为两阶段选择后的 1~K 条，**menu** 为菜单全量，**组合指标** 基于 recommended 重算；供业务选择并落地执行 SQL 生成名单。

**总结**：Stage2 为 **coverage-free**，不依赖覆盖率预测；评分使用差异、稳定性、多样性及 Pair 缺失惩罚。候选导出为去重后列表，与日志一致；Portfolio 明确区分 menu（菜单）与 recommended（最终推荐），组合指标与推荐集合一致。按 **综合得分** 与 **组合指标** 中的 **最小结构距离** 理解质量与区分度即可。
