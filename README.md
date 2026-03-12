# Marketing Expansion Engine (Core)

基于**统计特征（非明细数据）**进行客群差异挖掘与可解释规则组合，输出字段阈值与枚举条件，
供统计人员在后台拉取用户清单，用于营销扩展触达。

------------------------------------------------------------------------

## 一、项目定位

本项目适用于：

-   只有"全量客群 vs 圈定客群"的字段统计结果
-   无法获取用户级明细数据
-   目标是输出"可执行规则"而非训练黑盒模型

输出结果为：

-   连续字段阈值区间
-   离散字段枚举集合
-   2～3 条规则组合（k=2 或 k=3）的可解释营销扩展方案

------------------------------------------------------------------------

## 二、系统架构

### Stage1：差异特征挖掘

功能：

-   连续特征差异计算（均值、分位数、IQR、CV 等）
-   单侧尾部阈值推荐（避免常态区间）
-   离散特征差异与推荐类别集合
-   输出字段级差异结果

输出：

-   numeric_diff\_{cohort}\_{date}.csv
-   categorical_diff\_{cohort}\_{date}.csv

------------------------------------------------------------------------

### Stage2：营销扩展规则引擎

功能：

1.  生成原子规则（优先 Stage1 候选族；不足时可放宽离散稳定性或增加连续低尾规则）
2.  原子规则经精度过滤（max_base_cov、min_lift_atomic）与自动降级
3.  Coverage-free Beam Search 生成 k=2 或 k=3 规则组合（含 rule_feature_ids / rule_signature）；min_rules_per_segment、require_exact_k 控制是否允许 k=2
4.  候选按 rule_signature 去重，再经统一评分（unified_scoring）与 Pair 缺失惩罚筛选
5.  两阶段选择（elite 强制 + 结构距离/锚点限流/family 覆盖）得到最终推荐；可选 risk_scoring 供下游 AUC 对比
6.  输出多组推荐营销扩展方案

输出：

-   atomic_rules\_{cohort}\_{date}.csv/json
-   candidate_segments\_{cohort}\_{date}.csv/json（**去重后**，含 rule_signature 与 dedup metadata）
-   segment_portfolio\_{cohort}\_{date}.json（含 **recommended** 最终推荐、**menu** 菜单全量、**selection_trace**；组合指标基于 recommended 重算）

详细输出结构与处理逻辑见 `docs/Stage2_输出说明.md`、`docs/Stage2_处理逻辑与输入输出说明.md`。

------------------------------------------------------------------------

## 三、使用方法

### 1. Stage1

``` bash
python run_stage1.py   --stat-date 202510   --full-dir ./data/full_stats   --cohort-dir ./data/cohort_stats   --output-dir ./data/stage1_output
```

### 2. Stage2

``` bash
python run_stage2.py   --stat-date 202510   --cohort-name sub   --stage1-output-dir ./data/stage1_output   --output-dir ./data/stage2_output
```

------------------------------------------------------------------------

## 四、关键配置说明（config.json）

### 1）覆盖率控制（控制名单规模）

-   max_base_cov：单条规则最大覆盖率
-   max_combo_cov_est：组合规则最大覆盖率
-   min_sub_cov：单条规则最小圈定覆盖

### 2）精度控制（控制误触率）

-   expected_cohort_ratio：圈定占比（必须正确填写）
-   min_combo_precision：组合最小估计精度

### 3）结构控制（可解释性）

-   max_features_per_segment：每个客群最多规则条数（如 3）
-   min_rules_per_segment：每个客群至少规则数（设为 2 时允许 k=2 或 k=3）
-   require_exact_k：是否只保留严格 k=max_features_per_segment 的候选（false 时允许 k=2）

### 4）排序与风险分

-   unified_scoring：统一评分权重（w1_precision、w2_lift、w3_all_cov_penalty 等），控制候选排序
-   risk_scoring.use_rule_score_directly：为 true 时风险分直接用规则得分，便于与下游 AUC 对齐

------------------------------------------------------------------------

## 五、设计原则

1.  只使用统计数据，不依赖明细样本
2.  优先使用"尾部阈值"，避免常态区间
3.  强调小覆盖 + 高浓度组合
4.  输出多组方案供营销选择
5.  所有规则必须可解释、可执行

------------------------------------------------------------------------

## 六、重要说明

本项目不是分类模型，不以 ROC/AUC 为目标。

优化方向是：

-   更小的组合覆盖率
-   更高的估计精度
-   更低的误触率
-   更强的业务可解释性

最终营销效果需在线上或抽样回流验证。

------------------------------------------------------------------------

## 七、适用场景

-   携号转网用户扩展
-   新产品订购用户扩展
-   宽带订购用户扩展
-   风险类客群识别扩展
-   其他基于统计对比的营销扩展场景

------------------------------------------------------------------------

## 八、项目目标

构建一个：

> 可解释、可执行、可控规模、可控浓度的统计驱动营销扩展引擎
