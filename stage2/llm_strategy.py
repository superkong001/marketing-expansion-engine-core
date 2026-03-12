"""
LLM 受控选择与组合策略模块

从候选规则库中选择和组合规则，生成可运营的客群方案
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import requests
from dataclasses import dataclass, asdict

try:
    from .stage2_config import Stage2Config
    from .rule_combination import SegmentRule
    from .rule_conflict_checker import check_rule_conflicts
    from .rule_output import generate_sql_where_clause
except ImportError:
    from stage2_config import Stage2Config
    from rule_combination import SegmentRule
    from rule_conflict_checker import check_rule_conflicts
    from rule_output import generate_sql_where_clause

logger = logging.getLogger(__name__)

# Stage2 导出 CSV 使用中文表头，此处反向映射以便统一用内部列名
ATOMIC_RULES_HEADER_CN_TO_INTERNAL = {
    "字段ID": "column_id",
    "字段名称": "column_name",
    "特征类型": "rule_type_feature",
    "规则子类型": "rule_type",
    "区间下界": "rule_low",
    "区间上界": "rule_high",
    "推荐方向": "direction",
    "推荐类别集合": "rule_categories",
    "差异评分": "divergence_score",
    "稳定性评分": "stability_score",
    "分布类型": "distribution_type",
    "全量覆盖率": "base_cov",
    "圈定覆盖率": "sub_cov",
    "Lift": "lift",
    "精度估计": "precision_est",
    "误判率估计": "fp_rate_est",
    "覆盖率未知": "cov_unknown",
}
CANDIDATE_SEGMENTS_HEADER_CN_TO_INTERNAL = {
    "规则编号": "rule_id",
    "规则表达式": "rule_expression",
    "SQL筛选条件": "sql_where_clause",
    "差异评分": "divergence_score",
    "稳定性评分": "stability_score",
    "综合得分": "score",
    "包含字段数": "feature_count",
    "规则数": "k",
}


def _normalize_atomic_rules_df(df: pd.DataFrame) -> pd.DataFrame:
    """若 CSV 为中文表头，统一为内部列名，便于后续使用 column_id / rule_low 等。"""
    if "column_id" in df.columns:
        return df
    rename = {cn: internal for cn, internal in ATOMIC_RULES_HEADER_CN_TO_INTERNAL.items() if cn in df.columns}
    if rename:
        df = df.rename(columns=rename)
        logger.debug("原子规则表列名已从中文映射为内部列名: %s", list(rename.keys()))
    return df


def _normalize_candidate_segments_df(df: pd.DataFrame) -> pd.DataFrame:
    """若 CSV 为中文表头，统一为内部列名。"""
    if "rule_id" in df.columns:
        return df
    rename = {cn: internal for cn, internal in CANDIDATE_SEGMENTS_HEADER_CN_TO_INTERNAL.items() if cn in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


@dataclass
class LLMSegment:
    """LLM 生成的客群数据结构"""
    segment_name: str
    segment_desc: str
    rules: List[Dict[str, Any]]
    sql_where: Optional[str] = None
    rationale: str = ""
    coverage_status: str = "待验证"  # 禁止输出预测覆盖率，只能标记"待验证"
    coverage_rate: Optional[float] = None  # SQL实测覆盖率（可选）
    overlap_rate: Optional[float] = None  # 与其他客群的重叠率（可选）
    segment_definition: str = ""  # 根据筛选条件提炼的简短定义，用于客群编号后缀，如「亚健康人群」


class LLMConfig:
    """LLM 配置类"""
    def __init__(self, config_file: str = "llm_conf.env"):
        """从配置文件加载 LLM 配置"""
        config_path = Path(__file__).parent.parent / config_file
        if not config_path.exists():
            # 尝试从当前目录查找
            config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"LLM 配置文件不存在: {config_file}")
        
        # 读取环境变量文件
        self.model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen3-8B")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        
        # 从文件读取（覆盖环境变量）
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key == "LLM_MODEL_ID":
                        self.model_id = value
                    elif key == "LLM_API_KEY":
                        self.api_key = value
                    elif key == "LLM_BASE_URL":
                        self.base_url = value
                    elif key == "LLM_TIMEOUT":
                        self.timeout = int(value)
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")


def _truncate_for_log(text: str, head: int = 800, tail: int = 200) -> str:
    """日志用：过长则截断为 前 head + ... [总长 N 字符] + 后 tail。"""
    if not text or len(text) <= head + tail + 50:
        return text
    return f"{text[:head]}\n... [共 {len(text)} 字符，已截断] ...\n{text[-tail:]}"


def call_llm_api(prompt: str, config: LLMConfig, max_tokens: int = 2000) -> str:
    """
    调用 LLM API
    
    Args:
        prompt: 提示词
        config: LLM 配置
        max_tokens: 最大生成 token 数
    
    Returns:
        LLM 返回的文本
    """
    # prompt 较长时自动提高读超时，避免 60s 下长回复超时
    request_timeout = config.timeout
    if len(prompt) > 4000 and config.timeout < 120:
        request_timeout = 120
        logger.info("prompt 较长（%d 字符），请求超时已设为 %d 秒", len(prompt), request_timeout)
    logger.info("调用大模型: model=%s, base_url=%s, max_tokens=%d, timeout=%ds", config.model_id, config.base_url, max_tokens, request_timeout)
    logger.info("输入 prompt（摘要）:\n%s", _truncate_for_log(prompt))
    url = f"{config.base_url}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.model_id,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的客群分析专家，擅长从候选规则库中选择和组合规则，生成可运营的客群方案。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3  # 降低随机性，提高可控性
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=request_timeout)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            logger.info("大模型输出（摘要）:\n%s", _truncate_for_log(content))
            return content
        else:
            raise ValueError(f"LLM API 返回格式异常: {result}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API 调用失败: {e}")
        raise


def load_candidate_rules(
    atomic_rules_path: str,
    candidate_segments_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    加载候选规则库
    
    Args:
        atomic_rules_path: 原子规则库 CSV 路径
        candidate_segments_path: 候选客群规则 CSV 路径（可选）
    
    Returns:
        (atomic_rules_df, candidate_segments_df)
    """
    atomic_rules_df = pd.read_csv(atomic_rules_path, encoding='utf-8-sig')
    atomic_rules_df = _normalize_atomic_rules_df(atomic_rules_df)
    logger.info(f"加载原子规则库: {len(atomic_rules_df)} 条规则")
    
    candidate_segments_df = None
    if candidate_segments_path and Path(candidate_segments_path).exists():
        candidate_segments_df = pd.read_csv(candidate_segments_path, encoding='utf-8-sig')
        candidate_segments_df = _normalize_candidate_segments_df(candidate_segments_df)
        logger.info(f"加载候选客群规则: {len(candidate_segments_df)} 条规则")
    
    return atomic_rules_df, candidate_segments_df


def load_segment_portfolio(portfolio_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    可选加载 segment_portfolio JSON，用于在 prompt 中提供「当前统计推荐方案」参考。

    Args:
        portfolio_path: segment_portfolio_*.json 文件路径（可选）

    Returns:
        若路径存在且含 recommended / recommended_rule_details 则返回
        {"recommended": [...], "recommended_rule_details": [...], "metadata": {...}}；
        否则返回 None。
    """
    if not portfolio_path or not Path(portfolio_path).exists():
        return None
    try:
        with open(portfolio_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        recommended = data.get('recommended') or []
        details = data.get('recommended_rule_details') or []
        metadata = data.get('metadata') or {}
        if not recommended and not details:
            return None
        logger.info(f"加载统计推荐方案: {portfolio_path}，推荐条数 {len(recommended)}，明细条数 {len(details)}")
        return {
            "recommended": recommended,
            "recommended_rule_details": details,
            "metadata": metadata,
        }
    except Exception as e:
        logger.warning(f"加载 segment_portfolio 失败，将不注入统计推荐方案: {e}")
        return None


def load_metadata(metadata_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载元数据（字段中文名、枚举值释义、业务分组）
    
    Args:
        metadata_path: 元数据 JSON 文件路径（可选）
    
    Returns:
        元数据字典
    """
    metadata = {
        "column_names": {},  # column_id -> 中文名
        "enum_values": {},  # column_id -> {value: 释义}
        "business_groups": {}  # column_id -> 业务分组（消费/套餐/终端/位置等）
    }
    
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata.update(json.load(f))
        logger.info(f"加载元数据: {metadata_path}")
    else:
        logger.warning("未提供元数据文件，使用默认值")
    
    return metadata


def _format_portfolio_for_prompt(portfolio_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """将 recommended + recommended_rule_details 转成一段说明文字，用于拼进 prompt。"""
    recommended = portfolio_data.get("recommended") or []
    details = portfolio_data.get("recommended_rule_details") or []
    meta = portfolio_data.get("metadata") or {}
    lines = []
    # 按 rule_id 对齐 recommended 与 details
    details_by_id = {d.get("rule_id"): d for d in details if d.get("rule_id")}
    for i, rec in enumerate(recommended[:20], 1):
        rule_id = rec.get("rule_id", "")
        feat_ids = rec.get("rule_feature_ids", [])
        detail = details_by_id.get(rule_id, {})
        rule_desc = detail.get("rule_desc", "")
        rules_list = detail.get("rules", [])
        prec_est = detail.get("combo_precision_est")
        kappa = detail.get("kappa")
        parts = [f"  {i}. rule_id={rule_id}, 特征={feat_ids}"]
        if rule_desc:
            parts.append(f"     筛选条件: {rule_desc}")
        if rules_list:
            rule_parts = []
            for r in rules_list:
                fid = r.get("字段ID", r.get("column_id", ""))
                fname = r.get("字段名称", fid)
                typ = r.get("类型", "numeric")
                if typ == "numeric":
                    lo, hi = r.get("下界"), r.get("上界")
                    if hi is None or (isinstance(hi, float) and (hi == float('inf') or np.isposinf(hi))):
                        rule_parts.append(f"{fname}>={lo}")
                    else:
                        rule_parts.append(f"{fname}∈[{lo},{hi})")
                else:
                    cat = r.get("类别集合", r.get("categories", []))
                    rule_parts.append(f"{fname}∈{cat}")
            parts.append(f"     规则明细: {'; '.join(rule_parts)}")
        if prec_est is not None:
            parts.append(f"     组合精度估计: {prec_est:.3f}")
        if kappa is not None:
            parts.append(f"     kappa: {kappa:.3f}")
        lines.append("\n".join(parts))
    if meta:
        lines.append(f"  统计说明: selected_count={meta.get('selected_count')}, reason={meta.get('reason', '')}")
    return "\n".join(lines)


def build_candidate_prompt(
    atomic_rules_df: pd.DataFrame,
    candidate_segments_df: Optional[pd.DataFrame],
    metadata: Dict[str, Any],
    config: Stage2Config,
    max_segments: int = 5,
    segment_portfolio_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    构建受控的 LLM Prompt

    Args:
        atomic_rules_df: 原子规则库 DataFrame
        candidate_segments_df: 候选客群规则 DataFrame（可选）
        metadata: 元数据字典
        config: Stage2 配置对象
        max_segments: 最大生成客群数
        segment_portfolio_data: 可选，load_segment_portfolio 的返回值，用于注入「当前统计推荐方案」

    Returns:
        Prompt 字符串
    """
    # 构建原子规则库描述
    atomic_rules_desc = []
    for _, row in atomic_rules_df.iterrows():
        col_id = row.get('column_id', '')
        col_name = metadata.get("column_names", {}).get(col_id, row.get('column_name', col_id))
        rule_type = row.get('rule_type_feature', 'numeric')
        
        if rule_type == 'numeric':
            low = row.get('rule_low', np.nan)
            high = row.get('rule_high', np.nan)
            direction = row.get('direction', 'high')
            div_score = row.get('divergence_score', 0.0)
            stab_score = row.get('stability_score', 0.0)
            
            if pd.notna(low) and pd.notna(high):
                is_inf = high == float("inf") or (isinstance(high, str) and str(high).strip().lower() == "inf")
                if is_inf:
                    rule_desc = f"{col_name} >= {low}"
                else:
                    rule_desc = f"{col_name} ∈ [{low}, {high})"
            else:
                rule_desc = f"{col_name} (数值区间)"
            
            atomic_rules_desc.append(
                f"- 字段: {col_name} ({col_id}), 规则: {rule_desc}, "
                f"方向: {direction}, 差异评分: {div_score:.3f}, 稳定性: {stab_score:.3f}"
            )
        
        elif rule_type == 'categorical':
            categories = row.get('rule_categories', '')
            div_score = row.get('divergence_score', 0.0)
            stab_score = row.get('stability_score', 0.0)
            
            if categories is not None and not (isinstance(categories, float) and np.isnan(categories)):
                if isinstance(categories, str):
                    cat_list = [c.strip() for c in categories.split(',') if c.strip()]
                elif hasattr(categories, '__iter__') and not isinstance(categories, str):
                    cat_list = [str(c).strip() for c in categories]
                else:
                    cat_list = [str(categories)]
                rule_desc = f"{col_name} ∈ {{{', '.join(cat_list)}}}" if cat_list else f"{col_name} (类别集合)"
            else:
                rule_desc = f"{col_name} (类别集合)"
            
            atomic_rules_desc.append(
                f"- 字段: {col_name} ({col_id}), 规则: {rule_desc}, "
                f"差异评分: {div_score:.3f}, 稳定性: {stab_score:.3f}"
            )
    
    # 构建候选客群规则描述（如果存在）
    candidate_segments_desc = []
    if candidate_segments_df is not None and len(candidate_segments_df) > 0:
        for _, row in candidate_segments_df.iterrows():
            rule_id = row.get('rule_id', '')
            rule_expr = row.get('rule_expression', '')
            div_score = row.get('divergence_score', 0.0)
            stab_score = row.get('stability_score', 0.0)
            score = row.get('score', 0.0)
            
            candidate_segments_desc.append(
                f"- 规则ID: {rule_id}, 表达式: {rule_expr}, "
                f"差异评分: {div_score:.3f}, 稳定性: {stab_score:.3f}, 综合得分: {score:.3f}"
            )
    
    # 限制显示规则数量，避免 prompt 过长
    max_atomic_rules_show = min(50, len(atomic_rules_desc))
    atomic_rules_show = atomic_rules_desc[:max_atomic_rules_show]
    
    prompt = f"""你是一个专业的客群分析专家。请从以下候选规则库中选择和组合规则，生成可运营的客群方案。建议数量约 {max_segments} 个，可根据候选规则与统计推荐方案灵活调整：若组合空间有限可少于该数，若空间充足也可略多。

## 候选规则库

### 原子规则库（可组合使用）：
{chr(10).join(atomic_rules_show)}
{f'... (共 {len(atomic_rules_df)} 条原子规则，仅显示前 {max_atomic_rules_show} 条)' if len(atomic_rules_desc) > max_atomic_rules_show else ''}

"""
    
    if candidate_segments_desc:
        max_candidate_show = min(20, len(candidate_segments_desc))
        candidate_show = candidate_segments_desc[:max_candidate_show]
        prompt += f"""### 候选客群规则（可直接使用或参考）：
{chr(10).join(candidate_show)}
{f'... (共 {len(candidate_segments_df)} 条候选客群规则，仅显示前 {max_candidate_show} 条)' if len(candidate_segments_desc) > max_candidate_show else ''}

"""
    
    if segment_portfolio_data:
        portfolio_text = _format_portfolio_for_prompt(segment_portfolio_data, metadata)
        prompt += f"""## 当前统计推荐方案（请在此基础上优化或保留）

以下为 Stage2 两阶段选择得到的统计推荐方案，请在此基础上生成客群方案：可保留、微调或替换规则，并**根据每条客群的筛选条件起一个与之对应的、可运营的客群名称**。

```
{portfolio_text}
```

"""
    
    prompt += f"""## 约束条件（必须严格遵守）

### A. 候选空间约束（必须）
1. **只能从候选规则库中选择**：不得生成新字段、新阈值数值。
2. **数值阈值**：只能使用候选规则中的 (rule_low, rule_high) 区间。**重要**：规则「字段 >= 某值」（无上界）必须写成 low=该数值、high=null 或不写 high；不要写成 low=null、high=该数值（那是小于等于，会校验失败）。
3. **离散阈值**：只能使用候选规则中的 rule_categories 或其子集（限制最大类别数不超过 {config.max_categories}）。

### B. 规则结构约束（必须）
1. **字段数量限制**：每个客群最多 {config.max_features_per_segment} 个字段。
2. **字段唯一性**：同一字段在一个客群中只能出现一次（禁止重复约束）。
3. **无矛盾规则**：不允许互相矛盾的规则（如 age<30 且 age>=40）。

### C. 解释与命名约束（必须）
1. **客群命名**：必须**根据该客群的筛选条件（规则表达式）**起名，使名称与条件一一对应、可读可运营；不得使用模型术语（如"高KL散度人群"）。
2. **命名建议**：{{差异特征关键词}}{{场景词}}{{人群}}（例：甘油三酯与转氨酶偏高体重偏大人群、高ARPU重度流量用户）
3. **解释要求**：必须引用候选指标（差异评分/稳定性/差异方向），不得凭空编造。

### D. 禁止误导约束（必须）
1. **禁止输出覆盖率估计**：不得输出任何覆盖率/命中率的"估计值"。
2. **覆盖率状态**：若缺少 SQL 实测结果，只能输出"待验证"标签。

## 输出格式要求

请以 JSON 格式输出，每个客群包含以下字段：

```json
{{
  "segments": [
    {{
      "segment_name": "根据本客群筛选条件起的可运营名称（与规则表达式对应）",
      "segment_definition": "根据筛选条件提炼的简短定义，4-8字，如「亚健康人群」「高指标人群」，用于客群编号后缀（必填）",
      "segment_desc": "一句话解释（引用候选指标）",
      "rules": [
        {{
          "column_id": "字段ID（必须来自候选规则库）",
          "column_name": "字段中文名",
          "type": "numeric 或 categorical",
          "low": 数值下界（>= 某值时填该值；无下界时不填或null）,
          "high": 数值上界（<= 某值时填该值；无上界时不填或null）。例：triglyceride>=120 应写 low:120, high:null",
          "categories": ["类别1", "类别2"]（categorical类型，可选）,
          "direction": "high 或 low"（numeric类型，可选）
        }}
      ],
      "rationale": "选择理由（引用候选指标：差异评分、稳定性、差异方向等）"
    }}
  ]
}}
```

## 任务

请生成客群方案（数量可少可多，建议约 {max_segments} 个），确保：
1. 所有规则都来自候选规则库
2. 每个客群最多 {config.max_features_per_segment} 个字段
3. **每个客群的 segment_name 须根据该客群的筛选条件（规则）起名，名称与条件对应、可运营**
4. **每个客群必须填写 segment_definition**：根据本客群筛选条件提炼的简短定义（4-8 字），如「亚健康人群」「高指标人群」，将用于客群编号（格式：序号_定义，如 1_亚健康人群）
5. 解释引用候选指标
6. 不输出覆盖率估计值

请直接输出 JSON，不要包含其他说明文字。"""
    
    return prompt


def parse_llm_response(response_text: str) -> List[Dict[str, Any]]:
    """
    解析 LLM 返回的 JSON 响应
    
    Args:
        response_text: LLM 返回的文本
    
    Returns:
        客群列表
    """
    # 尝试提取 JSON 部分
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start == -1 or json_end == 0:
        raise ValueError("LLM 响应中未找到 JSON 格式")
    
    json_text = response_text[json_start:json_end]
    
    try:
        data = json.loads(json_text)
        if "segments" in data:
            return data["segments"]
        else:
            # 如果没有 segments 字段，尝试直接解析为列表
            if isinstance(data, list):
                return data
            else:
                raise ValueError("LLM 响应格式不正确：缺少 segments 字段")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        logger.error(f"响应文本: {response_text}")
        raise


def validate_segment_rules(
    segment: Dict[str, Any],
    atomic_rules_df: pd.DataFrame,
    config: Stage2Config
) -> Tuple[bool, str]:
    """
    验证客群规则是否符合约束
    
    Args:
        segment: 客群字典
        atomic_rules_df: 原子规则库 DataFrame
        config: Stage2 配置对象
    
    Returns:
        (是否有效, 错误信息)
    """
    rules = segment.get("rules", [])
    
    # 检查字段数量
    if len(rules) > config.max_features_per_segment:
        return False, f"字段数量超过限制: {len(rules)} > {config.max_features_per_segment}"
    
    # 检查字段唯一性
    column_ids = [r.get("column_id") for r in rules]
    if len(column_ids) != len(set(column_ids)):
        return False, "存在重复字段"
    
    # 检查规则是否来自候选规则库
    for rule in rules:
        col_id = rule.get("column_id")
        rule_type = rule.get("type")
        
        # 在原子规则库中查找匹配的规则（先 column_id，再 column_name，便于 LLM 写中文名如 weight(kg)）
        matching_rules = atomic_rules_df[
            (atomic_rules_df['column_id'] == col_id) &
            (atomic_rules_df['rule_type_feature'] == rule_type)
        ]
        if len(matching_rules) == 0 and 'column_name' in atomic_rules_df.columns:
            matching_rules = atomic_rules_df[
                (atomic_rules_df['column_name'] == col_id) &
                (atomic_rules_df['rule_type_feature'] == rule_type)
            ]
        
        if len(matching_rules) == 0:
            return False, f"字段 {col_id} 不在候选规则库中"
        
        # 验证数值区间是否匹配
        if rule_type == "numeric":
            low = rule.get("low")
            high = rule.get("high")
            direction = (rule.get("direction") or "").strip().lower()
            # 规范 LLM 输出的区间与原子库一致：high 方向 [X, +inf)，low 方向 (-inf, X]
            if low is None and high is not None:
                if direction == "high":
                    # LLM 误写 (None, X) 为「>= X」→ (X, inf)
                    low, high = high, None
                    logger.debug("校验兼容: 规则 (None, X) 按 direction=high 规范为 [X, +inf)")
                # direction=low 时 (None, X) 已是 (-inf, X]，不处理
            elif low is not None and high is None and direction == "low":
                # LLM 误写 (X, None) 为「<= X」→ (-inf, X]
                high, low = low, None
                logger.debug("校验兼容: 规则 (X, None) 按 direction=low 规范为 (-inf, X]")
            
            def _norm_bound(v: Any) -> Any:
                """CSV 可能将 inf 存成字符串，统一为 float。"""
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                if isinstance(v, str):
                    v = v.strip().lower()
                    if v == "inf":
                        return float("inf")
                    if v in ("-inf", "负无穷"):
                        return float("-inf")
                    try:
                        return float(v)
                    except ValueError:
                        return v
                return v

            # 检查是否与候选规则中的区间匹配
            matched = False
            for _, candidate_row in matching_rules.iterrows():
                candidate_low = _norm_bound(candidate_row.get("rule_low"))
                candidate_high = _norm_bound(candidate_row.get("rule_high"))
                
                # 处理 NaN 和 inf 的情况；上界「无」在库里可能是 inf，LLM 可能返回 None，视为等价
                def safe_compare(val1, val2, tolerance=1e-6):
                    """安全比较两个值（处理 NaN 和 inf）；None 与 ±inf 按无界等价。"""
                    if pd.isna(val1) and pd.isna(val2):
                        return True
                    if val1 is None or (isinstance(val1, float) and np.isnan(val1)):
                        val1 = None
                    if val2 is None or (isinstance(val2, float) and np.isnan(val2)):
                        val2 = None
                    if val1 is None and val2 is None:
                        return True
                    # 无上界：None 与 +inf 等价
                    if val1 is None and val2 == float("inf"):
                        return True
                    if val2 is None and val1 == float("inf"):
                        return True
                    # 无下界：None 与 -inf 等价
                    if val1 is None and val2 == float("-inf"):
                        return True
                    if val2 is None and val1 == float("-inf"):
                        return True
                    if val1 == float("inf") and val2 == float("inf"):
                        return True
                    if val1 == float("-inf") and val2 == float("-inf"):
                        return True
                    if val1 is None or val2 is None:
                        return False
                    if val1 == float("inf") or val2 == float("inf"):
                        return False
                    if val1 == float("-inf") or val2 == float("-inf"):
                        return False
                    return abs(float(val1) - float(val2)) < tolerance
                
                # 比较下界
                if not safe_compare(candidate_low, low):
                    continue
                
                # 比较上界
                if not safe_compare(candidate_high, high):
                    continue
                
                matched = True
                break
            
            if not matched:
                return False, f"字段 {col_id} 的数值区间 ({low}, {high}) 不在候选规则库中"
        
        # 验证类别集合是否匹配
        elif rule_type == "categorical":
            categories = set(rule.get("categories", []))
            
            # 检查是否与候选规则中的类别集合匹配（可以是子集）
            matched = False
            for _, candidate_row in matching_rules.iterrows():
                raw = candidate_row.get('rule_categories', '')
                if pd.isna(raw):
                    continue
                candidate_categories_str = str(raw).strip()
                if not candidate_categories_str or candidate_categories_str.lower() == 'nan':
                    continue
                candidate_categories = set(
                    [c.strip() for c in candidate_categories_str.split(',') if c.strip()]
                )
                
                # 允许是子集
                if categories.issubset(candidate_categories):
                    matched = True
                    break
            
            if not matched:
                return False, f"字段 {col_id} 的类别集合 {categories} 不在候选规则库中"
    
    return True, ""


def convert_to_segment_rule(segment: Dict[str, Any], metadata: Dict[str, Any]) -> SegmentRule:
    """
    将 LLM 生成的客群字典转换为 SegmentRule 对象
    
    Args:
        segment: 客群字典
        metadata: 元数据字典
    
    Returns:
        SegmentRule 对象
    """
    feature_rules = []
    
    for rule in segment.get("rules", []):
        col_id = rule.get("column_id")
        col_name = metadata.get("column_names", {}).get(
            col_id, rule.get("column_name", col_id)
        )
        
        feature_rule = {
            "column_id": col_id,
            "column_name": col_name,
            "type": rule.get("type")
        }
        
        if rule.get("type") == "numeric":
            r_low, r_high = rule.get("low"), rule.get("high")
            # 与校验一致：LLM 误写 (None, X) 视为 >= X，即 (X, null)
            if r_low is None and r_high is not None:
                r_low, r_high = r_high, None
            feature_rule["low"] = r_low
            feature_rule["high"] = r_high
            feature_rule["direction"] = rule.get("direction", "high")
        elif rule.get("type") == "categorical":
            feature_rule["categories"] = rule.get("categories", [])
        
        feature_rules.append(feature_rule)
    
    return SegmentRule(
        rule_id=f"llm_segment_{hash(str(segment.get('segment_name', '')))}",
        feature_rules=feature_rules,
        divergence_score=0.0,  # LLM 生成的规则不计算评分
        stability_score=0.0,
        diversity_score=0.0,
        score=0.0
    )


def generate_llm_segments(
    atomic_rules_path: str,
    candidate_segments_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    segment_portfolio_path: Optional[str] = None,
    config: Stage2Config = None,
    llm_config: Optional[LLMConfig] = None,
    max_segments: int = 5
) -> List[LLMSegment]:
    """
    使用 LLM 生成客群方案

    Args:
        atomic_rules_path: 原子规则库 CSV 路径
        candidate_segments_path: 候选客群规则 CSV 路径（可选）
        metadata_path: 元数据 JSON 路径（可选）
        segment_portfolio_path: segment_portfolio_*.json 路径（可选），注入「当前统计推荐方案」参考
        config: Stage2 配置对象
        llm_config: LLM 配置对象（可选）
        max_segments: 最大生成客群数

    Returns:
        LLM 生成的客群列表
    """
    if config is None:
        config = Stage2Config()

    if llm_config is None:
        llm_config = LLMConfig()

    # 加载数据
    logger.info("加载候选规则库和元数据...")
    atomic_rules_df, candidate_segments_df = load_candidate_rules(
        atomic_rules_path, candidate_segments_path
    )
    metadata = load_metadata(metadata_path)
    segment_portfolio_data = load_segment_portfolio(segment_portfolio_path)

    # 构建 Prompt
    logger.info("构建 LLM Prompt...")
    prompt = build_candidate_prompt(
        atomic_rules_df, candidate_segments_df, metadata, config, max_segments,
        segment_portfolio_data=segment_portfolio_data,
    )
    
    # 调用 LLM
    logger.info("调用 LLM API...")
    response_text = call_llm_api(prompt, llm_config)
    
    # 解析响应
    logger.info("解析 LLM 响应...")
    segments_data = parse_llm_response(response_text)
    
    # 验证和转换
    llm_segments = []
    validated_segment_rules = []  # 用于冲突检查
    
    for idx, segment_data in enumerate(segments_data):
        # 验证规则约束
        is_valid, error_msg = validate_segment_rules(segment_data, atomic_rules_df, config)
        if not is_valid:
            logger.warning(f"客群 '{segment_data.get('segment_name', '')}' 验证失败: {error_msg}")
            continue
        
        # 转换为 SegmentRule 对象（用于生成 SQL）
        segment_rule = convert_to_segment_rule(segment_data, metadata)
        
        # 检查规则冲突（与已生成的客群）
        conflicts = check_rule_conflicts(segment_rule, validated_segment_rules)
        
        if conflicts:
            logger.warning(f"客群 '{segment_data.get('segment_name', '')}' 存在规则冲突: {conflicts}")
            # 可以选择跳过或继续（这里选择继续，但记录警告）
        
        # 生成 SQL WHERE 子句
        column_name_map = metadata.get("column_names", {})
        sql_where = generate_sql_where_clause(segment_rule, column_name_map)
        
        # 构建 LLMSegment 对象
        llm_segment = LLMSegment(
            segment_name=segment_data.get("segment_name", ""),
            segment_desc=segment_data.get("segment_desc", ""),
            rules=segment_data.get("rules", []),
            sql_where=sql_where,
            rationale=segment_data.get("rationale", ""),
            coverage_status="待验证",  # 禁止输出预测覆盖率
            segment_definition=segment_data.get("segment_definition", "").strip() or segment_data.get("segment_name", ""),
        )
        
        llm_segments.append(llm_segment)
        validated_segment_rules.append(segment_rule)  # 添加到已验证列表
    
    logger.info(f"成功生成 {len(llm_segments)} 个客群方案")
    return llm_segments


def _sanitize_segment_id_definition(s: str, max_len: int = 20) -> str:
    """将 segment_definition 或 segment_name 转为客群编号后缀：去首尾空白、空格替为下划线、截断过长。"""
    if not s or not str(s).strip():
        return "客群"
    t = str(s).strip().replace(" ", "_").replace("\t", "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t[:max_len] if len(t) > max_len else t


def _llm_segment_to_menu_item(segment: LLMSegment, idx: int, rule_id: str, segment_id: str) -> Dict[str, Any]:
    """将 LLMSegment 转为与 stage2 segment_portfolio menu 单项同构的字典。"""
    rules_detail = []
    for r in segment.rules:
        typ = r.get("type", "numeric")
        item = {
            "字段ID": r.get("column_id", ""),
            "字段名称": r.get("column_name", r.get("column_id", "")),
            "类型": "numeric" if typ == "numeric" else "categorical",
        }
        if typ == "numeric":
            item["下界"] = r.get("low")
            item["上界"] = r.get("high")
            item["方向"] = r.get("direction", "high")
            item["direction"] = r.get("direction", "high")
        else:
            item["类别集合"] = r.get("categories", [])
            item["direction"] = None
        item["feature_family"] = "other"
        rules_detail.append(item)
    return {
        "客群编号": segment_id,
        "规则编号": rule_id,
        "tier": "llm",
        "k": len(segment.rules),
        "rule_desc": segment.segment_desc or (segment.sql_where or ""),
        "SQL筛选条件": segment.sql_where or "",
        "规则明细": rules_detail,
        "评分指标": {
            "差异评分": 0.0,
            "稳定性评分": 0.0,
            "多样性评分": 0.0,
            "综合得分": 0.0,
        },
        "客群名称": segment.segment_name,
        "客群说明": segment.segment_desc,
    }


def build_portfolio_output(
    llm_segments: List[LLMSegment],
    max_segments: int,
) -> Dict[str, Any]:
    """
    构建与 stage2 segment_portfolio_*.json 同构的字典，便于下游用同一套逻辑读取。
    """
    n = len(llm_segments)
    selected_segments = []
    menu_list = []
    selection_steps = []
    for i, seg in enumerate(llm_segments):
        rule_id = f"llm_segment_{i + 1}"
        definition = seg.segment_definition or seg.segment_name
        seg_suffix = _sanitize_segment_id_definition(definition)
        segment_id = f"{i + 1}_{seg_suffix}"
        rule_feature_ids = [r.get("column_id", "") for r in seg.rules if r.get("column_id")]
        selected_segments.append({
            "rule_id": rule_id,
            "segment_id": segment_id,
            "k": len(seg.rules),
            "rule_feature_ids": rule_feature_ids,
        })
        menu_list.append(_llm_segment_to_menu_item(seg, i, rule_id, segment_id))
        selection_steps.append({
            "step": i + 1,
            "rule_id": rule_id,
            "reason": "llm_generated",
            "rule_desc": seg.segment_desc or (seg.sql_where or ""),
        })
    metadata = {
        "pi_used": None,
        "pi_source": "llm",
        "target_segment_count": max_segments,
        "selected_count": n,
        "recommended_meaning": "LLM 生成的客群方案",
    }
    if n != max_segments:
        metadata["reason"] = "LLM 生成数量与请求不一致"
    portfolio_metrics_cn = {
        "客群数量": n,
        "平均差异评分": 0.0,
        "平均稳定性评分": 0.0,
        "平均多样性评分": 0.0,
        "平均综合得分": 0.0,
        "最小结构距离": 1.0,
    }
    return {
        "metadata": metadata,
        "selected_segments": selected_segments,
        "recommended": selected_segments,
        "menu": menu_list,
        "selection_trace": selection_steps,
        "selection_steps": selection_steps,
        "selection_reason": "LLM 生成，无两阶段选择",
        "elite_forced": False,
        "tiers": {
            "elite": [],
            "standard": [],
            "expand": menu_list,
        },
        "组合指标": portfolio_metrics_cn,
    }


def export_llm_segments(
    llm_segments: List[LLMSegment],
    output_path: Path,
    format: str = 'json',
    max_segments: int = 5,
) -> None:
    """
    导出 LLM 生成的客群方案。
    format='json' 时输出与 stage2 segment_portfolio_*.json 同构的 JSON，便于下游统一读取。
    
    Args:
        llm_segments: LLM 生成的客群列表
        output_path: 输出文件路径（不含扩展名）
        format: 输出格式（'json' 或 'csv'）
        max_segments: 请求的最大客群数（用于 metadata.target_segment_count）
    """
    output_path = Path(output_path)
    
    if format == 'json':
        file_path = output_path.with_suffix('.json')
        portfolio_data = build_portfolio_output(llm_segments, max_segments)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(portfolio_data, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM 客群方案已导出到（与 segment_portfolio 同构）: {file_path}")
    
    elif format == 'csv':
        file_path = output_path.with_suffix('.csv')
        records = []
        for segment in llm_segments:
            # 将规则列表转换为字符串
            rules_str = json.dumps(segment.rules, ensure_ascii=False)
            
            records.append({
                'segment_name': segment.segment_name,
                'segment_desc': segment.segment_desc,
                'rules': rules_str,
                'sql_where': segment.sql_where,
                'rationale': segment.rationale,
                'coverage_status': segment.coverage_status
            })
        
        df = pd.DataFrame(records)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        logger.info(f"LLM 客群方案已导出到: {file_path}")
    
    else:
        raise ValueError(f"不支持的格式: {format}")


def load_sql_validation_results(validation_file: str) -> Dict[str, Dict[str, float]]:
    """
    加载SQL实测结果
    
    Args:
        validation_file: 实测结果文件路径（JSON或CSV）
    
    Returns:
        字典：segment_name -> {coverage_rate, overlap_rate, ...}
    """
    validation_path = Path(validation_file)
    if not validation_path.exists():
        logger.warning(f"实测结果文件不存在: {validation_file}")
        return {}
    
    if validation_path.suffix == '.json':
        with open(validation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 假设格式：{"segments": [{"segment_name": "...", "coverage_rate": 0.1, ...}, ...]}
            if "segments" in data:
                return {s.get("segment_name"): s for s in data["segments"]}
            else:
                return data
    elif validation_path.suffix == '.csv':
        df = pd.read_csv(validation_path, encoding='utf-8-sig')
        return df.set_index('segment_name').to_dict('index')
    else:
        raise ValueError(f"不支持的实测结果文件格式: {validation_path.suffix}")


def refine_llm_segments(
    llm_segments: List[LLMSegment],
    validation_results: Dict[str, Dict[str, float]],
    atomic_rules_df: pd.DataFrame,
    metadata: Dict[str, Any],
    config: Stage2Config,
    llm_config: LLMConfig
) -> List[LLMSegment]:
    """
    基于SQL实测结果进行二次精修
    
    精修策略：
    - 覆盖率过低：删字段、换备选阈值
    - 重叠率过高：改命名、调整规则
    - 覆盖率过高：可能需要更窄的阈值
    
    Args:
        llm_segments: 初始LLM生成的客群列表
        validation_results: SQL实测结果字典
        atomic_rules_df: 原子规则库DataFrame
        metadata: 元数据字典
        config: Stage2配置对象
        llm_config: LLM配置对象
    
    Returns:
        精修后的客群列表
    """
    if not validation_results:
        logger.warning("没有实测结果，跳过精修")
        return llm_segments
    
    # 构建精修Prompt
    segments_info = []
    for segment in llm_segments:
        validation = validation_results.get(segment.segment_name, {})
        coverage_rate = validation.get('coverage_rate', None)
        overlap_rate = validation.get('overlap_rate', None)
        
        segments_info.append({
            'segment_name': segment.segment_name,
            'segment_desc': segment.segment_desc,
            'rules': segment.rules,
            'coverage_rate': coverage_rate,
            'overlap_rate': overlap_rate,
            'sql_where': segment.sql_where
        })
    
    prompt = f"""你是一个专业的客群分析专家。需要对以下客群方案进行精修，基于SQL实测结果优化规则和命名。

## 当前客群方案及实测结果

{json.dumps(segments_info, ensure_ascii=False, indent=2)}

## 精修要求

1. **覆盖率过低（< 0.05）**：考虑删除部分字段、使用更宽的备选阈值
2. **重叠率过高（> 0.3）**：调整规则、改命名以区分
3. **覆盖率过高（> 0.5）**：考虑使用更窄的备选阈值
4. **规则必须来自候选规则库**：只能使用atomic_rules中的规则和阈值
5. **命名必须可运营**：不得使用模型术语

## 候选规则库（部分）

{json.dumps(atomic_rules_df.head(20).to_dict('records'), ensure_ascii=False, indent=2)}

## 输出格式

请以JSON格式输出精修后的客群方案，格式与输入相同，但需要：
- 调整rules（删字段、换备选阈值）
- 更新segment_name和segment_desc（如重叠率过高）
- 更新rationale（说明精修理由）

请直接输出JSON，不要包含其他说明文字。"""
    
    # 调用LLM
    logger.info("调用LLM进行二次精修...")
    response_text = call_llm_api(prompt, llm_config, max_tokens=3000)
    
    # 解析响应
    segments_data = parse_llm_response(response_text)
    
    # 转换并验证
    refined_segments = []
    for segment_data in segments_data:
        # 验证规则约束
        is_valid, error_msg = validate_segment_rules(segment_data, atomic_rules_df, config)
        if not is_valid:
            logger.warning(f"精修后的客群 '{segment_data.get('segment_name', '')}' 验证失败: {error_msg}")
            continue
        
        # 转换为SegmentRule对象
        segment_rule = convert_to_segment_rule(segment_data, metadata)
        
        # 生成SQL WHERE子句
        column_name_map = metadata.get("column_names", {})
        sql_where = generate_sql_where_clause(segment_rule, column_name_map)
        
        # 获取实测结果
        original_name = segment_data.get('original_segment_name', segment_data.get('segment_name', ''))
        validation = validation_results.get(original_name, {})
        
        # 构建精修后的LLMSegment
        refined_segment = LLMSegment(
            segment_name=segment_data.get("segment_name", ""),
            segment_desc=segment_data.get("segment_desc", ""),
            rules=segment_data.get("rules", []),
            sql_where=sql_where,
            rationale=segment_data.get("rationale", ""),
            coverage_status="已验证" if validation.get('coverage_rate') is not None else "待验证",
            coverage_rate=validation.get('coverage_rate'),
            overlap_rate=validation.get('overlap_rate'),
            segment_definition=segment_data.get("segment_definition", "").strip() or segment_data.get("segment_name", ""),
        )
        
        refined_segments.append(refined_segment)
    
    logger.info(f"精修完成，生成 {len(refined_segments)} 个精修后的客群方案")
    return refined_segments


def main():
    """主函数：命令行入口。JSON 输出与 stage2 segment_portfolio_*.json 同构，下游可统一读取。"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM 受控选择与组合策略：从候选规则库生成可运营的客群方案。输出 JSON 与 segment_portfolio 同构。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行示例（需先配置 llm_conf.env 中的 LLM_API_KEY 等）:

  # 仅原子规则，输出 llm_portfolio.json（与 segment_portfolio 同结构）
  python -m stage2.llm_strategy --atomic-rules ./data/stage2_output/atomic_rules_sub_202512.csv --output ./data/stage2_output/llm_portfolio_sub_202512

  # 原子规则 + 候选客群，生成 5 条客群，并导出 CSV
  python -m stage2.llm_strategy --atomic-rules ./data/stage2_output/atomic_rules_sub_202512.csv --candidate-segments ./data/stage2_output/candidate_segments_sub_202512.csv --max-segments 5 --output ./data/stage2_output/llm_portfolio_sub_202512 --format both

  # 带统计推荐方案参考（让 LLM 在此基础上优化并起名）
  python -m stage2.llm_strategy --atomic-rules ./data/stage2_output/atomic_rules_sub_202512.csv --candidate-segments ./data/stage2_output/candidate_segments_sub_202512.csv --segment-portfolio ./data/stage2_output/segment_portfolio_sub_202512.json --output ./data/stage2_output/llm_portfolio_sub_202512

  # 带元数据与实测结果精修
  python -m stage2.llm_strategy --atomic-rules ./data/stage2_output/atomic_rules_sub_202512.csv --candidate-segments ./data/stage2_output/candidate_segments_sub_202512.csv --metadata ./data/column_meta.json --validation-results ./data/sql_validation.json --output ./data/stage2_output/llm_portfolio_sub_202512
"""
    )
    
    parser.add_argument(
        '--atomic-rules',
        type=str,
        required=True,
        help='原子规则库 CSV 文件路径'
    )
    
    parser.add_argument(
        '--candidate-segments',
        type=str,
        default=None,
        help='候选客群规则 CSV 文件路径（可选）'
    )
    
    parser.add_argument(
        '--segment-portfolio',
        type=str,
        default=None,
        help='segment_portfolio_*.json 文件路径（可选），注入当前统计推荐方案供 LLM 参考并在此基础上起名'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='元数据 JSON 文件路径（可选）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='llm_segments',
        help='输出文件路径（不含扩展名，默认: llm_segments）'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='json',
        help='输出格式（json/csv/both，默认: json）'
    )
    
    parser.add_argument(
        '--max-segments',
        type=int,
        default=5,
        help='最大生成客群数（默认: 5）'
    )
    
    parser.add_argument(
        '--llm-config',
        type=str,
        default='llm_conf.env',
        help='LLM 配置文件路径（默认: llm_conf.env）'
    )
    
    parser.add_argument(
        '--validation-results',
        type=str,
        default=None,
        help='SQL实测结果文件路径（可选，用于二次精修，支持JSON或CSV格式）'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    try:
        # 加载配置
        config = Stage2Config()
        llm_config = LLMConfig(args.llm_config)
        
        # 生成客群方案
        logger.info("=" * 60)
        logger.info("LLM 受控选择与组合策略")
        logger.info("=" * 60)
        
        # 生成初始客群方案
        logger.info("\n[1/2] 生成初始客群方案...")
        llm_segments = generate_llm_segments(
            atomic_rules_path=args.atomic_rules,
            candidate_segments_path=args.candidate_segments,
            metadata_path=args.metadata,
            segment_portfolio_path=args.segment_portfolio,
            config=config,
            llm_config=llm_config,
            max_segments=args.max_segments
        )
        
        logger.info(f"  初始方案生成完成，共 {len(llm_segments)} 个客群")
        
        # 如果提供了实测结果文件，进行二次精修
        if args.validation_results:
            logger.info("\n[2/2] 基于SQL实测结果进行二次精修...")
            
            # 加载实测结果
            validation_results = load_sql_validation_results(args.validation_results)
            
            if validation_results:
                logger.info(f"  加载实测结果: {len(validation_results)} 个客群的实测数据")
                
                # 重新加载原子规则库和元数据（精修需要）
                atomic_rules_df, _ = load_candidate_rules(args.atomic_rules, args.candidate_segments)
                metadata = load_metadata(args.metadata)
                
                # 执行二次精修
                llm_segments = refine_llm_segments(
                    llm_segments=llm_segments,
                    validation_results=validation_results,
                    atomic_rules_df=atomic_rules_df,
                    metadata=metadata,
                    config=config,
                    llm_config=llm_config
                )
                
                logger.info(f"  精修完成，共 {len(llm_segments)} 个精修后的客群")
            else:
                logger.warning("  实测结果文件为空或格式不正确，跳过精修")
        else:
            logger.info("\n[2/2] 跳过二次精修（未提供 --validation-results 参数）")
        
        # 导出结果（JSON 与 stage2 segment_portfolio_*.json 同构，便于下游统一读取）
        output_path = Path(args.output)
        if args.format in ['json', 'both']:
            export_llm_segments(llm_segments, output_path, format='json', max_segments=args.max_segments)
        if args.format in ['csv', 'both']:
            export_llm_segments(llm_segments, output_path, format='csv')
        
        logger.info("\n" + "=" * 60)
        logger.info("LLM 客群方案生成完成！")
        logger.info("=" * 60)
        logger.info(f"最终客群数: {len(llm_segments)}")
        logger.info(f"输出文件: {output_path}")
        if args.validation_results:
            logger.info(f"实测结果: {args.validation_results}")
    
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

