"""
方案1：统计规则 + LLM 仅命名

- 规则完全沿用 segment_portfolio 的 menu（全部客群），不增删改规则。
- 仅调用 LLM 为每条客群生成「客群名称」「客群说明」。
- 输出与 segment_portfolio 同构的 JSON，下游「任意一条命中→1」时效果与统计版一致（ROC 不低于统计版）。

用法:
  python -m stage2.stats_rules_llm_naming --segment-portfolio ./data/stage2_output/segment_portfolio_sub_202601.json --output ./data/stage2_output/segment_portfolio_llm_naming_sub_202601
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from .llm_strategy import LLMConfig, call_llm_api
except ImportError:
    from llm_strategy import LLMConfig, call_llm_api

logger = logging.getLogger(__name__)


def _sanitize_segment_id_suffix(s: str, max_len: int = 20) -> str:
    """名称转客群编号后缀：去空格、截断。"""
    if not s or not str(s).strip():
        return "客群"
    t = str(s).strip().replace(" ", "_").replace("\t", "_")
    t = re.sub(r"[^\w\u4e00-\u9fff_]", "", t)
    while "__" in t:
        t = t.replace("__", "_")
    return t[:max_len] if len(t) > max_len else t


def _sanitize_rule_id_suffix(segment_definition: str, max_len: int = 20) -> str:
    """简短定义转规则编号后缀：仅保留中文/英文/数字，最多 max_len 字，用于 序号_xxx 的 rule_id（保留可读）。"""
    if not segment_definition or not str(segment_definition).strip():
        return "客群"
    t = re.sub(r"[^\u4e00-\u9fff\w]", "", str(segment_definition).strip())
    return t[:max_len] if t else "客群"


def _menu_item_to_rule_feature_ids(menu_item: Dict[str, Any]) -> List[str]:
    """从 menu 项的规则明细提取字段 ID 列表（与 recommended 的 rule_feature_ids 一致）。"""
    rules = menu_item.get("规则明细") or []
    return [r.get("字段ID", r.get("column_id", "")) for r in rules if r.get("字段ID") or r.get("column_id")]


def build_naming_prompt(menu: List[Dict[str, Any]]) -> str:
    """构建「仅命名」的 prompt：以 rule_desc 组合为主，要求根据每条 rule_desc 为对应 rule_id 起合适名字。"""
    lines = []
    for i, item in enumerate(menu, 1):
        rule_id = item.get("规则编号", "")
        rule_desc = item.get("rule_desc", item.get("SQL筛选条件", ""))
        score_info = item.get("评分指标") or {}
        diff = score_info.get("差异评分")
        stab = score_info.get("稳定性评分")
        comp = score_info.get("综合得分")
        parts = [
            f"{i}. rule_id: {rule_id}",
            f"   rule_desc（筛选条件组合）: {rule_desc}",
        ]
        if diff is not None:
            parts.append(f"   差异评分: {diff:.3f}")
        if stab is not None:
            parts.append(f"   稳定性评分: {stab:.3f}")
        if comp is not None:
            parts.append(f"   综合得分: {comp:.3f}")
        lines.append("\n".join(parts))

    return f"""你是一个专业的客群分析专家。以下每条客群由 **rule_desc（筛选条件组合）** 唯一确定。请**根据每条 rule_desc 的组合内容**，为对应的 rule_id 起一个与之匹配的「客群名称」和一句「客群说明」。

## 规则列表（共 {len(menu)} 条）：每条名称必须严格对应其 rule_desc

{chr(10).join(lines)}

## 要求（必须遵守）

1. **可读性优先**：所有名称必须**一眼能看懂**，禁止过度缩写。例如用「低总费用电子支付未订电影」而不是「低费电付未影」。
2. **严格按 rule_desc 字面命名**：名称必须与 rule_desc 的**逻辑含义**一致，不得自行推断相反含义。
   - **有筛选的维度**：条件对该维度有明确截断或取值限制（如「某字段 < 上界」「某字段 IN (某值)」），名称中按该限制的语义写（如「低总费用」「电子支付」「未订电影」）。
   - **无筛选/冗余维度**：若某条件在取值域上等价于「不限制」（例如「某数值字段 >= 取值域最小值」、或「某二值字段 >= 0」即 0 和 1 都满足），该维度在名称中应写「不限[该维度语义]」，**禁止**根据字面推断成「非某类」「仅某类」等与“不限制”相反的表述。
3. **客群名称**：根据本条 rule_desc 的筛选条件起名，体现关键维度（有筛选的写限制语义，无筛选的写「不限某某」），**不同 rule_desc 必须得到不同名称**，不得重复或泛化。
4. **segment_definition（必填）**：根据该条 rule_desc 提炼的**可读的简短定义，约 6～14 字**。有筛选的维度写限制语义，**无筛选/冗余的维度统一用「不限某某」**，禁止用「非某某」等易误解表述。**每条 rule_desc 的 segment_definition 必须唯一**。
5. **客群说明**：一句话概括该 rule_desc 对应的客群特征；若某维度实为无筛选，说明中用「不限[该维度]」，不用「非某某」等易误解表述。
6. 输出顺序与上面列表一致（1～{len(menu)}），且每条必须带上对应的 rule_id。

## 输出格式（严格 JSON）

```json
{{
  "names": [
    {{ "rule_id": "规则编号", "segment_name": "可读的客群名称", "segment_desc": "客群说明", "segment_definition": "可读的6～14字简短定义（禁止缩写）" }},
    ...
  ]
}}
```

请直接输出上述 JSON，不要其他说明。"""


def _short_definition_fallback(name: str, max_chars: int = 14) -> str:
    """从客群名称去下划线后截取，作为 segment_definition 兜底（保留可读性，不强行缩写）。"""
    if not name or not str(name).strip():
        return "客群"
    s = str(name).strip().replace("_", "").replace(" ", "")
    s = re.sub(r"[^\u4e00-\u9fff\w]", "", s)
    return s[:max_chars] if s else "客群"


def parse_naming_response(response_text: str, menu: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    解析 LLM 返回的命名 JSON，得到 rule_id -> { segment_name, segment_desc, segment_definition }。
    若解析失败或缺少某 rule_id，则用原 menu 兜底；segment_definition 缺省时由 segment_name 截取。
    """
    result = {}
    for item in menu:
        rid = item.get("规则编号", "")
        result[rid] = {
            "segment_name": item.get("客群名称", ""),
            "segment_desc": item.get("客群说明", ""),
            "segment_definition": _short_definition_fallback(item.get("客群名称", "")),
        }

    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start == -1 or end <= start:
        logger.warning("LLM 响应中未找到 JSON，使用原 menu 名称与说明")
        return result

    try:
        data = json.loads(response_text[start:end])
        names = data.get("names", data.get("segments", []))
        if not isinstance(names, list):
            return result
        for i, entry in enumerate(names):
            if not isinstance(entry, dict):
                continue
            rid = entry.get("rule_id")
            if not rid and i < len(menu):
                rid = menu[i].get("规则编号", "")
            if rid:
                seg_name = str(entry.get("segment_name", result.get(rid, {}).get("segment_name", ""))).strip()
                seg_desc = str(entry.get("segment_desc", result.get(rid, {}).get("segment_desc", ""))).strip()
                seg_def = str(entry.get("segment_definition", "")).strip()
                if not seg_def:
                    seg_def = _short_definition_fallback(seg_name)
                # 保留可读长度，仅做合理上限（约 20 字），不强行截成难懂缩写
                if len(seg_def) > 20:
                    seg_def = seg_def[:20]
                result[rid] = {
                    "segment_name": seg_name,
                    "segment_desc": seg_desc,
                    "segment_definition": seg_def,
                }
    except json.JSONDecodeError as e:
        logger.warning("解析 LLM 命名 JSON 失败: %s，使用原 menu 名称与说明", e)

    return result


def apply_naming_to_menu_item(item: Dict[str, Any], naming: Dict[str, str], new_segment_id: str) -> Dict[str, Any]:
    """对单个 menu 项应用命名，并可选更新客群编号。"""
    out = dict(item)
    out["客群名称"] = naming.get("segment_name", out.get("客群名称", ""))
    out["客群说明"] = naming.get("segment_desc", out.get("客群说明", ""))
    if new_segment_id:
        out["客群编号"] = new_segment_id
    return out


def build_selected_and_recommended_from_menu(menu: List[Dict[str, Any]], naming_by_rule_id: Dict[str, Dict[str, str]]) -> tuple:
    """
    从 menu 构建 selected_segments 与 recommended。
    rule_id 使用「序号_可读简短定义」，segment_id 使用「序号_名称后缀」；保留 original_rule_id。
    若 LLM 返回的 segment_definition 重复，则对后续重复项追加 _2、_3 等以保证唯一且可读。
    """
    seen_suffix: Dict[str, int] = {}
    selected = []
    for i, item in enumerate(menu, 1):
        original_rule_id = item.get("规则编号", "")
        naming = naming_by_rule_id.get(original_rule_id) or {}
        seg_def = naming.get("segment_definition", _short_definition_fallback(naming.get("segment_name", "")))
        rule_id_suffix = _sanitize_rule_id_suffix(seg_def, max_len=20)
        # 去重：同一 suffix 已出现过则追加 _2、_3，保证 rule_id 唯一且仍可读
        if rule_id_suffix in seen_suffix:
            seen_suffix[rule_id_suffix] += 1
            rule_id_suffix = f"{rule_id_suffix}{seen_suffix[rule_id_suffix]}"
        else:
            seen_suffix[rule_id_suffix] = 0
        rule_id = f"{i}_{rule_id_suffix}"
        name = naming.get("segment_name", item.get("客群名称", ""))
        seg_suffix = _sanitize_segment_id_suffix(name)
        segment_id = f"{i}_{seg_suffix}" if seg_suffix else item.get("客群编号", f"segment_{i}")
        selected.append({
            "rule_id": rule_id,
            "original_rule_id": original_rule_id,
            "segment_id": segment_id,
            "k": item.get("k", 0),
            "rule_feature_ids": _menu_item_to_rule_feature_ids(item),
        })
    return selected, selected


def run(segment_portfolio_path: str, output_path: str, llm_config: LLMConfig = None) -> None:
    """
    加载统计 portfolio，用 LLM 仅生成名称与说明，写出与 segment_portfolio 同构的 JSON。
    """
    path = Path(segment_portfolio_path)
    if not path.exists():
        raise FileNotFoundError(f"segment_portfolio 不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    menu = list(data.get("menu") or [])
    if not menu:
        raise ValueError("segment_portfolio 的 menu 为空")

    if llm_config is None:
        llm_config = LLMConfig()

    # 1) 调用 LLM 仅命名
    prompt = build_naming_prompt(menu)
    logger.info("调用 LLM 仅为 %d 条客群生成名称与说明...", len(menu))
    response_text = call_llm_api(prompt, llm_config, max_tokens=3000)
    naming_by_rule_id = parse_naming_response(response_text, menu)

    # 2) 用 menu 构建 selected_segments / recommended；rule_id = 序号_简短定义，保留 original_rule_id
    selected_segments, recommended = build_selected_and_recommended_from_menu(menu, naming_by_rule_id)
    orig_to_sel = {s["original_rule_id"]: s for s in selected_segments}

    # 3) 更新 menu：规则编号改为 序号_简短定义，并写 原规则编号、客群编号、名称与说明
    new_menu = []
    for i, item in enumerate(menu, 1):
        original_rule_id = item.get("规则编号", "")
        sel = orig_to_sel.get(original_rule_id, {})
        rule_id = sel.get("rule_id", f"{i}_客群")
        segment_id = sel.get("segment_id", item.get("客群编号", f"segment_{i}"))
        naming = naming_by_rule_id.get(original_rule_id, {})
        new_item = apply_naming_to_menu_item(item, naming, segment_id)
        new_item["规则编号"] = rule_id
        new_item["原规则编号"] = original_rule_id
        new_item["客群编号"] = segment_id
        new_menu.append(new_item)

    # 同步更新 tiers：规则编号改为 序号_简短定义，原规则编号保留
    tiers = data.get("tiers") or {}
    for tier_name in ("elite", "standard", "expand"):
        tier_list = tiers.get(tier_name) or []
        if not tier_list:
            continue
        updated = []
        for item in tier_list:
            original_rule_id = item.get("规则编号", "")
            sel = orig_to_sel.get(original_rule_id, {})
            rule_id = sel.get("rule_id", original_rule_id)
            segment_id = sel.get("segment_id", item.get("客群编号", ""))
            naming = naming_by_rule_id.get(original_rule_id, {})
            t_item = apply_naming_to_menu_item(item, naming, segment_id)
            t_item["规则编号"] = rule_id
            t_item["原规则编号"] = original_rule_id
            t_item["客群编号"] = segment_id
            updated.append(t_item)
        tiers[tier_name] = updated

    # 4) 构建 recommended_rule_details：rule_id 为 序号_简短定义，含 original_rule_id 与 rule_desc
    recommended_rule_details = []
    details_by_orig = {d.get("rule_id"): d for d in (data.get("recommended_rule_details") or []) if d.get("rule_id")}
    for sel in selected_segments:
        orig = sel.get("original_rule_id", "")
        detail = details_by_orig.get(orig)
        menu_item = next((m for m in menu if m.get("规则编号") == orig), None)
        if detail:
            d = dict(detail)
            d["rule_id"] = sel["rule_id"]
            d["original_rule_id"] = orig
            d["rule_desc"] = d.get("rule_desc") or (menu_item and menu_item.get("rule_desc", "")) or (menu_item and menu_item.get("SQL筛选条件", ""))
            recommended_rule_details.append(d)
        else:
            recommended_rule_details.append({
                "rule_id": sel["rule_id"],
                "original_rule_id": orig,
                "rule_desc": menu_item.get("rule_desc", menu_item.get("SQL筛选条件", "")) if menu_item else "",
                "rules": menu_item.get("规则明细", []) if menu_item else [],
            })

    # 5) 输出
    out = {
        "metadata": {
            **data.get("metadata", {}),
            "pi_source": "llm_naming",
            "recommended_meaning": "统计规则+LLM命名（规则与统计版一致，仅名称与说明由LLM生成）",
            "selected_count": len(selected_segments),
        },
        "selected_segments": selected_segments,
        "recommended": recommended,
        "recommended_rule_details": recommended_rule_details,
        "menu": new_menu,
        "selection_trace": [
            {"step": 1, "reason": "stats_rules_llm_naming", "rule_desc": "沿用统计 menu 全部规则，LLM 仅命名"}
        ],
        "selection_steps": [
            {"step": 1, "reason": "stats_rules_llm_naming", "rule_desc": "沿用统计 menu 全部规则，LLM 仅命名"}
        ],
        "selection_reason": "方案1：统计规则+LLM仅命名，效果不低于统计版",
        "elite_forced": data.get("elite_forced", False),
        "tiers": tiers,
        "组合指标": {
            **(data.get("组合指标") or {}),
            "客群数量": len(selected_segments),
        },
    }

    out_path = Path(output_path).with_suffix(".json") if not str(output_path).endswith(".json") else Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("已写出: %s（共 %d 条客群，规则与统计版一致）", out_path, len(selected_segments))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="方案1：统计规则+LLM仅命名，输出与 segment_portfolio 同构的 JSON，ROC 不低于统计版。"
    )
    parser.add_argument("--segment-portfolio", type=str, required=True, help="segment_portfolio_*.json 路径")
    parser.add_argument("--output", type=str, default="segment_portfolio_llm_naming", help="输出 JSON 路径（可含扩展名 .json）")
    parser.add_argument("--llm-config", type=str, default="llm_conf.env", help="LLM 配置文件路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    llm_config = LLMConfig(args.llm_config)
    run(args.segment_portfolio, args.output, llm_config)


if __name__ == "__main__":
    main()
