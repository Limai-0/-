"""utils.py — 「数析」智能数据分析台核心模块

本文件的目标：把“LLM + Pandas”的一次性脚本，升级为可讲解、可扩展、可上线的组件化系统。

核心思想：
- **LLM 负责理解自然语言**：把用户问题解析为一个结构化执行计划（AnalysisPlan）。
- **代码负责真实执行**：根据计划调用 Pandas 工具链，得到确定的表格/图表数据。
- **统一输出协议**：最终返回一个 JSON dict，交给界面层（main.py）渲染。

你可以把它理解为一个极简版的“Agent 系统”：
- Planning（PlanEngine）：自然语言 -> steps（工具调用序列）
- Guard（RiskGuard）：识别高风险/敏感字段，决定是否拦截
- Execute（DataAnalyzer）：按 steps 执行工具链（DataFrame pipeline）
- Audit（AuditLogger）：记录每次请求的 trace_id、状态、用过哪些工具

按 7 大核心能力拆解：
  1. ToolRegistry      — 工具注册中心（有哪些工具、怎么调用、调用计数）
  2. PlanEngine        — LLM 规划器（自然语言 -> 结构化计划 AnalysisPlan）
  3. RiskGuard         — 风险识别（高危工具/敏感字段/大导出）
  4. AuditLogger       — 审计留痕（人机协同 + 复盘）
  5. DataAnalyzer      — 执行器（按计划真实跑 Pandas 计算并产出结构化结果）
  6. QualityOptimizer  — 质量优化建议（基于审计日志的简单复盘）
  7. ReportExporter    — 报告导出（HTML 报告字符串）

统一输出协议（供 main.py 渲染）：
- 纯文本：{"answer": "..."}
- 表格：  {"table": {"columns": [...], "data": [[...], ...]}}
- 条形图：{"bar":   {"columns": [x_col, y_col], "data": [[x,y], ...]}}
- 折线图：{"line":  {"x": "x列名", "y": "y列名", "columns": [...], "data": [[x,y], ...]}}
- 散点图：{"scatter":{...}}
并附带元信息：
- "_trace_id"：一次请求的追踪 ID
- "_executed_tools"：本次执行过的工具列表
"""

# ============================
# 说明：如何把 Pandas 能力“产品化”？
#
# 关键不是写脚本，而是把能力拆成可控组件：
# - ToolRegistry：工具注册/权限/超时/调用计数（治理入口）
# - PlanEngine：自然语言 → 结构化计划（保证口径一致）
# - RiskGuard + AuditLogger：高风险阻断 + 审计留痕（可上线）
# - DataAnalyzer：按计划执行工具链，最终产出统一 JSON（可渲染/可导出）
# - QualityOptimizer：基于审计日志做复盘与优化建议（持续迭代）
# - ReportExporter：把结果做成可交付物（HTML 报告）
#
# 使用方式（界面层 main.py 会这么用）：
#   facade = DataframeAgentFacade(dashscope_api_key=...)
#   resp = facade.analyze(df, query, human_confirmed=False)
# ============================

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from langchain_community.chat_models import ChatTongyi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. ToolRegistry — Agent 架构与工具链设计
# ─────────────────────────────────────────────────────────────
@dataclass
class ToolMeta:
    """工具元数据。

    设计目的：让“工具”从散落的函数，变成可治理的资源。

    字段解释：
    - name: 工具名（会暴露给 LLM 规划器使用）
    - description: 给人读的描述（也会进入 LLM prompt，帮助模型选工具）
    - allowed_roles: 哪些角色能用（用于风控/权限，当前 demo 默认 analyst）
    - timeout_sec: 预留的超时控制（demo 未做强制超时）
    - fn: 真实执行函数（签名约定：fn(df, **args) -> DataFrame | dict | str）
    """
    name: str
    description: str
    allowed_roles: List[str]  # e.g. ["analyst", "admin"]
    timeout_sec: float = 5.0
    fn: Optional[Callable] = None


class ToolRegistry:
    """工具注册中心：统一管理工具元数据、权限与调用入口。

    你可以把它理解为一个“工具商店”：
    - 注册：有哪些工具（ToolMeta）
    - 查询：给定 tool_name 找到工具实现
    - 计数：记录每个工具被调用了多少次（便于监控/优化）
    """

    def __init__(self):
        self._registry: Dict[str, ToolMeta] = {}
        self._call_counts: Dict[str, int] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """注册内置工具。

        关键点：
        - 这些工具名会被 PlanEngine 暴露给 LLM，成为“规划”阶段可选 action。
        - 每个工具都绑定一个可执行的 fn（真实跑 Pandas 逻辑）。
        - 工具的参数 schema 由 PlanEngine 的 prompt 约束，避免“模型乱填参数”。
        """
        builtins = [
            ToolMeta("df_shape", "返回数据集行数/列数", ["analyst", "admin"], fn=self._tool_df_shape),
            ToolMeta("head", "返回前 N 行数据", ["analyst", "admin"], fn=self._tool_head),
            ToolMeta("col_mean", "计算某一列的平均值", ["analyst", "admin"], fn=self._tool_col_mean),
            ToolMeta("numeric_means", "计算所有数值列的平均值", ["analyst", "admin"], fn=self._tool_numeric_means),
            ToolMeta("value_counts", "统计某列各类别数量", ["analyst", "admin"], fn=self._tool_value_counts),
            ToolMeta("safe_filter", "按条件过滤数据行", ["analyst", "admin"], fn=self._tool_safe_filter),
            ToolMeta("group_agg", "分组聚合统计", ["analyst", "admin"], fn=self._tool_group_agg),
            ToolMeta("top_n", "按列排序取 TopN 行", ["analyst", "admin"], fn=self._tool_top_n),
            ToolMeta("anomaly_detect", "异常值检测（z-score）", ["analyst", "admin"], fn=self._tool_anomaly_detect),
            ToolMeta("plot_bar", "生成条形图数据", ["analyst", "admin"], fn=self._tool_plot_bar),
            ToolMeta("plot_line", "生成折线图数据", ["analyst", "admin"], fn=self._tool_plot_line),
            ToolMeta("plot_scatter", "生成散点图数据", ["analyst", "admin"], fn=self._tool_plot_scatter),
            ToolMeta("export_report", "导出 HTML/PDF 报告", ["admin"]),
            ToolMeta("delete_rows", "删除数据行（高危）", ["admin"]),
            ToolMeta("update_schema", "修改字段结构（高危）", ["admin"]),
        ]
        for t in builtins:
            self._registry[t.name] = t

    # ----------------------------
    # 内置工具实现（真实执行）
    # 约定：工具返回类型可以是：DataFrame / dict(table|bar|line|scatter) / str
    # ----------------------------
    @staticmethod
    def _tool_df_shape(df: pd.DataFrame, **_: Any) -> Dict[str, Any]:
        """返回行列数（回答类输出）。"""
        return {"answer": f"数据集共有 {df.shape[0]} 行，{df.shape[1]} 列。"}

    @staticmethod
    def _tool_head(df: pd.DataFrame, n: int = 10, **_: Any) -> pd.DataFrame:
        """返回前 N 行（DataFrame 输出，便于后续继续 pipeline）。"""
        n = int(n)
        return df.head(n)

    @staticmethod
    def _tool_col_mean(df: pd.DataFrame, column: str, **_: Any) -> Dict[str, Any]:
        """计算某一列的均值（回答类输出）。

        说明：
        - 会做 to_numeric(coerce) 以兼容字符串数字。
        - 若全为空/全非数值，则返回可解释的文本。
        """
        if column not in df.columns:
            return {"answer": f"列 {column} 不存在。"}
        s = pd.to_numeric(df[column], errors="coerce")
        v = float(s.mean()) if s.notna().any() else None
        return {
            "answer": f"{column} 列的平均值是 {v:.4f}" if v is not None else f"{column} 列无法计算平均值（非数值或全为空）。"}

    @staticmethod
    def _tool_numeric_means(df: pd.DataFrame, **_: Any) -> Dict[str, Any]:
        """计算所有数值列均值（表格输出）。"""
        num_df = df.select_dtypes(include=["number"])  # type: ignore[arg-type]
        means = num_df.mean(numeric_only=True).to_dict()
        out = pd.DataFrame({"column": list(means.keys()), "mean": list(means.values())})
        return {"table": {"columns": ["column", "mean"], "data": out.values.tolist()}}

    @staticmethod
    def _tool_value_counts(df: pd.DataFrame, column: str, top: int = 50, **_: Any) -> pd.DataFrame:
        """类别分布统计（返回 DataFrame）。

        为什么返回 DataFrame：
        - 这样可以和 plot_bar 组合：value_counts -> plot_bar
        - 如果直接返回 table dict，会“断开” pipeline，不利于组合。
        """
        if column not in df.columns:
            raise ValueError(f"列 {column} 不存在。")
        vc = df[column].astype(str).value_counts(dropna=False).head(int(top))
        out = pd.DataFrame({column: vc.index.tolist(), "count": vc.values.tolist()})
        # 返回 DataFrame，便于后续 plot_bar 继续消费
        return out

    @staticmethod
    def _tool_safe_filter(df: pd.DataFrame, expr: str, **_: Any) -> pd.DataFrame:
        """安全过滤（pandas.query）。

        风险说明：
        - query 表达式是字符串，理论上可能被滥用。
        - demo 简化处理：只允许 pandas.query 语法，异常抛给上层。
        - 生产化可加入：表达式白名单/黑名单、最大行数限制等。
        """
        # 只允许 pandas.query 表达式；异常直接抛出给上层。
        return df.query(expr)

    @staticmethod
    def _tool_group_agg(df: pd.DataFrame, by: str, agg: Dict[str, str], **_: Any) -> pd.DataFrame:
        """分组聚合。

        参数：
        - by: 分组列
        - agg: 形如 {"price": "mean", "area": "max"}
        """
        if by not in df.columns:
            raise ValueError(f"group by 列不存在: {by}")
        for col in agg.keys():
            if col not in df.columns:
                raise ValueError(f"聚合列不存在: {col}")
        res = df.groupby(by).agg(agg).reset_index()
        return res

    @staticmethod
    def _tool_top_n(df: pd.DataFrame, n: int = 5, by: Optional[str] = None, ascending: bool = False,
                    **_: Any) -> pd.DataFrame:
        """按某列排序取 TopN。

        教学要点：
        - LLM 常把“最高/最大/TopN”映射到这个工具
        - by 会尽量 to_numeric 做数值排序（price 这类列）
        """
        n = int(n)
        if by and by in df.columns:
            key = pd.to_numeric(df[by], errors="coerce")
            tmp = df.copy()
            tmp["__sort_key__"] = key
            tmp = tmp.sort_values("__sort_key__", ascending=bool(ascending), na_position="last")
            tmp = tmp.drop(columns=["__sort_key__"])
            return tmp.head(n)
        return df.head(n)

    @staticmethod
    def _tool_anomaly_detect(df: pd.DataFrame, column: str, z_threshold: float = 3.0, top: int = 20, **_: Any) -> Dict[
        str, Any]:
        """异常值检测：z-score。

        输出：table（异常行）
        """
        if column not in df.columns:
            return {"answer": f"列 {column} 不存在。"}
        s = pd.to_numeric(df[column], errors="coerce")
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return {"answer": f"列 {column} 无法进行异常检测（方差为 0 或无有效数值）。"}
        z = (s - mu) / sigma
        mask = z.abs() >= float(z_threshold)
        out = df.loc[mask].copy().head(int(top))
        return {"table": {"columns": [str(c) for c in out.columns],
                          "data": out.astype(object).where(pd.notnull(out), None).values.tolist()}}

    @staticmethod
    def _tool_plot_bar(df: pd.DataFrame, x: str, y: str = "count", **_: Any) -> Dict[str, Any]:
        """生成条形图数据（bar）。

        约定：
        - 输入 df 通常来自 value_counts 或 group_agg
        - 输出结构为 {"bar": {"columns": [...], "data": [...]}}
        """
        # 约定：传入的 df 通常是 value_counts 的结果（两列：x + count）或 group_agg 的结果
        if x not in df.columns:
            raise ValueError(f"x 列不存在: {x}")
        if y not in df.columns:
            # 允许把第二列当 y
            y = df.columns[1] if len(df.columns) > 1 else y
        out = df[[x, y]].copy()
        return {"bar": {"columns": [str(x), str(y)], "data": out.values.tolist()}}

    @staticmethod
    def _tool_plot_line(df: pd.DataFrame, x: str, y: str, **_: Any) -> Dict[str, Any]:
        """生成折线图数据（line）。

        特殊语义：
        - x="index"：使用行号作为 x 轴（当数据集没有时间列时，展示“趋势”常用）
        """
        # 支持 x="index"：用行号作为趋势横轴
        if x == "index":
            out = df.copy()
            out["index"] = list(range(len(out)))
        else:
            out = df
        if x not in out.columns or y not in out.columns:
            raise ValueError(f"折线图列不存在: x={x}, y={y}")
        out2 = out[[x, y]].copy()
        return {"line": {"x": str(x), "y": str(y), "columns": [str(x), str(y)], "data": out2.values.tolist()}}

    @staticmethod
    def _tool_plot_scatter(df: pd.DataFrame, x: str, y: str, **_: Any) -> Dict[str, Any]:
        """生成散点图数据（scatter）。

        特殊语义：
        - x="index"：使用行号作为 x 轴
        """
        if x == "index":
            out = df.copy()
            out["index"] = list(range(len(out)))
        else:
            out = df
        if x not in out.columns or y not in out.columns:
            raise ValueError(f"散点图列不存在: x={x}, y={y}")
        out2 = out[[x, y]].copy()
        return {"scatter": {"x": str(x), "y": str(y), "columns": [str(x), str(y)], "data": out2.values.tolist()}}

    def register(self, tool: ToolMeta):
        self._registry[tool.name] = tool
        logger.info(f"[ToolRegistry] 注册工具: {tool.name}")

    def get(self, name: str) -> Optional[ToolMeta]:
        return self._registry.get(name)

    def list_tools(self, role: str = "analyst") -> List[str]:
        return [n for n, t in self._registry.items() if role in t.allowed_roles]

    def record_call(self, name: str):
        self._call_counts[name] = self._call_counts.get(name, 0) + 1

    def call_stats(self) -> Dict[str, int]:
        return dict(self._call_counts)


# ─────────────────────────────────────────────────────────────
# 2. PlanEngine — 自然语言转结构化计划
# ─────────────────────────────────────────────────────────────
@dataclass
class AnalysisPlan:
    trace_id: str
    intent: str
    steps: List[Dict[str, Any]]
    fields: List[str]
    output_type: str  # answer / table / bar / line / scatter
    export_rows: int = 0


class PlanEngine:
    """将自然语言查询转换为结构化 AnalysisPlan，再交由 DataAnalyzer 执行。

    这是“Planning”阶段：
    - 输入：df 元信息（列名/类型/样本） + 用户自然语言 query
    - 输出：AnalysisPlan（intent + steps + output_type）

    关键：
    - prompt 里必须给清楚“可用工具列表 + 参数 schema”，否则 LLM 很容易输出不可执行的 steps。
    - 本 demo 要求 LLM **只输出 JSON**，然后用 _safe_parse 做容错解析。
    """

    def __init__(self, model: ChatTongyi, registry: ToolRegistry):
        self.model = model
        self.registry = registry

    def parse(self, df_meta: Dict[str, Any], query: str) -> AnalysisPlan:
        """调用 LLM 生成结构化计划。

        返回的计划必须满足：
        - steps: [{"tool": "工具名", "args": {...}}]
        - output_type: answer/table/bar/line/scatter

        注意：
        - 这里是 100% 强制走 LLM（按你的要求已移除规则兜底）。
        - 课堂演示时如果 API Key 无效，会在这里直接抛错。
        """
        allowed = self.registry.list_tools(role="analyst")
        prompt = f"""你是数据分析规划师。请将用户问题拆解为结构化 JSON 执行计划，并且严格使用下方工具与参数约定。

DataFrame 信息:
- 形状: {df_meta['shape']}
- 列名: {df_meta['columns']}
- 字段类型: {df_meta['dtypes']}
- 样本(15行): {df_meta['sample']}

可用工具: {allowed}

工具参数约定（非常重要，必须严格遵守；只输出 JSON，不要解释）：
1) df_shape
   - args: {{}}
   - 用途：回答“多少行多少列”
2) head
   - args: {{"n": 10}}
   - 用途：显示前 N 行
3) col_mean
   - args: {{"column": "price"}}
   - 用途：某列平均值
4) numeric_means
   - args: {{}}
   - 用途：所有数值列的平均值（输出 table）
5) value_counts
   - args: {{"column": "furnishingstatus", "top": 50}}
   - 用途：类别分布统计（输出 table，列为 column + count）
6) safe_filter
   - args: {{"expr": "price > 5000000 and area >= 6000"}}
   - 用途：按条件过滤（pandas query 语法）
7) group_agg
   - args: {{"by": "furnishingstatus", "agg": {{"price": "mean"}}}}
   - 用途：分组聚合（agg 的 value 只能是: mean|sum|min|max|count）
8) top_n
   - args: {{"n": 5, "by": "price", "ascending": false}}
   - 用途：按列排序取 TopN（最高/最大 ascending=false；最低/最小 ascending=true）
9) plot_bar
   - args: {{"x": "furnishingstatus", "y": "count"}}
   - 用途：将当前 DataFrame 转为 bar 图数据
10) plot_line
   - args: {{"x": "index", "y": "price"}} 或 {{"x": "date", "y": "price"}}
   - 用途：折线图（必须确保 x/y 在当前 DataFrame 中存在）
11) plot_scatter
   - args: {{"x": "area", "y": "price"}}
   - 用途：散点图

用户问题: {query}

输出格式（仅返回 JSON，不要其他文字）:
{{
  "intent": "一句话概括用户意图",
  "steps": [
    {{"tool": "工具名", "args": {{"参数名": "参数值"}}}}
  ],
  "fields": ["涉及字段列表"],
  "output_type": "answer|table|bar|line|scatter",
  "export_rows": 0
}}"""
        raw = self.model.invoke(prompt).content
        data = self._safe_parse(raw)
        return AnalysisPlan(
            trace_id="t-" + uuid.uuid4().hex[:8],
            intent=data.get("intent", query),
            steps=data.get("steps", []),
            fields=data.get("fields", []),
            output_type=data.get("output_type", "answer"),
            export_rows=data.get("export_rows", 0),
        )

    @staticmethod
    def _safe_parse(raw: str) -> Dict[str, Any]:
        """尽可能把模型输出解析成 dict。

        兼容几类常见异常输出：
        - 正常 JSON：{"a":1}
        - JSON 外围带杂字符：xxx {...} yyy
        - 多个 JSON 拼接：{"answer":"..."}{"table":{...}}
        """
        if not isinstance(raw, str):
            return {}

        s = raw.strip()
        if not s:
            return {}

        # 1) 直接尝试解析
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

        # 2) 尝试从文本中抽取第一个 {...}
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            blob = m.group()
            try:
                obj = json.loads(blob)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass

        # 3) 解析“多个 JSON 对象拼接”的情况：{...}{...}
        #    用 JSONDecoder.raw_decode 逐段解析并 merge
        dec = json.JSONDecoder()
        i = 0
        merged: Dict[str, Any] = {}
        while i < len(s):
            # 找到下一个 '{'
            j = s.find('{', i)
            if j == -1:
                break
            try:
                obj, end = dec.raw_decode(s, j)
                if isinstance(obj, dict):
                    merged.update(obj)
                i = end
            except Exception:
                i = j + 1
        return merged


# ─────────────────────────────────────────────────────────────
# 3. RiskGuard — 工具治理与安全边界
# ─────────────────────────────────────────────────────────────
HIGH_RISK_TOOLS = {"export_report", "delete_rows", "update_schema"}
SENSITIVE_FIELDS = {"phone", "id_card", "salary", "password", "bank_account"}
LARGE_EXPORT_ROWS = 1000


class RiskGuard:
    """评估计划风险等级；阻断未授权的高危操作。"""

    def assess(self, plan: AnalysisPlan) -> Tuple[str, List[str]]:
        factors: List[str] = []
        tools_used = {s.get("tool", "") for s in plan.steps}
        fields_used = set(plan.fields)

        if tools_used & HIGH_RISK_TOOLS:
            factors.append("high_risk_tools=" + ",".join(sorted(tools_used & HIGH_RISK_TOOLS)))
        if fields_used & SENSITIVE_FIELDS:
            factors.append("sensitive_fields=" + ",".join(sorted(fields_used & SENSITIVE_FIELDS)))
        if plan.export_rows >= LARGE_EXPORT_ROWS:
            factors.append(f"large_export_rows={plan.export_rows}")

        level = "high" if factors else "low"
        return level, factors

    def validate_tools(self, plan: AnalysisPlan, registry: ToolRegistry) -> List[str]:
        """返回计划中未注册的工具名（空列表表示合法）。"""
        all_tools = set(registry._registry.keys())
        used = {s.get("tool", "") for s in plan.steps}
        return sorted(used - all_tools)


# ─────────────────────────────────────────────────────────────
# 4. AuditLogger — 人机协同审核 + 审计日志
# ─────────────────────────────────────────────────────────────
@dataclass
class AuditEvent:
    trace_id: str
    status: str  # pending / approved / cancelled / executed / failed
    risk_level: str
    risk_factors: List[str]
    intent: str
    executed_tools: List[str] = field(default_factory=list)
    result_keys: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class AuditLogger:
    """记录每次分析的审核状态与执行结果，支持复盘与质量回顾。"""

    def __init__(self):
        self._log: List[AuditEvent] = []

    def record(self, event: AuditEvent):
        self._log.append(event)
        logger.info(f"[Audit] {event.trace_id} status={event.status} risk={event.risk_level}")

    def all_events(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._log]

    def cancelled_count(self) -> int:
        return sum(1 for e in self._log if e.status == "cancelled")

    def executed_count(self) -> int:
        return sum(1 for e in self._log if e.status == "executed")


# ─────────────────────────────────────────────────────────────
# 5. DataAnalyzer — 数据分析能力集成（ReAct 执行层）
# ─────────────────────────────────────────────────────────────
class DataAnalyzer:
    """依据 AnalysisPlan 逐步执行工具调用，返回最终 JSON 结果。"""

    def __init__(self, model: ChatTongyi, registry: ToolRegistry):
        self.model = model
        self.registry = registry

    def execute(self, plan: AnalysisPlan, df: pd.DataFrame, df_meta: Dict[str, Any]) -> Dict[str, Any]:
        """执行计划（Execute 阶段）。

        核心机制：DataFrame pipeline
        - work_df：当前工作 DataFrame（初始为原始 df）
        - 每一步调用一个工具：out = tool.fn(work_df, **args)
        - 如果 out 是 DataFrame：更新 work_df（用于下一步继续处理）
        - 如果 out 是 dict/str：视为“结构化结果”或“文本答案”，不更新 work_df

        最终结果的选择规则：
        - 如果最后一次工具输出是 dict 且包含 answer/table/bar/line/scatter，直接作为最终结果
        - 否则按 plan.output_type 做默认输出（通常是把 work_df 转 table）

        这样设计的好处：
        - LLM 只负责选工具/传参，不负责“编数据”
        - 工具链可组合（例如 value_counts -> plot_bar）
        - 输出结构稳定，渲染层简单
        """
        executed_tools: List[str] = []
        work_df: pd.DataFrame = df
        last_obj: Any = None

        def _df_to_table(d: pd.DataFrame, limit: int = 200) -> Dict[str, Any]:
            dd = d.head(limit)
            cols = [str(c) for c in dd.columns]
            data = dd.astype(object).where(pd.notnull(dd), None).values.tolist()
            return {"table": {"columns": cols, "data": data}}

        for step in plan.steps:
            tool_name = step.get("tool", "")
            args = step.get("args", {}) or {}
            meta = self.registry.get(tool_name)
            if meta is None or meta.fn is None:
                logger.warning(f"[DataAnalyzer] 跳过未注册/无实现工具: {tool_name}")
                continue

            self.registry.record_call(tool_name)
            executed_tools.append(tool_name)
            logger.info(f"[DataAnalyzer] 执行工具: {tool_name} args={args}")

            # 统一调用：工具默认以当前 work_df 为输入
            out = meta.fn(work_df, **args)
            last_obj = out

            # pipeline 规则：返回 DataFrame 则更新 work_df；返回 dict/str 则不更新
            if isinstance(out, pd.DataFrame):
                work_df = out

        # 组装最终结果（只要是 dict 且含结构化 key，就直接返回）
        result: Dict[str, Any]
        if isinstance(last_obj, dict) and any(k in last_obj for k in ("answer", "table", "bar", "line", "scatter")):
            result = dict(last_obj)
        elif isinstance(last_obj, str):
            result = {"answer": last_obj}
        else:
            # 根据 output_type 决定默认输出
            if plan.output_type in ("bar", "line", "scatter"):
                # 若 LLM 没输出 plot_*，兜底为 table（避免前端空白）
                result = _df_to_table(work_df)
            elif plan.output_type == "table":
                result = _df_to_table(work_df)
            else:
                result = {"answer": "已完成分析，但未生成可展示的结构化结果。"}

        result["_executed_tools"] = executed_tools
        result["_trace_id"] = plan.trace_id
        return result

    def _call_llm_for_result(self, plan: AnalysisPlan, df_meta: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""你是数据分析执行助手。数据已加载为 Pandas DataFrame。

DataFrame 信息:
- 形状: {df_meta['shape']}
- 列名: {df_meta['columns']}
- 字段类型: {df_meta['dtypes']}
- 样本(15行): {df_meta['sample']}

分析意图: {plan.intent}
执行步骤: {json.dumps(plan.steps, ensure_ascii=False)}
期望输出类型: {plan.output_type}

输出格式（仅返回 JSON，不要其他文字）:
- 文字回答: {{"answer": "答案文字"}}
- 表格:     {{"table": {{"columns": ["列1","列2"], "data": [[v1,v2]]}}}}
- 条形图:   {{"bar":   {{"columns": ["类别","值"], "data": [["A",10]]}}}}
- 折线图:   {{"line":  {{"x":"x轴真实列名","y":"y轴真实列名","columns":["x轴列名","y轴列名"],"data":[[x,y]]}}}}
- 散点图:   {{"scatter":{{"x":"x轴真实列名","y":"y轴真实列名","columns":["x轴列名","y轴列名"],"data":[[x,y]]}}}}

重要：columns 中必须填写数据集中真实存在的列名，不能用 "x"/"y" 等占位符。
只返回 JSON，不要代码，不要解释。"""
        raw = self.model.invoke(prompt).content
        # LLM 偶发会返回被转义的 JSON（例如 {\"table\":...} 或 "{...}"），这里做一次反转义/二次解析兜底
        raw_str = raw.strip() if isinstance(raw, str) else str(raw)
        result = PlanEngine._safe_parse(raw_str)
        if not result and raw_str.startswith('"') and raw_str.endswith('"'):
            try:
                inner = json.loads(raw_str)
                if isinstance(inner, str):
                    result = PlanEngine._safe_parse(inner)
            except Exception:
                pass
        if not result and raw_str.startswith("{\\\""):
            result = PlanEngine._safe_parse(raw_str.replace('\\"', '"'))
        result = result or {"answer": raw_str}
        # 若同时返回了 answer + 结构化数据，删除 answer 避免展示层渲染原始 JSON
        if "answer" in result and any(k in result for k in ("table", "bar", "line", "scatter")):
            result.pop("answer")
        return result


# ─────────────────────────────────────────────────────────────
# 6. QualityOptimizer — 数据复盘与质量优化
# ─────────────────────────────────────────────────────────────
class QualityOptimizer:
    """分析审计日志，输出高频错误类型和优化建议。"""

    def __init__(self, audit: AuditLogger):
        self.audit = audit

    def suggest(self) -> List[str]:
        events = self.audit.all_events()
        if not events:
            return ["暂无分析记录，无法给出建议。"]
        suggestions: List[str] = []
        failed = [e for e in events if e["status"] == "failed"]
        cancelled = [e for e in events if e["status"] == "cancelled"]
        if len(cancelled) > 0:
            suggestions.append(f"有 {len(cancelled)} 次因高风险被拦截，建议审查敏感字段权限或降低导出量。")
        if len(failed) > 0:
            suggestions.append(f"有 {len(failed)} 次执行失败，建议检查 Prompt 模板或数据字段名。")
        if not suggestions:
            suggestions.append("执行质量良好，无明显异常。")
        return suggestions


# ─────────────────────────────────────────────────────────────
# 7. ReportExporter — 可视化与报告生成
# ─────────────────────────────────────────────────────────────
class ReportExporter:
    """将分析结果转换为可下载的 HTML 报告字符串。"""

    def export_html(self, query: str, result: Dict[str, Any], audit_events: List[Dict]) -> str:
        rows = ""
        if "answer" in result:
            rows += f"回答：{result['answer']}"
        if "table" in result:
            t = result["table"]
            header = "".join(f"{c}" for c in t.get("columns", []))
            body = "".join(
                "" + "".join(f"{v}" for v in row) + ""
                for row in t.get("data", [])
            )
            rows += f"{header}{body}"
        trace_id = result.get("_trace_id", "N/A")
        tools = ", ".join(result.get("_executed_tools", []))
        return f"""
数析报告

  「数析」智能数据分析报告
  查询：{query}
  Trace ID：{trace_id} | 执行工具：{tools}
  {rows}

  审计日志
  {json.dumps(audit_events, ensure_ascii=False, indent=2)}
"""


# ─────────────────────────────────────────────────────────────
# 门面：DataframeAgentFacade — 统一编排 7 大模块
# ─────────────────────────────────────────────────────────────
class DataframeAgentFacade:
    """对外暴露单一入口 analyze()，内部编排全部 7 个模块。"""

    def __init__(self, dashscope_api_key: str):
        # 初始化大模型（生产环境建议加超时/重试/成本控制/熔断，这里略）
        self.model = ChatTongyi(
            model="qwen3-max",
            dashscope_api_key=dashscope_api_key,
            temperature=0,
        )
        # 注册中心：统一管理工具、权限、超时、调用计数等
        self.registry = ToolRegistry()
        # 计划引擎：把自然语言 query 转成结构化 AnalysisPlan
        self.plan_eng = PlanEngine(self.model, self.registry)
        # 风险守卫：识别高危工具/敏感字段/大导出等
        self.risk = RiskGuard()
        # 审计记录：记录 cancelled/executed/failed
        self.audit = AuditLogger()
        # 执行器：按 plan.steps 调用工具，并最终让模型产出统一 JSON
        self.analyzer = DataAnalyzer(self.model, self.registry)
        # 优化器：基于审计日志输出可操作建议
        self.optimizer = QualityOptimizer(self.audit)
        # 导出器：把结果渲染成 HTML 报告
        self.exporter = ReportExporter()

    def analyze(
            self,
            df,
            query: str,
            human_confirmed: bool = False,
    ) -> Dict[str, Any]:
        import pandas as pd

        df_meta = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "sample": df.sample(n=min(15, len(df)), random_state=42).to_string(index=False),
        }

        # ① 生成结构化计划（第一阶段：Planning）
        # 说明：这里会调用 LLM（通义千问）来产生可执行的 steps。
        plan = self.plan_eng.parse(df_meta, query)
        logger.info(f"[Facade] Plan: intent={plan.intent} steps={len(plan.steps)}")

        # ② 风险评估：确定是否需要人工确认、是否包含未注册工具
        # - 风险等级 high：包含高危工具/敏感字段/大导出
        # - invalid_tools：LLM 可能输出了不存在的工具名（这里会记录 warning）
        risk_level, factors = self.risk.assess(plan)
        invalid_tools = self.risk.validate_tools(plan, self.registry)
        if invalid_tools:
            logger.warning(f"[Facade] 未注册工具: {invalid_tools}")

        # ③ 人机协同门控：高风险需要人工确认（Human-in-the-Loop）
        if risk_level == "high" and not human_confirmed:
            self.audit.record(AuditEvent(
                trace_id=plan.trace_id,
                status="cancelled",
                risk_level=risk_level,
                risk_factors=factors,
                intent=plan.intent,
            ))
            return {
                "blocked": True,
                "trace_id": plan.trace_id,
                "reason": "high_risk_not_confirmed",
                "risk_factors": factors,
            }

        # ④ 执行分析（第二阶段：Execute）
        # - 真正执行工具链（Pandas 计算）
        # - 写入审计日志（executed/failed）
        try:
            result = self.analyzer.execute(plan, df, df_meta)
            self.audit.record(AuditEvent(
                trace_id=plan.trace_id,
                status="executed",
                risk_level=risk_level,
                risk_factors=factors,
                intent=plan.intent,
                executed_tools=result.get("_executed_tools", []),
                result_keys=list(result.keys()),
            ))
        except Exception as e:
            self.audit.record(AuditEvent(
                trace_id=plan.trace_id,
                status="failed",
                risk_level=risk_level,
                risk_factors=factors,
                intent=plan.intent,
            ))
            raise ValueError(f"执行分析时出错: {e}") from e

        return result

    def export_report(self, query: str, result: Dict[str, Any]) -> str:
        return self.exporter.export_html(query, result, self.audit.all_events())

    def quality_suggestions(self) -> List[str]:
        return self.optimizer.suggest()


# ─────────────────────────────────────────────────────────────
# 向后兼容入口（main.py 仍可直接调用）
# ─────────────────────────────────────────────────────────────
def dataframe_agent(dashscope_api_key: str, df, query: str) -> Dict[str, Any]:
    facade = DataframeAgentFacade(dashscope_api_key)
    return facade.analyze(df, query, human_confirmed=True)
