"""main.py — 「数析」智能数据分析台 Streamlit 界面

对接 utils.py 的 DataframeAgentFacade，支持：
  - 人工确认高风险操作（Human-in-the-Loop）
  - 审计日志侧边栏查看
  - HTML 报告一键下载
  - 质量优化建议

设计要点：
  - 界面层只负责：数据上传、问题收集、结果渲染、交互控制
  - 业务逻辑全部委托给 DataframeAgentFacade（单一职责）
  - 使用 st.session_state 持久化 facade，保证审计日志与质量建议累积
  - 高风险操作默认阻断，必须用户勾选确认才能执行（合规要求）
"""

# ============================
# 目的：把 utils.py 的“产品化能力”用界面方式交付出来
#
# main.py 只负责三件事：
# 1) 读取/上传数据、收集用户自然语言问题
# 2) 调用 DataframeAgentFacade.analyze(...) 获取统一 JSON 结果
# 3) 根据 JSON 字段渲染：answer/table/bar/line/scatter，并展示审计日志
#
# 关键设计：
# - facade 对象放在 st.session_state，保证多轮分析时：审计日志/质量建议能累积
# - 高风险计划默认阻断，必须用户勾选 human_confirmed 才执行（Human-in-the-Loop）
# ============================

import json
import re

import pandas as pd
import streamlit as st

from utils import DataframeAgentFacade


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def render_chart(input_data: dict, chart_type: str):
    """通用图表渲染：bar / line / scatter。

    参数：
    - input_data: Agent 输出的标准化图表数据（columns/data/x/y/series）
    - chart_type: 图表类型（'bar'|'line'|'scatter'）

    处理逻辑：
    1) 将 input_data 转为 DataFrame
    2) 解析 x/y/series 列映射（处理 LLM 输出不一致问题）
    3) 根据图表类型调用 Streamlit 对应图表组件
    4) 异常时展示原始数据便于调试
    """
    # 约定：input_data 来自 utils.py 的工具输出，结构大致为：
    # - bar:     {"columns": [x_col, y_col], "data": [[x,y], ...]}
    # - line:    {"x": "真实x列名", "y": "真实y列名", "columns": [...], "data": [[x,y], ...]}
    # - scatter: 同 line
    #
    # 注意：LLM 可能会输出不一致的字段（例如 columns 里写成 ["x","y"] 占位符），
    # 所以这里做了列名解析与兜底渲染，保证页面不“白屏”。
    try:
        # input_data 是标准化的图表数据结构（由 Agent 输出），包含：
        # - columns: 列名
        # - data: 二维数组（行）或一维数组（值）
        # - x/y/series: 可选，指定 x/y 对应的列名
        columns  = input_data.get("columns", [])
        data     = input_data.get("data", [])
        x_col    = input_data.get("x")
        y_col    = input_data.get("y")
        series   = input_data.get("series")

        if not (isinstance(data, list) and data):
            raise ValueError("数据为空")

        if not isinstance(data[0], list):
            df_chart = pd.DataFrame({"值": data}, index=columns)
        else:
            df_chart = pd.DataFrame(data, columns=columns)

        # LLM 有时把 columns 写成 ["x","y"] 字面量，但 x/y 字段才是真实列名
        # 若 x_col/y_col 不在 df 列名中，尝试把对应位置的列重命名
        def resolve_col(col_hint, pos):
            """解析列名：优先使用 col_hint，否则使用位置 pos 的列名。"""
            if col_hint and col_hint in df_chart.columns:
                return col_hint
            if len(df_chart.columns) > pos:
                real = df_chart.columns[pos]
                if col_hint and col_hint != real:
                    df_chart.rename(columns={real: col_hint}, inplace=True)
                return col_hint if col_hint else real
            return None

        x_col = resolve_col(x_col, 0)
        y_col = resolve_col(y_col, 1)
        if series:
            series = resolve_col(series, 2)

        if chart_type == "bar":
            if x_col and y_col and x_col in df_chart.columns:
                st.bar_chart(df_chart.set_index(x_col)[y_col])
            else:
                st.bar_chart(df_chart)
        elif chart_type == "line":
            if x_col and y_col and x_col in df_chart.columns:
                df_sorted = df_chart.sort_values(x_col)
                st.line_chart(df_sorted, x=x_col, y=y_col)
            else:
                st.line_chart(df_chart)
        elif chart_type == "scatter":
            if x_col and y_col and x_col in df_chart.columns:
                st.scatter_chart(df_chart, x=x_col, y=y_col)
            else:
                st.scatter_chart(df_chart)
    except Exception as e:
        st.error(f"图表渲染失败: {e}")
        with st.expander("原始图表数据"):
            st.json(input_data)


def _try_unwrap_json_answer(result: dict) -> dict:
    """若 answer 字段是 JSON 字符串，尝试解析并合并回 result。

    背景：
    - 部分模型会把“最终 JSON”塞进 answer 字段（字符串形式），导致前端无法直接渲染 table/bar/line。
    - 这里做兼容：尽可能把 answer 里的 JSON 解析出来并 merge 回 result。

    处理策略：
    1) 直接解析 answer 字符串
    2) 若被包了一层字符串（"{...}"），再解析一次
    3) 若被转义（\"{...}\"），先反转义再解析
    4) 兜底：用正则抽取 {...} 部分再解析
    """
    answer = result.get("answer", "")
    if not isinstance(answer, str):
        return result
    stripped = answer.strip()
    # 兼容 3 类常见输出：
    # 1) 直接 JSON：{"table":{...}}
    # 2) 被包了一层字符串："{...}"（需要 loads 两次）
    # 3) JSON 被转义：\{"table":{...}\}（需要先反转义再 loads）
    candidates = []
    if stripped:
        candidates.append(stripped)
    if stripped.startswith('"') and stripped.endswith('"'):
        candidates.append(stripped)
    if stripped.startswith("{\\\""):
        candidates.append(stripped.replace('\\"', '"'))

    parsed = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            # 如果外层解析出来还是字符串，再解析一次
            if isinstance(obj, str) and obj.strip().startswith("{"):
                obj = json.loads(obj)
            if isinstance(obj, dict):
                parsed = obj
                break
        except Exception:
            continue
    if parsed is None:
        # 兜底：从 answer 中抽取 {...} 部分再尝试解析
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not m:
            return result
        blob = m.group()
        for cand in (blob, blob.replace('\\"', '"')):
            try:
                obj = json.loads(cand)
                if isinstance(obj, str) and obj.strip().startswith("{"):
                    obj = json.loads(obj)
                if isinstance(obj, dict):
                    parsed = obj
                    break
            except Exception:
                continue
        if parsed is None:
            return result
    if any(k in parsed for k in ("table", "bar", "line", "scatter", "answer")):
        merged = {k: v for k, v in result.items() if k != "answer"}
        merged.update(parsed)
        return merged
    return result


def render_result(result: dict):
    """统一渲染分析结果（answer / table / bar / line / scatter）。

    处理流程：
    1) 若结果被阻断，展示风险因子并返回
    2) 尝试从 answer 字段解析嵌套 JSON（兼容模型输出问题）
    3) 展示 trace_id 与执行工具信息
    4) 按优先级渲染：answer（仅当无结构化数据）→ table → 图表

    重要说明（教学讲解点）：
    - 后端（utils.py）返回的是“结构化 JSON”，而不是直接返回 DataFrame。
    - UI 只做渲染，不做业务计算：这能保证职责清晰、方便测试与复用。
    - 若渲染异常，可以通过展开“原始图表数据”、或查看 Trace ID 对应的审计日志来定位。
    """
    if result.get("blocked"):
        st.error("🚫 高风险操作已被拦截，需人工确认后才能执行。")
        st.json({"risk_factors": result.get("risk_factors", [])})
        return

    # LLM 有时把整个 JSON 塞进 answer 字段，尝试拆解
    result = _try_unwrap_json_answer(result)

    trace_id = result.get("_trace_id", "N/A")
    tools    = result.get("_executed_tools", [])
    st.caption(f"🔍 Trace ID: `{trace_id}` | 执行工具: {', '.join(tools) or '无'}")

    has_visual = any(k in result for k in ("table", "bar", "line", "scatter"))

    # 仅当没有结构化数据时才展示文字回答，避免把 JSON 字符串当文字渲染
    if "answer" in result and not has_visual:
        st.success(result["answer"])

    if "table" in result:
        t = result["table"]
        st.dataframe(pd.DataFrame(t["data"], columns=t["columns"]), use_container_width=True)

    for chart_type in ("bar", "line", "scatter"):
        if chart_type in result:
            st.subheader({"bar": "📊 条形图", "line": "📈 折线图", "scatter": "🔵 散点图"}[chart_type])
            render_chart(result[chart_type], chart_type)


# ─────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────
# 配置说明：
# - page_title: 浏览器标签页标题
# - page_icon: 标签页图标（emoji）
# - layout: 'wide' 让内容占满屏幕宽度（适合表格/图表展示）
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="「数析」智能数据分析台",
    page_icon="📊",
    layout="wide",
)
st.title("📊 「数析」智能数据分析台")
st.markdown("基于**通义千问 + 7 大核心能力架构**的智能数据分析平台。")

# ─────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────
# 侧边栏功能：
# 1) API Key 配置（必填，用于调用通义千问）
# 2) 安全设置：人工确认高风险操作（Human-in-the-Loop）
# 3) 审计日志：展示最近 5 条记录（状态/trace_id/意图）
# 4) 质量建议：基于审计日志生成优化建议
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input("通义千问 API 密钥", type="password",
                             help="阿里云 DashScope 控制台获取")
    st.markdown("[获取 API Key](https://dashscope.console.aliyun.com/apiKey)")

    # 人机协同：是否手动确认高风险操作
    st.divider()
    st.subheader("🛡️ 安全设置")
    human_confirmed = st.checkbox(
        "我已确认，允许执行高风险操作",
        value=False,
        help="涉及敏感字段、高危工具或大批量导出时，需勾选此项才能执行。",
    )

    # 审计日志
    st.divider()
    st.subheader("📋 审计日志")
    if "facade" in st.session_state:
        events = st.session_state["facade"].audit.all_events()
        st.caption(f"共 {len(events)} 条记录")
        for ev in events[-5:]:
            color = "🟢" if ev["status"] == "executed" else ("🔴" if ev["status"] == "cancelled" else "🟡")
            st.markdown(f"{color} `{ev['trace_id']}` {ev['status']} — {ev['intent'][:20]}")
    else:
        st.caption("暂无记录")

    # 质量优化建议
    st.divider()
    st.subheader("💡 质量建议")
    if st.button("生成优化建议") and "facade" in st.session_state:
        for tip in st.session_state["facade"].quality_suggestions():
            st.info(tip)

# ─────────────────────────────────────────────
# 主区域：文件上传
# ─────────────────────────────────────────────
# 功能说明：
# - 支持 CSV 文件上传（类型限制）
# - 上传后展示：行数、列数、文件大小、数据预览
# - 数据存入 st.session_state.df，供后续分析使用
# ─────────────────────────────────────────────
st.header("📁 数据上传")
uploaded = st.file_uploader("上传 CSV 数据文件", type="csv")

if uploaded:
    try:
        st.session_state["df"] = pd.read_csv(uploaded)
        df = st.session_state["df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("行数", df.shape[0])
        c2.metric("列数", df.shape[1])
        c3.metric("文件大小", f"{uploaded.size / 1024:.1f} KB")
        with st.expander("📊 原始数据预览", expanded=True):
            st.dataframe(df, height=280, use_container_width=True)
    except Exception as e:
        st.error(f"文件读取失败: {e}")

# ─────────────────────────────────────────────
# 主区域：智能查询
# ─────────────────────────────────────────────
# 查询流程：
# 1) 收集用户自然语言问题
# 2) 前置校验：API Key、数据文件、查询内容
# 3) 创建/更新 DataframeAgentFacade（存入 session_state）
# 4) 调用 facade.analyze(df, query, human_confirmed)
# 5) 渲染结果（阻断/成功/失败）
# 6) 若成功，提供 HTML 报告下载
# ─────────────────────────────────────────────
st.header("💬 智能查询")
query = st.text_area(
    "用自然语言描述分析需求：",
    placeholder="例如：近 30 天 GMV 趋势如何？ / 哪个渠道转化率最低？ / 用散点图展示价格与面积的关系",
    height=90,
)
btn = st.button("🚀 开始分析", type="primary")

if btn:
    if not api_key:
        st.warning("⚠️ 请先输入 API 密钥")
    elif "df" not in st.session_state:
        st.warning("⚠️ 请先上传数据文件")
    elif not query.strip():
        st.warning("⚠️ 请输入查询内容")
    else:
        # 每次点击都重新创建 facade，避免 session_state 缓存旧实例
        st.session_state["facade"]  = DataframeAgentFacade(api_key)
        st.session_state["_api_key"] = api_key

        facade: DataframeAgentFacade = st.session_state["facade"]  # type: ignore[assignment]

        with st.spinner("🤔 AI 正在分析，请稍候…"):
            try:
                result = facade.analyze(
                    st.session_state["df"],
                    query,
                    human_confirmed=human_confirmed,
                )
                st.header("📋 分析结果")
                render_result(result)

                # 报告下载
                if not result.get("blocked"):
                    html_report = facade.export_report(query, result)
                    st.download_button(
                        label="⬇️ 下载 HTML 报告",
                        data=html_report.encode("utf-8"),
                        file_name="analysis_report.html",
                        mime="text/html",
                    )

            except ValueError as e:
                st.error(f"❌ 分析失败：{e}")
                st.info("💡 建议：检查查询是否包含数据中存在的字段名")
            except Exception as e:
                st.error(f"❌ 系统错误：{e}")
