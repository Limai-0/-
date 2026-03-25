为了让面试官快速理解你的项目深度，这份 Markdown (MD) 文档不仅要展示功能，更要突出其**工业级架构设计**。你可以将此内容保存为 `README.md` 存放在你的项目根目录。

---

# 「数析」智能数据分析平台 (SmartData-Agent)

基于 **LangChain Agent** 与 **通义千问** 构建的工业级智能数据分析系统，旨在通过自然语言驱动复杂的 Pandas 数据处理流水线，实现从原始数据到决策洞察的自动化转化。



## 🌟 核心亮点

### 1. 结构化 Agent 编排架构 (The 7-Layer Architecture)
不同于简单的“LLM 拼接脚本”，本项目将数据分析能力拆解为 7 个解耦的模块，确保了系统的可扩展性与可维护性：
* **ToolRegistry**: 工具治理中心，实现工具权限控制与调用监控。
* **PlanEngine**: 基于 LLM 的执行规划器，将模糊意图转化为确定性的 JSON 执行步骤。
* **RiskGuard**: 安全守卫，实时识别高危 SQL/Pandas 操作及敏感字段导出。
* **DataAnalyzer**: 核心执行引擎，采用 **ReAct** 模式处理 DataFrame Pipeline。
* **AuditLogger**: 完整的审计链路，记录每一个 `trace_id` 的状态与风险因子。
* **QualityOptimizer**: 基于历史日志的质量复盘，提供 Prompt 优化建议。
* **ReportExporter**: 自动生成专业级 HTML 分析报告。

### 2. 人机协同安全机制 (Human-in-the-Loop)
系统引入了动态门控机制：当 `RiskGuard` 识别到高风险操作（如涉及薪资敏感字段或大批量数据导出）时，系统会自动触发**人工确认请求**，只有用户显式授权后方可执行，符合企业级合规要求。

### 3. 高鲁棒性解析引擎 (Robust Parser)
针对 LLM 输出不稳定的痛点，设计了多层解析策略：
* 支持 **Markdown 代码块清洗**。
* 支持 **多级 JSON 反转义解析**。
* 支持 **语义列名自动对齐**：当模型引用的列名与真实 DataFrame 存在微小偏差时，系统可依据 `resolve_col` 逻辑自动纠偏。

---

## 🛠️ 技术栈
* **大模型**: 通义千问 (Qwen-max)
* **框架**: LangChain (LCEL)
* **数据处理**: Pandas / NumPy
* **前端交互**: Streamlit
* **安全性**: 基于状态机的风险阻断机制

---

## 📖 核心模块实现细节 (代码精要)

### 执行流程示意图


### 1. 工具注册示例
通过装饰器模式与元数据定义，将 Pandas 操作“产品化”：
```python
ToolMeta("group_agg", "分组聚合统计", ["analyst"], fn=self._tool_group_agg)
# 优势：解耦了业务逻辑与大模型提示词，支持按角色授权调用
```

### 2. 统一输出协议
后端不返回原始 DataFrame，而是返回标准化的 JSON 协议，极大降低了前端渲染的复杂度：
```json
{
  "bar": {
    "columns": ["furnishingstatus", "price"],
    "data": [["furnished", 5000000], ["semi-furnished", 4800000]]
  },
  "_trace_id": "t-8af21d",
  "_executed_tools": ["group_agg", "plot_bar"]
}
```

---

## 🚀 快速开始

1. **安装依赖**:
   ```bash
   pip install pandas streamlit langchain_community dashscope
   ```
2. **配置 API Key**:
   在界面侧边栏输入您的阿里云 DashScope API Key。
3. **启动应用**:
   ```bash
   streamlit run main.py
   ```

---

## 🎯 未来规划
* [ ] 集成 **LangGraph** 实现更复杂的多轮对话循环。
* [ ] 支持多模态输入（如直接上传图表进行对比分析）。
* [ ] 对接数据库直连，支持大规模 SQL 分析。

---

> **项目自述**：本项目不仅是一个 Demo，更是一次关于“如何让 AI 稳定、安全地服务于数据分析”的深度工程实践。
