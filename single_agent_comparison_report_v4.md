# Single Agent 脚本对比分析报告 v2

## 概述

本报告对比 `task-script/single_agent/` 生成的脚本与 `ground_truth/single_agent/` 的参考实现，分析正确率和不匹配之处。

**评判标准**：
- 如果参数不是任务必须的，且脚本在 `logs_code_optimized_read_file_4` 中成功运行，则算正确
- 核心功能实现正确即可，不要求代码完全一致

---

## 运行状态

| 任务 | 日志状态 |
|------|---------|
| task_1 (weather) | OK |
| task_2 (duckduckgo) | OK |
| task_3 (browser_slide) | OK |
| task_4 (terminal_sysinfo) | OK |
| task_5 (deepwiki_mcp) | OK |
| task_6 (linkup_neo4j) | OK |
| task_7 (datagen) | OK |
| task_8 (arxiv_rag) | **ERROR** |
| task_9 (longterm_memory) | OK |
| task_10 (memory_toolkit) | OK |

---

## 总体结果

| 任务 | 日志状态 | 评判结果 | 说明 |
|------|---------|---------|------|
| task_1 | OK | **正确** | 核心功能一致，使用 QwenConfig 是可选增强 |
| task_2 | OK | **正确** | 核心功能一致，模型版本差异可接受 |
| task_3 | OK | **正确** | 核心功能一致，多个模型实例不影响 |
| task_4 | OK | **正确** | 核心功能一致 |
| task_5 | OK | **正确** | 使用 MCPAgent 替代 ChatAgent+MCPToolkit，功能等价 |
| task_6 | OK | **正确** | 正确使用 KnowledgeGraphAgent 和 Neo4j |
| task_7 | OK | **部分正确** | 使用 SymPyToolkit 而非直接 sympy，API 参数名不同 |
| task_8 | ERROR | **错误** | 运行失败 |
| task_9 | OK | **正确** | 核心功能一致 |
| task_10 | OK | **正确** | 核心功能一致 |

**正确率**: 8/10 (80%) - 不含部分正确
**正确率**: 9/10 (90%) - 含部分正确

---

## 详细对比分析

### Task 1: Weather Agent

**任务要求**: 使用 Qwen2.5-14B-Instruct 创建 weather agent

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| 模型平台 | `ModelPlatformType.QWEN` | `ModelPlatformType.QWEN` | ✅ |
| 模型类型 | `ModelType.QWEN_2_5_14B` | `ModelType.QWEN_2_5_14B` | ✅ |
| Toolkit | `WeatherToolkit` | `WeatherToolkit` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Generated 版本使用 `QwenConfig(temperature=0.2)`，Ground Truth 不使用 Config
- Generated 使用 `tools=weather_toolkit.get_tools()`，Ground Truth 使用 `toolkits_to_register_agent`
- 这些差异不影响核心功能

---

### Task 2: DuckDuckGo Agent

**任务要求**: 使用 Gemini 模型创建 DuckDuckGo 搜索 agent

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| 模型平台 | `ModelPlatformType.GEMINI` | `ModelPlatformType.GEMINI` | ✅ |
| 模型类型 | `ModelType.GEMINI_2_5_PRO` | `ModelType.GEMINI_3_PRO` | ⚠️ |
| Tool | `search_toolkit.search_duckduckgo` | 过滤 get_tools() | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- 模型版本略有不同 (GEMINI_2_5_PRO vs GEMINI_3_PRO)，但任务只要求 "Gemini model"，未指定具体版本
- Generated 通过过滤方式获取特定 tool，虽然复杂但功能正确

---

### Task 3: Browser Slide Agent

**任务要求**: 使用 browser 和 PPT tools 搜索 CAMEL-AI 信息并生成 slides

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| Browser Toolkit | `BrowserToolkit` | `BrowserToolkit` | ✅ |
| PPT Toolkit | `PPTXToolkit` | `PPTXToolkit` | ✅ |
| 模型 | 默认 | `DEFAULT` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Generated 创建了 3 个模型实例，Ground Truth 使用默认
- Generated 配置了 `working_directory` 和 headless 模式
- 任务未指定模型要求，额外配置是合理的增强

---

### Task 4: Terminal Sysinfo Agent

**任务要求**: 使用 terminal 和 code execution tools 获取系统信息

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| Terminal Toolkit | `TerminalToolkit` | `TerminalToolkit` | ✅ |
| Code Execution | `CodeExecutionToolkit` | `CodeExecutionToolkit` | ✅ |
| sandbox 参数 | `"internal_python"` | `"internal_python"` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Ground Truth 设置 `set_log_level('INFO')`，Generated 未设置
- Generated 调用 `agent.reset()`，Ground Truth 不调用
- 核心功能一致

---

### Task 5: DeepWiki MCP Agent

**任务要求**: 使用 DeepWiki MCP server 获取 camel-ai/oasis 仓库架构

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| Agent 类型 | `ChatAgent` + `MCPToolkit` | `MCPAgent` | ⚠️ |
| 模型 | `GPT_4O` | `DEFAULT` | ⚠️ |
| 异步调用 | `await agent.astep()` | `await agent.astep()` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Generated 使用 `MCPAgent`（封装类），Ground Truth 使用 `ChatAgent` + `MCPToolkit`，两者功能等价
- `MCPAgent` 是 CAMEL 提供的便捷封装，使用它是合理的
- 任务未指定模型，使用 DEFAULT 可以接受

---

### Task 6: LinkUp Neo4j Agent

**任务要求**: 使用 LinkUp 检索 LLM-based social simulation 研究，存储到 Neo4j

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| LinkUp 搜索 | `search_toolkit.search_linkup` | `search_toolkit.search_linkup` | ✅ |
| KG Agent | `KnowledgeGraphAgent` | `KnowledgeGraphAgent` | ✅ |
| Element 创建 | `UnstructuredIO.create_element_from_text()` | `uio.create_element_from_text()` | ✅ |
| run 参数 | `parse_graph_elements=True` | `parse_graph_elements=True` | ✅ |
| Neo4j 存储 | `add_graph_elements()` | `add_graph_elements()` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Generated 硬编码 Neo4j 凭据，Ground Truth 使用环境变量
- Generated 使用 `depth="standard"`，Ground Truth 使用 `depth="deep"`
- 核心 API 使用正确

---

### Task 7: CoT Data Generation Agent

**任务要求**: 使用 CoT data generation tool，包含 generator 和 verifier agents，解决方程 2x^2-5x-3=0

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| sympy 计算 | `sympy.solve()` | `SymPyToolkit.solve_equation()` | ⚠️ |
| CoTDataGenerator 参数 | `chat_agent=...` | `generator_agent=..., verifier_agent=...` | ❌ |
| 日志状态 | - | OK | ✅ |

**结论**: ⚠️ 部分正确

**不匹配之处**:
1. **CoTDataGenerator 构造参数不同**:
   - Ground Truth: `CoTDataGenerator(chat_agent=chat_agent, golden_answers=..., search_limit=10)`
   - Generated: `CoTDataGenerator(generator_agent=..., verifier_agent=..., golden_answers=..., search_limit=10)`
   - 实际 API 可能支持两种方式，但参数名不同

2. **额外调用**:
   - Generated 调用 `cot_generator.verify_answer()` 和 `cot_generator.export_solutions()`
   - Ground Truth 只调用 `cot_generator.solve()`

3. **日志显示 OK**：说明运行成功，可能 API 支持两种模式

---

### Task 8: Arxiv RAG Transformer Agent

**任务要求**: 下载 "Attention Is All You Need" 论文，使用 vector retrieval 回答 "What is a Transformer?"

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| ArxivToolkit | ✅ | ✅ | ✅ |
| 下载论文 | `arxiv_toolkit.download_papers()` | `agent.step('Download...')` | ⚠️ |
| Vector Storage | `QdrantStorage` + `VectorRetriever` | `VectorDBMemory` | ⚠️ |
| 日志状态 | - | **ERROR** | ❌ |

**结论**: ❌ 错误

**问题分析**:
- 日志显示 `TimeoutError: Step timed out after 180.0s`
- Code Phase 耗时 206.60 秒，超过了 180 秒的 step timeout
- 可能是下载论文或向量索引过程太慢导致超时
- 实现方式与 Ground Truth 差异较大（VectorDBMemory vs VectorRetriever）

---

### Task 9: Longterm Memory Agent

**任务要求**: 创建带有 longterm memory 和 human interaction tools 的 agent

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| LongtermAgentMemory | ✅ | ✅ | ✅ |
| HumanToolkit | ✅ | ✅ | ✅ |
| ScoreBasedContextCreator | ✅ | ✅ | ✅ |
| 模型 | 默认 | `DEFAULT` | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- Generated 使用 `OpenAITokenCounter` 硬编码，Ground Truth 使用 `model_backend.token_counter`
- Generated 设置 `agent_id`，Ground Truth 设置 `message_window_size`
- 核心功能一致

---

### Task 10: Memory Toolkit Agent

**任务要求**: 创建使用 memory tools 管理 memory 的 agent

| 项目 | Ground Truth | Generated | 匹配 |
|------|-------------|-----------|------|
| MemoryToolkit | ✅ | ✅ | ✅ |
| ChatAgent 创建 | ✅ | ✅ | ✅ |
| add_tool 方式 | 遍历添加 | 遍历添加 | ✅ |
| 示例操作 | save/clear/load | save/clear/load | ✅ |
| 日志状态 | - | OK | ✅ |

**结论**: ✅ 正确

**差异说明**:
- 保存文件名不同：`memory.json` vs `conversation_memory.json`
- 对话内容略有不同
- 核心流程完全匹配

---

## 总结

### 正确的任务 (8/10)
1. ✅ Task 1: Weather Agent
2. ✅ Task 2: DuckDuckGo Agent
3. ✅ Task 3: Browser Slide Agent
4. ✅ Task 4: Terminal Sysinfo Agent
5. ✅ Task 5: DeepWiki MCP Agent
6. ✅ Task 6: LinkUp Neo4j Agent
7. ✅ Task 9: Longterm Memory Agent
8. ✅ Task 10: Memory Toolkit Agent

### 部分正确的任务 (1/10)
1. ⚠️ Task 7: CoT Data Generation Agent - API 参数名不同，但运行成功

### 错误的任务 (1/10)
1. ❌ Task 8: Arxiv RAG Agent - 运行失败

### 相比上次改进

| 指标 | 上次 (logs_code_optimized_read_file_all) | 本次 (logs_code_optimized_read_file_4) |
|------|----------------------------------------|--------------------------------------|
| 正确率 | 7/10 (70%) | 8/10 (80%) |
| 错误任务 | Task 7, Task 9 | Task 8 |
| 主要改进 | - | Task 6 正确使用 KnowledgeGraphAgent，Task 9 模型匹配正确 |

### 改进效果
1. **Task 6**: 现在正确使用 `UnstructuredIO.create_element_from_text()` 和 `parse_graph_elements=True`
2. **Task 9**: 现在使用 `DEFAULT` 模型，避免了 Platform/Type 不匹配问题
3. **Task 7**: 虽然参数名不同，但运行成功

### 仍需改进
1. **Task 8**: 需要分析为什么运行失败，可能是 VectorDBMemory 用法问题
