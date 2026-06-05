# AI 智能对话 Agent v2.2

具备 **长期记忆 + MCP 工具池 + 四状态机 + 工作流** 的对话智能体框架。基于 Python 和 LLM（DeepSeek/GPT-4o 等）实现，支持 LLM 自主创建工具、三层记忆管理和后台工作流。

## 特性

| 能力 | 说明 |
|------|------|
| 🧠 **四状态机** | THINK → EXECUTE ⇄ REFLECT → END，出错自动反思恢复 |
| 🔧 **MCP 工具池** | LLM 自主判断工具是否存在，不存在时 `create_tool` 创建并立即使用 （非手动创建）|
| 💾 **三层记忆** | 工作记忆(Token预算) → 短期记忆(LLM摘要) → 长期记忆(向量检索) |
| 🔍 **语义检索** | TF-IDF + OpenAI Embedding 混合召回 |
| ⏳ **记忆遗忘** | importance 评分 × 时间衰减 → 自动清理不重要的记忆 |
| 🔒 **安全检查** | 拦截 LLM 生成代码中的危险操作（os/subprocess/eval 等） |
| ⚙️ **后台工作流** | 复杂任务自动拆分多步，异步执行 |
| 🗑️ **过期清理** | 启动时自动删除 14 天未使用的工具 |
| 🎛️ **/ 命令系统** | 运行时切换模型、查看配置、调整参数 |
| 🌐 **多模型** | OpenAI / DeepSeek / Ollama / SiliconFlow 统一适配 |

## 项目结构

```
├── main.py                   # 交互入口 + / 命令系统
├── config.py                 # 配置加载器
├── config.example.json       # 配置模板（提交到 git）
├── config.json               # 实际配置（含密钥，gitignore）
│
├── brain/                    # 智能体大脑
│   ├── adapters.py           #   多模型适配器
│   ├── state.py              #   四状态机定义
│   ├── brain.py              #   AgentBrain 核心
│   ├── call.py               #   API 调用封装
│   └── example.py            #   使用示例
│
├── memory/                   # 三层记忆系统
│   ├── models.py             #   数据模型（Message/Memory/LongTermMemory）
│   ├── manager.py            #   三层编排器
│   ├── embeddings.py         #   向量引擎（TF-IDF + OpenAI）
│   ├── utils.py              #   旧 API 兼容
│   ├── cleanup.py            #   AI 评分记忆删除
│   └── archives/             #   记忆存档（运行时生成）
│
├── tools/                    # MCP 工具池
│   ├── mcp_pool.py           #   工具池核心
│   ├── manage_tools.py       #   维护脚本
│   ├── mcp_tools.json        #   工具元数据
│   └── tool_add/tool_direct/ #   7 个核心工具
│
└── test/                     # 测试（94/94 通过）
    ├── test_mcp_tools.py     #   工具池测试 34/34
    ├── test_memory.py        #   记忆系统测试 60/60
    ├── test_brain.py         #   大脑基础测试
    └── test_llm_create_tool.py  # LLM 集成测试
```

## 快速开始

### 1. 配置

```bash
cp config.example.json config.json
# 编辑 config.json，填入你的 API Key
```

```json
{
  "llm": {
    "provider": "deepseek",
    "api_key": "sk-your-key-here",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  }
}
```

### 2. 运行

```bash
python main.py
```

### 3. 对话

```
User：/help                    # 查看所有命令
User：/model                   # 查看当前模型
User：计算 2 的 10 次方        # 正常对话
🤖 2的10次方 = 1024
```

## / 命令系统

| 命令 | 功能 |
|------|------|
| `/help` `/`? | 显示帮助 |
| `/model [name]` | 查看/切换模型（如 `/model gpt-4o`） |
| `/config` | 查看当前配置（密钥隐藏） |
| `/tools` | 查看 MCP 工具池状态 |
| `/memory` | 查看三层记忆统计 |
| `/system [text]` | 查看/修改系统提示词 |
| `/temp [value]` | 查看/调整 temperature |
| `/clear` | 清空当前对话记忆 |
| `/save` | 手动保存记忆 |
| `/exit` `/quit` | 退出 |

## 四状态机

```
首次问题 → THINK(思考)
              │
     ┌───────┼───────┐
     ▼       │       ▼
  EXECUTE    │      END
  (执行)     │     (结束)
     │       │
     ▼       │
  REFLECT ←──┘
  (反思·出错时)
     │
     └──→ EXECUTE(重试) 或 THINK(重新分析)
```

每个状态由 LLM 驱动决策，不是固定顺序：
- **THINK**：分析任务 + 历史操作 → 判断 `done=true/false`
- **EXECUTE**：调用工具，出错自动进入 REFLECT
- **REFLECT**：分析错误根因 → 修正方案 → 重试或放弃
- **END**：输出最终答案

## MCP 工具池

### LLM 自主创建工具

```
用户："帮我获取今日天气"
  │
  ├─ LLM 检查池中是否有 weather 工具
  ├─ 没有 → 调用 create_tool 生成 Python 代码
  ├─ 系统：安全检查 → 语法验证 → 写入 .py → 注册 JSON
  ├─ 新工具立即可用
  └─ LLM 调用新工具 → 返回结果
```

### 工具持久化

LLM 创建的工具体现在两个文件中，**重启后自动加载**：

| 存储 | 位置 |
|------|------|
| 代码 | `tools/tool_add/tool_direct/{name}.py` |
| 元数据 | `tools/mcp_tools.json` |

### 工具生命周期

```
创建 → 使用 → 累计 usage_count
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
 LLM主动删除   14天未用自动清理   受保护(7个核心工具)
```

### 内置工具（7个核心操作）

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件 |
| `write_file` | 写入文件（覆盖/追加） |
| `create_file` | 创建文件 |
| `delete_file` | 删除文件 |
| `create_directory` | 创建目录 |
| `run_command` | 执行系统命令（Windows/Linux 自适应） |
| `delete_tool` | 从 MCP 池删除工具 |

> 其他功能（计算、爬虫、工作流等）由 LLM 通过 `create_tool` 按需创建，不预置。

## 三层记忆

```
L1 工作记忆 (Working Memory)
  ├── 当前对话上下文
  ├── Token 预算管理 (默认4000)
  └── 超出80% → 触发溢出

       ↓ 溢出

L2 短期记忆 (Episodic Memory)
  ├── LLM 摘要 + 结构化事实提取
  ├── 最多 200 项
  └── importance × 时间衰减

       ↓ 巩固

L3 长期记忆 (Semantic Memory)
  ├── 向量 Embedding + 关键词混合检索
  ├── 最多 1000 项
  └── 跨会话持久化
```

### 使用示例

```python
from memory import MemoryManager

mgr = MemoryManager(client=llm_client, model="deepseek-chat")

# 对话
mgr.add_user_message("我叫小明，喜欢Python")
mgr.add_assistant_message("记住了！")

# 检索
results = mgr.recall("小明喜欢什么编程语言")

# 获取上下文注入 LLM
ctx = mgr.get_context_for_llm()
```

## 使用 AgentBrain API

```python
from config import config
from brain import create_agent, ModelConfig, ModelProvider

configs = [
    ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    ),
]

brain = create_agent(
    model_configs=configs,
    system_prompt="你是一个智能助手...",
    max_iterations=10,
    openai_client=config.create_llm_client(),
)

# 注册自定义工具
brain.tools.register("my_tool", "我的工具", my_function)

# 执行任务
result = brain.run("请帮我完成...")
print(result)  # 四状态机自动流转直到 END

# 查看
print(brain.get_history())      # 状态流转历史
print(brain.get_memory_count()) # 记忆统计
```

## 配置项说明

| 配置路径 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `llm.api_key` | string | — | API 密钥 |
| `llm.base_url` | string | `https://api.deepseek.com` | API 地址 |
| `llm.model` | string | `deepseek-chat` | 模型名 |
| `llm.temperature` | float | `0.7` | 创意度 |
| `llm.max_tokens` | int | `4096` | 最大输出 token |
| `brain.max_iterations` | int | `100` | 最大轮次 |
| `brain.max_retries` | int | `3` | 最大重试次数 |
| `brain.system_prompt` | string | — | 系统提示词 |
| `memory.working_token_budget` | int | `4000` | 工作记忆 token 预算 |
| `memory.short_term_max_items` | int | `200` | 短期记忆上限 |
| `memory.long_term_max_items` | int | `1000` | 长期记忆上限 |
| `tools.maintenance_days_unused` | int | `7` | 过期工具天数阈值 |
| `logging.level` | string | `INFO` | 日志级别 |

## 运行测试

```bash
python test/test_mcp_tools.py        # 工具池 34/34
python test/test_memory.py           # 记忆系统 60/60
python test/test_brain.py            # 大脑基础
python test/test_llm_create_tool.py  # LLM 集成（需 API）
```

## 维护

```bash
# 查看工具池状态
python tools/manage_tools.py

# 执行清理 + LLM 冗余评估
python tools/manage_tools.py --execute --api-key YOUR_KEY
```

## 依赖

- Python 3.10+
- openai
- pydantic

```bash
pip install openai pydantic
```
